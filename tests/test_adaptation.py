"""Offline tests for the depth-labelled dataset contract, the pinned DIODE record table and reader, the seeded
scan-level split, the aligned AbsRel / δ1 metrics with the two priors, BYOD loaders (directory and zip), the
injected-runner evaluation path, artifact-manifest rejections and adapt() argument validation. No model
library is loaded by the fake pipeline; the corpus is served through an injected fetcher of small synthetic
PNG / npy triples."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import zipfile

import numpy as np
import pytest
from PIL import Image

from depth_anything_depth_estimation_pipeline import (
    ARTIFACT_FORMAT,
    EVAL_DEPTH_RANGE_M,
    MODEL_ID,
    MODEL_REVISION,
    POLICY_ZERO_SHOT,
    SAMPLE_RECORDS,
    SAMPLE_SPLIT,
    TRANSFORMER_BLOCKS,
    DepthAnythingPipeline,
    align_inverse_depth,
    aligned_abs_rel,
    aligned_delta1,
    build_sample_dataset,
    check_split_disjoint,
    constant_prior,
    dataset_digest,
    depth_metrics,
    fetch_sample_dataset,
    image_digest,
    load_byod_dataset,
    prior_baselines,
    scan_summary,
    split_dataset,
    validate_dataset,
    vertical_gradient_prediction,
    write_dataset_csv,
)
from depth_anything_depth_estimation_pipeline import pipeline as pl
from depth_anything_depth_estimation_pipeline import samples as sm
from depth_anything_depth_estimation_pipeline.samples import fetch_corpus, read_corpus

H, W = 24, 32


def _scene(seed: int) -> tuple[Image.Image, np.ndarray, np.ndarray]:
    """A synthetic scene: depth = a tilted plane (nearer at the bottom) plus a box; shading follows depth."""
    rng = np.random.default_rng(seed)
    rows = np.linspace(6.0, 1.5, H)[:, None]
    depth = np.repeat(rows, W, axis=1) + rng.uniform(0, 0.05, size=(H, W))
    depth[8:16, 10:22] = 1.0 + 0.1 * (seed % 3)
    mask = np.ones((H, W), dtype=bool)
    mask[:2, :] = False
    shade = (255 * (1.0 - (depth - depth.min()) / (depth.max() - depth.min()))).astype(np.uint8)
    image = Image.fromarray(np.stack([shade, shade // 2 + 60, np.full_like(shade, 90)], axis=-1))
    return image, depth.astype(np.float32), mask


def _records(n=12):
    return [
        {
            "id": f"r{i:02d}",
            "image": img,
            "depth": depth,
            "mask": mask,
            "scan": f"scan-{i // 2}",
            "domain": "indoors" if i % 2 else "outdoor",
        }
        for i, (img, depth, mask) in ((i, _scene(i)) for i in range(n))
    ]


def _npy(array):
    buffer = io.BytesIO()
    np.save(buffer, array)
    return buffer.getvalue()


def _png(image):
    buffer = io.BytesIO()
    image.save(buffer, "PNG")
    return buffer.getvalue()


def _fake_runner(image):
    """A fake backend: inverse depth rising towards the bottom of the frame (the vertical-gradient prior)."""
    return vertical_gradient_prediction(image.height, image.width) + 0.5


def _pipeline_without_model():
    return DepthAnythingPipeline(_fake_runner, "cpu")


def _pin(monkeypatch, views_per_scan=2, scans_per_domain=4):
    """Replace the pinned table with synthetic scenes via an injected fetcher (files keyed by URL)."""
    files = {}
    table = []
    for d, domain in enumerate(("indoors", "outdoor")):
        for s in range(scans_per_domain):
            scene, scan = f"scene_{d:02d}{s:02d}", f"scan_{d:02d}{s:02d}"
            for v in range(views_per_scan):
                stem = f"{scene}_{scan}_{v:03d}"
                image, depth, mask = _scene(100 * d + 10 * s + v)
                blobs = {
                    ".png": _png(image),
                    "_depth.npy": _npy(depth[..., None]),
                    "_depth_mask.npy": _npy(mask.astype(np.float32)),
                }
                for suffix, data in blobs.items():
                    files[sm.file_url(domain, scene, scan, stem, suffix)] = data
                row = [f"view-{len(table):03d}", domain, scene, scan, stem]
                for suffix in sm.FILE_SUFFIXES:
                    row += [len(blobs[suffix]), hashlib.sha256(blobs[suffix]).hexdigest()]
                table.append(tuple(row))
    monkeypatch.setattr(sm, "SAMPLE_RECORDS", tuple(table))
    return files


# --- pinned table and reader --------------------------------------------------------------------------


def test_pinned_record_table_is_complete_and_traceable():
    assert len(SAMPLE_RECORDS) == 40
    assert all(len(r) == 11 and r[1] in sm.DOMAINS for r in SAMPLE_RECORDS)
    assert all(len(r[k]) == 64 for r in SAMPLE_RECORDS for k in (6, 8, 10))
    assert all(
        r[5] > 100_000 and r[7] == 3_145_856 and r[9] in (3_145_856, 6_291_584) for r in SAMPLE_RECORDS
    )
    assert len({r[0] for r in SAMPLE_RECORDS}) == 40 and len({r[4] for r in SAMPLE_RECORDS}) == 40
    scans = {}
    for r in SAMPLE_RECORDS:
        scans.setdefault((r[1], r[3]), 0)
        scans[(r[1], r[3])] += 1
    assert len(scans) == 20 and set(scans.values()) == {2}  # two views of every scan
    assert sum(1 for d, _ in scans if d == "indoors") == 10
    assert sm.file_url(*SAMPLE_RECORDS[0][1:5], ".png").startswith(sm.CORPUS_BASE_URL)
    assert sm.CORPUS_COMMIT in sm.CORPUS_BASE_URL and len(sm.CORPUS_COMMIT) == 40
    assert sum(r[5] + r[7] + r[9] for r in SAMPLE_RECORDS) == sm.CORPUS_BYTES
    assert sum(SAMPLE_SPLIT.values()) == 10 and set(SAMPLE_SPLIT) == {"train", "validation", "test"}


def test_fetch_corpus_verifies_each_file_and_caches(tmp_path, monkeypatch, forbid_model_imports):
    files = _pin(monkeypatch)
    calls = []

    def fetcher(url):
        calls.append(url)
        return files[url]

    corpus = fetch_corpus(cache_dir=tmp_path, fetcher=fetcher)
    assert len(corpus) == 16 and all(set(v) == set(sm.FILE_SUFFIXES) for v in corpus.values())
    assert fetch_corpus(cache_dir=tmp_path, fetcher=fetcher) == corpus and len(calls) == 48
    assert all(u.startswith(sm.CORPUS_BASE_URL) for u in calls)
    with pytest.raises(ValueError, match="pinned"):
        fetch_corpus(cache_dir=tmp_path / "other", fetcher=lambda url: b"tampered")


def test_read_corpus_decodes_records_with_provenance(tmp_path, monkeypatch, forbid_model_imports):
    files = _pin(monkeypatch)
    corpus = read_corpus(fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: files[url]))
    first = corpus[0]
    assert first["id"] == "view-000" and first["domain"] == "indoors" and first["scan"] == "scan_0000"
    assert isinstance(first["image"], Image.Image) and first["image"].size == (W, H)
    assert first["depth"].shape == (H, W) and first["depth"].dtype == np.float32
    assert first["mask"].dtype == bool and not first["mask"][0, 0] and first["mask"][10, 10]
    assert first["source_url"].endswith(".png")
    with pytest.raises(ValueError, match="missing"):
        read_corpus({})


def test_sample_split_is_by_scan_seeded_and_disjoint(tmp_path, monkeypatch, forbid_model_imports):
    files = _pin(monkeypatch)
    corpus = read_corpus(fetch_corpus(cache_dir=tmp_path, fetcher=lambda url: files[url]))
    sizes = {"train": 2, "validation": 1, "test": 1}
    splits = build_sample_dataset(corpus, seed=42, sizes=sizes)
    assert check_split_disjoint(splits) == {"train": 8, "validation": 4, "test": 4}
    summary = scan_summary(splits)
    assert summary["train"]["scans"] == 4 and summary["test"]["domains"] == {"indoors": 2, "outdoor": 2}
    assert [r["id"] for r in splits["train"]][:2] == ["train-000", "train-001"]
    assert (
        build_sample_dataset(corpus, seed=42, sizes=sizes)["test"][0]["source_id"]
        == splits["test"][0]["source_id"]
    )
    assert (
        build_sample_dataset(corpus, seed=7, sizes=sizes)["test"][0]["source_id"]
        != splits["test"][0]["source_id"]
    )
    with pytest.raises(ValueError, match="only"):
        build_sample_dataset(corpus, sizes={"train": 4, "validation": 1, "test": 1})
    leaky = {"train": splits["train"], "test": [splits["train"][0]]}
    with pytest.raises(ValueError, match="appears in both"):
        check_split_disjoint(leaky)
    other_view = next(r for r in splits["train"][1:] if r["scan"] == splits["train"][0]["scan"])
    with pytest.raises(ValueError, match="views in both"):
        check_split_disjoint({"train": [splits["train"][0]], "test": [other_view]})


def test_fetch_sample_dataset_end_to_end_with_injected_fetcher(tmp_path, monkeypatch, forbid_model_imports):
    files = _pin(monkeypatch)
    splits = fetch_sample_dataset(
        cache_dir=tmp_path, fetcher=lambda url: files[url], sizes={"train": 2, "validation": 1, "test": 1}
    )
    manifests = {name: validate_dataset(part) for name, part in splits.items()}
    assert manifests["train"]["n_records"] == 8 and manifests["train"]["scans"] == 4
    assert manifests["train"]["domain_counts"] == {"indoors": 4, "outdoor": 4}
    assert len({m["digest"] for m in manifests.values()}) == 3


# --- dataset contract ----------------------------------------------------------------------------------


def test_validate_dataset_reports_and_rejects(tmp_path, forbid_model_imports):
    records = _records()
    report = validate_dataset(records)
    assert report["n_records"] == 12 and report["scans"] == 6 and report["image_side"] == {"min": W, "max": W}
    assert 0.9 < report["valid_fraction"]["min"] <= report["valid_fraction"]["max"] <= 1.0
    assert 1.0 < report["median_depth_m"]["min"] < 6.0 and report["model_id"] == MODEL_ID
    assert len(report["digest"]) == 64 and report["digest"] == dataset_digest(records)
    image, depth, _mask = _scene(99)
    np.save(tmp_path / "d.npy", depth)
    image.save(tmp_path / "i.png")
    from_paths = validate_dataset(
        [{"id": "p", "image": tmp_path / "i.png", "depth": tmp_path / "d.npy"}, *records[:3]]
    )
    assert from_paths["records"][0]["mask"].all() and from_paths["records"][0]["depth"].shape == (H, W)
    bad = [
        ({**records[0], "id": "bad id"}, "id must match"),
        ({**records[0], "image": "nope.png"}, "image file not found"),
        ({**records[0], "image": Image.new("RGB", (pl.MAX_IMAGE_SIDE + 1, 20))}, "image side"),
        ({**records[0], "image": Image.new("RGB", (200, 20))}, "aspect ratio"),
        ({**records[0], "depth": records[0]["depth"][:-1]}, "depth shape"),
        ({**records[0], "depth": records[0]["depth"].astype(str)}, "must be numeric"),
        ({**records[0], "mask": records[0]["mask"][:, :-1]}, "mask shape"),
        ({**records[0], "mask": np.zeros((H, W), dtype=bool)}, "valid depth"),
        ({**records[0], "depth": records[0]["depth"] * 1000}, "MAX_DEPTH_M"),
        ({"id": "x", "image": records[0]["image"]}, "missing 'depth'"),
    ]
    for record, message in bad:
        with pytest.raises(ValueError, match=message):
            validate_dataset([record, *records[1:]])
    with pytest.raises(ValueError, match="duplicate id"):
        validate_dataset([records[0], records[0], *records[2:]])
    with pytest.raises(ValueError, match="4..2000"):
        validate_dataset(records[:3])
    with pytest.raises(ValueError, match="list of"):
        validate_dataset({"id": "x"})


def test_split_dataset_groups_by_scan_deduplicates_and_is_seeded(forbid_model_imports):
    records = _records(20)
    duplicated = [*records, {**records[0], "id": "dup", "scan": "other"}]
    splits = split_dataset(duplicated, val_fraction=0.2, test_fraction=0.2, seed=1)
    assert sum(len(part) for part in splits.values()) == 20
    check_split_disjoint(splits)
    assert {
        r["id"] for r in split_dataset(duplicated, val_fraction=0.2, test_fraction=0.2, seed=1)["test"]
    } == {r["id"] for r in splits["test"]}
    assert len(splits["test"]) == 4 and len(splits["validation"]) == 4  # whole scans of two views
    with pytest.raises(ValueError, match="fractions"):
        split_dataset(records, val_fraction=0.5, test_fraction=0.6)


# --- metrics and priors ---------------------------------------------------------------------------------


def test_alignment_metrics_and_priors(forbid_model_imports):
    _image, depth, mask = _scene(3)
    exact = 2.0 / depth + 0.3  # an affine image of the true inverse depth
    estimate, ref, fit = align_inverse_depth(exact, depth, mask)
    assert (
        np.allclose(estimate, ref, rtol=1e-4)
        and abs(fit["scale"] - 0.5) < 1e-6
        and fit["n_valid"] == mask.sum()
    )
    assert aligned_abs_rel(exact, depth, mask) < 1e-5 and aligned_delta1(exact, depth, mask) == 1.0
    noisy = exact + np.random.default_rng(0).normal(0, 0.05, size=exact.shape)
    assert 0 < aligned_abs_rel(noisy, depth, mask) < 0.2
    with pytest.raises(ValueError, match="shape mismatch"):
        aligned_abs_rel(exact[:-1], depth, mask)
    with pytest.raises(ValueError, match="valid reference pixels"):
        aligned_abs_rel(exact, depth, np.zeros_like(mask))
    far = depth.copy()
    far[5, 5] = EVAL_DEPTH_RANGE_M[1] * 2  # outside the scored range: ignored, not an unbounded error
    assert aligned_abs_rel(exact, far, mask) < 1e-5
    records = _records(4)
    metrics = depth_metrics([2.0 / r["depth"] + 0.3 for r in records], records)
    assert metrics["n"] == 4 and metrics["abs_rel"] < 1e-5 and metrics["delta1"] == 1.0
    assert set(metrics["per_domain"]) == {"indoors", "outdoor"} and len(metrics["per_image"]) == 4
    constant = constant_prior(records[0])
    assert constant["abs_rel"] > 0.2 and 0 < constant["delta1"] < 1 and constant["constant_m"] > 1
    priors = prior_baselines(records)
    assert priors["constant_prior"]["n"] == 4 and priors["vertical_gradient_prior"]["n"] == 4
    assert (
        priors["vertical_gradient_prior"]["abs_rel"] < priors["constant_prior"]["abs_rel"]
    )  # planes are tilted
    with pytest.raises(ValueError, match="predictions for"):
        depth_metrics([], records)


# --- BYOD ------------------------------------------------------------------------------------------------


def test_byod_directory_and_zip_round_trip_and_rejections(tmp_path, forbid_model_imports):
    records = _records(4)
    folder = tmp_path / "byod"
    folder.mkdir()
    with open(folder / "records.csv", "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "image", "depth", "mask", "group"])
        writer.writeheader()
        for record in records:
            record["image"].save(folder / f"{record['id']}.png")
            np.save(folder / f"{record['id']}_depth.npy", record["depth"])
            np.save(folder / f"{record['id']}_mask.npy", record["mask"])
            writer.writerow(
                {
                    "id": record["id"],
                    "image": f"{record['id']}.png",
                    "depth": f"{record['id']}_depth.npy",
                    "mask": f"{record['id']}_mask.npy",
                    "group": record["scan"],
                }
            )
    loaded = load_byod_dataset(folder)
    assert [r["id"] for r in loaded] == [r["id"] for r in records] and loaded[0]["group"] == "scan-0"
    assert (
        np.array_equal(loaded[0]["depth"], records[0]["depth"]) and validate_dataset(loaded)["n_records"] == 4
    )
    archive = tmp_path / "byod.zip"
    with zipfile.ZipFile(archive, "w") as zf:
        for path in folder.iterdir():
            zf.write(path, f"inner/{path.name}")
    from_zip = load_byod_dataset(archive)
    assert image_digest(from_zip[1]["image"]) == image_digest(records[1]["image"])
    assert not list(tmp_path.glob("inner"))  # decoded from the archive, never extracted
    exported = write_dataset_csv(records, tmp_path / "out" / "train.csv")
    rows = list(csv.DictReader(exported.read_text(encoding="utf-8").splitlines()))
    assert rows[0]["group"] == "scan-0" and rows[0]["depth"].endswith("_depth.npy")
    (tmp_path / "empty.zip").write_bytes(b"")
    with pytest.raises((ValueError, zipfile.BadZipFile)):
        load_byod_dataset(tmp_path / "empty.zip")
    with pytest.raises(ValueError, match="records.csv"):
        load_byod_dataset(tmp_path / "missing")


# --- injected-runner evaluation, adapt() guards, artifact refusals ---------------------------------------


def test_evaluate_through_the_injected_runner(forbid_model_imports):
    pipe = _pipeline_without_model()
    metrics = pipe.evaluate(_records(4))
    assert metrics["n"] == 4 and metrics["policy"] == POLICY_ZERO_SHOT and metrics["adapted"] is False
    assert metrics["verdict"] == "measured-small-sample" and 0 < metrics["abs_rel"] < 1
    assert metrics["abs_rel"] == pytest.approx(
        prior_baselines(_records(4))["vertical_gradient_prior"]["abs_rel"]
    )
    with pytest.raises(ValueError, match="1..2000"):
        pipe.evaluate([])


def test_adapt_and_artifacts_need_a_loaded_model(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    with pytest.raises(ValueError, match="head_epochs"):
        pipe.adapt(_records(), head_epochs=-1)
    with pytest.raises(ValueError, match="epochs"):
        pipe.adapt(_records(), epochs=21)
    with pytest.raises(ValueError, match="lr"):
        pipe.adapt(_records(), lr=1.0)
    with pytest.raises(ValueError, match="trainable_blocks"):
        pipe.adapt(_records(), trainable_blocks=TRANSFORMER_BLOCKS + 1)
    with pytest.raises(ValueError, match="from_pretrained"):
        pipe.adapt(_records())
    with pytest.raises(ValueError, match="call adapt"):
        pipe.save_artifact(tmp_path)


def test_load_artifact_rejects_bad_manifests_before_touching_weights(tmp_path, forbid_model_imports):
    pipe = _pipeline_without_model()
    manifest = {
        "format": ARTIFACT_FORMAT,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": pl.WEIGHT_SHA256},
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["head.conv1.weight", "neck.fusion_stage.layers.0.projection.weight"],
        "adapter": {"policy": pl.POLICY_FROZEN, "trainable_blocks": 2},
    }
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps({**manifest, "format": "other"}))
    with pytest.raises(ValueError, match="artifact format"):
        pipe.load_artifact(tmp_path)
    bad_base = {**manifest, "base_model": {**manifest["base_model"], "weight_sha256": "0" * 64}}
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(bad_base))
    with pytest.raises(ValueError, match="different base model"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)
    (tmp_path / pl.ARTIFACT_WEIGHTS_NAME).write_bytes(b"x")
    with pytest.raises(ValueError, match="digest or size mismatch"):
        pipe.load_artifact(tmp_path)


def test_load_artifact_refuses_unsupported_versions_extra_files_traversal_and_policies(
    tmp_path, forbid_model_imports
):
    pipe = _pipeline_without_model()
    good = {
        "format": ARTIFACT_FORMAT,
        "format_version": pl.ARTIFACT_FORMAT_VERSION,
        "base_model": {"id": MODEL_ID, "revision": MODEL_REVISION, "weight_sha256": pl.WEIGHT_SHA256},
        "files": [{"path": pl.ARTIFACT_WEIGHTS_NAME, "bytes": 1, "sha256": "0" * 64}],
        "tensors": ["head.conv1.weight"],
        "adapter": {"policy": pl.POLICY_FROZEN, "trainable_blocks": 2},
    }

    def write(manifest):
        (tmp_path / pl.ARTIFACT_MANIFEST_NAME).write_text(json.dumps(manifest))

    write({**good, "format_version": "0.9"})
    with pytest.raises(ValueError, match="format_version"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": good["files"] * 2})
    with pytest.raises(ValueError, match="exactly one file"):
        pipe.load_artifact(tmp_path)
    write({**good, "files": [{**good["files"][0], "path": "../" + pl.ARTIFACT_WEIGHTS_NAME}]})
    with pytest.raises(ValueError, match="must name exactly|inside the artifact directory"):
        pipe.load_artifact(tmp_path)
    write({**good, "base_model": {**good["base_model"], "weight_file": "pytorch_model.bin"}})
    with pytest.raises(ValueError, match="different base weight file"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {**good["adapter"], "policy": "something else"}})
    with pytest.raises(ValueError, match="not a canonical policy"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {"policy": pl.POLICY_UNFROZEN.format(k=3), "trainable_blocks": 2}})
    with pytest.raises(ValueError, match="not a canonical policy"):
        pipe.load_artifact(tmp_path)
    write({**good, "adapter": {**good["adapter"], "trainable_blocks": 99}})
    with pytest.raises(ValueError, match="trainable_blocks"):
        pipe.load_artifact(tmp_path)
    write(good)  # every manifest check passes; the weights file is still missing, and no model was imported
    with pytest.raises(FileNotFoundError, match="artifact weights missing"):
        pipe.load_artifact(tmp_path)


def test_targets_reject_a_sparse_mask():
    import numpy as np

    torch = pytest.importorskip("torch")

    record = {
        "id": "sparse",
        "depth": np.ones((8, 8), dtype=np.float32),
        "mask": np.zeros((8, 8), dtype=np.float32),
    }
    record["mask"][0, 0] = 1.0
    with pytest.raises(ValueError, match="valid pixels"):
        pl.DepthAnythingPipeline._targets(record, (8, 8), "cpu")
    record["mask"][0, 1] = 1.0
    target, valid = pl.DepthAnythingPipeline._targets(record, (8, 8), "cpu")
    assert int(valid.sum()) == 2 and bool(torch.isfinite(target).all())


def test_byod_requires_a_group_on_every_row(tmp_path, forbid_model_imports):
    records = _records(4)
    folder = tmp_path / "ungrouped"
    folder.mkdir()
    rows = []
    for record in records:
        record["image"].save(folder / f"{record['id']}.png")
        np.save(folder / f"{record['id']}_depth.npy", record["depth"])
        np.save(folder / f"{record['id']}_mask.npy", record["mask"])
        rows.append(
            {
                "id": record["id"],
                "image": f"{record['id']}.png",
                "depth": f"{record['id']}_depth.npy",
                "mask": f"{record['id']}_mask.npy",
                "group": "",
            }
        )

    def write_rows(table):
        with open(folder / "records.csv", "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["id", "image", "depth", "mask", "group"])
            writer.writeheader()
            writer.writerows(table)

    write_rows(rows)
    with pytest.raises(ValueError, match="has no `group`"):
        load_byod_dataset(folder)
    assert len(load_byod_dataset(folder, require_group=False)) == 4  # the explicit opt-out
    grouped = [{**row, "group": "scan-" + row["id"][-1]} for row in rows]
    write_rows(grouped)
    assert {r["group"] for r in load_byod_dataset(folder)} == {"scan-" + r["id"][-1] for r in records}
    write_rows(grouped + [grouped[0]])
    with pytest.raises(ValueError, match="more than once"):
        load_byod_dataset(folder)


def test_byod_refuses_escaping_paths_and_duplicate_zip_basenames(tmp_path, forbid_model_imports):
    records = _records(4)
    folder = tmp_path / "byod"
    folder.mkdir()
    records[0]["image"].save(tmp_path / "outside.png")
    (folder / "records.csv").write_text(
        "id,image,depth,mask,group\nr0,../outside.png,d.npy,m.npy,g0\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="leaves the dataset directory"):
        load_byod_dataset(folder)
    duplicate = tmp_path / "dup.zip"
    with zipfile.ZipFile(duplicate, "w") as zf:
        zf.writestr("a/records.csv", "id\n")
        zf.writestr("a/x.png", b"x")
        zf.writestr("b/x.png", b"y")
    with pytest.raises(ValueError, match="more than one member named"):
        load_byod_dataset(duplicate)
