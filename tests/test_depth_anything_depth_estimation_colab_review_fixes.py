"""Regression tests for the 2026-10-02 Notebook Review Framework v1 review of
tutorials/depth_anything_depth_estimation_colab.ipynb (DAC-M2, DAC-M3, DAC-m1..m4).

DAC-M1 (one-pass Run all) was fixed by the uv isolated environment on main (78f01a2) and is covered by
tests/test_colab_uv_isolated_environment.py. Everything here runs inside CI's install budget (numpy, pillow,
pytest; no torch): the notebook's own cells are executed with inert stand-ins for the model, and the BYOD cell
with the carried samples module on small synthetic RGB-D records.
"""
# ruff: noqa: E501  -- notebook source fragments and stand-in records are kept on one line

from __future__ import annotations

import importlib.util
import io
import json
import re
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from depth_anything_depth_estimation_pipeline import samples as samples_module

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = json.loads((ROOT / "tutorials" / TEMPLATE["notebook_name"]).read_text(encoding="utf-8"))
CELLS = NOTEBOOK["cells"]


def _src(cell: dict) -> str:
    return "".join(cell["source"]) if isinstance(cell["source"], list) else cell["source"]


def _section(number: int) -> tuple[str, str]:
    for index, cell in enumerate(CELLS):
        if cell["cell_type"] == "markdown" and f"## {number}. " in _src(cell):
            code = next(_src(c) for c in CELLS[index + 1 :] if c["cell_type"] == "code")
            return _src(cell), code
    raise AssertionError(f"no section {number}")


MARKDOWN = "\n".join(_src(c) for c in CELLS if c["cell_type"] == "markdown")


# ---------------------------------------------------------------- DAC-M2: every pass starts from the base


class _FakePipe:
    def __init__(self, adapted: bool):
        self.adapter = {"policy": "unfrozen last 2 blocks + DPT neck and head"} if adapted else None


def _reset_namespace(pipe):
    loads = []

    class FakePipeline:
        @staticmethod
        def from_pretrained(weights_dir):
            loads.append(weights_dir)
            return _FakePipe(adapted=False)

    cleared = []
    torch = types.SimpleNamespace(cuda=types.SimpleNamespace(is_available=lambda: True, empty_cache=lambda: cleared.append(1)))
    ns = {"pipe": pipe, "torch": torch, "DepthAnythingPipeline": FakePipeline, "WEIGHTS_DIR": Path("weights/x")}
    _md, code = _section(5)
    helper = code[: code.index("\nreset_to_pretrained()\n")]
    exec(compile(helper, "<section 5 helper>", "exec"), ns)
    return ns, loads, cleared


def test_reset_to_pretrained_reloads_only_an_adapted_model(capsys) -> None:
    ns, loads, cleared = _reset_namespace(_FakePipe(adapted=True))
    ns["reset_to_pretrained"]()
    assert loads == [Path("weights/x")] and cleared == [1]
    assert ns["pipe"].adapter is None
    assert "reloaded" in capsys.readouterr().out
    untouched = ns["pipe"]
    ns["reset_to_pretrained"]()
    assert ns["pipe"] is untouched and loads == [Path("weights/x")]


@pytest.mark.parametrize("number", [5, 6, 7])
def test_sections_5_to_7_reset_before_they_use_the_model(number: int) -> None:
    _md, code = _section(number)
    body = re.sub(r"def reset_to_pretrained\(\):\n(?:    [^\n]*\n|\n)*", "", code)
    call = body.index("\nreset_to_pretrained()\n")
    uses = [body.find(token) for token in ("pipe.", "prior_baselines(") if token in body]
    assert uses and call < min(uses)


def test_section_6_refuses_to_report_an_adapted_model_as_zero_shot() -> None:
    _md, code = _section(6)
    assert "if zero_shot_test['adapted']:\n    raise RuntimeError(" in code
    assert "'adapted': zero_shot_test['adapted']" in code


def test_section_7_refuses_to_adapt_an_adapted_model() -> None:
    _md, code = _section(7)
    guard = code.index("if pipe.adapter is not None:\n    raise RuntimeError(")
    assert code.index("reset_to_pretrained()") < guard < code.index("adapt_result = pipe.adapt(")


def test_rerun_instructions_name_run_after_and_the_reset() -> None:
    assert "re-run from that cell" not in MARKDOWN
    byod = TEMPLATE["byod"]
    assert "Runtime → Run after" in byod and "reload the pretrained model first" in byod
    section10 = MARKDOWN.split("## 10. Your turn", 1)[1].split("## Interpretation", 1)[0]
    assert "set `LEARNING_RATE = 3e-5`" in section10 and "Runtime → Run after" in section10


# ---------------------------------------------------------------- DAC-M3: the guided layer


def test_guided_layer_elements_present() -> None:
    for marker in (
        "**Who this is for.**",
        "**Input → Model → Output.**",
        "**How to use this notebook.**",
        "**Roadmap:**",
        "## 10. Your turn — change one thing: the learning rate",
        "## Troubleshooting",
        "## Glossary",
        "## Conclusion (your notes)",
    ):
        assert marker in MARKDOWN, marker
    for number in range(4, 10):
        md, _code = _section(number)
        assert "**Predict before running:**" in md.split(f"## {number}. ", 1)[1], number
    assert MARKDOWN.count("<details><summary>Check your reasoning</summary>") >= 7
    glossary = MARKDOWN.split("## Glossary", 1)[1].split("## Conclusion", 1)[0]
    for term in ("Relative inverse depth", "Alignment", "AbsRel", "δ1", "DPT neck and head", "Policy", "Epoch / epoch 0"):
        assert f"- **{term}" in glossary, term
    trouble = MARKDOWN.split("## Troubleshooting", 1)[1].split("## Glossary", 1)[0]
    for topic in ("restart", "download", "memory", "digest", "BYOD is refused"):
        assert topic in trouble, topic


def test_infrastructure_labels_and_collapsed_carried_modules() -> None:
    assert TEMPLATE["infrastructure_labels"] is True
    labelled = [c for c in CELLS if c["cell_type"] == "markdown" and "> **Infrastructure.**" in _src(c)]
    assert len(labelled) == 3  # Sections 1, 2 (the carried modules) and 3
    modules = [c for c in CELLS if c.get("metadata", {}).get("dimer", {}).get("embedded_module")]
    assert len(modules) == 3 and all(c["metadata"]["jupyter"]["source_hidden"] is True for c in modules)


# ---------------------------------------------------------------- DAC-m1: timing claims name their environment


def test_timing_claims_name_their_environment() -> None:
    for stale in ("about four minutes of model time", "The build record measured about 0.4 s", "About ten seconds on CPU"):
        assert stale not in MARKDOWN
    for figure in ("725.3 s", "249.7 s", "92.6 s", "3.2 s"):
        for match in re.finditer(re.escape(figure), MARKDOWN):
            context = MARKDOWN[max(0, match.start() - 400) : match.end() + 100]
            assert re.search(r"Colab|Windows|workstation|pre-flight", context), figure
    assert "(an estimate, not a measurement)" in TEMPLATE["run_all"]
    _md, code = _section(7)
    assert "row['elapsed_s']" in code and "'rough_seconds_per_epoch'" in code


# ---------------------------------------------------------------- DAC-m3: no doubled braces


def test_no_doubled_braces_and_the_enforced_id_pattern() -> None:
    assert "{{" not in MARKDOWN and "}}" not in MARKDOWN
    prerequisites = next(_src(c) for c in CELLS if _src(c).startswith("## Prerequisites"))
    assert "`{id, image, depth, mask}`" in prerequisites
    assert "`[A-Za-z0-9_.:-]{1,64}`" in prerequisites
    assert samples_module._ID_RE.pattern == r"^[A-Za-z0-9_.:-]{1,64}$"


# ---------------------------------------------------------------- DAC-m2: BYOD group count, upload guards, BYOD_PATH


def _record(rid: str, group: str, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    image = Image.fromarray(rng.integers(0, 255, (24, 32, 3), dtype=np.uint8))
    depth = rng.uniform(1.0, 20.0, (24, 32)).astype(np.float32)
    return {"id": rid, "image": image, "depth": depth, "mask": np.ones((24, 32), dtype=bool), "group": group}


def _write_byod_dir(root: Path, n_views: int, n_groups: int) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    rows = ["id,image,depth,mask,group"]
    for k in range(n_views):
        r = _record(f"v{k}", f"g{k % n_groups}", k)
        r["image"].save(root / f"v{k}.png")
        np.save(root / f"v{k}_d.npy", r["depth"])
        np.save(root / f"v{k}_m.npy", r["mask"])
        rows.append(f"v{k},v{k}.png,v{k}_d.npy,v{k}_m.npy,{r['group']}")
    (root / "records.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return root


def _section4_namespace(tmp_path: Path, monkeypatch, *, byod_path: str = "", upload: dict | None = None) -> tuple[dict, str]:
    monkeypatch.chdir(tmp_path)
    ns: dict = {"__name__": "__main__", "np": np, "Path": Path, "os": __import__("os")}
    ns.update({k: getattr(samples_module, k) for k in dir(samples_module) if not k.startswith("__")})
    if upload is not None:
        colab = types.ModuleType("google.colab")
        colab.files = types.SimpleNamespace(upload=lambda: dict(upload))
        google = types.ModuleType("google")
        google.colab = colab
        monkeypatch.setitem(sys.modules, "google", google)
        monkeypatch.setitem(sys.modules, "google.colab", colab)
    _md, code = _section(4)
    code = code.replace("USE_BYOD = False", "USE_BYOD = True").replace('BYOD_PATH = ""', f"BYOD_PATH = {byod_path!r}")
    return ns, code


def test_byod_path_directory_runs_section_4_without_google_colab(tmp_path, monkeypatch, capsys) -> None:
    data = _write_byod_dir(tmp_path / "set", n_views=8, n_groups=4)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    ns, code = _section4_namespace(tmp_path, monkeypatch, byod_path=str(data))
    exec(compile(code, "<section 4>", "exec"), ns)
    assert {k: len(v) for k, v in ns["splits"].items()} == {"test": 2, "validation": 2, "train": 4}
    assert ns["data_source"] == "BYOD (set)"
    assert (tmp_path / "outputs" / "depth_anything_depth_estimation_train.csv").is_file()
    assert "'probe': 'too small'" in capsys.readouterr().out


@pytest.mark.parametrize("n_groups", [1, 2, 3])
def test_byod_with_fewer_than_four_groups_is_refused_naming_the_count(tmp_path, monkeypatch, n_groups: int) -> None:
    data = _write_byod_dir(tmp_path / "set", n_views=8, n_groups=n_groups)
    ns, code = _section4_namespace(tmp_path, monkeypatch, byod_path=str(data))
    with pytest.raises(ValueError, match=rf"has {n_groups} distinct group\(s\).*needs at least 4"):
        exec(compile(code, "<section 4>", "exec"), ns)


def test_byod_with_too_few_training_views_names_the_training_split(tmp_path, monkeypatch) -> None:
    data = _write_byod_dir(tmp_path / "set", n_views=4, n_groups=4)
    ns, code = _section4_namespace(tmp_path, monkeypatch, byod_path=str(data))
    with pytest.raises(ValueError, match="training records; at least 4 are required"):
        exec(compile(code, "<section 4>", "exec"), ns)


@pytest.mark.parametrize("upload", [{}, {"a.zip": b"", "b.zip": b""}])
def test_empty_or_multiple_upload_asks_for_one_file(tmp_path, monkeypatch, upload) -> None:
    ns, code = _section4_namespace(tmp_path, monkeypatch, upload=upload)
    with pytest.raises(ValueError, match=rf"Upload exactly one \.zip file \(received {len(upload)}\).*BYOD_PATH"):
        exec(compile(code, "<section 4>", "exec"), ns)


def test_uploaded_zip_still_goes_through_the_loader(tmp_path, monkeypatch) -> None:
    data = _write_byod_dir(tmp_path / "set", n_views=8, n_groups=4)
    buf = io.BytesIO()
    import zipfile

    with zipfile.ZipFile(buf, "w") as zf:
        for f in data.iterdir():
            zf.writestr(f.name, f.read_bytes())
    ns, code = _section4_namespace(tmp_path, monkeypatch, upload={"mine.zip": buf.getvalue()})
    exec(compile(code, "<section 4>", "exec"), ns)
    assert ns["data_source"] == "BYOD (mine.zip)" and len(ns["train_records"]) == 4


def test_missing_byod_path_is_refused(tmp_path, monkeypatch) -> None:
    ns, code = _section4_namespace(tmp_path, monkeypatch, byod_path=str(tmp_path / "nope"))
    with pytest.raises(FileNotFoundError, match="does not exist"):
        exec(compile(code, "<section 4>", "exec"), ns)


# ---------------------------------------------------------------- DAC-m4: the learner's own run is summarised


def _metrics(abs_rel: float, per_view: list[float]) -> dict:
    return {
        "abs_rel": abs_rel,
        "delta1": 0.7,
        "n": len(per_view),
        "per_domain": {"indoors": {"abs_rel": abs_rel, "delta1": 0.7, "n": len(per_view)}},
        "per_image": [{"id": f"t{k}", "abs_rel": v} for k, v in enumerate(per_view)],
    }


def test_section_8_prints_a_run_summary_and_keeps_a_run_history(tmp_path, monkeypatch, capsys) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "outputs").mkdir()
    zero = _metrics(0.2, [0.1, 0.2, 0.3, 0.2])
    adapted = _metrics(0.19, [0.09, 0.2, 0.31, 0.15])

    class Pipe:
        def evaluate(self, records):
            return adapted

    ns = {
        "json": json,
        "pipe": Pipe(),
        "test_records": [], "val_records": [],
        "priors": {"constant_prior": {"abs_rel": 0.3, "delta1": 0.5}, "vertical_gradient_prior": {"abs_rel": 0.25, "delta1": 0.55}},
        "zero_shot_test": zero,
        "adapt_result": {"policy": "unfrozen last 2 blocks + DPT neck and head", "best_epoch": 5, "history": [{"val": {"abs_rel": 0.2426}}]},
        "MODEL_ID": "m", "MODEL_REVISION": "r", "MODEL_KEY": "k", "data_source": "d", "dataset_manifests": {}, "disjoint": {},
        "summary": {}, "report": {}, "adapt_seconds": 1.0, "USE_BYOD": False,
        "HEAD_EPOCHS": 2, "EPOCHS": 3, "LEARNING_RATE": 1e-5, "TRAINABLE_BLOCKS": 2,
    }
    _md, code = _section(8)
    exec(compile(code, "<section 8>", "exec"), ns)
    summary = ns["run_summary"]
    assert summary["selected_policy"].startswith("unfrozen") and summary["delta_abs_rel"] == -0.01
    assert (summary["test_views_improved"], summary["test_views_worsened"], summary["test_views_unchanged"]) == (2, 1, 1)
    ns["LEARNING_RATE"] = 3e-5
    exec(compile(code, "<section 8>", "exec"), ns)
    assert [h["learning_rate"] for h in ns["run_history"]] == [1e-5, 3e-5]
    saved = json.loads((tmp_path / "outputs" / "depth_anything_depth_estimation_evaluation_report.json").read_text(encoding="utf-8"))
    assert saved["run_summary"] == summary
    assert "'run_summary'" in capsys.readouterr().out
    _md9, code9 = _section(9)
    assert code9.rstrip().endswith("print({'run_summary': run_summary})")


def test_closing_phrases_build_record_figures_as_reference_answers() -> None:
    closing = MARKDOWN.split("## Interpretation and limits", 1)[1].split("## Troubleshooting", 1)[0]
    head = closing.split("<details>", 1)[0]
    assert "run summary" in head and "0.167" not in head and "0.1669" not in head
    assert "and watch the neck-and-head policy win" not in MARKDOWN
    experiments = closing.split("**Optional experiments", 1)[1]
    assert experiments.count("<details><summary>Reference answer</summary>") >= 3


# ---------------------------------------------------------------- the validator enforces the above


def test_validator_rejects_a_section_without_the_reset_or_a_prediction() -> None:
    import copy

    validator = _load("validate_release_assets")
    path = ROOT / "tutorials" / TEMPLATE["notebook_name"]
    validator._validate_guided_layer(path, NOTEBOOK)
    broken = copy.deepcopy(NOTEBOOK)
    for index, cell in enumerate(broken["cells"]):
        if cell["cell_type"] == "markdown" and "## 6. " in _src(cell):
            code = broken["cells"][index + 1]
            code["source"] = _src(code).replace("reset_to_pretrained()\n", "", 1)
    with pytest.raises(validator.ValidationError, match="Section 6 must call reset_to_pretrained"):
        validator._validate_guided_layer(path, broken)
    broken = copy.deepcopy(NOTEBOOK)
    for cell in broken["cells"]:
        if cell["cell_type"] == "markdown" and "## 8. " in _src(cell):
            cell["source"] = _src(cell).replace("**Predict before running:**", "Predict:")
    with pytest.raises(validator.ValidationError, match="Section 8 must ask for a prediction"):
        validator._validate_guided_layer(path, broken)
