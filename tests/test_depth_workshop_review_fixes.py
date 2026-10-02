"""Regression tests for the 2026-10-02 review of the depth-estimation WORKSHOP notebook (DEP-M1, m1..m7).

The notebook's carried worker (``workshop.py``) and carried reference modules are materialised from the
notebook and exercised on CPU with synthetic RGB-D data and stand-in models (a deterministic function of
the pixels).
These are interface and contract tests within CI's install budget (numpy, pillow, pytest); they are not
pretrained-model or hosted-runtime evidence.
"""

from __future__ import annotations

import ast
import importlib.metadata
import importlib.util
import json
import re
import sys
import time
import types
import zipfile
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_MultiModel_Depth_Estimation_Workshop.ipynb"
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = {cell["id"]: cell for cell in NB["cells"]}
SOURCE = {cid: "".join(cell["source"]) for cid, cell in CELLS.items()}
MARKDOWN = "\n".join(s for cid, s in SOURCE.items() if CELLS[cid]["cell_type"] == "markdown")


def _carried() -> dict[str, str]:
    cell = next(s for s in SOURCE.values() if "CARRIED_FILES = " in s)
    body = ast.parse(cell).body
    node = next(n for n in body if isinstance(n, ast.Assign) and n.targets[0].id == "CARRIED_FILES")
    return ast.literal_eval(node.value)


CARRIED = _carried()
WORKER = CARRIED["workshop.py"]


class StandIn:
    """Inverse-depth-like output that depends on the pixels, so a brightness change changes it."""

    def __init__(self, metric: bool = False):
        self.metric = metric

    def predict(self, image):
        values = np.asarray(image.convert("RGB"), dtype=np.float32).mean(axis=2) / 255.0
        ramp = np.linspace(0.2, 1.0, values.shape[0], dtype=np.float32)[:, None]
        inverse = ramp + 0.05 * values
        return {"depth": (1.0 / inverse) if self.metric else inverse}

    @classmethod
    def from_artifact(cls, *args, **kwargs):
        return cls()

    @classmethod
    def from_pretrained(cls, *args, **kwargs):
        return cls(metric=True)


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    carried = tmp_path_factory.mktemp("carried")
    for name, text in CARRIED.items():
        (carried / name).parent.mkdir(parents=True, exist_ok=True)
        (carried / name).write_bytes(text.encode("utf-8"))
    patch = pytest.MonkeyPatch()
    patch.syspath_prepend(str(carried))
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(
        synchronize=lambda *a: None, is_available=lambda: False, empty_cache=lambda: None
    )
    patch.setitem(sys.modules, "torch", fake_torch)
    for name in [m for m in sys.modules if m.split(".")[0] in ("depth_reference", "zoe_reference")]:
        patch.delitem(sys.modules, name)
    spec = importlib.util.spec_from_file_location("dep_review_workshop_worker", carried / "workshop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    patch.setattr(module, "device", lambda: "cpu")
    real_version = importlib.metadata.version
    patch.setattr(
        importlib.metadata, "version", lambda n: real_version(n) if n in ("numpy", "Pillow") else "stand-in"
    )
    yield module
    patch.undo()


def _make_byod(directory: Path) -> list[dict]:
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    records = []
    for role, count in (("train", 6), ("validation", 2), ("test", 4)):
        for k in range(count):
            rid = f"{role[:2]}{k:02d}"
            Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(directory / f"{rid}.png")
            np.save(directory / f"{rid}_depth.npy", (1 + 4 * rng.random((48, 64))).astype(np.float32))
            np.save(directory / f"{rid}_mask.npy", np.ones((48, 64), dtype=bool))
            records.append(
                {
                    "id": rid,
                    "image": f"{rid}.png",
                    "depth": f"{rid}_depth.npy",
                    "mask": f"{rid}_mask.npy",
                    "group_id": f"{role}-g{k % 2}",
                    "role": role,
                }
            )
    (directory / "manifest.json").write_text(json.dumps({"units": "metres", "records": records}))
    return records


def _low_coverage(directory: Path, records: list[dict]) -> None:
    mask = np.zeros((48, 64), dtype=bool)
    mask[0, :61] = True  # 61 of 3072 pixels: below the 5% coverage rule
    np.save(directory / records[7]["mask"], mask)


def _missing_depth(directory: Path, records: list[dict]) -> None:
    (directory / records[5]["depth"]).unlink()


def _float64_depth(directory: Path, records: list[dict]) -> None:
    path = directory / records[3]["depth"]
    np.save(path, np.load(path).astype(np.float64))


def _set(index: int, key: str, value):
    def mutate(directory: Path, records: list[dict]) -> None:
        if value is None:
            records[index].pop(key)
        else:
            records[index][key] = value
        (directory / "manifest.json").write_text(json.dumps({"units": "metres", "records": records}))

    return mutate


# ---- DEP-M1: BYOD refusals name the record/file and run before any model stage ---------------------------


def test_valid_byod_passes_load_and_prepare(ws, tmp_path):
    _make_byod(tmp_path / "byod")
    assert len(ws.load_byod(tmp_path / "byod")) == 12
    ws.prepare(tmp_path / "run", str(tmp_path / "byod"))
    prepared = json.loads((tmp_path / "run/outputs/prepare.json").read_text())
    assert prepared["counts"] == {"train": 6, "validation": 2, "test": 4}
    assert prepared["private_byod"] is True


@pytest.mark.parametrize(
    ("mutate", "needles"),
    [
        (_low_coverage, ["'va01'", "5%"]),
        (_missing_depth, ["'tr05'", "tr05_depth.npy", "not found"]),
        (_float64_depth, ["'tr03'", "tr03_depth.npy", "float32", "float64"]),
        (_set(9, "group_id", None), ["'te01'", "group_id"]),
        (_set(11, "group_id", "train-g0"), ["'te03'", "'train-g0'", "'train'", "'test'"]),
        (_set(2, "id", "bad id"), ["'bad id'", "A-Z"]),
        (_set(4, "role", "holdout"), ["'tr04'", "'holdout'"]),
        (_set(6, "image", "../escape.png"), ["'va00'", "escapes"]),
    ],
)
def test_byod_refusal_names_failing_record_before_model_work(ws, tmp_path, mutate, needles):
    records = _make_byod(tmp_path / "byod")
    mutate(tmp_path / "byod", records)
    with pytest.raises(ValueError) as caught:
        ws.load_byod(tmp_path / "byod")  # byod_run calls this before creating any child stage
    message = str(caught.value)
    assert "records[0]" not in message
    for needle in needles:
        assert needle in message, message


def test_byod_contract_is_stated_before_data_is_supplied():
    section = next(s for s in SOURCE.values() if "## 10. Optional: your own labelled RGB-D data" in s)
    for needle in (
        '"units": "metres"',
        "A-Za-z0-9_-",
        "5%",
        "1000 m",
        "0.6–350 m",
        "EXIF orientation 1",
        "`float32` and `bool`",
        '"records"',
        "Files",
        "nothing is sent to a DIMER service",
        "authorized",
    ):
        assert needle in section, needle
    byod_cell = SOURCE["code-22"]
    assert "summary.md" in byod_cell and "FileLink" not in byod_cell


# ---- DEP-m1 / DEP-m6: infrastructure labelling, glossary, terminology ------------------------------------


def test_infrastructure_cells_are_titled_and_collapsed():
    for cid in ("code-02", "code-03", "code-04"):
        assert SOURCE[cid].startswith("# @title Infrastructure"), cid
        assert CELLS[cid]["metadata"].get("cellView") == "form", cid
    assert "**Infrastructure**" in SOURCE["md-01"]


def test_glossary_and_notebook_terminology():
    assert SOURCE["md-00b"].startswith("### Key terms")
    for term in (
        "Inverse depth",
        "AbsRel",
        "Delta-1",
        "Affine alignment",
        "Backbone",
        "Neck and head",
        "Adapter",
    ):
        assert f"**{term}" in SOURCE["md-00b"], term
    for term in ("AdamW", "Scale-and-shift-invariant loss", "Reload parity"):
        assert f"**{term}**" in SOURCE["md-00b"], term
    assert not re.search(r"(?<![`\w])workshop(?![`\w])", MARKDOWN, re.I)
    assert "Depth workshop measured results" not in WORKER


# ---- DEP-m2: scoring code readable from a learner cell ---------------------------------------------------


def test_scoring_code_cell_prints_the_three_functions(tmp_path, capsys):
    (tmp_path / "workshop.py").write_text(WORKER, encoding="utf-8")
    exec(compile(SOURCE["code-10c"], "code-10c", "exec"), {"ROOT": tmp_path})
    printed = capsys.readouterr().out
    for name in ("def support(", "def aligned_values(", "def score("):
        assert name in printed


# ---- DEP-m3: training evidence and honest figure labels --------------------------------------------------


def test_depth_cell_shows_per_epoch_pre_restoration_change():
    cell = SOURCE["code-12"]
    assert "max_weight_delta_before_selection" in cell and "epoch_observations" in cell
    assert "Max weight change before selection" in cell


def test_select_figures_labels(ws):
    test = [{"id": f"t{i}", "domain": "indoor" if i < 2 else "outdoor"} for i in range(4)]
    unchanged = ws.select_figures(test, [{"id": r["id"], "abs_rel_delta": 0.0} for r in test])
    assert not any("best" in v or "worst" in v for v in unchanged.values())
    assert set(unchanged) == {"t0", "t3", "t2"}  # same figure set as the original rule
    changed = ws.select_figures(
        test, [{"id": r["id"], "abs_rel_delta": d} for r, d in zip(test, (0, -0.1, 0.2, 0), strict=True)]
    )
    assert changed["t1"] == "best adaptation delta" and changed["t2"] == "worst adaptation delta"


# ---- DEP-m5 / DEP-m7: unlabelled refusals, brightness bounds ---------------------------------------------


def test_unlabelled_image_refusals_are_actionable(ws, tmp_path):
    big = tmp_path / "phone.jpg"
    Image.new("RGB", (4032, 3024), "grey").save(big)
    with pytest.raises(ValueError, match=r"phone\.jpg: 4032x3024 px .*2048"):
        ws.bounded_image(big)
    exif = Image.Exif()
    exif[274] = 6
    rotated = tmp_path / "rotated.jpg"
    Image.new("RGB", (640, 480), "grey").save(rotated, exif=exif)
    with pytest.raises(ValueError, match="EXIF orientation is 6") as caught:
        ws.bounded_image(rotated)
    assert "reference" not in str(caught.value)
    section = SOURCE["md-23"]
    assert "2048" in section and "section 5" in section


def test_brightness_bounds_and_canonical_factors(ws):
    image = Image.new("RGB", (8, 8), (100, 150, 200))
    out, info = ws.brightness(image, 0.8)
    assert info["factor"] == 0.8 and np.asarray(out)[0, 0].tolist() == [80, 120, 160]
    for bad in (0.1, 2.5):
        with pytest.raises(ValueError, match="outside 0.2..2.0"):
            ws.brightness(image, bad)
    assert ws.CONFIG["brightness_factors"] == [0.6, 1.0, 1.4]
    assert WORKER.count("for factor in (0.6, 1.0, 1.4):") == 1
    cell = SOURCE["code-19c"]
    assert 'RUN_OWN_FACTOR = False  # @param {type:"boolean"}' in cell
    assert "0.2 <= float(OWN_BRIGHTNESS_FACTOR) <= 2.0" in cell


# ---- the fixed worker end to end on CPU stand-ins, plus the learner cells that read its outputs ----------


@pytest.fixture(scope="module")
def run(ws, tmp_path_factory):
    base = tmp_path_factory.mktemp("flow")
    root = base / "run"
    for name, text in CARRIED.items():  # the worker reads source.json, the lock and manifests from root
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_bytes(text.encode("utf-8"))
    _make_byod(base / "byod")

    def stage(name, action):
        state = ws.begin_stage(root, name)
        action()
        ws.finish_stage(root, name, state, 0.01)

    def held_out():
        return [r for r in ws.load_records(root) if r["role"] != "train"]

    def depth():
        ws.predict_rows(root, StandIn(), held_out(), "depth_frozen", True)
        history = {
            "history": [
                {"epoch": e, "train_loss": None if e == 0 else 0.1, "val": {"abs_rel": 0.3, "delta1": 0.6}}
                for e in range(4)
            ],
            "best_epoch": 0,
            "actual_optimizer_steps": 18,
            "epoch_observations": [
                {"epoch": e, "max_weight_delta_before_selection": 3.7e-4, "actual_steps": 6 * e}
                for e in (1, 2, 3)
            ],
            "selected_max_weight_delta": 0.0,
            "trainable_parameters": 1,
            "training_pid": -1,
        }
        ws.write_json(root / "outputs/training_history.json", history)
        ws.write_json(root / "outputs/selection.json", {"selected_epoch": 0})
        ws.write_json(root / "outputs/adapter/manifest.json", {"stand_in": True})
        (root / "outputs/adapter/adapter.safetensors").write_bytes(b"stand-in")
        ws.predict_rows(root, StandIn(), held_out(), "depth_selected_live", True)

    def reload():
        rows = ws.predict_rows(root, StandIn(), held_out(), "depth_reloaded", True)
        ws.write_json(
            root / "outputs/reload_parity.json", {"passed": True, "checks": [r["id"] for r in rows]}
        )

    stage("prepare", lambda: ws.prepare(root, str(base / "byod")))
    stage("depth", depth)
    stage("reload", reload)
    stage("zoe", lambda: ws.predict_rows(root, StandIn(metric=True), held_out(), "zoe_metrics", False))
    import depth_reference.pipeline as depth_pipeline
    import zoe_reference

    patch = pytest.MonkeyPatch()
    patch.setattr(depth_pipeline, "DepthAnythingPipeline", StandIn)
    patch.setattr(zoe_reference, "ZoeDepthMetricPipeline", StandIn)
    stage("activity", lambda: ws.activity(root))
    stage("report", lambda: ws.report(root))
    yield types.SimpleNamespace(root=root, base=base)
    patch.undo()


def test_byod_report_describes_its_own_split(run):
    summary = (run.root / "outputs/summary.md").read_text()
    assert summary.startswith("# Depth estimation notebook: measured results")
    assert "Your test split has 4 records in 2 acquisition groups" in summary
    assert "Default test views" not in summary
    selection = json.loads((run.root / "outputs/figure_selection.json").read_text())
    assert not any("best" in v or "worst" in v for v in selection.values())
    with zipfile.ZipFile(run.root / "outputs/depth_results.zip") as bundle:
        assert "summary.md" in bundle.namelist()


def test_own_activity_is_isolated_and_bounded(ws, run):
    receipts = (run.root / "stage_receipts.json").read_text()
    bundle = ws.digest(run.root / "outputs/depth_results.zip")
    canonical = ws.digest(run.root / "outputs/brightness_experiment.json")
    with pytest.raises(ValueError, match="outside 0.2..2.0"):
        ws.own_activity(run.root, 2.5)
    ws.own_activity(run.root, 0.8)
    own = json.loads((run.root / "own_activity_output.json").read_text())
    assert Path(own["root"]).parent == run.root / "own_activity"
    assert own["comparison"]["improved"] + own["comparison"]["unchanged"] + own["comparison"]["worsened"] == 2
    assert (Path(own["root"]) / "outputs/own_factor.png").is_file()
    assert (run.root / "stage_receipts.json").read_text() == receipts
    assert ws.digest(run.root / "outputs/depth_results.zip") == bundle
    assert ws.digest(run.root / "outputs/brightness_experiment.json") == canonical
    with pytest.raises(RuntimeError, match="section 8"):
        ws.own_activity(run.base / "empty", 0.8)


def test_unlabelled_inference_writes_a_preview_and_checks_its_prerequisite(ws, run):
    photo = run.base / "photo.png"
    Image.fromarray(np.random.default_rng(1).integers(0, 255, (300, 400, 3), dtype=np.uint8)).save(photo)
    ws.infer(run.root, str(photo))
    result = json.loads((run.root / "unlabelled_output.json").read_text())
    assert result["abs_rel"] is None and "preview.png" in result["outputs"]
    with Image.open(Path(result["root"]) / "preview.png") as preview:
        assert preview.size == (1260, 350)
    with pytest.raises(RuntimeError, match="section 5"):
        ws.infer(run.base / "no_run", str(photo))


def test_learner_cells_run_against_the_stand_in_outputs(ws, run, monkeypatch, capsys):
    shown: list[str] = []
    display = types.ModuleType("IPython.display")
    display.display = lambda obj: shown.append(getattr(obj, "text", "image"))
    display.Markdown = lambda text: types.SimpleNamespace(text=text)

    def image(filename):
        assert Path(filename).is_file(), filename
        return types.SimpleNamespace()

    display.Image = image
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    ws.own_activity(run.root, 1.2)
    photo = run.base / "photo2.png"
    Image.new("RGB", (64, 48), "grey").save(photo)
    ws.infer(run.root, str(photo))
    (run.root / "byod_output.json").write_text(json.dumps({"root": str(run.root)}))
    namespace = {
        "ROOT": run.root,
        "json": json,
        "Path": Path,
        "time": time,
        "SESSION_START": time.perf_counter(),
        "BOOTSTRAP_SECONDS": 1.0,
        "CARRIED_HASHES": {},
        "run_stage": lambda *a, **k: None,
        "show_json": lambda rel: json.loads((run.root / rel).read_text()),
        "show_previews": lambda pattern: None,
        "show_metric_tables": lambda *a, **k: None,
    }
    switches = {
        "code-19c": {"RUN_OWN_FACTOR = False": "RUN_OWN_FACTOR = True"},
        "code-22": {"RUN_BYOD = False": "RUN_BYOD = True", "BYOD_PATH = ''": "BYOD_PATH = 'x'"},
        "code-24": {"RUN_UNLABELLED = False": "RUN_UNLABELLED = True", "IMAGE_PATH = ''": "IMAGE_PATH = 'x'"},
    }
    for cid in ("code-12", "code-19c", "code-20", "code-22", "code-24"):
        source = SOURCE[cid]
        for old, new in switches.get(cid, {}).items():
            assert source.count(old) == 1
            source = source.replace(old, new)
        exec(compile(source, cid, "exec"), namespace)
    printed = capsys.readouterr().out
    table = next(text for text in shown if text.startswith("| Epoch"))
    assert "| 0 | baseline |" in table and "0 (starting weights)" in table and "3.700e-04" in table
    assert "Files pane" in printed and "peak GPU" in printed and "free disk" in printed
    session = json.loads((run.root / "session_summary.json").read_text())
    assert {"stage_resources", "run_folder_bytes", "disk_free_bytes"} <= set(session)
    assert any(text.startswith("# Depth estimation notebook") for text in shown)  # BYOD summary displayed
    final = SOURCE["code-20"]
    assert "FileLink" not in final and "peak_gpu_bytes" in final
    for cid in ("code-19c", "code-22", "code-24"):  # defaults leave the optional branches off
        exec(compile(SOURCE[cid], cid, "exec"), dict(namespace))
    assert "disabled" in capsys.readouterr().out


def test_review_revision_is_recorded():
    revision = NB["metadata"]["dimer"]["review_revisions"][-1]
    assert revision["reviewed_commit"] == "5f6b516c0ca532a6fefa379a280e9e127cec5d37"
    assert revision["findings_fixed"] == [
        "DEP-M1",
        "DEP-m1",
        "DEP-m2",
        "DEP-m3",
        "DEP-m4",
        "DEP-m5",
        "DEP-m6",
        "DEP-m7",
    ]
    assert NB["metadata"]["dimer"]["release_status"] == "Candidate"
    assert "403 s on 2026-09-27" in SOURCE["md-00"]
