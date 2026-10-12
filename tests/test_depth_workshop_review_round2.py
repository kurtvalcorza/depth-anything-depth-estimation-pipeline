"""Regression tests for the 2026-10-12 review of the depth-estimation WORKSHOP notebook (DEP2-m1..m5, m7).

The notebook's own ``run_stage`` helper (cell code-04) runs the carried worker in a subprocess with this
interpreter, exactly as in Colab apart from the interpreter. Refusals happen before any model loads, so no
model or GPU is needed. These are CPU interface tests within CI's install budget (numpy, pillow, pytest);
they are not hosted-runtime or pretrained-model evidence.
"""

from __future__ import annotations

import ast
import importlib.util
import json
import os
import re
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_MultiModel_Depth_Estimation_Workshop.ipynb"
NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = {cell["id"]: cell for cell in NB["cells"]}
SOURCE = {cid: "".join(cell["source"]) for cid, cell in CELLS.items()}


def _carried() -> dict[str, str]:
    node = next(
        n
        for n in ast.parse(SOURCE["code-03"]).body
        if isinstance(n, ast.Assign) and n.targets[0].id == "CARRIED_FILES"
    )
    return ast.literal_eval(node.value)


CARRIED = _carried()


def _materialise(directory: Path) -> Path:
    for name, text in CARRIED.items():
        (directory / name).parent.mkdir(parents=True, exist_ok=True)
        (directory / name).write_bytes(text.encode("utf-8"))
    return directory


def _run_stage_namespace(root: Path, python: str = sys.executable) -> dict:
    """The notebook's run_stage, defined from cell code-04, bound to `root` and this interpreter."""
    tree = ast.parse(SOURCE["code-04"])
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "run_stage")
    namespace = {
        "ROOT": root,
        "PYTHON": Path(python),
        "ENV": dict(os.environ, CUDA_VISIBLE_DEVICES="-1", PYTHONIOENCODING="utf-8"),
        "subprocess": subprocess,
        "json": json,
        "Path": Path,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "code-04:run_stage", "exec"), namespace)
    return namespace


def _make_byod(directory: Path, float64_index: int | None = None) -> list[dict]:
    directory.mkdir(parents=True)
    rng = np.random.default_rng(0)
    records = []
    roles = ["train"] * 6 + ["validation"] * 2 + ["test"] * 4
    groups = ["g1", "g2", "g1", "g2", "g1", "g2", "v1", "v2", "t1", "t2", "t1", "t2"]
    for index, (role, group) in enumerate(zip(roles, groups, strict=True)):
        rid = f"r{index:02d}"
        Image.fromarray(rng.integers(0, 255, (48, 64, 3), dtype=np.uint8)).save(directory / f"{rid}.png")
        dtype = np.float64 if index == float64_index else np.float32
        np.save(directory / f"{rid}_depth.npy", rng.uniform(1, 20, (48, 64)).astype(dtype))
        np.save(directory / f"{rid}_mask.npy", np.ones((48, 64), dtype=bool))
        records.append(
            {
                "id": rid,
                "image": f"{rid}.png",
                "depth": f"{rid}_depth.npy",
                "mask": f"{rid}_mask.npy",
                "group_id": group,
                "role": role,
            }
        )
    _write_manifest(directory, records)
    return records


def _write_manifest(directory: Path, records) -> None:
    (directory / "manifest.json").write_text(json.dumps({"units": "metres", "records": records}))


@pytest.fixture(scope="module")
def worker(tmp_path_factory):
    """The carried workshop.py imported in-process (no torch needed for the BYOD loading path)."""
    carried = _materialise(tmp_path_factory.mktemp("carried_round2"))
    patch = pytest.MonkeyPatch()
    patch.syspath_prepend(str(carried))
    patch.setitem(sys.modules, "torch", types.ModuleType("torch"))
    for name in [m for m in sys.modules if m.split(".")[0] in ("depth_reference", "zoe_reference")]:
        patch.delitem(sys.modules, name)
    spec = importlib.util.spec_from_file_location("dep2_round2_worker", carried / "workshop.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    yield module
    for name in [m for m in sys.modules if m.split(".")[0] in ("depth_reference", "zoe_reference")]:
        patch.delitem(sys.modules, name, raising=False)
    patch.undo()


# ---- DEP2-m3: run_stage streams each line once and reports the outcome -----------------------------------

FAKE_WORKER = """import sys
stage = sys.argv[sys.argv.index('--stage') + 1]
print('line one')
print('line two')
if stage == 'refuse':
    print('Refused: record x: bad. Fix it and run this cell again.')
    raise SystemExit(65)
if stage == 'boom':
    raise RuntimeError('genuine failure')
"""


def test_run_stage_prints_each_line_once_and_returns_the_outcome(tmp_path, capsys):
    (tmp_path / "workshop.py").write_text(FAKE_WORKER)
    run_stage = _run_stage_namespace(tmp_path)["run_stage"]
    assert run_stage("ok") is True
    out = capsys.readouterr().out
    assert out.count("line one") == 1 and out.count("line two") == 1
    assert run_stage("ok", echo=False) is True
    assert "line one" not in capsys.readouterr().out
    assert run_stage("refuse") is False
    out = capsys.readouterr().out
    assert out.count("Refused:") == 1 and out.count("line one") == 1 and "Traceback" not in out
    assert run_stage("refuse", echo=False) is False
    out = capsys.readouterr().out
    assert out.count("Refused:") == 1 and "line one" not in out
    with pytest.raises(RuntimeError, match="boom failed with exit 1"):
        run_stage("boom")
    out = capsys.readouterr().out
    assert out.count("Traceback (most recent call last)") == 1
    assert out.count("RuntimeError: genuine failure") == 1


def test_report_summary_is_shown_once_rendered():
    cell = SOURCE["code-20"]
    assert "run_stage('report', echo=False)" in cell
    assert cell.count("outputs/summary.md") == 1 and "display(Markdown(" in cell


# ---- DEP2-m2: a refused input is one 'Refused:' line, no traceback, no notebook error --------------------


def test_byod_refusal_is_one_line_and_the_cell_stops_quietly(tmp_path, capsys):
    root = _materialise(tmp_path / "root")
    byod = tmp_path / "byod"
    _make_byod(byod, float64_index=7)
    namespace = _run_stage_namespace(root)
    cell = SOURCE["code-22"]
    switches = (("RUN_BYOD = False", "RUN_BYOD = True"), ("BYOD_PATH = ''", f"BYOD_PATH = {str(byod)!r}"))
    for old, new in switches:
        assert cell.count(old) == 1
        cell = cell.replace(old, new)
    exec(compile(cell, "code-22", "exec"), namespace)  # must not raise
    out = capsys.readouterr().out
    assert "Traceback" not in out and "RuntimeError" not in out
    refusals = [line for line in out.splitlines() if line.startswith("Refused:")]
    assert len(refusals) == 1, out
    assert "record 'r07'" in refusals[0] and "r07_depth.npy" in refusals[0] and "float64" in refusals[0]
    assert out.count("float64") == 1
    assert "BYOD_PATH" in refusals[0] and "run this cell again" in refusals[0]
    assert not (root / "byod_output.json").exists() and not (root / "byod_runs").exists()


def test_empty_paths_are_refused_without_an_error(tmp_path, capsys):
    namespace = _run_stage_namespace(tmp_path)
    for cid, switch in (("code-22", "RUN_BYOD = False"), ("code-24", "RUN_UNLABELLED = False")):
        cell = SOURCE[cid]
        assert cell.count(switch) == 1
        exec(compile(cell.replace(switch, switch.replace("False", "True")), cid, "exec"), dict(namespace))
        out = capsys.readouterr().out
        assert out.startswith("Refused:") and out.count("Refused:") == 1 and "PATH is empty" in out


@pytest.mark.parametrize(
    ("make", "needles"),
    [
        (lambda p: Image.new("RGB", (4032, 3024), "grey").save(p), ["4032x3024", "2048"]),
        (lambda p: p.write_bytes(b"not an image"), ["could not be opened as an image"]),
        (lambda p: None, ["file not found"]),  # missing file
    ],
)
def test_unlabelled_refusal_is_one_line(tmp_path, capsys, make, needles):
    root = _materialise(tmp_path / "root")
    (root / "outputs/adapter").mkdir(parents=True)
    (root / "outputs/adapter/manifest.json").write_text("{}")  # the prerequisite check passes
    photo = tmp_path / "photo.png"
    make(photo)
    namespace = _run_stage_namespace(root)
    cell = SOURCE["code-24"]
    for old, new in (
        ("RUN_UNLABELLED = False", "RUN_UNLABELLED = True"),
        ("IMAGE_PATH = ''", f"IMAGE_PATH = {str(photo)!r}"),
    ):
        assert cell.count(old) == 1
        cell = cell.replace(old, new)
    exec(compile(cell, "code-24", "exec"), namespace)  # must not raise
    out = capsys.readouterr().out
    refusals = [line for line in out.splitlines() if line.startswith("Refused:")]
    assert len(refusals) == 1 and "Traceback" not in out, out
    assert "photo.png" in refusals[0] and "IMAGE_PATH" in refusals[0]
    for needle in needles:
        assert needle in refusals[0], refusals[0]
    assert not (root / "unlabelled_output.json").exists()


def test_genuine_worker_failure_still_raises(tmp_path, capsys):
    root = _materialise(tmp_path / "root")
    run_stage = _run_stage_namespace(root)["run_stage"]
    with pytest.raises(RuntimeError, match="own_activity failed with exit 1"):
        run_stage("own_activity", "--factor", "1.2")  # section 8 has not run: a genuine precondition failure
    out = capsys.readouterr().out
    assert out.count("Traceback (most recent call last)") == 1 and "Refused:" not in out


def test_main_refuses_only_documented_input_rules(worker, monkeypatch, capsys, tmp_path):
    def documented(*args):
        raise worker.InputRefused("record 'r01': role must be 'train', 'validation' or 'test', not a list")

    def bug(*args):
        raise ValueError("an internal defect")

    for stage, target, option in (("byod", "byod_run", "--byod"), ("infer", "infer", "--image")):
        argv = ["workshop.py", "--root", str(tmp_path), "--stage", stage, option, "x"]
        monkeypatch.setattr(sys, "argv", argv)
        monkeypatch.setattr(worker, target, documented)
        with pytest.raises(SystemExit) as stopped:
            worker.main()
        assert stopped.value.code == worker.REFUSED_EXIT == 65
        assert capsys.readouterr().out.startswith("Refused: record 'r01'")
        monkeypatch.setattr(worker, target, bug)
        with pytest.raises(ValueError, match="an internal defect"):
            worker.main()


def _list_role(directory, records):
    records[4]["role"] = ["train"]
    _write_manifest(directory, records)


def _dict_group(directory, records):
    records[9]["group_id"] = {"scene": 1}
    _write_manifest(directory, records)


def _bool_group(directory, records):
    records[2]["group_id"] = True
    _write_manifest(directory, records)


def _manifest_list(directory, records):
    (directory / "manifest.json").write_text(json.dumps(records))


def _manifest_not_json(directory, records):
    (directory / "manifest.json").write_bytes(b"\xff{not json")


def _corrupt_image(directory, records):
    (directory / records[3]["image"]).write_bytes(b"\x89PNG\r\n\x1a\n broken")


def _corrupt_depth(directory, records):
    (directory / records[5]["depth"]).write_bytes(b"\x93NUMPY garbage")


@pytest.mark.parametrize(
    ("mutate", "needles"),
    [
        (_list_role, ["'r04'", "role", "list"]),
        (_dict_group, ["'r09'", "group_id", "dict"]),
        (_bool_group, ["'r02'", "group_id", "bool"]),
        (_manifest_list, ["manifest.json", "JSON object", "list"]),
        (_manifest_not_json, ["manifest.json", "UTF-8 JSON"]),
        (_corrupt_image, ["'r03'"]),
        (_corrupt_depth, ["'r05'", "r05_depth.npy"]),
    ],
)
def test_malformed_byod_fields_are_refused_before_use(worker, tmp_path, mutate, needles):
    records = _make_byod(tmp_path / "byod")
    mutate(tmp_path / "byod", records)
    with pytest.raises(worker.InputRefused) as caught:
        worker.load_byod(tmp_path / "byod")
    message = str(caught.value)
    # Not chained: the learner sees one message, never 'During handling of the above exception'.
    assert caught.value.__cause__ is None
    assert caught.value.__context__ is None or caught.value.__suppress_context__
    for needle in needles:
        assert needle in message, message


def test_valid_byod_still_loads_with_integer_groups(worker, tmp_path):
    records = _make_byod(tmp_path / "byod")
    for record, group in zip(records, [1, 2, 1, 2, 1, 2, 3, 4, 5, 6, 5, 6], strict=True):
        record["group_id"] = group
    _write_manifest(tmp_path / "byod", records)
    assert len(worker.load_byod(tmp_path / "byod")) == 12


# ---- DEP2-m1: versions and device are printed in the kernel ----------------------------------------------


def test_version_probe_output_is_captured_and_printed():
    cell = SOURCE["code-04"]
    tree = ast.parse(cell)
    probe = next(
        n
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and "__version__" in (ast.get_source_segment(cell, n) or "")
    )
    segment = ast.get_source_segment(cell, probe)
    assert {k.arg for k in probe.keywords} >= {"capture_output", "text"}
    assert "check" not in {k.arg for k in probe.keywords}
    for needle in ("torch.__version__", "torch.version.cuda", "transformers.__version__", "numpy", "pillow",
                   "get_device_name"):
        assert needle in segment, needle
    assert "print(versions.stdout.strip())" in cell and "raise RuntimeError(" in cell
    announce = cell.index("Creating the isolated Python 3.12.12 environment")
    assert announce < cell.index("'venv', '--managed-python'")


# ---- DEP2-m4: quoted runtimes name their revision; the fetch is announced --------------------------------


def test_quoted_runtimes_are_labelled_and_the_fetch_is_announced():
    opening = SOURCE["md-00"]
    for sentence in re.split(r"(?<=[.;])\s+", opening):
        for seconds, blob in (("823 s", "66021da652f7"), ("403 s", "aa6a35259d6f")):
            if seconds in sentence:
                assert blob in sentence, sentence
    assert "823 s" in opening and "630 s" in opening and "not re-run for this one" in opening
    assert "10 minutes" in SOURCE["md-05"] and "1aaeacf" in SOURCE["md-05"]
    prepare_cell = SOURCE["code-06"]
    assert prepare_cell.index("10 minutes") < prepare_cell.index("run_stage('prepare')")


# ---- DEP2-m5: the adaptation result is scoped as one fixed recipe ----------------------------------------


def test_adaptation_is_scoped_as_one_fixed_recipe():
    assert "one adaptation recipe fixed in advance" in SOURCE["md-13"]
    assert "not that neck/head adaptation cannot help" in SOURCE["md-13"]
    assert "Adaptation with one fixed recipe" in SOURCE["md-21"]
    assert "other adaptation settings" in SOURCE["md-21"]


# ---- DEP2-m7: notebook kind and use notice (declared spec stays 2.2) -------------------------------------


def test_kind_and_use_notice():
    dimer = NB["metadata"]["dimer"]
    assert dimer["notebook_kind"] == "WORKSHOP" and dimer["notebook_spec"] == "2.2"
    assert list(dimer)[:4] == ["notebook_spec", "notebook_profile", "notebook_mode", "notebook_kind"]
    first_code = next(i for i, cell in enumerate(NB["cells"]) if cell["cell_type"] == "code")
    opening = SOURCE["md-00"]
    assert NB["cells"].index(CELLS["md-00"]) < first_code
    notice = opening.index("> **⚠️ Notice.** For research, training and evaluation only.")
    assert notice < opening.index("**Profile:**")
    assert "on your own Colab account" in opening and "safety-critical intended use" in opening
    revision = NB["metadata"]["dimer"]["review_revisions"][-1]
    assert revision["reviewed_commit"] == "ea2026e378c50db6b2cb6c9c864b83fd17787ae8"
    assert revision["open_owner_decisions"] == ["DEP2-m6", "DEP2-CX1", "DEP2-CX2"]


# ---- peer review of PR #14 (CX7, CX4, CX3) ----------------------------------------------------------------


def test_unexpected_failure_inside_the_loaders_validation_raises(worker, tmp_path, monkeypatch):
    """CX7: only a documented input rule is a refusal; a defect inside a validation call keeps its error."""
    import zoe_reference

    _make_byod(tmp_path / "byod")

    def broken_digest(array):
        raise ValueError("internal defect in a helper")

    monkeypatch.setattr(worker, "array_digest", broken_digest)  # called inside validate_roles
    with pytest.raises(ValueError, match="internal defect") as caught:
        worker.load_byod(tmp_path / "byod")
    assert not isinstance(caught.value, worker.InputRefused)
    monkeypatch.undo()

    def broken_validator(image):
        return int("not a number")  # a ValueError raised by a library call, not by a rule

    monkeypatch.setattr(zoe_reference, "validate_image", broken_validator)
    with pytest.raises(ValueError, match="invalid literal") as caught:
        worker.load_byod(tmp_path / "byod")
    assert not isinstance(caught.value, worker.InputRefused)


def test_unexpected_failure_inside_infer_validation_raises(worker, tmp_path, monkeypatch):
    (tmp_path / "outputs/adapter").mkdir(parents=True)
    (tmp_path / "outputs/adapter/manifest.json").write_text("{}")
    photo = tmp_path / "photo.png"
    Image.new("RGB", (64, 48), "grey").save(photo)

    def broken(path):
        return float("nan-ish")

    monkeypatch.setattr(worker, "bounded_image", broken)
    with pytest.raises(ValueError, match="could not convert") as caught:
        worker.infer(tmp_path, str(photo))
    assert not isinstance(caught.value, worker.InputRefused)


def test_documented_rules_are_still_refused(worker):
    def rule():
        raise ValueError("a rule")

    try:
        worker.support(np.zeros((2, 2), dtype=np.float32), np.zeros((2, 2), dtype=bool))
    except ValueError as error:
        assert worker.input_rule_violation(error)
    try:
        rule()
    except ValueError as error:
        assert not worker.input_rule_violation(error)


def test_unlabelled_metre_colours_are_labelled_the_right_way_round(worker, tmp_path):
    """CX4: colour() maps low (near) to blue and high (far) to red; the caption and colour bar say so."""
    worker_source = CARRIED["workshop.py"]
    assert "blue = near; red = far" in worker_source and "blue = far)" not in worker_source
    relative = np.linspace(0, 1, 48 * 64, dtype=np.float32).reshape(48, 64)
    metric = np.linspace(1, 100, 48 * 64, dtype=np.float32).reshape(48, 64)
    path = tmp_path / "preview.png"
    worker.unlabelled_preview(Image.new("RGB", (64, 48), "grey"), relative, metric, path)
    with Image.open(path) as preview:
        assert preview.size == (1260, 380)
        near, far = preview.getpixel((890 + 1, 355)), preview.getpixel((890 + 298, 355))
    assert near[2] > 200 and near[0] < 50  # blue at 0.6 m
    assert far[0] > 200 and far[2] < 50  # red at 350 m


def test_fit_diagnostics_are_shown_beside_the_metric_tables(monkeypatch, capsys, tmp_path):
    """CX3: negative-scale, degenerate and far-cap fits and the fewest scored pixels are shown and flagged."""
    shown: list[str] = []
    display = types.ModuleType("IPython.display")
    display.display = lambda obj: shown.append(obj.text)
    display.Markdown = lambda text: types.SimpleNamespace(text=text)
    monkeypatch.setitem(sys.modules, "IPython", types.ModuleType("IPython"))
    monkeypatch.setitem(sys.modules, "IPython.display", display)
    tree = ast.parse(SOURCE["code-04"])
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef)
                 and n.name in ("show_fit_diagnostics", "show_metric_tables")]
    namespace = {"ROOT": tmp_path, "json": json}
    exec(compile(ast.Module(body=functions, type_ignores=[]), "code-04", "exec"), namespace)
    fit = {"rank": 2, "degenerate": False, "negative_scale": False, "far_cap_fraction": 0.0}
    aligned_protocol = {"role": "test", "protocol": "reference-assisted affine alignment"}
    rows = [
        {"id": "a", "model": "m", "abs_rel": 0.1, "delta1": 0.9, "n_valid": 500, **aligned_protocol, **fit},
        {"id": "b", "model": "m", "abs_rel": 0.0, "delta1": 1.0, "n_valid": 2, **aligned_protocol, **fit,
         "negative_scale": True, "far_cap_fraction": 0.25},
        {"id": "c", "role": "test", "model": "z", "protocol": "unaligned metric metres", "abs_rel": 0.4,
         "delta1": 0.4, "n_valid": 70},
    ]
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs/rows.json").write_text(json.dumps({"rows": rows}))
    namespace["show_metric_tables"](["rows.json"])
    printed = capsys.readouterr().out
    diagnostics = [text for text in shown if text.startswith("*Scoring support and fit diagnostics")]
    assert len(diagnostics) == 2
    aligned = next(text for text in diagnostics if "affine" in text)
    assert "| m | 2 | 2 (b) | 1 | 0 | 0.1250 / 0.2500 |" in aligned
    assert "| z | 1 | 70 (c) | no fit | no fit | no fit |" in next(t for t in diagnostics if "unaligned" in t)
    assert printed.count("Flagged images") == 1 and "m: b [NC]" in printed
