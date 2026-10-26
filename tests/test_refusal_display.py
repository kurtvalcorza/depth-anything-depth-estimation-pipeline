"""DAT19/DAT20: a refused BYOD input shows one clean line, ``Refused: <file>: <reason>``, with no traceback.

The router cell carries ml-worker's refusal channel (template ``dat20-router-refusal/1``) and Section 4 wraps
its whole BYOD branch so that every user-input problem raises ``BYODRefusal``. These tests drive both from the
notebook JSON with CI's dependencies only:

- Section 4 is executed verbatim (the review-fix harness) for each refusal path;
- the router's kernel loop is driven with a queued stub connection on every platform;
- on POSIX the real worker process runs Section 4, so the full display path is checked end to end.
"""
# ruff: noqa: E501  -- notebook source literals are matched verbatim

from __future__ import annotations

import json
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from test_depth_anything_depth_estimation_colab_review_fixes import _section4_namespace, _write_byod_dir

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "depth_anything_depth_estimation_colab.ipynb"
STEM = "depth_anything_depth_estimation"

NB = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
CELLS = ["".join(c["source"]) if isinstance(c["source"], list) else c["source"] for c in NB["cells"]]
CODE = [i for i, c in enumerate(NB["cells"]) if c["cell_type"] == "code"]


def _cell(marker: str) -> int:
    found = [i for i in CODE if marker in CELLS[i]]
    assert len(found) == 1, (marker, found)
    return found[0]


SEC4 = _cell("USE_BYOD = False  # @param")
ROUTER = _cell("_WORKER_SOURCE = r")


def refusing_section4_source(path: str) -> str:
    src = CELLS[SEC4].replace("USE_BYOD = False", "USE_BYOD = True", 1)
    return src.replace('BYOD_PATH = ""', f"BYOD_PATH = {path!r}", 1)


def _refusal(tmp_path, monkeypatch, **kwargs) -> str:
    ns, code = _section4_namespace(tmp_path, monkeypatch, **kwargs)
    with pytest.raises(ValueError) as info:
        exec(compile(code, "<section 4>", "exec"), ns)
    exc = info.value
    assert getattr(exc, "dimer_refusal", False) is True, repr(exc)
    lines = exc._render_traceback_()
    assert len(lines) == 1 and lines[0].startswith("Refused: ") and "Traceback" not in lines[0], lines
    return lines[0]


# ---- each refusal path in Section 4 is one clean line naming the file ---------------------------------------


def test_acquisition_refusals(tmp_path, monkeypatch):
    assert _refusal(tmp_path, monkeypatch, upload={}).startswith("Refused: (no file): upload exactly one .zip file (received 0)")
    assert _refusal(tmp_path, monkeypatch, upload={"a.zip": b"", "b.zip": b""}).startswith("Refused: a.zip, b.zip: upload exactly one .zip file")
    missing = str(tmp_path / "nope")
    assert _refusal(tmp_path, monkeypatch, byod_path=missing).startswith(f"Refused: {missing}: BYOD_PATH does not exist")
    monkeypatch.setitem(sys.modules, "google.colab", None)
    assert "needs Google Colab" in _refusal(tmp_path, monkeypatch)


def test_content_and_contract_refusals_name_the_file(tmp_path, monkeypatch):
    sample_data = tmp_path / "sample_data"
    sample_data.mkdir()
    (sample_data / "california_housing_test.csv").write_text("longitude,latitude\n1,2\n", encoding="utf-8")
    assert _refusal(tmp_path, monkeypatch, byod_path=str(sample_data)).startswith("Refused: sample_data: ")
    hostname = tmp_path / "hostname"
    hostname.write_text("runtime\n", encoding="utf-8")
    assert "must be a directory or a .zip holding records.csv" in _refusal(tmp_path, monkeypatch, byod_path=str(hostname))
    assert _refusal(tmp_path, monkeypatch, upload={"broken.zip": b"not a zip"}).startswith("Refused: broken.zip: ")
    line = _refusal(tmp_path, monkeypatch, byod_path=str(_write_byod_dir(tmp_path / "two", n_views=8, n_groups=2)))
    assert line.startswith("Refused: two: BYOD set has 2 distinct group(s)")
    assert line.endswith("Fix the file, then select this cell and choose Runtime → Run after.")
    small = _write_byod_dir(tmp_path / "small", n_views=4, n_groups=4)
    assert "training records; at least 4 are required" in _refusal(tmp_path, monkeypatch, byod_path=str(small))


def test_a_refusal_removes_the_earlier_runs_data_and_results(tmp_path, monkeypatch):
    data = _write_byod_dir(tmp_path / "set", n_views=8, n_groups=4)
    monkeypatch.delitem(sys.modules, "google.colab", raising=False)
    ns, code = _section4_namespace(tmp_path, monkeypatch, byod_path=str(data))
    exec(compile(code, "<section 4>", "exec"), ns)
    assert ns["train_records"] and list((tmp_path / "outputs").glob(f"{STEM}_*"))
    (tmp_path / "outputs" / f"{STEM}_adapter").mkdir()
    with pytest.raises(ValueError, match="does not exist"):
        exec(compile(refusing_section4_source(str(tmp_path / "nope")), "<section 4>", "exec"), ns)
    assert not list((tmp_path / "outputs").glob(f"{STEM}_*"))
    for name in ("records", "splits", "train_records", "val_records", "test_records", "data_source", "summary"):
        assert name not in ns, name


def test_a_sample_path_failure_still_raises_unchanged(tmp_path, monkeypatch):
    def broken(*args, **kwargs):
        raise ValueError("internal sample check failed")

    ns, _code = _section4_namespace(tmp_path, monkeypatch)
    ns["fetch_corpus"] = broken
    with pytest.raises(ValueError, match="internal sample check failed") as info:
        exec(compile(CELLS[SEC4], "<section 4>", "exec"), ns)
    assert not getattr(info.value, "dimer_refusal", False)


WORKER_SETUP = [
    f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r})",
    "import os\nfrom pathlib import Path\n\nimport numpy as np\nimport depth_anything_depth_estimation_pipeline.samples as _samples\nglobals().update({k: getattr(_samples, k) for k in dir(_samples) if not k.startswith('__')})",
]
BAD_NAME, BAD_BYTES, BAD_RULE, DISCARDED = "hostname", b"runtime\n", "must be a directory or a .zip holding records.csv", "train_records"
_UNUSED = types


# ---- the router: a refusal crosses the worker boundary as one line; a genuine error keeps its traceback -----


def _router_namespace() -> dict:
    """The router cell's definitions (everything before it starts the worker), with a stand-in IPython."""
    src = CELLS[ROUTER]
    head = src[: src.index("\nif SKIP_INSTALL:")]
    ipython = types.ModuleType("IPython")
    ipython.get_ipython = lambda: None
    saved = sys.modules.get("IPython")
    sys.modules["IPython"] = ipython
    try:
        ns: dict = {"__name__": "__main__", "os": os, "subprocess": subprocess, "sys": sys}
        exec(compile(head, f"<cell {ROUTER}>", "exec"), ns)
    finally:
        if saved is None:
            sys.modules.pop("IPython", None)
        else:
            sys.modules["IPython"] = saved
    return ns


class _Queued:
    def __init__(self, messages):
        self.messages = list(messages)
        self.sent = []

    def recv(self):
        return self.messages.pop(0)

    def send(self, message):
        self.sent.append(message)


def _stub_runtime(ns, messages):
    runtime = object.__new__(ns["IsolatedRuntime"])
    runtime._recv, runtime._send, runtime._display = _Queued(messages), _Queued([]), lambda *a, **k: None
    return runtime


def test_router_carries_the_refusal_channel():
    src = CELLS[ROUTER]
    assert 'if getattr(exc, "dimer_refusal", False):' in src and '_send.send(("refusal", str(exc)))' in src
    assert 'elif kind == "refusal":' in src and "class IsolatedCellRefusal(Exception):" in src


def test_router_kernel_shows_a_refusal_as_one_line_and_an_error_with_its_traceback(capsys):
    ns = _router_namespace()
    with pytest.raises(ns["IsolatedCellRefusal"]) as info:
        _stub_runtime(ns, [("refusal", "data.csv: a column is missing")]).run("cell")
    assert info.value._render_traceback_() == ["Refused: data.csv: a column is missing"]
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    trace = "Traceback (most recent call last):\n  File x\nZeroDivisionError: division by zero\n"
    with pytest.raises(ns["IsolatedCellError"], match="ZeroDivisionError"):
        _stub_runtime(ns, [("error", trace, "ZeroDivisionError: division by zero")]).run("cell")
    assert capsys.readouterr().err == trace


@pytest.mark.skipif(sys.platform == "win32", reason="the router's worker uses pass_fds and POSIX sessions")
def test_real_worker_shows_the_section4_refusal_as_one_line_and_keeps_running(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    ns = _router_namespace()
    runtime = ns["IsolatedRuntime"](sys.executable, display=lambda *a, **k: None)
    try:
        for source in WORKER_SETUP:
            runtime.run(source)
        bad = tmp_path / BAD_NAME
        bad.write_bytes(BAD_BYTES)
        capsys.readouterr()
        with pytest.raises(ns["IsolatedCellRefusal"]) as info:
            runtime.run(refusing_section4_source(str(bad)))
        lines = info.value._render_traceback_()
        captured = capsys.readouterr()
        assert len(lines) == 1 and lines[0].startswith(f"Refused: {BAD_NAME}: "), lines
        assert BAD_RULE in lines[0], lines
        assert "Traceback" not in captured.out + captured.err, captured
        runtime.run(f"print('still running', {DISCARDED!r} in globals())")
        assert capsys.readouterr().out == "still running False\n"
        with pytest.raises(ns["IsolatedCellError"]):
            runtime.run("1 / 0")
        assert "Traceback" in capsys.readouterr().err
    finally:
        runtime.close()
