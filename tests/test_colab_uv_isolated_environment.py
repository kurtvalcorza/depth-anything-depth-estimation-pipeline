"""The depth colab tutorial runs in the uv isolated environment (generator /2.1, 2026-10-03).

A Colab CLI T4 run of blob b3e9ed6b stopped at the in-kernel install guard ("Core dependencies changed ...
numpy: loaded=2.1.3, installed=2.5.3"): Colab imports NumPy before the first cell. The notebook now installs a
hash-locked environment with a pinned uv into a managed CPython and routes every later cell to one persistent
worker there, so nothing is installed into the kernel and Run all needs no restart.
"""
# ruff: noqa: E501  -- assertion messages are kept on one line

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import os
import re
import signal
import subprocess
import sys
from multiprocessing.connection import Connection
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


build = _load("build_notebook")
TEMPLATE = _load("notebook_template").TEMPLATE
NOTEBOOK = ROOT / "tutorials" / TEMPLATE["notebook_name"]
LOCK = ROOT / TEMPLATE["lock"]
WORKSHOP = ROOT / "tutorials" / "DIMER_MultiModel_Depth_Estimation_Workshop.ipynb"


@pytest.fixture(scope="module")
def code() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return ["".join(c["source"]) for c in notebook["cells"] if c["cell_type"] == "code"]


@pytest.fixture(scope="module")
def notebook_lines() -> list[str]:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    return [line for c in notebook["cells"] for line in "".join(c["source"]).split("\n")]


def _locked(text: str) -> dict[str, str]:
    return {m.group(1).lower(): m.group(2) for m in re.finditer(r"^([A-Za-z0-9._-]+)==([^\s\\]+)", text, re.M)}


def test_template_selects_the_fleet_uv_mechanism() -> None:
    assert TEMPLATE["isolated_runtime"] is True
    assert TEMPLATE["managed_python"] == "3.12.12"
    assert TEMPLATE["uv"]["version"] == "0.12.15" and TEMPLATE["uv"]["bytes"] == 20081404
    assert TEMPLATE["lock"] == "tutorials/requirements-colab.lock.txt"
    assert build.GENERATOR_VERSION == "build_notebook.py/2.1"


def test_lock_pins_the_pyproject_dependencies_with_hashes() -> None:
    text = LOCK.read_text(encoding="utf-8")
    build.check_lock(build._pins(ROOT, TEMPLATE), text)  # every direct pin at its version, every entry hashed
    blocks = [b for b in re.split(r"\n(?=[A-Za-z0-9])", text) if "==" in b.split("\n", 1)[0] and not b.startswith("#")]
    assert len(blocks) == len(_locked(text)) == 47
    assert all("--hash=sha256:" in b for b in blocks)
    assert "--python-platform x86_64-manylinux_2_28 --generate-hashes --only-binary :all:" in text.split("\n", 2)[1]


def test_lock_matches_the_workshop_versions() -> None:
    """Both notebooks of this repository install the same files: every package the workshop locks is locked here at
    the same version; only torchaudio (a pyproject pin the workshop does not need) is extra."""
    workshop = json.loads(WORKSHOP.read_text(encoding="utf-8"))
    carried = next("".join(c["source"]) for c in workshop["cells"] if "CARRIED_FILES = {" in "".join(c["source"]))
    node = next(n for n in ast.parse(carried).body if isinstance(n, ast.Assign) and getattr(n.targets[0], "id", "") == "CARRIED_FILES")
    workshop_text = ast.literal_eval(node.value)["requirements.txt"]
    mine, theirs = _locked(LOCK.read_text(encoding="utf-8")), _locked(workshop_text)
    assert len(theirs) == 46
    assert {k: v for k, v in mine.items() if k not in theirs} == {"torchaudio": "2.11.0"}
    assert {k: (mine[k], v) for k, v in theirs.items() if mine.get(k) != v} == {}


def test_install_cell_carries_the_lock_byte_for_byte(code: list[str]) -> None:
    install = code[0]
    text = LOCK.read_text(encoding="utf-8")
    assert f"LOCK_SHA256 = {hashlib.sha256(text.encode('utf-8')).hexdigest()!r}" in install
    assert f"LOCKED_PACKAGES = {len(_locked(text))}" in install
    assert "LOCK_TEXT = r'''" + text + "'''" in install


def test_install_requires_hashes_and_binary_wheels(code: list[str]) -> None:
    install = code[0]
    assert '"--require-hashes", "--only-binary", ":all:"' in install
    assert '"--managed-python", "--python", MANAGED_PYTHON' in install
    assert "UV_BYTES = 20081404" in install and TEMPLATE["uv"]["sha256"] in install
    assert 'platform.system() != "Linux" or platform.machine() != "x86_64"' in install


def test_nothing_is_installed_into_the_kernel(code: list[str]) -> None:
    kernel = [i for i, src in enumerate(code) if "# dimer: kernel cell" in src]
    assert kernel == [0, 1]
    for src in (code[0], code[1]):
        assert '"-m", "pip"' not in src and "'-m', 'pip'" not in src, "a kernel cell runs pip in the kernel's Python"
        assert "import importlib.metadata" not in src and "invalidate_caches" not in src
        assert "Restart the runtime" not in src
    # The runtime cell's pip install and stale-import guard only run when DIMER_NOTEBOOK_CI_PREINSTALLED is unset;
    # the worker always sets it, so in the routed path neither the install nor the restart instruction can run.
    runtime = code[2]
    assert "if not SKIP_INSTALL:" in runtime and "Restart the runtime" in runtime
    assert 'DIMER_NOTEBOOK_CI_PREINSTALLED="1"' in code[1]


def test_every_later_cell_runs_in_the_venv_python(code: list[str]) -> None:
    install, router = code[0], code[1]
    assert 'ISOLATED_PYTHON = ISOLATED_ENV / "bin" / "python"' in install
    assert "_DIMER_ISOLATED_RUNTIME = IsolatedRuntime(ISOLATED_PYTHON)" in router
    assert "[str(python), \"-c\", _WORKER_SOURCE" in router
    assert "_ip.input_transformers_cleanup.append(_route_to_isolated_runtime)" in router
    assert all("# dimer: kernel cell" not in src for src in code[2:])


def test_learner_cells_use_the_worker_display(code: list[str]) -> None:
    """IPython is not in the isolated environment: the previews use the display() the worker provides."""
    learner = "\n".join(code[2:])
    assert "from IPython" not in learner
    assert learner.count("show_image = display") == 1 and "show_image(sheet.reduce(3))" in learner
    # `show` is the list of two test records kept in Section 6; the display handle must not shadow it.
    assert "show = test_records[:2]" in learner and "show = display" not in learner


def test_no_cell_line_over_2000_characters(notebook_lines: list[str]) -> None:
    longest = max(len(line) for line in notebook_lines)
    assert longest <= 2000, longest


def test_workshop_notebook_is_untouched() -> None:
    raw = WORKSHOP.read_bytes()
    assert hashlib.sha1(b"blob %d\0" % len(raw) + raw).hexdigest().startswith("66021da652f7")


@pytest.mark.skipif(os.name != "posix", reason="the worker uses POSIX fd passing, as on Colab/Kaggle")
def test_persistent_worker_runs_cells_in_one_namespace(code: list[str]) -> None:
    """Start the notebook's own worker with this interpreter and feed it cells: state persists, printed output and
    display() bundles come back, and an error is raised in the kernel."""
    router = code[1]
    source = router[router.index('_WORKER_SOURCE = r"""') : router.index("def _is_user_cell():")]
    ns = {"os": os, "sys": sys, "subprocess": subprocess, "signal": signal, "Connection": Connection}
    exec(compile(source, "<router>", "exec"), ns)
    shown: list[dict] = []
    worker = ns["IsolatedRuntime"](sys.executable, display=lambda bundle, raw=True: shown.append(bundle))
    try:
        worker.run("value = 41\nimport os\nprint(os.environ['DIMER_NOTEBOOK_CI_PREINSTALLED'])")
        worker.run("value += 1\ndisplay({'value': value})\nvalue")
        assert [b["text/plain"] for b in shown] == ["{'value': 42}", "42"]
        with pytest.raises(ns["IsolatedCellError"], match="ZeroDivisionError"):
            worker.run("1 / 0")
        worker.run("display(value)")
        assert shown[-1]["text/plain"] == "42"
    finally:
        worker.close()
