"""The depth workshop notebook is rebuilt byte for byte from tracked sources (DEP2-m6, round 2026-10-12)."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _builder():
    path = ROOT / "tools" / "build_depth_workshop.py"
    spec = importlib.util.spec_from_file_location("build_depth_workshop", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_notebook_is_rebuilt_byte_for_byte():
    builder = _builder()
    assert builder.build() == builder.NOTEBOOK.read_bytes()
    assert builder.main(["--check"]) == 0


def test_package_modules_are_carried_from_the_package_itself():
    skeleton = json.loads((ROOT / "tools/depth_workshop/notebook.json").read_text(encoding="utf-8"))
    carried = dict(skeleton["metadata"]["dimer"]["generated_from"]["carried"])
    package = "src/depth_anything_depth_estimation_pipeline"
    assert carried["depth_reference/pipeline.py"] == f"{package}/pipeline.py"
    assert carried["licenses/depth-code.txt"] == "LICENSE"
    assert carried["workshop.py"] == "tools/depth_workshop/carried/workshop.py"
    for source in carried.values():
        assert (ROOT / source).is_file(), source
