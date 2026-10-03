"""WORKSHOP notebook source layout: no cell line over 2,000 characters, and the carrier split round-trips."""

from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOK = ROOT / "tutorials" / "DIMER_MultiModel_Depth_Estimation_Workshop.ipynb"
TOOL = ROOT / "tools" / "split_workshop_carrier.py"
MAX_LINE = 2000


def _load_tool():
    spec = importlib.util.spec_from_file_location("split_workshop_carrier", TOOL)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_no_workshop_cell_line_over_limit() -> None:
    notebook = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for index, cell in enumerate(notebook["cells"]):
        longest = max((len(line) for line in "".join(cell["source"]).split("\n")), default=0)
        assert longest <= MAX_LINE, f"cell {index} has a {longest}-character line"
    assert _load_tool().long_lines(notebook) == []


def test_carried_literal_round_trips() -> None:
    tool = _load_tool()
    files = {
        "empty.txt": "",
        "long.txt": "x" * 2500 + "\n",
        "multi.py": "first line\n\nthird 'quoted' line\r\nno trailing newline é",
    }
    literal = tool.carried_literal(files)
    assert ast.literal_eval(literal) == files
    assert max(len(line) for line in literal.split("\n")) <= tool.CARRIER_PIECE + 20


def test_split_carrier_preserves_surrounding_source() -> None:
    tool = _load_tool()
    source = "# @title Infrastructure\nCARRIED_FILES = " + repr({"a.py": "y" * 2500}) + "\nprint(1)\n"
    new = tool.split_carrier(source)
    assert new.startswith("# @title Infrastructure\nCARRIED_FILES = {")
    assert new.endswith("\nprint(1)\n")
    assert ast.literal_eval(ast.parse(new).body[0].value) == {"a.py": "y" * 2500}
