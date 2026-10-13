#!/usr/bin/env python3
"""Build tutorials/DIMER_MultiModel_Depth_Estimation_Workshop.ipynb from tracked sources.

Sources (all tracked in this repository):
- tools/depth_workshop/notebook.json: the notebook skeleton (metadata, cell order, ids, cell metadata);
  each cell's "source" names a file under tools/depth_workshop/cells/;
- tools/depth_workshop/cells/*.md|*.py: the cell texts; cells/code-03.py is the carrier template, whose
  @@CARRIED_FILES@@ and @@CARRIED_HASHES@@ placeholders are filled here;
- the carried files, listed in order in notebook.json under metadata.dimer.generated_from.carried: the Depth
  Anything package modules, LICENSE and manifest come from src/, LICENSE and weights/ (so they cannot drift
  from the package), and the runner, ZoeDepth module, lock and the rest from tools/depth_workshop/carried/.

The carried source.json, CARRIED_HASHES and metadata.dimer.generated_from.files are computed from the carried
bytes. generator_sha256 "computed" means the SHA-256 of this file (LF line endings). Line endings of every
source are normalised to LF. Output is json.dumps(indent=1, ensure_ascii=False) + LF.

Usage: python tools/build_depth_workshop.py [--check]
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCES = ROOT / "tools" / "depth_workshop"
NOTEBOOK = ROOT / "tutorials" / "DIMER_MultiModel_Depth_Estimation_Workshop.ipynb"


def _text(path: Path) -> str:
    return path.read_bytes().decode("utf-8").replace("\r\n", "\n")


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _carried_literal():
    path = ROOT / "tools" / "split_workshop_carrier.py"
    spec = importlib.util.spec_from_file_location("split_workshop_carrier", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.carried_literal


def carried_files(generated: dict) -> dict[str, str]:
    """Carried name -> text, in the declared order, with source.json computed last."""
    files = {}
    for name, source in generated["carried"]:
        files[name] = _text(ROOT / source)
    keys = ("repository", "base_revision", "generator", "generator_sha256")
    provenance = {key: generated[key] for key in keys}
    if provenance["generator_sha256"] == "computed":
        provenance["generator_sha256"] = _sha256(_text(Path(__file__)))
    provenance["files"] = {name: _sha256(text) for name, text in files.items()}
    files["source.json"] = json.dumps(provenance, indent=2)
    return files, provenance


def build() -> bytes:
    skeleton = json.loads(_text(SOURCES / "notebook.json"))
    notebook = copy.deepcopy(skeleton)
    generated = notebook["metadata"]["dimer"]["generated_from"]
    files, provenance = carried_files(generated)
    hashes = {name: _sha256(text) for name, text in files.items()}
    for cell in notebook["cells"]:
        text = _text(SOURCES / cell["source"])
        if cell["id"] == "code-03":
            for marker in ("@@CARRIED_FILES@@", "@@CARRIED_HASHES@@"):
                if text.count(marker) != 1:
                    raise SystemExit(f"code-03 template: {marker} must occur exactly once")
            text = text.replace("@@CARRIED_FILES@@", _carried_literal()(files))
            text = text.replace("@@CARRIED_HASHES@@", repr(hashes))
        cell["source"] = text.splitlines(keepends=True)
    del generated["carried"]
    generated.update(provenance)
    return (json.dumps(notebook, indent=1, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--check", action="store_true", help="fail if the committed notebook differs from a rebuild"
    )
    args = parser.parse_args(argv)
    built = build()
    if args.check:
        if NOTEBOOK.read_bytes() != built:
            message = f"FAIL: {NOTEBOOK.name} differs from tools/build_depth_workshop.py output; rebuild it"
            print(message, file=sys.stderr)
            return 1
        print(f"OK: {NOTEBOOK.name} is up to date ({len(built)} bytes)")
        return 0
    NOTEBOOK.write_bytes(built)
    print(f"wrote {NOTEBOOK.name} ({len(built)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
