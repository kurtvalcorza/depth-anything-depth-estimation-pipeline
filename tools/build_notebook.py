#!/usr/bin/env python3
"""Generate a STANDALONE DIMER tutorial notebook (NOTEBOOK_SPEC 1.1 §3.6) from repository sources.

The notebook carries the repository's pipeline module verbatim (ST2), the model identity and
snapshot manifest inline (ST3), the exact runtime pins (ENV2/PAR2), and records what it was
generated from (ST5). Regeneration from the same committed state is byte-identical, so
``--check`` can gate CI (PAR3).

Usage (from the repository root, or with --repo):
    python tools/build_notebook.py            # write tutorials/<notebook_name>
    python tools/build_notebook.py --check    # exit 1 if the committed notebook differs
    python tools/build_notebook.py --out PATH # write elsewhere (review copies)

The per-repository template is ``tools/notebook_template.py`` and exposes ``TEMPLATE`` (see
``template_contract`` below). This file is vendored per repository; the fleet copy lives in the
relay ``shared/`` directory and is the one to edit first.
"""
# ruff: noqa: E501  -- learner-facing prose is kept on single lines so the rendered markdown stays readable
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

GENERATOR_VERSION = "build_notebook.py/1"
NOTEBOOK_SPEC = "1.1"

# ST2: the only permitted differences between the repository module and the embedded cell.
# Each rule is (regex on one line, replacement). Keep this list short and documented; the
# parity test applies exactly these rules before comparing.
REWRITES: tuple[tuple[str, str], ...] = (
    (
        r"^DEFAULT_WEIGHTS_DIR = Path\(__file__\)[^\n]*$",
        'DEFAULT_WEIGHTS_DIR = Path.cwd() / "weights" / MODEL_KEY'
        "  # standalone rewrite (build_notebook.py): working-directory-relative",
    ),
)

# Stale-import guard shared with the repository-installing notebooks (ENV6).
_INSTALL_GUARD = '''
def _installed_version(distribution):
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None

if not SKIP_INSTALL:
    # Capture every distribution already imported in this runtime, whatever its module name
    # (PIL -> pillow), so a pinned install that replaces a loaded package is detected and the
    # notebook stops with a restart instruction instead of continuing with mixed versions.
    _module_dists = importlib.metadata.packages_distributions()
    _loaded = sorted({d for m in list(sys.modules) for d in _module_dists.get(m.partition('.')[0], ())})
    loaded = {distribution: _installed_version(distribution) for distribution in _loaded}
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', *PINS], check=True)
    importlib.invalidate_caches()
    stale = []
    for distribution, before in loaded.items():
        installed = _installed_version(distribution)
        if before is not None and before != installed:
            stale.append(f'{distribution}: loaded={before}, installed={installed}')
    if stale:
        raise RuntimeError('Core dependencies changed while older modules were loaded: ' + '; '.join(stale) + '. Restart the runtime, then rerun from the top.')
'''


def template_contract() -> dict[str, str]:
    """Keys ``TEMPLATE`` must define (documentation for template authors)."""
    return {
        "package": "import name of the repository package, e.g. resnet50_classification_pipeline",
        "repo_name": "GitHub repository name",
        "stem": "output file stem, e.g. resnet50_classification (notebook, outputs/ files)",
        "notebook_name": "tutorials/<notebook_name>",
        "profile": "TASK-INFERENCE | MULTI-CAPABILITY | E2E | ARTIFACT-INFERENCE",
        "pipeline_class": "public class exposing from_pretrained(weights_dir=...)",
        "runtime_imports": "list of principal libraries whose versions the runtime cell prints, e.g. ['torch', 'timm']",
        "title": "H1 text",
        "badges": "list of (alt, image_url, link_url)",
        "capability": "one-line capability statement",
        "intro": "markdown paragraphs after the header block (no heading)",
        "learning_objectives": "one markdown paragraph starting after the bold label",
        "exclusions": "one markdown sentence after the bold label",
        "prerequisites": "list of markdown bullets; the generator appends the External access bullet",
        "cells": "list of {'md': str, 'code': str} stage cells inserted after the model cell; may use {stem}, {MODEL_ID}, {MODEL_REVISION}",
        "closing": "markdown for Interpretation and limits + References",
        "weights_key": "MODEL_KEY value (weights/<key>/dimer-base-manifest.json)",
    }


def load_template(path: Path) -> dict[str, Any]:
    spec = importlib.util.spec_from_file_location("notebook_template", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    template = module.TEMPLATE
    missing = [k for k in template_contract() if k not in template]
    if missing:
        raise SystemExit(f"template missing keys: {missing}")
    return template


def apply_rewrites(module_text: str) -> str:
    text = module_text
    for pattern, replacement in REWRITES:
        text, n = re.subn(pattern, replacement, text, flags=re.M)
        if n != 1:
            raise SystemExit(f"rewrite rule matched {n} times (expected 1): {pattern}")
    return text.rstrip("\n") + "\n"


def _pins(repo: Path) -> list[str]:
    text = (repo / "pyproject.toml").read_text(encoding="utf-8")
    block = re.search(r"^dependencies\s*=\s*\[(.*?)^\]", text, re.M | re.S)
    if not block:
        raise SystemExit("pyproject.toml: dependencies block not found")
    pins = re.findall(r'"([^"]+)"', block.group(1))
    bad = [p for p in pins if "==" not in p]
    if bad:
        raise SystemExit(f"unpinned runtime dependency (ENV2): {bad}")
    return pins


def _head_revision(repo: Path) -> str:
    """The repository revision the notebook is generated from (ST5): HEAD at generation time.

    It is a provenance label only; the parity anchor is the module SHA-256, so a later commit that
    carries the regenerated notebook does not invalidate it (see ``--check``).
    """
    try:
        out = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        out = ""
    return out or "uncommitted"


def recorded_revision(notebook_path: Path) -> str | None:
    """`generated_from.revision` of an existing notebook, or None."""
    if not notebook_path.exists():
        return None
    try:
        meta = json.loads(notebook_path.read_text(encoding="utf-8"))["metadata"]["dimer"]["generated_from"]
        return str(meta["revision"])
    except (KeyError, ValueError, TypeError):
        return None


def load_context(repo: Path, template: dict[str, Any], revision: str | None = None) -> dict[str, Any]:
    pkg = template["package"]
    module_rel = f"src/{pkg}/pipeline.py"
    module_text = (repo / module_rel).read_text(encoding="utf-8")
    manifest_path = repo / "weights" / template["weights_key"] / "dimer-base-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    ident = {
        k: re.search(rf'^{k} = "([^"]+)"$', module_text, re.M).group(1)  # type: ignore[union-attr]
        for k in ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")
    }
    if manifest["modelId"] != ident["MODEL_ID"] or manifest["revision"] != ident["MODEL_REVISION"]:
        raise SystemExit("manifest identity != module identity")
    if template["weights_key"] != ident["MODEL_KEY"]:
        raise SystemExit("template weights_key != MODEL_KEY")
    return {
        "pkg": pkg,
        "module_rel": module_rel,
        "module_text": module_text,
        "embedded_text": apply_rewrites(module_text),
        "module_sha256": hashlib.sha256(module_text.encode("utf-8")).hexdigest(),
        "module_revision": revision or _head_revision(repo),
        "manifest": manifest,
        "pins": _pins(repo),
        **ident,
    }


def _md(source: str, cell_id: str) -> dict[str, Any]:
    return {"cell_type": "markdown", "id": cell_id, "metadata": {}, "source": source.rstrip("\n")}


def _code(source: str, cell_id: str, metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cell_id,
        "metadata": metadata or {},
        "outputs": [],
        "source": source.rstrip("\n"),
    }


def render(repo: Path, template: dict[str, Any], revision: str | None = None) -> dict[str, Any]:
    ctx = load_context(repo, template, revision)
    stem = template["stem"]
    fmt = {"stem": stem, **{k: ctx[k] for k in ("MODEL_ID", "MODEL_REVISION", "MODEL_LICENSE", "MODEL_KEY")}}
    cells: list[dict[str, Any]] = []
    n = 0

    def add(cell: dict[str, Any]) -> None:
        nonlocal n
        cell["id"] = f"{stem}-{n:02d}"
        n += 1
        cells.append(cell)

    badges = " ".join(f"[![{alt}]({img})]({link})" for alt, img, link in template["badges"])
    total_mb = ctx["manifest"]["totalBytes"] / 1e6
    header = (
        f"# {template['title']}\n\n{badges}\n\n"
        f"**Profile:** `{template['profile']}`  \n"
        f"**Notebook specification:** DIMER Notebook Specification {NOTEBOOK_SPEC} — **standalone** (§3.6)  \n"
        f"**Capability:** {template['capability']}\n\n"
        f"**This notebook is standalone.** It carries the repository's pipeline module "
        f"(`{ctx['module_rel']}` at revision `{ctx['module_revision'][:12]}`) verbatim in Section 2, the pinned model "
        f"identity and the per-file SHA-256 manifest in Section 3, and the exact runtime pins in Section 1, so it keeps "
        f"working after export even if the repository changes or disappears. Its only external dependencies are the "
        f"pinned PyPI distributions and the Hugging Face Hub at the immutable revision `{ctx['MODEL_REVISION']}` "
        f"(~{total_mb:.0f} MB, digest-verified before loading). It was generated by `tools/build_notebook.py` "
        f"({GENERATOR_VERSION}); edit the repository and regenerate rather than editing cells.\n\n"
        f"{template['intro'].strip()}\n\n"
        f"**Learning objectives:** {template['learning_objectives'].strip()}\n\n"
        f"**This notebook does not demonstrate:** {template['exclusions'].strip()}"
    )
    add(_md(header, ""))

    prereq = list(template["prerequisites"]) + [
        f"- **External access:** the Hugging Face Hub only, to fetch the pinned `{ctx['MODEL_ID']}` snapshot "
        f"(~{total_mb:.0f} MB) at revision `{ctx['MODEL_REVISION'][:12]}…`. No GitHub access and no credentials are "
        f"required; nothing is installed from this repository."
    ]
    add(_md("## Prerequisites\n\n" + "\n".join(prereq), ""))

    pins_literal = "PINS = [\n" + "".join(f"    {p!r},\n" for p in ctx["pins"]) + "]"
    imports = template["runtime_imports"]
    ident_print = ", ".join(f"'{m}': {m}.__version__" for m in imports)
    install_md = (
        "## 1. Install the pinned runtime\n\n"
        "The dependency set is pinned exactly (the same `==` pins as the repository's `pyproject.toml` at the "
        "generating revision) and installed directly — there is no repository clone and no package install. If a pin "
        "replaces a distribution this runtime has already imported, the cell stops with a restart instruction rather "
        "than continuing with mixed versions. Look for a dictionary reporting the notebook's source revision, Python, "
        + ", ".join(f"`{m}`" for m in imports)
        + " versions, and whether CUDA is available."
    )
    install_code = (
        "import importlib\nimport importlib.metadata\nimport os\nimport platform\nimport subprocess\nimport sys\n\n"
        f"{pins_literal}\n"
        "NOTEBOOK_SOURCE = {\n"
        f"    'repository': {template['repo_name']!r},\n"
        f"    'repository_revision': {ctx['module_revision']!r},\n"
        f"    'embedded_module': {ctx['module_rel']!r},\n"
        f"    'module_sha256': {ctx['module_sha256']!r},\n"
        f"    'generator': {GENERATOR_VERSION!r},\n"
        f"    'notebook_spec': {NOTEBOOK_SPEC!r},\n"
        "}\n"
        "SKIP_INSTALL = os.environ.get('DIMER_NOTEBOOK_CI_PREINSTALLED') == '1'\n"
        f"{_INSTALL_GUARD}\n"
        f"import {', '.join(imports)}\n"
        f"print({{'notebook_source': NOTEBOOK_SOURCE, 'python': platform.python_version(), {ident_print}, "
        "'cuda': torch.cuda.is_available()})"
    )
    add(_md(install_md, ""))
    add(_code(install_code, ""))

    embed_md = (
        f"## 2. Pipeline code (carried verbatim from `{ctx['module_rel']}` @ `{ctx['module_revision'][:12]}`)\n\n"
        "This cell **is** the repository's pipeline module: the pinned identity constants, snapshot verification "
        "(`verify_snapshot`), staged download (`stage_missing_files`), the named operational ceilings, the public "
        "validation and evaluation helpers, and the pipeline class. The text is the module's, byte for byte, except "
        f"for the rewrite rules listed in `tools/build_notebook.py` (currently {len(REWRITES)}: the default weights "
        "directory becomes working-directory-relative because a notebook has no `__file__`). The repository's parity "
        "test (`tests/test_notebook_parity.py`) fails whenever this cell and the module diverge, so what you run here "
        "is what the repository tests. Nothing in this cell runs a model yet."
    )
    add(_md(embed_md, ""))
    add(
        _code(
            ctx["embedded_text"],
            "",
            {"dimer": {"embedded_module": ctx["module_rel"], "module_sha256": ctx["module_sha256"]}},
        )
    )

    manifest_literal = json.dumps(ctx["manifest"], indent=2, ensure_ascii=False)
    n_files = len(ctx["manifest"]["files"])
    model_md = (
        "## 3. Pin, stage and verify the model\n\n"
        f"The model identity is carried twice — `MODEL_ID`/`MODEL_REVISION` in the module above and the "
        f"`{n_files}`-file manifest below (paths, byte sizes, SHA-256) — and the cell first asserts they agree. "
        "It writes the manifest into the working-directory snapshot, then `stage_missing_files(..., allow_download=True)` "
        f"fetches exactly the entries that are absent from the Hugging Face Hub **at revision `{ctx['MODEL_REVISION'][:12]}…`** "
        "(never `main`), `verify_snapshot` re-hashes every file and raises on the first size or digest mismatch, and only "
        f"then does `{template['pipeline_class']}.from_pretrained(weights_dir=WEIGHTS_DIR)` load the verified files. There "
        "is no fallback to a different download and no remote model code is executed. The effective identity, device "
        "and weight source are printed before any inference."
    )
    model_code = (
        "import json\n\n"
        f"MANIFEST = {manifest_literal}\n\n"
        "if (MANIFEST['modelId'], MANIFEST['revision']) != (MODEL_ID, MODEL_REVISION):\n"
        "    raise RuntimeError('inline manifest does not name the identity carried by the pipeline module; the notebook was not regenerated after a change')\n"
        "WEIGHTS_DIR = DEFAULT_WEIGHTS_DIR\n"
        "WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)\n"
        "with open(WEIGHTS_DIR / MANIFEST_NAME, 'w', encoding='utf-8') as handle:\n"
        "    json.dump(MANIFEST, handle, indent=2)\n"
        "print({'model_id': MODEL_ID, 'revision': MODEL_REVISION, 'license': MODEL_LICENSE, 'files': len(MANIFEST['files']), 'total_bytes': MANIFEST['totalBytes']})\n"
        "fetched = stage_missing_files(WEIGHTS_DIR, allow_download=True)\n"
        "print({'weights_dir': str(WEIGHTS_DIR), 'fetched': fetched})\n"
        "snapshot = verify_snapshot(WEIGHTS_DIR)\n"
        "_files = snapshot.get('files', []) if isinstance(snapshot, dict) else []\n"
        "print({'verified_files': len(_files) if isinstance(_files, list) else _files, 'revision': snapshot.get('revision', MODEL_REVISION) if isinstance(snapshot, dict) else MODEL_REVISION})\n"
        f"pipe = {template['pipeline_class']}.from_pretrained(weights_dir=WEIGHTS_DIR)\n"
        "print({'device': getattr(pipe, 'device', None), 'source': getattr(pipe, 'source', 'local-snapshot')})"
    )
    add(_md(model_md, ""))
    add(_code(model_code, ""))

    for stage in template["cells"]:
        add(_md(stage["md"].format(**fmt), ""))
        if stage.get("code"):
            add(_code(stage["code"].format(**fmt), ""))

    add(_md(template["closing"].format(**fmt), ""))

    return {
        "cells": cells,
        "metadata": {
            "dimer": {
                "notebook_profile": template["profile"],
                "notebook_spec": NOTEBOOK_SPEC,
                "standalone": True,
                "generated_from": {
                    "repository": template["repo_name"],
                    "revision": ctx["module_revision"],
                    "module": ctx["module_rel"],
                    "module_sha256": ctx["module_sha256"],
                    "generator": GENERATOR_VERSION,
                },
            },
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.12"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def to_bytes(notebook: dict[str, Any]) -> bytes:
    return (json.dumps(notebook, indent=1, sort_keys=True, ensure_ascii=False) + "\n").encode("utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--template", type=Path, default=None, help="default: <repo>/tools/notebook_template.py")
    parser.add_argument("--out", type=Path, default=None, help="default: <repo>/tutorials/<notebook_name>")
    parser.add_argument("--check", action="store_true", help="exit 1 if the existing notebook differs (PAR3)")
    args = parser.parse_args(argv)
    repo = args.repo.resolve()
    template = load_template(args.template or repo / "tools" / "notebook_template.py")
    out = args.out or repo / "tutorials" / template["notebook_name"]
    if args.check:
        # PAR3/PAR4: the recorded revision is a provenance label and is carried through the check;
        # drift is caught by content (module text, manifest, pins) — a changed module changes the
        # rendered cell and its SHA-256, so the byte comparison fails regardless of the label.
        rendered = to_bytes(render(repo, template, recorded_revision(out)))
        # Compare on LF: a Windows checkout with core.autocrlf rewrites the file to CRLF.
        current = out.read_bytes().replace(b"\r\n", b"\n") if out.exists() else b""
        if current != rendered:
            print(f"STALE: {out} differs from the generator output; run tools/build_notebook.py", file=sys.stderr)
            return 1
        print(f"OK: {out} is up to date ({len(rendered)} bytes)")
        return 0
    rendered = to_bytes(render(repo, template))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(rendered)
    print(f"wrote {out} ({len(rendered)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
