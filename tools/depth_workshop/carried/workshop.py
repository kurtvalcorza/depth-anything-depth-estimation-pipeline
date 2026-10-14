"""Standalone depth workshop worker; heavy model imports occur only in GPU stages."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import uuid
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

SCENES = {
    "scene_00020": "train",
    "scene_00023": "train",
    "scene_00019": "validation",
    "scene_00024": "validation",
    "scene_00021": "test",
    "scene_00022": "test",
}
STAGES = ["prepare", "depth", "reload", "zoe", "activity", "report"]
REQUIRES = {
    "prepare": [],
    "depth": ["prepare"],
    "reload": ["depth"],
    "zoe": ["prepare"],
    "activity": ["reload"],
    "report": ["prepare", "depth", "reload", "zoe", "activity"],
}
CONFIG = {
    "trainable_blocks": 0,
    "head_epochs": 3,
    "epochs": 0,
    "lr": 1e-5,
    "seed": 42,
    "batch_size": 1,
    "weight_decay": 0.01,
    "gradient_clip": 1.0,
    "precision": "float32",
    "depth_range_metres": [0.6, 350],
    "brightness_factors": [0.6, 1.0, 1.4],
    "qualification": "Candidate",
    "target_resampling": "source nearest-neighbour depth and mask, then inverse depth",
    "prediction_preflight": "reject nonfinite on fixed support; metric predictions positive",
    "selection": "validation aligned AbsRel rounded to six decimals, earliest tie incl epoch 0",
}


# Exit code for a refused learner input (a BYOD directory or an unlabelled image). The worker prints the reason
# once and the notebook cell stops without raising; any other failure keeps its traceback and still raises.
REFUSED_EXIT = 65


class InputRefused(ValueError):
    """A BYOD manifest or file, or an unlabelled image, breaks a documented input rule (not a worker defect)."""


def refuse(error: Exception, action: str) -> None:
    print(f"Refused: {error}. {action} and run this cell again.", flush=True)
    raise SystemExit(REFUSED_EXIT)


# Functions (by file) whose own raise statements state a documented input rule (parse, schema, range or unit) for a
# BYOD directory or an unlabelled image. A ValueError raised anywhere else, for example inside a library or a helper
# these functions call, is a genuine failure: it keeps its traceback and is never shown as a refusal.
INPUT_RULES = {
    "workshop.py": frozenset(
        {"load_byod", "safe_file", "bounded_image", "bounded_array", "check_source_contract", "validate_roles", "support"}
    ),
    "samples.py": frozenset({"validate_dataset", "_check_record", "_as_array"}),
    "pipeline.py": frozenset({"validate_image"}),
    "zoe_reference.py": frozenset({"validate_image"}),
}


def input_rule_violation(error: BaseException) -> bool:
    """True only if `error` (and each `raise ... from` cause) was raised by a raise statement of an INPUT_RULES function."""
    while isinstance(error, ValueError):
        tb = error.__traceback__
        if tb is None:
            return False
        while tb.tb_next is not None:
            tb = tb.tb_next
        code = tb.tb_frame.f_code
        if code.co_name not in INPUT_RULES.get(Path(code.co_filename).name, ()):
            return False
        if error.__cause__ is None:
            return True
        error = error.__cause__
    return False


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def array_digest(array: np.ndarray) -> str:
    array = np.ascontiguousarray(array)
    return hashlib.sha256(str((array.shape, array.dtype.str)).encode() + array.tobytes()).hexdigest()


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


# DEP2-CX2 (owner decision B, 2026-10-13): an image is an informative evaluation only with at least
# INFORMATIVE_MIN_PIXELS scored pixels AND at least INFORMATIVE_MIN_FRACTION of its pixels scored. The
# reference-assisted alignment fits two parameters per image, so a handful of scored pixels can be matched almost
# exactly without showing any structure; 1,000 pixels keeps every fit heavily overdetermined (500 observations per
# parameter), and 1% keeps a small patch from standing in for a whole view. Both were fixed before the default
# sample's counts were inspected. Uninformative images are scored, kept and listed, but excluded from every mean,
# count and selection score.
INFORMATIVE_MIN_PIXELS = 1000
INFORMATIVE_MIN_FRACTION = 0.01


def informative(n_valid: int, valid_fraction: float) -> bool:
    return n_valid >= INFORMATIVE_MIN_PIXELS and valid_fraction >= INFORMATIVE_MIN_FRACTION


def support(depth: np.ndarray, mask: np.ndarray) -> np.ndarray:
    depth, mask = np.asarray(depth), np.asarray(mask)
    if depth.ndim != 2 or mask.shape != depth.shape or mask.dtype != np.bool_:
        raise ValueError("depth must be HxW and mask must be a matching boolean array")
    valid = mask & np.isfinite(depth) & (depth >= 0.6) & (depth <= 350)
    if int(valid.sum()) < 2:
        raise ValueError("need at least two valid reference pixels in [0.6,350] metres")
    return valid


def aligned_values(pred: np.ndarray, depth: np.ndarray, mask: np.ndarray) -> tuple:
    valid = support(depth, mask)
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape != depth.shape or not np.isfinite(pred[valid]).all():
        raise ValueError("prediction shape or finite fixed-support contract failed")
    p, ref = pred[valid], np.asarray(depth, dtype=np.float64)[valid]
    design = np.stack([p, np.ones_like(p)], axis=1)
    coefficients, _, rank, singular = np.linalg.lstsq(design, 1 / ref, rcond=None)
    a, b = coefficients
    inverse = a * p + b
    estimate = 1 / np.maximum(inverse, 1 / 350)
    fit = {
        "scale": float(a),
        "shift": float(b),
        "rank": int(rank),
        "singular_values": singular.tolist(),
        "degenerate": bool(rank < 2),
        "negative_scale": bool(a < 0),
        "far_cap_fraction": float(np.mean(inverse <= 1 / 350)),
    }
    return estimate, ref, valid, fit


def score(pred: np.ndarray, depth: np.ndarray, mask: np.ndarray, aligned: bool = False) -> dict:
    valid = support(depth, mask)
    pred = np.asarray(pred, dtype=np.float64)
    if pred.shape != depth.shape or not np.isfinite(pred[valid]).all():
        raise ValueError("prediction shape or finite fixed-support contract failed")
    fit = {}
    if aligned:
        estimate, ref, valid, fit = aligned_values(pred, depth, mask)
    else:
        estimate, ref = pred[valid], np.asarray(depth, dtype=np.float64)[valid]
        if np.any(estimate <= 0):
            raise ValueError("metric predictions must be positive on fixed support")
    return {
        "abs_rel": float(np.mean(np.abs(estimate - ref) / ref)),
        "delta1": float(np.mean(np.maximum(estimate / ref, ref / estimate) < 1.25)),
        "n_valid": int(valid.sum()),
        "valid_fraction": float(valid.mean()),
        "informative": informative(int(valid.sum()), float(valid.mean())),
        "excluded_predictions": 0,
        "protocol": "reference-assisted affine alignment" if aligned else "unaligned metric metres",
        **fit,
    }


def aggregate(rows: list[dict]) -> dict:
    if not rows:
        return {"n": 0, "abs_rel": None, "delta1": None, "reason": "no records"}
    # DEP2-CX2: uninformative images are listed, not averaged (rows from before this rule count as informative).
    excluded = [
        {key: r.get(key) for key in ("id", "model", "n_valid", "valid_fraction")}
        for r in rows
        if not r.get("informative", True)
    ]
    rows = [r for r in rows if r.get("informative", True)]
    if not rows:
        raise RuntimeError(
            f"every image in this group is uninformative (fewer than {INFORMATIVE_MIN_PIXELS} scored pixels or under "
            f"{INFORMATIVE_MIN_FRACTION:.0%} of the image scored), so no mean can be computed: {excluded}"
        )
    result = {
        "n": len(rows),
        "uninformative_excluded": excluded,
        "abs_rel": float(np.mean([r["abs_rel"] for r in rows])),
        "delta1": float(np.mean([r["delta1"] for r in rows])),
        "aggregation": "equal-image mean",
    }
    for key in ("scene", "domain"):
        result[f"per_{key}"] = {}
        for value in sorted({str(r.get(key, "unspecified")) for r in rows}):
            group = [r for r in rows if str(r.get(key, "unspecified")) == value]
            result[f"per_{key}"][value] = {
                "n": len(group),
                "abs_rel": float(np.mean([r["abs_rel"] for r in group])),
                "delta1": float(np.mean([r["delta1"] for r in group])),
            }
    return result


def brightness(image: Image.Image, factor: float) -> tuple[Image.Image, dict]:
    # The canonical activity uses exactly 0.6, 1.0 and 1.4; the optional own-factor activity may use 0.2..2.0.
    if not 0.2 <= factor <= 2.0:
        raise ValueError(f"brightness factor {factor} is outside 0.2..2.0")
    values = np.asarray(image.convert("RGB"), dtype=np.float32) * factor
    output = np.round(np.clip(values, 0, 255)).astype(np.uint8)
    return Image.fromarray(output), {
        "factor": factor,
        "clipped_pixel_fraction": float(np.any((values < 0) | (values > 255), axis=2).mean()),
        "input_sha256": array_digest(output),
    }


def verify_parity(before: np.ndarray, after: np.ndarray) -> dict:
    before, after = np.asarray(before), np.asarray(after)
    if before.shape != after.shape or not np.isfinite(before).all() or not np.isfinite(after).all():
        raise ValueError("reload shape/nonfinite failure")
    error = float(np.max(np.abs(before.astype(float) - after.astype(float))))
    if not np.allclose(before, after, atol=1e-5, rtol=1e-4):
        raise ValueError(f"reload prediction parity failed: max abs {error}")
    return {"max_absolute_error": error, "atol": 1e-5, "rtol": 1e-4, "passed": True}


def validate_roles(records: list[dict], byod: bool = False) -> dict:
    ids, images, contents, groups = set(), {}, {}, {}
    counts = dict.fromkeys(("train", "validation", "test"), 0)
    for r in records:
        rid, role = r.get("id"), r.get("role")
        if not isinstance(rid, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", rid):
            raise ValueError(f"record id {rid!r}: ids must be 1-64 characters from A-Z a-z 0-9 _ -")
        if rid in ids:
            raise ValueError(f"record id {rid!r} appears more than once")
        ids.add(rid)
        if role not in counts:
            raise ValueError(f"record {rid!r}: role {role!r} must be 'train', 'validation' or 'test'")
        counts[role] += 1
        group = r.get("group_id") or (str(r.get("domain", "")) + "/" + str(r.get("scene", "")))
        if not group or group == "/":
            raise ValueError(f"record {rid!r}: an acquisition group (group_id) is required")
        if group in groups and groups[group] != role:
            raise ValueError(
                f"record {rid!r}: group {group!r} appears in both {groups[group]!r} and {role!r}; "
                "keep every record of an acquisition group in one role"
            )
        groups[group] = role
        try:
            valid = support(r["depth"], r["mask"])
        except ValueError as error:
            raise ValueError(f"record {rid!r}: {error}") from error
        if r["depth"].shape != (r["image"].height, r["image"].width):
            raise ValueError(
                f"record {rid!r}: depth shape {r['depth'].shape} differs from the image "
                f"(height, width) {(r['image'].height, r['image'].width)}"
            )
        rgb_hash = array_digest(np.asarray(r["image"].convert("RGB")))
        rgbd_hash = hashlib.sha256(
            (rgb_hash + array_digest(r["depth"]) + array_digest(r["mask"])).encode()
        ).hexdigest()
        for table, key in ((images, rgb_hash), (contents, rgbd_hash)):
            if key in table:
                raise ValueError(f"record {rid!r}: decoded content duplicates record {table[key]!r}")
            table[key] = rid
        r.update(
            group_id=group,
            image_sha256=rgb_hash,
            rgbd_sha256=rgbd_hash,
            n_valid=int(valid.sum()),
            valid_fraction=float(valid.mean()),
            reference_above_80m=int(np.sum(valid & (r["depth"] > 80))),
        )
    expected = (
        {"train": 6, "validation": 2, "test": 4} if byod else {"train": 14, "validation": 6, "test": 20}
    )
    if any(counts[k] < v if byod else counts[k] != v for k, v in expected.items()):
        raise ValueError(
            f"wrong role counts: {counts}"
            + ("; BYOD needs at least 6 train, 2 validation and 4 test records" if byod else "")
        )
    group_counts = {role: sum(v == role for v in groups.values()) for role in counts}
    if any(n < 2 for n in group_counts.values()):
        raise ValueError(
            f"at least two independent acquisition groups required in every role; found {group_counts}"
        )
    return {
        "counts": counts,
        "group_counts": group_counts,
        "exclusions": [],
        "reference_above_80m_by_role": {
            role: sum(r["reference_above_80m"] for r in records if r["role"] == role) for role in counts
        },
    }


def safe_file(root: Path, name: str) -> Path:
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise ValueError(f"{name!r}: only relative paths inside the BYOD directory, with '/', are allowed")
    rel = Path(name)
    path = (root / rel).resolve()
    if rel.is_absolute() or ".." in rel.parts or not path.is_relative_to(root.resolve()):
        raise ValueError(f"{name!r}: path escapes the BYOD directory")
    if not path.is_file():
        raise ValueError(f"{name!r}: file not found in the BYOD directory")
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError(f"{name!r}: file exceeds 32 MiB")
    return path


def bounded_image(path: Path) -> Image.Image:
    if not path.is_file():
        raise InputRefused(f"{path.name}: file not found")
    if path.stat().st_size > 32 * 1024**2:
        raise ValueError(f"{path.name}: image file exceeds 32 MiB")
    try:
        opened = Image.open(path)
    except (OSError, SyntaxError) as error:  # PIL.UnidentifiedImageError is an OSError
        raise InputRefused(f"{path.name}: the file could not be opened as an image ({error})") from None
    with opened as image:
        w, h = image.size
        if min(w, h) < 32 or max(w, h) > 2048 or max(w, h) / min(w, h) > 4:
            raise ValueError(
                f"{path.name}: {w}x{h} px is outside the limits (sides 32..2048 px, aspect ratio at most 4); "
                "save a resized copy whose longest side is at most 2048 px"
            )
        orientation = image.getexif().get(274, 1)
        if orientation != 1:
            raise ValueError(
                f"{path.name}: EXIF orientation is {orientation}; save a copy with the rotation applied "
                "(for example PIL.ImageOps.exif_transpose) and, for labelled data, rotate its depth and mask "
                "arrays the same way"
            )
        try:
            return image.convert("RGB")
        except (OSError, SyntaxError) as error:  # truncated or corrupt pixel data
            raise InputRefused(f"{path.name}: the image data could not be decoded ({error})") from None


def bounded_array(path: Path, shape: tuple, dtype: np.dtype) -> np.ndarray:
    with path.open("rb") as handle:
        try:
            version = np.lib.format.read_magic(handle)
            header = None
            if version == (1, 0):
                header = np.lib.format.read_array_header_1_0(handle)
            elif version == (2, 0):
                header = np.lib.format.read_array_header_2_0(handle)
        except (ValueError, EOFError) as error:
            raise InputRefused(f"{path.name}: not a readable NPY file ({error})") from None
        if header is None:
            raise ValueError(f"{path.name}: only NPY format versions 1.0 and 2.0 are supported")
        got_shape, _, got_dtype = header
        if got_shape != shape or got_dtype != dtype or got_dtype.hasobject:
            raise ValueError(
                f"{path.name}: expected a {dtype} array of shape {shape} (the image's height x width), "
                f"found {got_dtype} {got_shape}"
            )
        if int(np.prod(got_shape)) * got_dtype.itemsize > 32 * 1024**2:
            raise ValueError(f"{path.name}: array allocation exceeds 32 MiB")
    try:
        return np.load(path, allow_pickle=False)
    except (ValueError, EOFError, OSError) as error:  # truncated or corrupt array data
        raise InputRefused(f"{path.name}: the array data could not be read ({error})") from None


def check_source_contract(record: dict) -> None:
    """The carried Depth Anything per-record contract (>=5% valid coverage, depth <= 1000 m), unprefixed."""
    from depth_reference import samples

    try:
        samples.validate_dataset([record], min_records=1)
    except ValueError as error:
        raise ValueError(str(error).removeprefix("records[0]: ").removeprefix("records[0] ")) from error


def load_byod(root: Path) -> list[dict]:
    """Read and check a BYOD directory. Every documented input rule it breaks raises InputRefused."""
    import zoe_reference
    from depth_reference import pipeline as source

    root = Path(root)
    try:
        manifest = safe_file(root, "manifest.json")
    except ValueError as error:
        if not input_rule_violation(error):
            raise
        raise InputRefused(f"BYOD_PATH {str(root)!r}: {error}") from None
    if manifest.stat().st_size > 1024**2:
        raise InputRefused("manifest.json exceeds 1 MiB")
    try:
        data = read_json(manifest)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise InputRefused(f"manifest.json is not valid UTF-8 JSON ({error})") from None
    if not isinstance(data, dict):
        raise InputRefused(
            f"manifest.json must be a JSON object with 'units' and 'records', found a {type(data).__name__}"
        )
    if data.get("units") != "metres" or not isinstance(data.get("records"), list):
        raise InputRefused("manifest requires units='metres' and records list")
    # DEP2-CX1: references must be z-depth (distance along the optical axis), the convention of the default DIODE
    # sample (planarity test, round 2026-10-12) and of ZoeDepth's output; ray distances are refused, not converted.
    convention = data.get("depth_convention")
    if convention != "z_depth":
        found = "no depth_convention" if convention is None else f"depth_convention {convention!r}"
        raise InputRefused(
            'manifest.json must declare "depth_convention": "z_depth" (depth along the camera\'s optical axis, the'
            f" convention of the default sample and of ZoeDepth), found {found}; convert ray distances first with"
            " z = ray_distance / sqrt(1 + ((u - cx) / fx)**2 + ((v - cy) / fy)**2)"
        )
    if not 12 <= len(data["records"]) <= 128:
        raise InputRefused("BYOD requires 12..128 records")
    records, pixels = [], 0
    for index, row in enumerate(data["records"]):
        if not isinstance(row, dict):
            raise InputRefused(f"record {index}: each record must be a JSON object")
        label = f"record {row.get('id', index)!r}"
        try:
            if not row.get("group_id"):
                raise ValueError("an explicit group_id is required")
            group = row["group_id"]
            if isinstance(group, bool) or not isinstance(group, (str, int)):
                raise ValueError(f"group_id must be a string or an integer, not a {type(group).__name__}")
            for key in ("image", "depth", "mask", "role"):
                if key not in row:
                    raise ValueError(f"missing required key {key!r}")
            if not isinstance(row["role"], str):
                raise ValueError(
                    f"role must be 'train', 'validation' or 'test', not a {type(row['role']).__name__}"
                )
            image = bounded_image(safe_file(root, row["image"]))
            shape = (image.height, image.width)
            pixels += image.height * image.width
            if pixels > 64_000_000:
                raise ValueError("total image pixels across records exceed 64 million")
            depth = bounded_array(safe_file(root, row["depth"]), shape, np.dtype("float32"))
            mask = bounded_array(safe_file(root, row["mask"]), shape, np.dtype("bool"))
            record = {
                **row,
                "image": image,
                "depth": depth,
                "mask": mask,
                "scene": row["group_id"],
                "domain": str(row.get("domain", "BYOD")),
            }
            check_source_contract(record)
            # The image checks prepare() applies, so nothing is refused after a model stage has started.
            source.validate_image(image)
            zoe_reference.validate_image(image)
        except ValueError as error:
            # Only a documented rule is the learner's to fix; any other ValueError is a defect and keeps its traceback.
            if not input_rule_violation(error):
                raise
            raise InputRefused(f"{label}: {error}") from None
        records.append(record)
    try:
        validate_roles(records, byod=True)
    except ValueError as error:
        if not input_rule_violation(error):
            raise
        raise InputRefused(str(error)) from None
    return records


def record_metadata(r: dict) -> dict:
    return {k: v for k, v in r.items() if k not in ("image", "depth", "mask")}


def load_records(root: Path) -> list[dict]:
    rows = read_json(root / "data/records.json")
    return [
        {
            **r,
            "image": Image.open(root / r["image"]).convert("RGB"),
            "depth": np.load(root / r["depth"], allow_pickle=False),
            "mask": np.load(root / r["mask"], allow_pickle=False),
        }
        for r in rows
    ]


def panel(root: Path, record: dict, name: str, predictions: dict | None = None) -> None:
    valid = support(record["depth"], record["mask"])
    images = [("RGB", record["image"])]

    def heat(
        array: np.ndarray, low: float, high: float, use_mask: bool = True, log_scale: bool = False
    ) -> Image.Image:
        if log_scale:
            scaled = np.log(np.maximum(array, low) / low) / np.log(high / low)
        else:
            scaled = (array - low) / (high - low)
        scaled = np.clip(scaled, 0, 1)
        scaled = np.nan_to_num(scaled)
        rgb = np.stack([scaled, 1 - np.abs(2 * scaled - 1), 1 - scaled], axis=-1)
        rgb[~valid if use_mask else np.zeros_like(valid)] = 0.35
        return Image.fromarray(np.round(rgb * 255).astype(np.uint8))

    images += [
        ("Reference metres (log colour: 0.6 to 350)", heat(record["depth"], 0.6, 350, log_scale=True)),
        ("Valid reference support", Image.fromarray(valid.astype(np.uint8) * 255).convert("RGB")),
    ]
    for label, (prediction, aligned) in (predictions or {}).items():
        if aligned:
            values, _, _, _ = aligned_values(prediction, record["depth"], record["mask"])
            raw = np.asarray(prediction)
            images.append(
                (
                    label + " arbitrary units (display min/max)",
                    heat(raw, float(raw.min()), float(raw.max()) + 1e-12, False),
                )
            )
            metric = np.full(valid.shape, np.nan)
            metric[valid] = values
        else:
            metric = prediction
            images.append(
                (label + " metres (log colour: 0.6 to 350)", heat(metric, 0.6, 350, log_scale=True))
            )
        images.append(
            (label + " absolute error metres [0,50]", heat(np.abs(metric - record["depth"]), 0, 50))
        )
    columns = min(3, len(images))
    rows = (len(images) + columns - 1) // columns
    canvas = Image.new("RGB", (420 * columns, 370 * rows + 65), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, image) in enumerate(images):
        image = image.copy()
        image.thumbnail((410, 300))
        x, y = (i % columns) * 420, (i // columns) * 370
        canvas.paste(image, (x, y + 40))
        draw.text((x + 4, y + 4), title, fill="black")
    legend_y = rows * 370
    for column, caption in enumerate(
        ("Metric metres (log)", "Absolute error metres (linear)", "Relative display (normalized)")
    ):
        if column >= columns:
            break
        x = column * 420
        draw.text((x + 4, legend_y), caption, fill="black")
        for pixel in range(400):
            fraction = pixel / 399
            colour = (int(255 * fraction), int(255 * (1 - abs(2 * fraction - 1))), int(255 * (1 - fraction)))
            draw.line((x + pixel, legend_y + 18, x + pixel, legend_y + 28), fill=colour)
        ticks = ((0.6, 2, 10, 50, 350), (0, 12.5, 25, 37.5, 50), (0, 0.25, 0.5, 0.75, 1))[column]
        for value in ticks:
            position = (
                np.log(value / 0.6) / np.log(350 / 0.6) if column == 0 else value / (50 if column == 1 else 1)
            )
            draw.text((x + min(374, int(position * 399)), legend_y + 33), str(value), fill="black")
    path = root / "outputs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def prepare(root: Path, byod: str | None = None) -> None:
    import zoe_reference
    from depth_reference import pipeline as source
    from depth_reference import samples

    if byod:
        records = load_byod(Path(byod))
    else:
        for attempt in range(3):
            try:
                records = samples.read_corpus(samples.fetch_corpus(cache_dir=root / "cache"))
                break
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == 2:
                    raise
                time.sleep(2**attempt)
        for r in records:
            r["role"] = SCENES[r["scene"]]
    result = validate_roles(records, byod=bool(byod))
    for r in records:
        try:
            source.validate_image(r["image"])
            zoe_reference.validate_image(r["image"])
            # Preserve source adaptation validation, including >=5% sensor-valid coverage.
            check_source_contract(r)
        except ValueError as error:
            raise ValueError(f"record {r['id']!r}: {error}") from error
    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in records:
        image_path, depth_path, mask_path = (
            f"data/{r['id']}{s}" for s in (".png", "_depth.npy", "_mask.npy")
        )
        r["image"].save(root / image_path)
        np.save(root / depth_path, r["depth"], allow_pickle=False)
        np.save(root / mask_path, r["mask"], allow_pickle=False)
        rows.append({**record_metadata(r), "image": image_path, "depth": depth_path, "mask": mask_path})
    write_json(data_dir / "records.json", rows)
    manifest = {
        **result,
        "private_byod": bool(byod),
        "units": "metres",
        "records": [
            {
                **record_metadata(r),
                "shape": list(r["depth"].shape),
                "depth_sha256": array_digest(r["depth"]),
                "mask_sha256": array_digest(r["mask"]),
            }
            for r in records
        ],
        "dataset": None
        if byod
        else {
            "release": samples.CORPUS_RELEASE,
            "url": samples.CORPUS_URL,
            "archive_bytes": samples.CORPUS_ARCHIVE_BYTES,
            "license": samples.CORPUS_LICENSE,
            "bytes": samples.CORPUS_BYTES,
        },
    }
    write_json(root / "outputs/sample_manifest.json", manifest)
    write_json(root / "outputs/config.json", CONFIG)
    medians = [
        float(np.median(r["depth"][support(r["depth"], r["mask"])])) for r in records if r["role"] == "train"
    ]
    constant = float(np.median(medians))
    baseline_rows = []
    for r in records:
        if r["role"] == "train":
            continue
        shape = r["depth"].shape
        oracle = float(np.median(r["depth"][support(r["depth"], r["mask"])]))
        ramp = np.repeat(np.linspace(0, 1, shape[0], dtype=np.float32)[:, None], shape[1], axis=1)
        for label, prediction, aligned in (
            ("training_median", np.full(shape, constant), False),
            ("oracle_median", np.full(shape, oracle), False),
            ("vertical_ramp", ramp, True),
        ):
            row = {**record_metadata(r), "model": label, **score(prediction, r["depth"], r["mask"], aligned)}
            if label == "oracle_median":
                row["protocol"] = "reference-assisted oracle median"
            baseline_rows.append(row)
    write_json(root / "outputs/baselines.json", {"training_constant_metres": constant, "rows": baseline_rows})
    write_json(
        root / "outputs/prepare.json",
        {**result, "private_byod": bool(byod), "training_constant_metres": constant},
    )
    for domain in sorted({r["domain"] for r in records}):
        example = next(r for r in records if r["domain"] == domain)
        panel(root, example, f"reference_{example['id']}.png")
    print("Prepared", result, flush=True)


def device() -> str:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("This model stage requires a Colab CUDA GPU; CPU validation never runs models")
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    return "cuda:0"


def predict_rows(root: Path, model: Any, records: list[dict], label: str, aligned: bool) -> list[dict]:
    import torch

    rows = []
    folder = root / "outputs/raw" / label
    folder.mkdir(parents=True, exist_ok=True)
    for r in records:
        torch.cuda.synchronize()
        started = time.perf_counter()
        prediction = model.predict(r["image"])["depth"]
        torch.cuda.synchronize()
        row = {
            **record_metadata(r),
            "model": label,
            "seconds": time.perf_counter() - started,
            **score(prediction, r["depth"], r["mask"], aligned),
        }
        np.save(folder / f"{r['id']}.npy", prediction, allow_pickle=False)
        rows.append(row)
    write_json(
        root / "outputs" / f"{label}.json",
        {
            "rows": rows,
            "test": aggregate([r for r in rows if r["role"] == "test"]),
            "validation": aggregate([r for r in rows if r["role"] == "validation"]),
        },
    )
    return rows


def tensor_hashes(model: Any, names: list[str]) -> dict:
    state = model.state_dict()
    return {name: array_digest(state[name].detach().cpu().numpy()) for name in names}


def depth(root: Path) -> None:
    import torch
    from depth_reference.pipeline import DepthAnythingPipeline

    runtime_device = device()
    records = load_records(root)
    train = [r for r in records if r["role"] == "train"]
    val = [r for r in records if r["role"] == "validation"]
    evaluation = [r for r in records if r["role"] != "train"]
    model = DepthAnythingPipeline.from_pretrained(
        device=runtime_device, weights_dir=root / "weights/depth", allow_download=True
    )
    model.predict(val[0]["image"])
    predict_rows(root, model, evaluation, "depth_frozen", True)
    params = dict(model._model.named_parameters())
    names = model._trainable_names(0)
    initial = {n: params[n].detach().cpu().clone() for n in names}
    frozen_names = [n for n in model._model.state_dict() if n not in names]
    before_frozen = tensor_hashes(model._model, frozen_names)
    updates, observations = [0], []
    real_adamw = torch.optim.AdamW

    def counted_adamw(parameters: Any, *args: Any, **kwargs: Any) -> Any:
        parameters = list(parameters)
        if {id(p) for p in parameters} != {id(params[n]) for n in names}:
            raise RuntimeError("unexpected trainable tensor set")
        optimiser = real_adamw(parameters, *args, **kwargs)

        def step_hook(*_: Any) -> None:
            updates[0] += 1

        optimiser.register_step_post_hook(step_hook)
        return optimiser

    def progress(entry: dict) -> None:
        delta = max(float((params[n].detach().cpu() - initial[n]).abs().max()) for n in names)
        observations.append({**entry, "max_weight_delta_before_selection": delta, "actual_steps": updates[0]})
        print("Epoch", entry["epoch"], "loss", entry["train_loss"], "validation", entry["val"], flush=True)
        write_json(root / "outputs/training_progress.json", observations)

    # Strict workshop support replaces only evaluation; source rounding/selection remains unchanged.
    def strict_evaluate(rows: list[dict]) -> dict:
        metrics = [score(model.predict(r["image"])["depth"], r["depth"], r["mask"], True) for r in rows]
        return aggregate(metrics)

    model.evaluate = strict_evaluate
    torch.optim.AdamW = counted_adamw
    try:
        history = model.adapt(
            train, val, trainable_blocks=0, head_epochs=3, epochs=0, lr=1e-5, seed=42, progress=progress
        )
    finally:
        torch.optim.AdamW = real_adamw
    if (
        updates[0] != 3 * len(train)
        or len(observations) != 3
        or not any(r["max_weight_delta_before_selection"] > 0 for r in observations)
    ):
        raise RuntimeError("expected real optimizer updates were not observed")
    after_frozen = tensor_hashes(model._model, frozen_names)
    if before_frozen != after_frozen:
        raise RuntimeError("frozen tensors changed")
    history.update(
        actual_optimizer_steps=updates[0],
        epoch_observations=observations,
        selected_max_weight_delta=max(
            float((params[n].detach().cpu() - initial[n]).abs().max()) for n in names
        ),
        frozen_tensor_hashes_before=before_frozen,
        frozen_tensor_hashes_after=after_frozen,
        training_pid=os.getpid(),
        trainable_tensor_names=names,
        trainable_parameters=sum(params[n].numel() for n in names),
        selection_rounding="source rounds to six decimals; earliest tie; epoch zero competes",
    )
    write_json(root / "outputs/training_history.json", history)
    write_json(
        root / "outputs/selection.json",
        {
            "selected_epoch": history["best_epoch"],
            "epoch_zero_selected": history["best_epoch"] == 0,
            "selection": CONFIG["selection"],
        },
    )
    model.save_artifact(
        root / "outputs/adapter", metadata={"workshop": CONFIG, "source": read_json(root / "source.json")}
    )
    predict_rows(root, model, evaluation, "depth_selected_live", True)
    example = evaluation[0]
    panel(
        root,
        example,
        "depth_first.png",
        {"Selected depth": (np.load(root / f"outputs/raw/depth_selected_live/{example['id']}.npy"), True)},
    )


def reload(root: Path) -> None:
    from depth_reference.pipeline import DepthAnythingPipeline

    if read_json(root / "outputs/training_history.json")["training_pid"] == os.getpid():
        raise RuntimeError("reload must run in a new process")
    model = DepthAnythingPipeline.from_artifact(
        root / "outputs/adapter", device=device(), weights_dir=root / "weights/depth"
    )
    records = [r for r in load_records(root) if r["role"] != "train"]
    rows = predict_rows(root, model, records, "depth_reloaded", True)
    checks = []
    original = read_json(root / "outputs/depth_selected_live.json")
    for r in records:
        checks.append(
            {
                "id": r["id"],
                **verify_parity(
                    np.load(root / f"outputs/raw/depth_selected_live/{r['id']}.npy", allow_pickle=False),
                    np.load(root / f"outputs/raw/depth_reloaded/{r['id']}.npy", allow_pickle=False),
                ),
            }
        )
    drift = {}
    for role in ("validation", "test"):
        measured = aggregate([r for r in rows if r["role"] == role])
        for key in ("abs_rel", "delta1"):
            value = abs(measured[key] - original[role][key])
            drift[f"{role}_{key}"] = value
            if value > 1e-6:
                raise RuntimeError("independently recomputed metric reload parity failed")
    write_json(
        root / "outputs/reload_parity.json",
        {
            "passed": True,
            "checks": checks,
            "metric_drift": drift,
            "metric_atol": 1e-6,
            "reload_pid": os.getpid(),
        },
    )


def zoe(root: Path) -> None:
    from zoe_reference import ZoeDepthMetricPipeline

    model = ZoeDepthMetricPipeline.from_pretrained(
        device=device(), weights_dir=root / "weights/zoe", allow_download=True
    )
    records = [r for r in load_records(root) if r["role"] != "train"]
    model.predict(records[0]["image"])
    predict_rows(root, model, records, "zoe_metrics", False)
    example = records[0]
    panel(
        root,
        example,
        "zoe_first.png",
        {"ZoeDepth": (np.load(root / f"outputs/raw/zoe_metrics/{example['id']}.npy"), False)},
    )


def paired(before: list[dict], after: list[dict]) -> dict:
    old = {r["id"]: r for r in before}
    rows = [
        {
            "id": r["id"],
            "scene": r.get("scene", "unspecified"),
            "domain": r.get("domain", "unspecified"),
            "abs_rel_delta": r["abs_rel"] - old[r["id"]]["abs_rel"],
            "delta1_delta": r["delta1"] - old[r["id"]]["delta1"],
            "informative": bool(r.get("informative", True) and old[r["id"]].get("informative", True)),
        }
        for r in after
    ]
    # DEP2-CX2: counts and means use informative pairs only; the uninformative ones are listed.
    excluded = [r["id"] for r in rows if not r["informative"]]
    every_row, rows = rows, [r for r in rows if r["informative"]]
    return {
        "rows": every_row,
        "uninformative_excluded": excluded,
        "improved": sum(r["abs_rel_delta"] < -1e-12 for r in rows),
        "unchanged": sum(abs(r["abs_rel_delta"]) <= 1e-12 for r in rows),
        "worsened": sum(r["abs_rel_delta"] > 1e-12 for r in rows),
        "per_scene": {
            scene: {
                "n": sum(r["scene"] == scene for r in rows),
                "mean_abs_rel_delta": float(
                    np.mean([r["abs_rel_delta"] for r in rows if r["scene"] == scene])
                ),
            }
            for scene in sorted({r["scene"] for r in rows})
        },
    }


def activity(root: Path) -> None:
    from depth_reference.pipeline import DepthAnythingPipeline

    model = DepthAnythingPipeline.from_artifact(
        root / "outputs/adapter", device=device(), weights_dir=root / "weights/depth"
    )
    records = [r for r in load_records(root) if r["role"] == "validation"]
    all_rows = []
    for factor in (0.6, 1.0, 1.4):
        changed, metadata = [], {}
        for r in records:
            image, info = brightness(r["image"], factor)
            metadata[r["id"]] = info
            changed.append({**r, "image": image})
        rows = predict_rows(root, model, changed, f"brightness_{factor}", True)
        all_rows.extend([{**r, **metadata[r["id"]]} for r in rows])
    base = [r for r in all_rows if r["factor"] == 1.0]
    comparisons = {str(f): paired(base, [r for r in all_rows if r["factor"] == f]) for f in (0.6, 1.4)}
    write_json(
        root / "outputs/brightness_experiment.json",
        {
            "rows": all_rows,
            "comparisons": comparisons,
            "role": "reused validation; descriptive activity only",
            "selection_changed": False,
        },
    )
    example = records[0]
    panel(
        root,
        example,
        "activity_first.png",
        {
            f"factor {f}": (np.load(root / f"outputs/raw/brightness_{f}/{example['id']}.npy"), True)
            for f in (0.6, 1.0, 1.4)
        },
    )


def select_figures(test: list[dict], change_rows: list[dict]) -> dict:
    """Fixed figure rule: first test record, lowest/highest paired AbsRel change, first record per domain.

    When adaptation changed no test image (epoch 0 selected), the extremes are only the lowest/highest id
    and are labelled that way instead of best/worst.
    """
    ranked = sorted(change_rows, key=lambda r: (r["abs_rel_delta"], r["id"]))
    changed = any(abs(r["abs_rel_delta"]) > 1e-12 for r in change_rows)
    selected = {
        test[0]["id"]: "first test record",
        ranked[0]["id"]: "best adaptation delta" if changed else "no adaptation change; lowest id",
        ranked[-1]["id"]: "worst adaptation delta" if changed else "no adaptation change; highest id",
    }
    for domain in sorted({r["domain"] for r in test}):
        selected.setdefault(
            next(r["id"] for r in test if r["domain"] == domain), "first test record in domain"
        )
    return selected


def report(root: Path) -> None:
    outputs = root / "outputs"
    records = load_records(root)
    rows = read_json(outputs / "baselines.json")["rows"]
    for label in ("depth_frozen", "depth_reloaded", "zoe_metrics"):
        rows += read_json(outputs / f"{label}.json")["rows"]
    summaries = {
        label: aggregate([r for r in rows if r["model"] == label and r["role"] == "test"])
        for label in sorted({r["model"] for r in rows})
    }
    write_json(outputs / "metrics.json", {"rows": rows, "test_equal_image_means": summaries})
    fields = [
        "id",
        "role",
        "scene",
        "domain",
        "model",
        "protocol",
        "abs_rel",
        "delta1",
        "n_valid",
        "valid_fraction",
        "informative",
    ]
    with (outputs / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    with (outputs / "metrics.csv").open(encoding="utf-8", newline="") as handle:
        restored = list(csv.DictReader(handle))
    if len(restored) != len(rows) or any(
        float(a["abs_rel"]) != b["abs_rel"] for a, b in zip(restored, rows, strict=True)
    ):
        raise RuntimeError("exported table reload failed")
    change = paired(
        [r for r in rows if r["model"] == "depth_frozen" and r["role"] == "test"],
        [r for r in rows if r["model"] == "depth_reloaded" and r["role"] == "test"],
    )
    write_json(outputs / "paired_changes.json", change)
    sample = read_json(outputs / "sample_manifest.json")
    if sample["private_byod"]:
        sample_note = (
            f"Your test split has {sample['counts']['test']} records in {sample['group_counts']['test']} "
            "acquisition groups; overlap with upstream pretraining data is unknown."
        )
    else:
        sample_note = (
            "Default test views cover only two scenes; "
            "contamination from upstream pretraining is not ruled out."
        )
    lines = [
        "# Depth estimation notebook: measured results",
        "",
        "Candidate: execution does not confer release approval.",
        "",
    ]
    for heading, labels in (
        (
            "Reference-assisted affine alignment / oracle baseline",
            ["depth_frozen", "depth_reloaded", "vertical_ramp", "oracle_median"],
        ),
        ("Unaligned metric metres", ["zoe_metrics", "training_median"]),
    ):
        lines += [
            "## " + heading,
            "",
            "| Method | Views | AbsRel (lower) | Delta-1 (higher) |",
            "|---|---:|---:|---:|",
        ]
        for label in labels:
            row = summaries[label]
            lines.append(f"| {label} | {row['n']} | {row['abs_rel']:.6f} | {row['delta1']:.6f} |")
        lines.append("")
    excluded = sorted({(e["id"], e["n_valid"]) for s in summaries.values() for e in s.get("uninformative_excluded", [])})
    lines.append(
        f"Uninformative test images (fewer than {INFORMATIVE_MIN_PIXELS} scored pixels or under "
        f"{INFORMATIVE_MIN_FRACTION:.0%} of the image scored; scored and kept, excluded from these means): "
        + (", ".join(f"{i} ({n} pixels)" for i, n in excluded) or "none")
        + "."
    )
    lines.append("")
    lines += [
        "Equal-image means; see metrics.json for scene/domain means. No cross-protocol ranking.",
        "Reference alignment uses test references and is not a deployable metre calibration.",
        sample_note,
        f"Adaptation: {change['improved']} improved, {change['unchanged']} unchanged, "
        f"{change['worsened']} worsened (AbsRel).",
    ]
    (outputs / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    test = [r for r in records if r["role"] == "test"]
    selected = select_figures(test, change["rows"])
    write_json(outputs / "figure_selection.json", selected)
    for rid in selected:
        r = next(r for r in test if r["id"] == rid)
        panel(
            root,
            r,
            f"figures/{rid}.png",
            {
                label: (
                    np.load(outputs / f"raw/{label}/{rid}.npy", allow_pickle=False),
                    label != "zoe_metrics",
                )
                for label in ("depth_frozen", "depth_reloaded", "zoe_metrics")
            },
        )
    private = read_json(outputs / "sample_manifest.json")["private_byod"]
    provenance = {
        "python": sys.version,
        "packages": {
            name: importlib.metadata.version(name)
            for name in (
                "numpy",
                "Pillow",
                "torch",
                "torchvision",
                "transformers",
                "safetensors",
                "huggingface-hub",
            )
        },
        "source": read_json(root / "source.json"),
        "worker_sha256": digest(Path(__file__)),
        "lock_sha256": digest(root / "requirements.txt"),
        "model_manifests": {
            name: read_json(root / f"weights/{name}/dimer-base-manifest.json") for name in ("depth", "zoe")
        },
        "stage_receipts_snapshot_before_report_completion": read_json(root / "stage_receipts.json"),
        "strict_prediction_support": CONFIG["prediction_preflight"],
    }
    write_json(outputs / "provenance.json", provenance)
    write_json(
        outputs / "run_summary.json",
        {
            "execution": "required model stages completed and parity passed",
            "qualification": (
                "Candidate; fresh hosted run and representative labelled BYOD require maintainer review"
            ),
            "selected_epoch": read_json(outputs / "selection.json")["selected_epoch"],
            "actual_updates": read_json(outputs / "training_history.json")["actual_optimizer_steps"],
            "protocols": ["reference-assisted affine alignment", "unaligned metric metres"],
            "private_byod": private,
        },
    )
    allow = [
        "sample_manifest.json",
        "config.json",
        "prepare.json",
        "baselines.json",
        "training_history.json",
        "selection.json",
        "reload_parity.json",
        "brightness_experiment.json",
        "paired_changes.json",
        "metrics.json",
        "metrics.csv",
        "summary.md",
        "figure_selection.json",
        "run_summary.json",
        "provenance.json",
        "adapter/manifest.json",
        "adapter/adapter.safetensors",
    ]
    if not private:
        allow += [str(p.relative_to(outputs)).replace("\\", "/") for p in (outputs / "figures").glob("*.png")]
    checksums = {name: digest(outputs / name) for name in allow}
    write_json(outputs / "checksums.json", checksums)
    archive = outputs / "depth_results.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        for name in [*allow, "checksums.json"]:
            bundle.write(outputs / name, name)
    with zipfile.ZipFile(archive) as bundle:
        for name, checksum in checksums.items():
            if hashlib.sha256(bundle.read(name)).hexdigest() != checksum:
                raise RuntimeError("ZIP checksum mismatch")
    # Raw arrays remain separate from displays/shareable ZIP; reread every array for export integrity.
    raw_checks = {}
    for path in (outputs / "raw").glob("*/*.npy"):
        array = np.load(path, allow_pickle=False)
        if array.ndim != 2 or not np.isfinite(array).all():
            raise RuntimeError("raw array export invalid")
        raw_checks[str(path.relative_to(outputs))] = digest(path)
    write_json(outputs / "raw_checksums.json", raw_checks)
    print((outputs / "summary.md").read_text(encoding="utf-8"), flush=True)


def stage_signature(root: Path) -> str:
    paths = [root / "outputs/sample_manifest.json", root / "outputs/config.json", root / "data/records.json"]
    paths += [root / "source.json", root / "requirements.txt", Path(__file__).resolve()]
    paths += sorted((root / "weights").glob("*/dimer-base-manifest.json"))
    paths += sorted((root / "data").glob("*.npy")) + sorted((root / "data").glob("*.png"))
    return hashlib.sha256("".join(digest(p) for p in paths).encode()).hexdigest()


def begin_stage(root: Path, stage: str) -> dict:
    path = root / "stage_receipts.json"
    state = read_json(path) if path.exists() else {"run_id": uuid.uuid4().hex, "stages": {}}
    if stage == "prepare":
        state = {"run_id": uuid.uuid4().hex, "stages": {}}
    else:
        for required in REQUIRES[stage]:
            if state["stages"].get(required, {}).get("status") != "complete":
                raise RuntimeError(f"missing current prerequisite {required}")
            products = state["stages"][required].get("product_sha256", {})
            for relative, expected in products.items():
                product_path = root / "outputs" / relative
                if not product_path.is_file() or digest(product_path) != expected:
                    raise RuntimeError(f"prerequisite output changed: {relative}; rerun {required}")
        if state.get("data_signature") != stage_signature(root):
            raise RuntimeError("prepared data/config changed; rerun prepare")
        descendants = {stage}
        for _ in STAGES:
            descendants.update(s for s, parents in REQUIRES.items() if any(p in descendants for p in parents))
        for item in descendants:
            state["stages"].pop(item, None)
    state["stages"][stage] = {"status": "running", "pid": os.getpid()}
    write_json(path, state)
    return state


def finish_stage(root: Path, stage: str, state: dict, elapsed: float) -> None:
    receipt = {"status": "complete", "pid": os.getpid(), "seconds": elapsed}
    if stage == "prepare":
        state["data_signature"] = stage_signature(root)
    if "torch" in sys.modules and sys.modules["torch"].cuda.is_available():
        torch = sys.modules["torch"]
        torch.cuda.synchronize()
        receipt.update(
            gpu=torch.cuda.get_device_name(0),
            peak_gpu_bytes=torch.cuda.max_memory_allocated(),
            precision="float32",
        )
    try:
        import resource

        receipt["peak_host_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
    except ImportError:
        receipt["peak_host_rss_bytes"] = None
    products = {
        "prepare": ["sample_manifest.json", "config.json", "prepare.json", "baselines.json"],
        "depth": [
            "depth_frozen.json",
            "depth_selected_live.json",
            "training_history.json",
            "selection.json",
            "adapter/*",
            "raw/depth_frozen/*.npy",
            "raw/depth_selected_live/*.npy",
        ],
        "reload": ["depth_reloaded.json", "reload_parity.json", "raw/depth_reloaded/*.npy"],
        "zoe": ["zoe_metrics.json", "raw/zoe_metrics/*.npy"],
        "activity": ["brightness_experiment.json", "brightness_*.json", "raw/brightness_*/*.npy"],
        "report": ["metrics.json", "metrics.csv", "summary.md", "run_summary.json", "depth_results.zip"],
    }
    outputs = root / "outputs"
    receipt["product_sha256"] = {
        path.relative_to(outputs).as_posix(): digest(path)
        for pattern in products[stage]
        for path in outputs.glob(pattern)
        if path.is_file()
    }
    state["stages"][stage] = receipt
    write_json(root / "stage_receipts.json", state)


def byod_run(root: Path, directory: str) -> None:
    load_byod(Path(directory))  # Refuse malformed data before any child model stage.
    child = root / "byod_runs" / uuid.uuid4().hex
    child.mkdir(parents=True)
    (child / "weights").symlink_to((root / "weights").resolve(), target_is_directory=True)
    for name in ("source.json", "requirements.txt"):
        (child / name).write_bytes((root / name).read_bytes())
    for stage in STAGES:
        command = [sys.executable, str(Path(__file__).resolve()), "--root", str(child), "--stage", stage]
        if stage == "prepare":
            command += ["--byod", directory]
        subprocess.run(command, check=True)
    write_json(
        root / "byod_output.json", {"root": str(child), "archive": str(child / "outputs/depth_results.zip")}
    )


def unlabelled_preview(image: Image.Image, relative: np.ndarray, metric: np.ndarray, path: Path) -> None:
    """RGB | relative inverse depth (display min/max, arbitrary units) | ZoeDepth metres (log 0.6..350)."""

    def colour(scaled: np.ndarray) -> Image.Image:
        scaled = np.clip(np.nan_to_num(scaled), 0, 1)
        rgb = np.stack([scaled, 1 - np.abs(2 * scaled - 1), 1 - scaled], axis=-1)
        return Image.fromarray(np.round(rgb * 255).astype(np.uint8))

    rel = np.asarray(relative, dtype=np.float64)
    low, high = float(rel.min()), float(rel.max())
    panels = [
        ("RGB input", image.convert("RGB")),
        (
            "Depth Anything: relative inverse depth\n(arbitrary units, display min/max; larger = nearer)",
            colour((rel - low) / (high - low + 1e-12)),
        ),
        (
            "ZoeDepth: predicted metres\n(log colour 0.6 to 350 m; blue = near; red = far)",
            colour(np.log(np.maximum(np.asarray(metric, dtype=np.float64), 0.6) / 0.6) / np.log(350 / 0.6)),
        ),
    ]
    canvas = Image.new("RGB", (420 * len(panels), 380), "white")
    draw = ImageDraw.Draw(canvas)
    for i, (title, panel_image) in enumerate(panels):
        panel_image = panel_image.copy()
        panel_image.thumbnail((410, 300))
        canvas.paste(panel_image, (i * 420, 40))
        draw.text((i * 420 + 4, 4), title, fill="black")
    # Metre colour bar under the ZoeDepth panel, on the same log scale as the map (display only).
    left = (len(panels) - 1) * 420 + 50
    canvas.paste(colour(np.tile(np.linspace(0, 1, 300), (12, 1))), (left, 350))
    draw.text((left - 44, 350), "0.6 m", fill="black")
    draw.text((left + 146, 364), "14.5 m", fill="black")
    draw.text((left + 304, 350), "350 m", fill="black")
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def own_activity(root: Path, factor: float) -> None:
    """Optional learner-chosen brightness factor on the reused validation views.

    Writes only under ``own_activity/<run>/`` and ``own_activity_output.json``; canonical outputs, stage
    receipts and the results ZIP are untouched, and nothing here can change model selection.
    """
    if not 0.2 <= factor <= 2.0:
        raise ValueError(f"OWN_BRIGHTNESS_FACTOR {factor} is outside 0.2..2.0; choose a value in that range")
    receipts = root / "stage_receipts.json"
    stages = read_json(receipts)["stages"] if receipts.is_file() else {}
    experiment = root / "outputs/brightness_experiment.json"
    if stages.get("activity", {}).get("status") != "complete" or not experiment.is_file():
        raise RuntimeError("Run the section 8 brightness activity first (Run all), then choose a factor")
    from depth_reference.pipeline import DepthAnythingPipeline

    base = [r for r in read_json(experiment)["rows"] if r["factor"] == 1.0]
    target = root / "own_activity" / uuid.uuid4().hex
    model = DepthAnythingPipeline.from_artifact(
        root / "outputs/adapter", device=device(), weights_dir=root / "weights/depth"
    )
    records = [r for r in load_records(root) if r["role"] == "validation"]
    changed, inputs = [], {}
    for r in records:
        image, info = brightness(r["image"], factor)
        inputs[r["id"]] = info
        changed.append({**r, "image": image})
    rows = predict_rows(target, model, changed, "own_factor", True)
    result = {
        "root": str(target),
        "factor": factor,
        "role": "reused validation; descriptive activity only",
        "selection_changed": False,
        "compared_with": "factor 1.0 rows of outputs/brightness_experiment.json",
        "inputs": inputs,
        "comparison": paired(base, rows),
    }
    write_json(target / "own_activity.json", result)
    write_json(root / "own_activity_output.json", result)
    example = records[0]
    panel(
        target,
        example,
        "own_factor.png",
        {
            "factor 1.0": (np.load(root / f"outputs/raw/brightness_1.0/{example['id']}.npy"), True),
            f"factor {factor}": (np.load(target / f"outputs/raw/own_factor/{example['id']}.npy"), True),
        },
    )


def infer(root: Path, image_path: str) -> None:
    import gc

    if not (root / "outputs/adapter/manifest.json").is_file():
        raise RuntimeError(
            "No selected adapter yet: complete Run all through section 5 (adaptation) before this cell"
        )
    from depth_reference.pipeline import DepthAnythingPipeline, validate_image
    from zoe_reference import ZoeDepthMetricPipeline
    from zoe_reference import validate_image as validate_zoe

    name = Path(image_path).name
    try:
        image = bounded_image(Path(image_path))
        validate_image(image)
        validate_zoe(image)
    except ValueError as error:
        if not input_rule_violation(error):
            raise
        message = str(error)
        raise InputRefused(message if message.startswith(name) else f"{name}: {message}") from None
    target = root / "unlabelled" / uuid.uuid4().hex
    target.mkdir(parents=True)
    model = DepthAnythingPipeline.from_artifact(
        root / "outputs/adapter", device=device(), weights_dir=root / "weights/depth"
    )
    relative = model.predict(image)["depth"]
    if not np.isfinite(relative).all():
        raise ValueError("nonfinite relative prediction")
    np.save(target / "relative_inverse_depth.npy", relative, allow_pickle=False)
    del model
    gc.collect()
    import torch

    torch.cuda.empty_cache()
    model = ZoeDepthMetricPipeline.from_pretrained(
        device=device(), weights_dir=root / "weights/zoe", allow_download=True
    )
    metric = model.predict(image)["depth"]
    if not np.isfinite(metric).all() or np.any(metric <= 0):
        raise ValueError("invalid metric prediction")
    np.save(target / "predicted_metres.npy", metric, allow_pickle=False)
    unlabelled_preview(image, relative, metric, target / "preview.png")
    result = {
        "root": str(target),
        "input_sha256": array_digest(np.asarray(image)),
        "abs_rel": None,
        "delta1": None,
        "reason": "no reference; accuracy not measurable",
        "alignment": "none",
        "outputs": {
            "relative_inverse_depth.npy": "relative inverse depth; arbitrary units",
            "predicted_metres.npy": "ZoeDepth predicted metres",
            "preview.png": "display only: RGB, relative inverse depth (display min/max), metres (log colour)",
        },
    }
    write_json(target / "result.json", result)
    write_json(root / "unlabelled_output.json", result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=[*STAGES, "byod", "infer", "own_activity"], required=True)
    parser.add_argument("--byod")
    parser.add_argument("--image")
    parser.add_argument("--factor", type=float)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    if args.stage == "own_activity":
        if args.factor is None:
            parser.error("--factor required")
        own_activity(root, args.factor)
        return
    if args.stage == "byod":
        if not args.byod:
            parser.error("--byod directory required")
        try:
            byod_run(root, args.byod)
        except ValueError as error:
            # Only a documented input rule (InputRefused) is a refusal; any other error is a genuine failure.
            if not isinstance(error, InputRefused):
                raise
            refuse(error, "Fix that file or manifest entry in BYOD_PATH")
        return
    if args.stage == "infer":
        if not args.image:
            parser.error("--image path required")
        try:
            infer(root, args.image)
        except ValueError as error:
            if not isinstance(error, InputRefused):
                raise
            refuse(error, "Save a corrected copy, set IMAGE_PATH to it")
        return
    state = begin_stage(root, args.stage)
    start = time.perf_counter()
    try:
        if args.stage == "prepare":
            prepare(root, args.byod)
        else:
            globals()[args.stage](root)
        finish_stage(root, args.stage, state, time.perf_counter() - start)
    except BaseException as error:
        state["stages"][args.stage] = {
            "status": "failed",
            "pid": os.getpid(),
            "error_type": type(error).__name__,
        }
        write_json(root / "stage_receipts.json", state)
        raise


if __name__ == "__main__":
    main()
