"""Depth metrics for the adaptation contract, implemented here with no external scorer.

The model emits **relative inverse depth** (disparity up to an unknown per-image scale and shift), so every
prediction is first aligned to the reference by least squares in inverse-depth space over the valid pixels
(``1 / depth = s * pred + t``), inverted to metres, and then scored:

- **AbsRel** — ``mean(|est - ref| / ref)`` over valid pixels (lower is better; 0 is perfect);
- **δ1** — the fraction of valid pixels with ``max(est / ref, ref / est) < 1.25`` (higher is better;
  1 is perfect).

Only reference pixels inside ``EVAL_DEPTH_RANGE_M`` (0.6..350 m, the DIODE sensor range) are scored, and the
aligned inverse depth is floored at ``1 / 350 m`` so a pixel the fit pushes past the far cap counts as 350 m
rather than as an unbounded error — the standard cap of relative-depth evaluation. Both metrics are reported
per image and as the mean over images (each image weighs the same regardless of its valid pixel count), plus
per domain when records carry one. Two trivial priors frame the numbers: the **constant
prior** (every pixel at the image's own median reference depth — an oracle constant, the strongest flat
guess) and the **vertical-gradient prior** (inverse depth rising linearly from the top row to the bottom row,
aligned exactly like a model prediction — "the ground is nearer at the bottom of the frame").
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

DELTA_THRESHOLD = 1.25
MIN_VALID_PIXELS = 2
EVAL_DEPTH_RANGE_M = (0.6, 350.0)  # reference pixels scored; also the far cap of the aligned estimate

METRIC_DEFINITIONS = {
    "alignment": (
        "per image, least-squares scale and shift mapping the prediction onto 1 / reference depth over the "
        "valid pixels inside EVAL_DEPTH_RANGE_M; the aligned inverse depth is floored at 1 / the far cap"
    ),
    "abs_rel": "mean over valid pixels of |aligned estimate - reference| / reference, then mean over images",
    "delta1": (
        "fraction of valid pixels with max(estimate / reference, reference / estimate) < 1.25, "
        "then mean over images"
    ),
    "per_domain": "the same means restricted to the records of one domain (indoors / outdoor when present)",
}


def _valid(pred: np.ndarray, depth: np.ndarray, mask: np.ndarray | None) -> np.ndarray:
    pred = np.asarray(pred, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if pred.shape != depth.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs depth {depth.shape}")
    low, high = EVAL_DEPTH_RANGE_M
    valid = np.isfinite(depth) & (depth >= low) & (depth <= high) & np.isfinite(pred)
    if mask is not None:
        mask = np.asarray(mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError(f"shape mismatch: mask {mask.shape} vs depth {depth.shape}")
        valid &= mask
    if int(valid.sum()) < MIN_VALID_PIXELS:
        raise ValueError(f"need at least {MIN_VALID_PIXELS} valid reference pixels")
    return valid


def align_inverse_depth(
    pred: np.ndarray, depth: np.ndarray, mask: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Return (aligned metric estimate, reference) over valid pixels and the fitted scale / shift."""
    valid = _valid(pred, depth, mask)
    p = np.asarray(pred, dtype=np.float64)[valid]
    ref = np.asarray(depth, dtype=np.float64)[valid]
    target = 1.0 / ref
    design = np.stack([p, np.ones_like(p)], axis=1)
    (scale, shift), *_ = np.linalg.lstsq(design, target, rcond=None)
    estimate = 1.0 / np.clip(scale * p + shift, 1.0 / EVAL_DEPTH_RANGE_M[1], None)
    return estimate, ref, {"scale": float(scale), "shift": float(shift), "n_valid": int(valid.sum())}


def aligned_abs_rel(pred: np.ndarray, depth: np.ndarray, mask: np.ndarray | None = None) -> float:
    estimate, ref, _ = align_inverse_depth(pred, depth, mask)
    return float(np.mean(np.abs(estimate - ref) / ref))


def aligned_delta1(
    pred: np.ndarray, depth: np.ndarray, mask: np.ndarray | None = None, *, threshold: float = DELTA_THRESHOLD
) -> float:
    estimate, ref, _ = align_inverse_depth(pred, depth, mask)
    ratio = np.maximum(estimate / ref, ref / estimate)
    return float(np.mean(ratio < threshold))


def image_metrics(pred: np.ndarray, depth: np.ndarray, mask: np.ndarray | None = None) -> dict[str, Any]:
    estimate, ref, fit = align_inverse_depth(pred, depth, mask)
    ratio = np.maximum(estimate / ref, ref / estimate)
    return {
        "abs_rel": float(np.mean(np.abs(estimate - ref) / ref)),
        "delta1": float(np.mean(ratio < DELTA_THRESHOLD)),
        "n_valid": fit["n_valid"],
        "scale": fit["scale"],
        "shift": fit["shift"],
    }


def depth_metrics(preds: Sequence[np.ndarray], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Aggregate per-image metrics for aligned predictions over `{id, depth, mask[, domain]}` records."""
    if len(preds) != len(records):
        raise ValueError(f"{len(preds)} predictions for {len(records)} records")
    if not records:
        raise ValueError("at least one record is required")
    per_image = []
    for pred, record in zip(preds, records, strict=True):
        row = image_metrics(pred, record["depth"], record.get("mask"))
        per_image.append({"id": record["id"], "domain": str(record.get("domain", "unspecified")), **row})
    domains = sorted({r["domain"] for r in per_image})
    per_domain = {
        d: {
            "n": sum(1 for r in per_image if r["domain"] == d),
            "abs_rel": float(np.mean([r["abs_rel"] for r in per_image if r["domain"] == d])),
            "delta1": float(np.mean([r["delta1"] for r in per_image if r["domain"] == d])),
        }
        for d in domains
    }
    return {
        "n": len(per_image),
        "abs_rel": float(np.mean([r["abs_rel"] for r in per_image])),
        "delta1": float(np.mean([r["delta1"] for r in per_image])),
        "per_domain": per_domain,
        "per_image": per_image,
        "definitions": dict(METRIC_DEFINITIONS),
    }


def constant_prior(record: Mapping[str, Any]) -> dict[str, Any]:
    """The oracle constant: every pixel at the image's own median reference depth (no alignment needed)."""
    depth = np.asarray(record["depth"], dtype=np.float64)
    valid = _valid(depth, depth, record.get("mask"))
    ref = depth[valid]
    constant = float(np.median(ref))
    ratio = np.maximum(constant / ref, ref / constant)
    return {
        "abs_rel": float(np.mean(np.abs(constant - ref) / ref)),
        "delta1": float(np.mean(ratio < DELTA_THRESHOLD)),
        "n_valid": int(valid.sum()),
        "constant_m": constant,
    }


def vertical_gradient_prediction(height: int, width: int) -> np.ndarray:
    """Relative inverse depth rising from 0 at the top row to 1 at the bottom row (the ground-plane prior)."""
    rows = np.linspace(0.0, 1.0, num=height, dtype=np.float32)
    return np.repeat(rows[:, None], width, axis=1)


def prior_baselines(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Both trivial priors scored on the same records as a model, each with the same aggregation."""
    constant_rows = []
    gradient_preds = []
    for record in records:
        constant_rows.append(
            {"id": record["id"], "domain": str(record.get("domain", "unspecified")), **constant_prior(record)}
        )
        h, w = np.asarray(record["depth"]).shape
        gradient_preds.append(vertical_gradient_prediction(h, w))
    gradient = depth_metrics(gradient_preds, records)
    return {
        "constant_prior": {
            "baseline": "every pixel at the image's own median reference depth (oracle constant)",
            "n": len(constant_rows),
            "abs_rel": float(np.mean([r["abs_rel"] for r in constant_rows])),
            "delta1": float(np.mean([r["delta1"] for r in constant_rows])),
            "per_image": constant_rows,
        },
        "vertical_gradient_prior": {
            "baseline": (
                "inverse depth rising linearly from the top row to the bottom row, "
                "affine-aligned like a prediction"
            ),
            **{k: gradient[k] for k in ("n", "abs_rel", "delta1", "per_domain", "per_image")},
        },
    }
