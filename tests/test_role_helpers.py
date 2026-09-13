"""Offline tests for the public validation and evaluation stage helpers (DAT24 / EVAL21)."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from depth_anything_depth_estimation_pipeline import (
    INPUT_SCHEMA,
    MAX_ASPECT_RATIO,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_REVISION,
    evaluation_report,
    validate_inputs,
)


def _image(width: int = 64, height: int = 48, mode: str = "RGB") -> Image.Image:
    return Image.new(mode, (width, height), 120 if mode == "L" else (10, 20, 30))


def _reference(height: int = 4, width: int = 5) -> np.ndarray:
    """A strictly positive metric depth map that ramps from 1 m to 5 m."""
    return np.tile(np.linspace(1.0, 5.0, width, dtype=np.float64), (height, 1))


def _result(depth: np.ndarray) -> dict:
    return {
        "depth": depth,
        "depth_kind": "relative",
        "depth_min": float(depth.min()),
        "depth_max": float(depth.max()),
        "height": depth.shape[0],
        "width": depth.shape[1],
    }


def test_validate_inputs_returns_manifest_with_schema_and_identity() -> None:
    manifest = validate_inputs([_image(), _image(120, 40)], names=["a", "b"])
    assert manifest["verdict"] == "accepted"
    assert manifest["findings"] == []
    assert manifest["schema"] == INPUT_SCHEMA
    assert manifest["schema"]["short_side_px"] == [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE]
    assert manifest["schema"]["aspect_ratio"] == [1.0, MAX_ASPECT_RATIO]
    assert manifest["n_images"] == 2
    assert manifest["inputs"] == [
        {"id": "a", "mode": "RGB", "size": [64, 48], "aspect_ratio": 1.333},
        {"id": "b", "mode": "RGB", "size": [120, 40], "aspect_ratio": 3.0},
    ]
    assert (manifest["model_id"], manifest["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_validate_inputs_single_image_default_ids_and_reports_source_mode() -> None:
    manifest = validate_inputs(_image(mode="L"))
    assert [entry["id"] for entry in manifest["inputs"]] == ["image-0"]
    assert manifest["inputs"][0]["mode"] == "L"  # the input's own mode, before the RGB conversion
    assert manifest["n_images"] == 1


def test_validate_inputs_rejects_like_predict() -> None:
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        validate_inputs(_image(MIN_IMAGE_SIDE - 1, 64))
    with pytest.raises(ValueError, match="MAX_ASPECT_RATIO"):
        validate_inputs(_image(64 * int(MAX_ASPECT_RATIO) + 64, 64))
    with pytest.raises(TypeError):
        validate_inputs("not an image")
    with pytest.raises(ValueError, match="names must have one entry per image"):
        validate_inputs([_image()], names=["a", "b"])
    with pytest.raises(ValueError, match="at least one image"):
        validate_inputs([])


def test_evaluation_report_not_measurable_without_reference_depth() -> None:
    report = evaluation_report(_result(np.ones((4, 5), dtype=np.float32)))
    assert report["verdict"] == "not-measurable"
    assert report["metrics"] == []
    assert report["baselines"] == []
    assert "metric depth map in metres" in report["needs"]
    assert "not metres" in report["score_semantics"]
    assert report["n_pixels"] == 20
    assert (report["model_id"], report["model_revision"]) == (MODEL_ID, MODEL_REVISION)


def test_evaluation_report_sample_sanity_with_reference_depth() -> None:
    reference = _reference()
    prediction = (1.0 / reference).astype(np.float32)  # a perfectly aligned inverse-depth map
    report = evaluation_report(_result(prediction), reference, sample_kind="BYOD")
    assert report["verdict"] == "sample-sanity"
    assert report["sample_kind"] == "BYOD"
    (metric,) = report["metrics"]
    assert metric["id"] == "abs_rel"  # the repository's own helper name (EVAL2)
    assert metric["align"] is True
    assert metric["n_valid_pixels"] == reference.size
    assert metric["value"] == pytest.approx(0.0, abs=1e-6)
    assert metric["estimation"]


def test_evaluation_report_metric_rises_for_a_wrong_prediction() -> None:
    reference = _reference()
    good = evaluation_report(_result((1.0 / reference).astype(np.float32)), reference)["metrics"][0]
    bad = evaluation_report(_result(np.flip(1.0 / reference, axis=1).astype(np.float32)), reference)
    assert bad["metrics"][0]["value"] > good["value"]
