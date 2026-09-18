"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): the zero-shot
evaluation on synthetic scenes, a one-epoch neck/head stage plus a one-epoch unfreeze of the last block, and
the artifact round trip with `predict` parity. Skipped when the weights are absent."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from depth_anything_depth_estimation_pipeline import DEFAULT_WEIGHTS_DIR, DepthAnythingPipeline
from depth_anything_depth_estimation_pipeline import pipeline as pl

pytest.importorskip("transformers")
if not (DEFAULT_WEIGHTS_DIR / pl.WEIGHTS_FILE).is_file():
    pytest.skip("snapshot not staged", allow_module_level=True)

H, W = 84, 112


def _scene(seed: int) -> dict:
    rng = np.random.default_rng(seed)
    rows = np.linspace(6.0, 1.5, H)[:, None]
    depth = np.repeat(rows, W, axis=1) + rng.uniform(0, 0.05, size=(H, W))
    depth[30:54, 40:80] = 1.0 + 0.1 * (seed % 3)
    shade = (255 * (1.0 - (depth - depth.min()) / (depth.max() - depth.min()))).astype(np.uint8)
    image = Image.fromarray(np.stack([shade, shade // 2 + 60, np.full_like(shade, 90)], axis=-1))
    return {
        "id": f"s{seed:02d}",
        "image": image,
        "depth": depth.astype(np.float32),
        "scan": f"scan-{seed // 2}",
    }


RECORDS = [_scene(i) for i in range(8)]


@pytest.fixture(scope="module")
def pipe():
    return DepthAnythingPipeline.from_pretrained(device="cpu")


def test_zero_shot_evaluation_is_finite_and_labelled(pipe):
    metrics = pipe.evaluate(RECORDS[6:])
    assert metrics["n"] == 2 and metrics["policy"] == pl.POLICY_ZERO_SHOT and metrics["adapted"] is False
    assert 0.0 <= metrics["abs_rel"] < 5.0 and 0.0 <= metrics["delta1"] <= 1.0


def test_neck_head_then_one_epoch_unfreeze_and_artifact_round_trip(pipe, tmp_path):
    result = pipe.adapt(RECORDS[:6], RECORDS[6:], trainable_blocks=1, head_epochs=1, epochs=1, lr=1e-5)
    assert result["history"][0]["stage"] == pl.POLICY_ZERO_SHOT and len(result["history"]) == 3
    assert result["history"][1]["stage"] == pl.POLICY_FROZEN and result["history"][2]["stage"].startswith(
        "unfrozen last 1"
    )
    assert result["n_trainable_head"] == pl.NECK_HEAD_PARAMETERS and result["n_trainable_blocks"] == 1_775_232
    assert result["n_total"] == pl.PARAMETER_COUNT and result["best_epoch"] in (0, 1, 2)
    assert result["policy"] == result["history"][result["best_epoch"]]["stage"]
    metrics = pipe.evaluate(RECORDS[6:])
    assert metrics["adapted"] is True and metrics["policy"] == result["policy"]
    assert metrics["abs_rel"] == pytest.approx(
        result["history"][result["best_epoch"]]["val"]["abs_rel"], abs=1e-5
    )
    artifact = pipe.save_artifact(tmp_path / "adapter", {"note": "test"})
    manifest = json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert any(t.startswith("neck.") for t in manifest["tensors"]) and any(
        t.startswith("head.") for t in manifest["tensors"]
    )
    assert any(t.startswith(pl.BLOCK_PREFIX) for t in manifest["tensors"]) == result["policy"].startswith(
        "unfrozen"
    )
    reloaded = DepthAnythingPipeline.from_artifact(artifact, device="cpu")
    before, after = pipe.predict(RECORDS[0]["image"])["depth"], reloaded.predict(RECORDS[0]["image"])["depth"]
    assert np.array_equal(before, after) and reloaded.adapter["best_epoch"] == result["best_epoch"]
    assert reloaded.evaluate(RECORDS[6:])["abs_rel"] == pytest.approx(metrics["abs_rel"], abs=1e-6)
