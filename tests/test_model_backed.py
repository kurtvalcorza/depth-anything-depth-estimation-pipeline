"""Model-backed checks that run only where the pinned snapshot is staged (local pre-flight): the zero-shot
evaluation on synthetic scenes, a one-epoch neck/head stage plus a one-epoch unfreeze of the last block, and
the artifact round trip with `predict` parity. Skipped when the weights are absent."""

from __future__ import annotations

import json

import numpy as np
import pytest
from PIL import Image

from depth_anything_depth_estimation_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    POLICY_FROZEN,
    POLICY_ZERO_SHOT,
    DepthAnythingPipeline,
)
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


def test_load_artifact_refuses_a_tensor_set_that_differs_from_the_recorded_policy(pipe, tmp_path):
    """A manifest that claims the frozen policy but carries block tensors, or records other blocks than the
    tensors it lists, is refused before any tensor is applied; a zero-shot artifact must equal the base."""
    import json as _json
    import shutil

    result = pipe.adapt(RECORDS[:6], None, trainable_blocks=1, head_epochs=1, epochs=1, lr=1e-5)
    assert result["policy"].startswith("unfrozen")
    artifact = pipe.save_artifact(tmp_path / "ok")
    manifest = _json.loads((artifact / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["no_op"] is False and any(
        t.startswith("backbone.encoder.layer.11.") for t in manifest["tensors"]
    )
    claims_frozen = tmp_path / "claims_frozen"
    shutil.copytree(artifact, claims_frozen)
    adapter = {**manifest["adapter"], "policy": POLICY_FROZEN}
    (claims_frozen / "manifest.json").write_text(_json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded policy"):
        DepthAnythingPipeline.from_artifact(claims_frozen, device="cpu")
    other_blocks = tmp_path / "other_blocks"
    shutil.copytree(artifact, other_blocks)
    adapter = {
        **manifest["adapter"],
        "policy": "unfrozen last 2 blocks + DPT neck and head",
        "trainable_blocks": 2,
    }
    (other_blocks / "manifest.json").write_text(_json.dumps({**manifest, "adapter": adapter}))
    with pytest.raises(ValueError, match="does not match its recorded policy"):
        DepthAnythingPipeline.from_artifact(other_blocks, device="cpu")
    zero = DepthAnythingPipeline.from_pretrained(device="cpu")
    zero.adapt(RECORDS[:6], RECORDS[6:], trainable_blocks=0, head_epochs=0, epochs=0)
    assert zero.adapter["policy"] == POLICY_ZERO_SHOT
    no_op = zero.save_artifact(tmp_path / "zero")
    zero_manifest = _json.loads((no_op / "manifest.json").read_text(encoding="utf-8"))
    assert zero_manifest["no_op"] is True and "byte copies" in zero_manifest["note"]
    reloaded = DepthAnythingPipeline.from_artifact(no_op, device="cpu")
    assert reloaded.adapter["policy"] == POLICY_ZERO_SHOT
    assert np.array_equal(
        reloaded.predict(RECORDS[0]["image"])["depth"], zero.predict(RECORDS[0]["image"])["depth"]
    )


def test_adapt_is_transactional_when_the_progress_callback_raises(pipe):
    """A failure inside training leaves the base exactly as it was, frozen, with no adapter attached."""
    import torch

    before = {k: v.clone() for k, v in pipe._model.state_dict().items()}

    def boom(entry):
        if entry["epoch"] == 1:
            raise RuntimeError("boom")

    with pytest.raises(RuntimeError, match="boom"):
        pipe.adapt(RECORDS[:6], None, trainable_blocks=1, head_epochs=1, epochs=1, lr=1e-5, progress=boom)
    after = pipe._model.state_dict()
    assert all(torch.equal(before[k], after[k]) for k in before) and pipe.adapter is None
    assert not any(p.requires_grad for p in pipe._model.parameters())
