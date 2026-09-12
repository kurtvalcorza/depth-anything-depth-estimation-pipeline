import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from depth_anything_depth_estimation_pipeline import (
    DEFAULT_WEIGHTS_DIR,
    MAX_ASPECT_RATIO,
    MAX_IMAGE_SIDE,
    MIN_IMAGE_SIDE,
    MODEL_ID,
    MODEL_KEY,
    MODEL_REVISION,
    DepthAnythingPipeline,
    abs_rel,
    verify_snapshot,
)

HEX40 = re.compile(r"^[0-9a-f]{40}$")
REPO = Path(__file__).resolve().parents[1]


def test_identity_constants():
    assert HEX40.match(MODEL_REVISION)
    assert MODEL_ID == "depth-anything/Depth-Anything-V2-Small-hf"
    assert DEFAULT_WEIGHTS_DIR == REPO / "weights" / MODEL_KEY
    manifest = REPO / "weights" / MODEL_KEY / "dimer-base-manifest.json"
    if manifest.is_file():
        data = json.loads(manifest.read_text(encoding="utf-8"))
        assert data["modelId"] == MODEL_ID
        assert data["revision"] == MODEL_REVISION


def _write_snapshot(root: Path, content: bytes, sha: str | None = None, size: int | None = None) -> None:
    (root / "config.json").write_bytes(content)
    manifest = {
        "modelId": MODEL_ID,
        "revision": MODEL_REVISION,
        "files": [
            {
                "path": "config.json",
                "bytes": len(content) if size is None else size,
                "sha256": hashlib.sha256(content).hexdigest() if sha is None else sha,
            }
        ],
        "totalBytes": len(content),
    }
    (root / "dimer-base-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_verify_snapshot_accepts_matching_manifest(tmp_path):
    _write_snapshot(tmp_path, b'{"model_type": "depth_anything"}')
    info = verify_snapshot(tmp_path)
    assert info["revision"] == MODEL_REVISION and info["files"] == 1


def test_verify_snapshot_rejects_tampered_digest(tmp_path):
    content = b'{"model_type": "depth_anything"}'
    good = hashlib.sha256(content).hexdigest()
    flipped = ("0" if good[0] != "0" else "1") + good[1:]
    _write_snapshot(tmp_path, content, sha=flipped)
    with pytest.raises(ValueError, match="sha256"):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_size_and_missing_file(tmp_path):
    _write_snapshot(tmp_path, b"abc", size=99)
    with pytest.raises(ValueError, match="size"):
        verify_snapshot(tmp_path)
    (tmp_path / "config.json").unlink()
    with pytest.raises(FileNotFoundError):
        verify_snapshot(tmp_path)


def test_verify_snapshot_rejects_wrong_revision(tmp_path):
    _write_snapshot(tmp_path, b"abc")
    manifest = json.loads((tmp_path / "dimer-base-manifest.json").read_text())
    manifest["revision"] = "0" * 40
    (tmp_path / "dimer-base-manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="revision"):
        verify_snapshot(tmp_path)


def _fake_pipeline(calls: list | None = None) -> DepthAnythingPipeline:
    def runner(image: Image.Image) -> np.ndarray:
        if calls is not None:
            calls.append(image.mode)
        height, width = image.height, image.width
        return np.linspace(0, 1, height * width, dtype=np.float32).reshape(height, width)

    return DepthAnythingPipeline(runner, "cpu")


def test_predict_output_fields():
    calls: list = []
    pipe = _fake_pipeline(calls)
    result = pipe.predict(Image.new("L", (40, 30)))
    assert result["depth"].shape == (30, 40) and result["depth"].dtype == np.float32
    assert result["depth_kind"] == "relative"
    assert result["depth_min"] == 0.0 and result["depth_max"] == 1.0
    assert (result["height"], result["width"]) == (30, 40)
    assert result["model_id"] == MODEL_ID and result["model_revision"] == MODEL_REVISION
    assert calls == ["RGB"]


def test_predict_rejects_bad_inputs():
    pipe = _fake_pipeline()
    with pytest.raises(TypeError):
        pipe.predict(np.zeros((30, 40, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="MIN_IMAGE_SIDE"):
        pipe.predict(Image.new("RGB", (MIN_IMAGE_SIDE - 1, 64)))
    with pytest.raises(ValueError, match="MAX_IMAGE_SIDE"):
        pipe.predict(Image.new("RGB", (MAX_IMAGE_SIDE + 1, 64)))
    with pytest.raises(ValueError, match="MAX_ASPECT_RATIO"):
        pipe.predict(Image.new("RGB", (int(64 * MAX_ASPECT_RATIO) + 1, 64)))


def test_predict_rejects_backend_shape_mismatch():
    pipe = DepthAnythingPipeline(lambda image: np.zeros((3, 3), dtype=np.float32), "cpu")
    with pytest.raises(RuntimeError):
        pipe.predict(Image.new("RGB", (40, 30)))


def test_abs_rel_alignment():
    rng = np.random.default_rng(0)
    ref_depth = rng.uniform(1.0, 10.0, size=(16, 16))
    pred = 3.0 * (1.0 / ref_depth) + 0.25  # affine-transformed inverse depth
    assert abs_rel(pred, ref_depth) == pytest.approx(0.0, abs=1e-9)
    assert abs_rel(ref_depth * 1.1, ref_depth, align=False) == pytest.approx(0.1)
    with pytest.raises(ValueError):
        abs_rel(pred[:8], ref_depth)
