from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
MODEL_REVISION = "5426e4f0f36572d16453bbda7a8389317b1bef99"
MODEL_LICENSE = "apache-2.0"
MODEL_KEY = "depth-anything-v2-small"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"

# Input ceilings. The DPT processor rescales the image so that its sides are multiples of 14 close
# to 518 px, and the raw prediction is interpolated back to the caller's resolution, so the cost that
# grows with the caller's image is the aspect ratio (backbone tokens) and the output interpolation.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 14
MAX_ASPECT_RATIO = 4.0
DEPTH_KIND = "relative"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_snapshot(path: str | Path | None = None) -> dict[str, Any]:
    """Check a local snapshot against its DIMER manifest; raise naming the first mismatch."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID:
        raise ValueError(f"manifest modelId {manifest.get('modelId')!r} != {MODEL_ID!r}")
    if manifest.get("revision") != MODEL_REVISION:
        raise ValueError(f"manifest revision {manifest.get('revision')!r} != {MODEL_REVISION!r}")
    for entry in manifest["files"]:
        file_path = root / entry["path"]
        if not file_path.is_file():
            raise FileNotFoundError(f"snapshot file missing: {file_path}")
        size = file_path.stat().st_size
        if size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: size {size} != manifest {entry['bytes']}")
        digest = _sha256(file_path)
        if digest != entry["sha256"]:
            raise ValueError(f"{entry['path']}: sha256 {digest} != manifest {entry['sha256']}")
    return {
        "path": str(root),
        "model_id": manifest["modelId"],
        "revision": manifest["revision"],
        "files": len(manifest["files"]),
        "total_bytes": manifest.get("totalBytes"),
    }


def _hub_download(relative_path: str, root: Path) -> None:
    """Fetch one manifest-listed file at MODEL_REVISION straight into the snapshot directory."""
    from huggingface_hub import hf_hub_download

    hf_hub_download(MODEL_ID, relative_path, revision=MODEL_REVISION, local_dir=str(root))


def stage_missing_files(
    path: str | Path | None = None,
    *,
    allow_download: bool = False,
    downloader: Callable[[str, Path], None] | None = None,
) -> list[str]:
    """Fetch manifest-listed files that are absent locally (a fresh clone commits the manifest but
    git-ignores the weights). Returns the relative paths fetched; `verify_snapshot` still runs after."""
    root = Path(path) if path is not None else DEFAULT_WEIGHTS_DIR
    manifest_path = root / MANIFEST_NAME
    if not manifest_path.is_file():
        raise FileNotFoundError(f"manifest not found: {manifest_path}")
    with open(manifest_path, encoding="utf-8") as fh:
        manifest = json.load(fh)
    if manifest.get("modelId") != MODEL_ID or manifest.get("revision") != MODEL_REVISION:
        raise ValueError(
            f"manifest names {manifest.get('modelId')}@{manifest.get('revision')}, "
            f"package pins {MODEL_ID}@{MODEL_REVISION}; refusing to stage"
        )
    missing = [entry["path"] for entry in manifest["files"] if not (root / entry["path"]).is_file()]
    if not missing:
        return []
    if not allow_download:
        raise FileNotFoundError(
            f"snapshot at {root} is missing {missing}; "
            f"pass allow_download=True to fetch them at {MODEL_REVISION}"
        )
    fetch = downloader or _hub_download
    for relative_path in missing:
        fetch(relative_path, root)
    return missing


def abs_rel(pred: np.ndarray, ref_depth: np.ndarray, *, align: bool = True) -> float:
    """Absolute relative error of a relative inverse-depth map against caller-supplied metric depth.

    The model emits relative inverse depth (disparity up to an unknown scale and shift), so the
    prediction is first aligned to ``1 / ref_depth`` by least squares over valid pixels
    (``ref_depth > 0``), inverted to depth, and scored as ``mean(|est - ref| / ref)``. With
    ``align=False`` the arrays are compared as given, which is only meaningful for metric input.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref_depth = np.asarray(ref_depth, dtype=np.float64)
    if pred.shape != ref_depth.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs ref {ref_depth.shape}")
    valid = np.isfinite(ref_depth) & (ref_depth > 0) & np.isfinite(pred)
    if valid.sum() < 2:
        raise ValueError("need at least 2 valid reference pixels (ref_depth > 0)")
    if align:
        target = 1.0 / ref_depth[valid]
        design = np.stack([pred[valid], np.ones(int(valid.sum()))], axis=1)
        (scale, shift), *_ = np.linalg.lstsq(design, target, rcond=None)
        estimate = 1.0 / np.clip(scale * pred[valid] + shift, 1e-6, None)
    else:
        estimate = pred[valid]
    return float(np.mean(np.abs(estimate - ref_depth[valid]) / ref_depth[valid]))


def validate_image(image: Any) -> Image.Image:
    """Type- and size-check a caller image and return it as RGB."""
    if not isinstance(image, Image.Image):
        raise TypeError(f"image must be a PIL.Image.Image, got {type(image).__name__}")
    width, height = image.size
    short, long = min(width, height), max(width, height)
    if short < MIN_IMAGE_SIDE:
        raise ValueError(f"image side {short} px < MIN_IMAGE_SIDE {MIN_IMAGE_SIDE}")
    if long > MAX_IMAGE_SIDE:
        raise ValueError(f"image side {long} px > MAX_IMAGE_SIDE {MAX_IMAGE_SIDE}")
    if long / short > MAX_ASPECT_RATIO:
        raise ValueError(f"aspect ratio {long / short:.2f} > MAX_ASPECT_RATIO {MAX_ASPECT_RATIO}")
    return image.convert("RGB")


@dataclass
class DepthAnythingPipeline:
    """Monocular relative depth estimation over the pinned Depth Anything V2 Small checkpoint."""

    _runner: Callable[[Image.Image], np.ndarray]
    device: str

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> DepthAnythingPipeline:
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs = str(root), {"local_files_only": True}
        elif allow_download:
            source, kwargs = MODEL_ID, {}
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        processor = AutoImageProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = AutoModelForDepthEstimation.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = model.to(resolved_device).eval()

        def runner(image: Image.Image) -> np.ndarray:
            inputs = processor(images=image, return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                predicted = model(**inputs).predicted_depth
            resized = torch.nn.functional.interpolate(
                predicted.unsqueeze(1), size=image.size[::-1], mode="bicubic", align_corners=False
            )
            return resized[0, 0].float().cpu().numpy()

        return cls(runner, resolved_device)

    def predict(self, image: Image.Image) -> dict[str, Any]:
        """Return relative inverse depth as a float32 H x W array at the input resolution."""
        rgb = validate_image(image)
        depth = np.asarray(self._runner(rgb), dtype=np.float32)
        if depth.shape != (rgb.height, rgb.width):
            raise RuntimeError(f"backend returned shape {depth.shape}, expected {(rgb.height, rgb.width)}")
        return {
            "depth": depth,
            "depth_kind": DEPTH_KIND,
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
            "height": rgb.height,
            "width": rgb.width,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
