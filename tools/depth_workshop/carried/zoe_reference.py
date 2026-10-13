"""Monocular metric depth estimation with the pinned ``Intel/zoedepth-nyu-kitti`` checkpoint (ZoeDepth).

The class loads the processor and model only from a digest-verified local snapshot (``weights/<key>/``)
or, when explicitly allowed, from the Hugging Face Hub at the pinned revision — always with
``trust_remote_code=False``: the ZoeDepth architecture (a BEiT-large DPT backbone with metric bin heads
for the NYU and KITTI ranges) comes from the pinned ``transformers`` release, the weights are
SafeTensors, and no model-repository code is executed. The output is depth in metres — a metric
estimate with no confidence, not a measurement — at the caller's resolution.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

MODEL_ID = "Intel/zoedepth-nyu-kitti"
MODEL_REVISION = "f364d4c7936e91f465abba182208dd68142bf0ca"
MODEL_LICENSE = "mit"
MODEL_KEY = "zoedepth-nyu-kitti"
DEFAULT_WEIGHTS_DIR = Path(__file__).resolve().parents[2] / "weights" / MODEL_KEY
MANIFEST_NAME = "dimer-base-manifest.json"
ARTIFACT_FORMAT = "org.valcorza.zoedepth.metric-head-adapter"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_MANIFEST_NAME = "manifest.json"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
MAX_ARTIFACT_MANIFEST_BYTES = 64 * 1024
MAX_ARTIFACT_HEADER_BYTES = 1024 * 1024
TRAINABLE_PREFIXES = ("metric_head.",)
MAX_ADAPTATION_RECORDS = 128
MAX_ADAPTATION_PIXELS = 32 * 1024 * 1024

# Input ceilings. The ZoeDepth processor resizes the image to fit 384x512 while keeping the aspect
# ratio (sides rounded to multiples of 32) and pads, so the backbone cost grows with the aspect ratio,
# not the pixel count; the prediction is interpolated back to the caller's resolution.
MAX_IMAGE_SIDE = 4096
MIN_IMAGE_SIDE = 32
MAX_ASPECT_RATIO = 4.0
DEPTH_KIND = "metric"
DEPTH_UNIT = "metres"
# The checkpoint's two metric heads (config.json bin_configurations): NYU indoor 0.001-10 m and KITTI
# outdoor 0.001-80 m; the model routes each image to one head by its own domain classifier.
DEPTH_RANGES_M = {"nyu": (0.001, 10.0), "kitti": (0.001, 80.0)}
# Optional horizontal-flip test-time augmentation (the upstream evaluation setting): two forward
# passes, predictions averaged. Off by default; a caller-owned request parameter.
FLIP_AUGMENTATION = False
# delta1 accuracy threshold (the depth-estimation convention): max(pred/ref, ref/pred) < 1.25.
DELTA_THRESHOLD = 1.25


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


def _valid_mask(pred: np.ndarray, ref_depth: np.ndarray) -> np.ndarray:
    if pred.shape != ref_depth.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs ref {ref_depth.shape}")
    valid = np.isfinite(ref_depth) & (ref_depth > 0) & np.isfinite(pred) & (pred > 0)
    if valid.sum() < 2:
        raise ValueError("need at least 2 valid reference pixels (ref_depth > 0 and pred > 0)")
    return valid


def abs_rel(pred: np.ndarray, ref_depth: np.ndarray) -> float:
    """Absolute relative error ``mean(|pred - ref| / ref)`` of metric depth against metric reference depth.

    Both arrays are in metres at the same H x W; no scale or shift alignment is applied, because the
    model claims metric output — a scale error therefore shows up in the number, as it should.
    """
    pred = np.asarray(pred, dtype=np.float64)
    ref_depth = np.asarray(ref_depth, dtype=np.float64)
    valid = _valid_mask(pred, ref_depth)
    return float(np.mean(np.abs(pred[valid] - ref_depth[valid]) / ref_depth[valid]))


def delta1(pred: np.ndarray, ref_depth: np.ndarray, *, threshold: float = DELTA_THRESHOLD) -> float:
    """Fraction of valid pixels whose ratio ``max(pred/ref, ref/pred)`` is below ``threshold`` (1.25)."""
    pred = np.asarray(pred, dtype=np.float64)
    ref_depth = np.asarray(ref_depth, dtype=np.float64)
    valid = _valid_mask(pred, ref_depth)
    ratio = np.maximum(pred[valid] / ref_depth[valid], ref_depth[valid] / pred[valid])
    return float(np.mean(ratio < threshold))


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


def _check_flip(value: Any) -> bool:
    if not isinstance(value, bool):
        raise TypeError("flip_augmentation must be a bool")
    return value


INPUT_SCHEMA: dict[str, Any] = {
    "input": "PIL.Image.Image, or a sequence of them for the validation stage; any mode, converted to RGB",
    "short_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "long_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "aspect_ratio": [1.0, MAX_ASPECT_RATIO],
    "flip_augmentation": "bool; two forward passes (image and its mirror) averaged when true",
    "output": (
        f"{DEPTH_KIND} depth in {DEPTH_UNIT}, float32 H x W at the input resolution (larger = farther); "
        "an estimate with no confidence, routed to the NYU (0.001-10 m) or KITTI (0.001-80 m) head by "
        "the model's own domain classifier"
    ),
    "preprocessing": (
        "convert to RGB; the ZoeDepth processor resizes to fit 384x512 keeping the aspect ratio "
        "(sides rounded to multiples of 32), pads, normalises with mean/std 0.5; the prediction is "
        "interpolated back to the input resolution by post_process_depth_estimation"
    ),
}


def validate_inputs(
    images: Any, *, flip_augmentation: bool = FLIP_AUGMENTATION, names: Sequence[str] | None = None
) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, request, verdict).

    Each image is routed through the public ``validate_image`` that ``predict`` itself calls, so a
    rejection here raises exactly what ``predict`` would; a caller that wants the finding recorded
    catches the exception and stores ``str(exc)`` under ``findings``.
    """
    batch = [images] if isinstance(images, Image.Image) else images
    if not isinstance(batch, Sequence) or isinstance(batch, str | bytes):
        raise TypeError("images must be a PIL.Image.Image or a sequence of them")
    if len(batch) < 1:
        raise ValueError("at least one image is required")
    if names is not None and len(names) != len(batch):
        raise ValueError("names must have one entry per image")
    flip = _check_flip(flip_augmentation)
    inputs = []
    for index, candidate in enumerate(batch):
        rgb = validate_image(candidate)
        width, height = rgb.size
        long_side, short_side = max(width, height), min(width, height)
        inputs.append(
            {
                "id": names[index] if names else f"image-{index}",
                "mode": getattr(candidate, "mode", rgb.mode),
                "size": [width, height],
                "aspect_ratio": round(long_side / short_side, 3),
            }
        )
    return {
        "schema": dict(INPUT_SCHEMA),
        "inputs": inputs,
        "n_images": len(inputs),
        "flip_augmentation": flip,
        "verdict": "accepted",
        "findings": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }


def evaluation_report(
    result: Mapping[str, Any],
    reference_depth: Any | None = None,
    *,
    sample_kind: str = "synthetic",
) -> dict[str, Any]:
    """Evaluation stage: a machine-readable report even when nothing is measurable.

    With ``reference_depth`` (metric depth in metres, same H x W as the prediction, non-positive or
    non-finite pixels ignored) the report carries ``abs_rel`` and ``delta1`` computed without any
    alignment — the model claims metres, so scale errors count — as sample-sanity evidence. Without it
    the verdict is ``not-measurable`` and the report says what ground truth would make the task
    measurable: a depth map in metres has no intrinsic score.
    """
    depth = np.asarray(result["depth"])
    base = {
        "task": "monocular metric depth estimation",
        "score_semantics": (
            f"{result.get('depth_kind', DEPTH_KIND)} depth in {DEPTH_UNIT} with no confidence; the value "
            "is an estimate whose scale depends on the model having recognised the scene's domain and "
            "camera, and a blank image still yields a depth map"
        ),
        "sample_kind": sample_kind,
        "flip_augmentation": bool(result.get("flip_augmentation", FLIP_AUGMENTATION)),
        "n_images": 1,
        "n_pixels": int(depth.size),
        "depth_min_m": float(depth.min()),
        "depth_max_m": float(depth.max()),
        "depth_median_m": float(np.median(depth)),
        "baselines": [],
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
    }
    if reference_depth is None:
        return {
            **base,
            "metrics": [],
            "verdict": "not-measurable",
            "reason": "no metric reference depth was supplied for the evaluated image",
            "needs": (
                "a metric depth map in metres with the same height and width as the image, from a depth "
                "sensor, LiDAR, stereo or an RGB-D benchmark, scored with abs_rel and delta1 against a "
                "constant-depth prior (the reference's median) as the trivial baseline"
            ),
        }
    ref = np.asarray(reference_depth, dtype=np.float64)
    valid = np.isfinite(ref) & (ref > 0)
    n_valid = int(valid.sum())
    constant = np.full_like(ref, float(np.median(ref[valid])) if n_valid else 1.0)
    return {
        **base,
        "metrics": [
            {
                "id": "abs_rel",
                "value": abs_rel(depth, ref),
                "align": False,
                "n_valid_pixels": n_valid,
                "estimation": "single image, no alignment (metric output as-is), no dispersion estimate",
            },
            {
                "id": "delta1",
                "value": delta1(depth, ref),
                "threshold": DELTA_THRESHOLD,
                "n_valid_pixels": n_valid,
                "estimation": "single image, fraction of valid pixels within the ratio threshold",
            },
        ],
        "baselines": [
            {
                "id": "constant_median_depth",
                "abs_rel": abs_rel(constant, ref),
                "delta1": delta1(constant, ref),
                "note": "every pixel predicted at the reference's median depth",
            }
        ],
        "verdict": "sample-sanity",
        "reason": "one image with caller-supplied metric depth; not a benchmark",
        "needs": "a held-out set of metric depth maps from the deployment domain for any generalisable claim",
    }


def validate_depth_dataset(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Validate paired RGB images and positive metric-depth targets for adaptation."""
    if isinstance(records, str | bytes) or not isinstance(records, Sequence):
        raise TypeError("records must be a sequence of mappings")
    if not 2 <= len(records) <= MAX_ADAPTATION_RECORDS:
        raise ValueError(f"record count {len(records)} outside 2..{MAX_ADAPTATION_RECORDS}")
    ids: list[str] = []
    digest = hashlib.sha256()
    valid_pixels = 0
    depth_values: list[np.ndarray] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise TypeError(f"record {index} must be a mapping")
        record_id = record.get("id")
        if not isinstance(record_id, str) or not record_id.strip():
            raise ValueError(f"record {index} id must be a non-empty string")
        if record_id in ids:
            raise ValueError(f"duplicate record id: {record_id}")
        ids.append(record_id)
        image = validate_image(record.get("image"))
        depth = np.asarray(record.get("depth_m"), dtype=np.float32)
        if depth.shape != (image.height, image.width):
            raise ValueError(
                f"record {record_id} depth shape {depth.shape} != image {(image.height, image.width)}"
            )
        if not np.all(np.isfinite(depth)) or np.any(depth <= 0):
            raise ValueError(f"record {record_id} depth_m must contain finite positive metres")
        if float(depth.max()) > 80.0:
            raise ValueError(f"record {record_id} depth_m exceeds the 80 m model ceiling")
        valid_pixels += int(depth.size)
        if valid_pixels > MAX_ADAPTATION_PIXELS:
            raise ValueError(
                f"dataset has {valid_pixels} pixels, exceeding MAX_ADAPTATION_PIXELS "
                f"{MAX_ADAPTATION_PIXELS}"
            )
        depth_values.append(depth.reshape(-1))
        digest.update(record_id.encode("utf-8"))
        digest.update(
            json.dumps(
                {"image_shape": [image.height, image.width, 3], "depth_shape": list(depth.shape)},
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        digest.update(np.asarray(image, dtype=np.uint8).tobytes())
        digest.update(depth.tobytes())
    all_depth = np.concatenate(depth_values)
    return {
        "records": len(records),
        "unique_ids": len(ids),
        "valid_depth_pixels": valid_pixels,
        "depth_min_m": float(all_depth.min()),
        "depth_max_m": float(all_depth.max()),
        "depth_median_m": float(np.median(all_depth)),
        "dataset_sha256": digest.hexdigest(),
        "verdict": "accepted",
    }


def _depth_record_content_sha256(record: Mapping[str, Any]) -> str:
    """Fingerprint one validated RGB/depth pair without trusting its caller-supplied ID."""
    image = validate_image(record["image"])
    depth = np.asarray(record["depth_m"], dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"image_shape": [image.height, image.width, 3], "depth_shape": list(depth.shape)},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(np.asarray(image, dtype=np.uint8).tobytes())
    digest.update(depth.tobytes())
    return digest.hexdigest()


def _depth_record_rgb_sha256(record: Mapping[str, Any]) -> str:
    """Fingerprint a validated RGB input independently of its ID and depth target."""
    image = validate_image(record["image"])
    digest = hashlib.sha256()
    digest.update(
        json.dumps(
            {"image_shape": [image.height, image.width, 3]},
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    )
    digest.update(np.asarray(image, dtype=np.uint8).tobytes())
    return digest.hexdigest()


def _prepare_depth_target(
    depth_m: Any,
    processor: Any,
    *,
    output_size: tuple[int, int],
    device: Any,
) -> Any:
    """Apply ZoeDepth's spatial pad/resize geometry to a metric-depth target."""
    import torch
    import torch.nn.functional as F

    depth = np.asarray(depth_m, dtype=np.float32)
    if getattr(processor, "do_pad", False):
        pad_image = getattr(processor, "pad_image", None)
        if not callable(pad_image):
            raise RuntimeError("ZoeDepth processor enables padding but exposes no pad_image method")
        depth = np.asarray(
            pad_image(
                depth[..., None],
                input_data_format="channels_last",
                data_format="channels_last",
            ),
            dtype=np.float32,
        )[..., 0]
    target = torch.from_numpy(np.ascontiguousarray(depth)).to(device)
    # ZoeDepth's image resize uses align_corners=True. Bilinear depth resampling preserves the same
    # coordinate grid while avoiding image-specific bicubic overshoot in metric targets.
    return F.interpolate(
        target[None, None], size=output_size, mode="bilinear", align_corners=True
    )[:, 0]


@dataclass
class ZoeDepthMetricPipeline:
    """Monocular metric depth estimation over the pinned ZoeDepth NYU+KITTI checkpoint."""

    _runner: Callable[[Image.Image, bool], np.ndarray]
    device: str
    model: Any | None = None
    processor: Any | None = None
    adaptation_config: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ZoeDepthMetricPipeline:
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
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import ZoeDepthForDepthEstimation, ZoeDepthImageProcessor

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        processor = ZoeDepthImageProcessor.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, **kwargs
        )
        model = ZoeDepthForDepthEstimation.from_pretrained(
            source, revision=MODEL_REVISION, trust_remote_code=False, dtype=torch.float32, **kwargs
        )
        model = model.to(resolved_device).eval()

        def runner(image: Image.Image, flip: bool) -> np.ndarray:
            inputs = processor(images=image, return_tensors="pt").to(resolved_device)
            with torch.inference_mode():
                outputs = model(**inputs)
                extra = {}
                if flip:
                    mirrored = torch.flip(inputs["pixel_values"], dims=[3])
                    extra["outputs_flipped"] = model(pixel_values=mirrored)
            # The pinned processor's post-processing un-pads, un-flips (when given) and interpolates
            # the prediction back to the source size; it returns metres.
            post = processor.post_process_depth_estimation(
                outputs, source_sizes=[(image.height, image.width)], **extra
            )
            return post[0]["predicted_depth"].float().cpu().numpy()

        return cls(runner, resolved_device, model=model, processor=processor)

    def freeze_for_adaptation(self) -> dict[str, int]:
        """Freeze ZoeDepth except its metric head."""
        if self.model is None:
            raise RuntimeError("cannot configure adaptation without an underlying torch model")
        trainable = frozen = 0
        for name, parameter in self.model.named_parameters():
            parameter.requires_grad = name.startswith(TRAINABLE_PREFIXES)
            if parameter.requires_grad:
                trainable += parameter.numel()
            else:
                frozen += parameter.numel()
        if trainable == 0:
            raise RuntimeError("ZoeDepth adaptation selected no trainable parameters")
        self.adaptation_config = {
            "method": "frozen-backbone-metric-head-gradient-adaptation",
            "trainable_prefixes": list(TRAINABLE_PREFIXES),
            "trainable_parameters": trainable,
            "frozen_parameters": frozen,
        }
        return {"trainable_parameters": trainable, "frozen_parameters": frozen}

    def finetune(
        self,
        train_records: Sequence[Mapping[str, Any]],
        val_records: Sequence[Mapping[str, Any]] | None = None,
        *,
        epochs: int = 2,
        learning_rate: float = 1e-5,
        seed: int = 42,
    ) -> list[dict[str, Any]]:
        """Run bounded metric-head adaptation using log-depth L1 loss."""
        import random

        import torch
        from torch.optim import AdamW

        if self.model is None or self.processor is None:
            raise RuntimeError("cannot fine-tune without the underlying model and processor")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 1 <= epochs <= 20:
            raise ValueError("epochs must be an int in 1..20")
        if not isinstance(learning_rate, int | float) or isinstance(learning_rate, bool):
            raise TypeError("learning_rate must be numeric")
        if not 0 < float(learning_rate) <= 1e-2:
            raise ValueError("learning_rate must be in (0, 1e-2]")
        train_manifest = validate_depth_dataset(train_records)
        val_manifest = validate_depth_dataset(val_records) if val_records else None
        total_records = train_manifest["records"] + (
            val_manifest["records"] if val_manifest is not None else 0
        )
        if total_records > MAX_ADAPTATION_RECORDS:
            raise ValueError(
                f"combined train and validation record count {total_records} exceeds "
                f"MAX_ADAPTATION_RECORDS {MAX_ADAPTATION_RECORDS}"
            )
        total_pixels = train_manifest["valid_depth_pixels"] + (
            val_manifest["valid_depth_pixels"] if val_manifest is not None else 0
        )
        if total_pixels > MAX_ADAPTATION_PIXELS:
            raise ValueError(
                f"combined train and validation data has {total_pixels} pixels, exceeding "
                f"MAX_ADAPTATION_PIXELS {MAX_ADAPTATION_PIXELS}"
            )
        if val_records:
            train_ids = {str(record["id"]) for record in train_records}
            val_ids = {str(record["id"]) for record in val_records}
            overlapping_ids = sorted(train_ids & val_ids)
            if overlapping_ids:
                raise ValueError(
                    f"train and validation records overlap by id: {overlapping_ids[:5]}"
                )
            train_rgb = {_depth_record_rgb_sha256(record) for record in train_records}
            val_rgb = {_depth_record_rgb_sha256(record) for record in val_records}
            if train_rgb & val_rgb:
                raise ValueError("train and validation records overlap by RGB content")
            train_content = {_depth_record_content_sha256(record) for record in train_records}
            val_content = {_depth_record_content_sha256(record) for record in val_records}
            if train_content & val_content:
                raise ValueError("train and validation records overlap by RGB/depth content")
        if not self.adaptation_config:
            self.freeze_for_adaptation()
        if any(
            parameter.requires_grad and not name.startswith(TRAINABLE_PREFIXES)
            for name, parameter in self.model.named_parameters()
        ):
            raise RuntimeError("parameters outside the declared ZoeDepth adapter surface are trainable")

        torch.manual_seed(seed)
        device = torch.device(self.device)
        self.model.to(device).eval()
        self.model.metric_head.train()
        trainable = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        if not trainable:
            raise RuntimeError("model has no trainable parameters")
        before = {name: parameter.detach().cpu().clone() for name, parameter in trainable.items()}
        optimizer_betas = (0.9, 0.999)
        optimizer_epsilon = 1e-8
        optimizer_weight_decay = 0.01
        optimizer = AdamW(
            list(trainable.values()),
            lr=float(learning_rate),
            betas=optimizer_betas,
            eps=optimizer_epsilon,
            weight_decay=optimizer_weight_decay,
        )

        def loss_for(record: Mapping[str, Any]) -> torch.Tensor:
            inputs = self.processor(
                images=validate_image(record["image"]), return_tensors="pt"
            ).to(device)
            prediction = self.model(pixel_values=inputs["pixel_values"]).predicted_depth
            target = _prepare_depth_target(
                record["depth_m"],
                self.processor,
                output_size=tuple(prediction.shape[-2:]),
                device=device,
            )
            return torch.mean(torch.abs(torch.log(prediction.clamp_min(1e-3)) - torch.log(target)))

        def validation_loss() -> float | None:
            if not val_records:
                return None
            self.model.eval()
            with torch.inference_mode():
                value = float(np.mean([float(loss_for(record).item()) for record in val_records]))
            self.model.metric_head.train()
            return value

        baseline_val_loss = validation_loss()
        history: list[dict[str, Any]] = []
        previous_config = dict(self.adaptation_config)
        for key in (
            "epochs",
            "learning_rate",
            "batch_size",
            "seed",
            "optimizer",
            "optimizer_betas",
            "optimizer_epsilon",
            "optimizer_weight_decay",
            "loss",
            "train_manifest",
            "validation_manifest",
            "training_median_depth_m",
            "baseline_validation_log_l1",
            "history",
            "weight_delta_l2",
        ):
            self.adaptation_config.pop(key, None)
        try:
            for epoch in range(1, epochs + 1):
                order = list(range(len(train_records)))
                random.Random(seed + epoch * 19).shuffle(order)
                total_loss = 0.0
                for index in order:
                    optimizer.zero_grad(set_to_none=True)
                    loss = loss_for(train_records[index])
                    if not bool(torch.isfinite(loss)):
                        raise RuntimeError("fine-tuning produced a non-finite loss")
                    loss.backward()
                    optimizer.step()
                    if any(
                        not bool(torch.isfinite(parameter).all())
                        for parameter in trainable.values()
                    ):
                        raise RuntimeError("fine-tuning produced non-finite adapter weights")
                    total_loss += float(loss.item())
                epoch_data: dict[str, Any] = {
                    "epoch": epoch,
                    "train_log_l1": round(total_loss / len(train_records), 6),
                    "optimizer_steps": len(train_records),
                }
                current_val = validation_loss()
                if current_val is not None:
                    if not math.isfinite(current_val):
                        raise RuntimeError("fine-tuning produced a non-finite validation loss")
                    epoch_data["val_log_l1"] = round(current_val, 6)
                history.append(epoch_data)

            delta_sq = 0.0
            for name, parameter in trainable.items():
                delta_sq += float(
                    torch.sum((parameter.detach().cpu() - before[name]) ** 2).item()
                )
            weight_delta_l2 = delta_sq**0.5
            if not math.isfinite(weight_delta_l2):
                raise RuntimeError("fine-tuning produced a non-finite weight delta")
            if weight_delta_l2 == 0.0:
                raise RuntimeError("fine-tuning completed without changing adapter weights")
            self.model.eval()
            self.adaptation_config.update(
                {
                    "epochs": epochs,
                    "learning_rate": float(learning_rate),
                    "batch_size": 1,
                    "seed": seed,
                    "optimizer": "AdamW",
                    "optimizer_betas": list(optimizer_betas),
                    "optimizer_epsilon": optimizer_epsilon,
                    "optimizer_weight_decay": optimizer_weight_decay,
                    "loss": "mean-absolute-log-depth-error",
                    "train_manifest": train_manifest,
                    "validation_manifest": val_manifest,
                    "training_median_depth_m": train_manifest["depth_median_m"],
                    "baseline_validation_log_l1": baseline_val_loss,
                    "history": history,
                    "weight_delta_l2": weight_delta_l2,
                }
            )
        except Exception:
            with torch.no_grad():
                for name, parameter in trainable.items():
                    parameter.copy_(before[name].to(device=parameter.device, dtype=parameter.dtype))
            self.adaptation_config = previous_config
            self.model.eval()
            raise
        return history

    def evaluate_adaptation(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Score metric depth on labelled records against a constant-median baseline."""
        validate_depth_dataset(records)
        train_median = self.adaptation_config.get("training_median_depth_m")
        if not isinstance(train_median, int | float) or train_median <= 0:
            raise RuntimeError("adaptation metadata does not contain a positive training median")
        model_abs_rel_sum = 0.0
        model_delta1_hits = 0
        baseline_abs_rel_sum = 0.0
        baseline_delta1_hits = 0
        valid_pixels = 0
        for record in records:
            prediction = np.asarray(self.predict(record["image"])["depth"], dtype=np.float64)
            reference = np.asarray(record["depth_m"], dtype=np.float64)
            baseline = np.full_like(reference, float(train_median))
            valid = _valid_mask(prediction, reference)
            count = int(valid.sum())
            valid_pixels += count
            model_abs_rel_sum += float(
                np.sum(np.abs(prediction[valid] - reference[valid]) / reference[valid])
            )
            model_ratio = np.maximum(
                prediction[valid] / reference[valid], reference[valid] / prediction[valid]
            )
            model_delta1_hits += int(np.sum(model_ratio < DELTA_THRESHOLD))
            baseline_abs_rel_sum += float(
                np.sum(np.abs(baseline[valid] - reference[valid]) / reference[valid])
            )
            baseline_ratio = np.maximum(
                baseline[valid] / reference[valid], reference[valid] / baseline[valid]
            )
            baseline_delta1_hits += int(np.sum(baseline_ratio < DELTA_THRESHOLD))
        model_abs_rel = model_abs_rel_sum / valid_pixels
        model_delta1 = model_delta1_hits / valid_pixels
        baseline_abs_rel = baseline_abs_rel_sum / valid_pixels
        baseline_delta1 = baseline_delta1_hits / valid_pixels
        return {
            "records": len(records),
            "valid_pixels": valid_pixels,
            "abs_rel": model_abs_rel,
            "delta1": model_delta1,
            "constant_median_baseline_abs_rel": baseline_abs_rel,
            "constant_median_baseline_delta1": baseline_delta1,
            "abs_rel_delta_vs_baseline": model_abs_rel - baseline_abs_rel,
            "delta1_delta_vs_baseline": model_delta1 - baseline_delta1,
        }

    def save_artifact(self, output_dir: str | Path, *, producer_revision: str) -> Path:
        """Write safe metric-head weights plus a closed integrity manifest."""
        from safetensors.torch import save_file

        weight_delta = self.adaptation_config.get("weight_delta_l2")
        training_median = self.adaptation_config.get("training_median_depth_m")
        history = self.adaptation_config.get("history")
        if (
            self.model is None
            or not isinstance(weight_delta, int | float)
            or isinstance(weight_delta, bool)
            or not math.isfinite(float(weight_delta))
            or weight_delta <= 0
            or not isinstance(training_median, int | float)
            or isinstance(training_median, bool)
            or not math.isfinite(float(training_median))
            or training_median <= 0
            or not isinstance(history, list)
            or not history
        ):
            raise RuntimeError("artifact export requires completed finite fine-tuning metadata")
        if len(producer_revision) != 40 or any(ch not in "0123456789abcdef" for ch in producer_revision):
            raise ValueError("producer_revision must be a lowercase 40-hex Git commit")
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        if any(root.iterdir()):
            raise FileExistsError(f"artifact directory is not empty: {root}")
        state = {
            name: tensor.detach().cpu().contiguous()
            for name, tensor in self.model.state_dict().items()
            if name.startswith(TRAINABLE_PREFIXES)
        }
        if not state:
            raise RuntimeError("no adapter tensors selected for export")
        weights_path = root / ARTIFACT_WEIGHTS_NAME
        save_file(state, str(weights_path))
        manifest = {
            "artifactSpec": "1.0",
            "format": ARTIFACT_FORMAT,
            "formatVersion": ARTIFACT_FORMAT_VERSION,
            "artifactClass": "ADAPTER",
            "artifactKind": "zoedepth-metric-head-adapter",
            "producer": {
                "pipelineId": "zoedepth-metric-depth-pipeline",
                "revision": producer_revision,
            },
            "createdAtUtc": datetime.now(UTC).isoformat(),
            "baseModel": {"id": MODEL_ID, "revision": MODEL_REVISION},
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "adaptation": dict(self.adaptation_config),
            "trainablePrefixes": list(TRAINABLE_PREFIXES),
            "retainedData": {"containsTrainingRecords": False, "containsSupportRecords": False},
            "serialization": "safetensors",
        }
        (root / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return root

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify and load a ZoeDepth adapter without code-capable deserialization."""
        from safetensors.torch import load_file

        if self.model is None:
            raise RuntimeError("cannot load an artifact without an underlying torch model")
        root = Path(artifact_dir)
        manifest_path = root / ARTIFACT_MANIFEST_NAME
        if not manifest_path.is_file():
            raise FileNotFoundError(f"artifact manifest not found: {manifest_path}")
        expected_files = {ARTIFACT_MANIFEST_NAME, ARTIFACT_WEIGHTS_NAME}
        actual_entries = {path.name for path in root.iterdir()}
        if actual_entries != expected_files:
            raise ValueError(
                f"artifact directory must contain exactly {sorted(expected_files)}, "
                f"found {sorted(actual_entries)}"
            )
        if manifest_path.stat().st_size > MAX_ARTIFACT_MANIFEST_BYTES:
            raise ValueError("artifact manifest exceeds the permitted size")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"unrecognized artifact format: {manifest.get('format')}")
        if manifest.get("formatVersion") != ARTIFACT_FORMAT_VERSION:
            raise ValueError(f"unsupported artifact formatVersion: {manifest.get('formatVersion')}")
        if manifest.get("baseModel") != {"id": MODEL_ID, "revision": MODEL_REVISION}:
            raise ValueError("artifact base model identity is incompatible")
        if manifest.get("trainablePrefixes") != list(TRAINABLE_PREFIXES):
            raise ValueError("artifact trainable prefixes do not match this pipeline")
        files = manifest.get("files")
        if not isinstance(files, list) or len(files) != 1:
            raise ValueError("artifact manifest must inventory exactly one weights file")
        entry = files[0]
        if entry.get("path") != ARTIFACT_WEIGHTS_NAME:
            raise ValueError("artifact manifest names an unexpected weights path")
        weights_path = root / ARTIFACT_WEIGHTS_NAME
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights not found: {weights_path}")
        weights_bytes = weights_path.stat().st_size
        if weights_bytes != entry.get("bytes"):
            raise ValueError("artifact weights failed size or SHA-256 verification")
        adaptation = manifest.get("adaptation")
        if not isinstance(adaptation, dict):
            raise ValueError("artifact adaptation metadata must be a mapping")
        destination = self.model.state_dict()
        expected = {name for name in destination if name.startswith(TRAINABLE_PREFIXES)}
        expected_payload_bytes = sum(
            destination[name].numel() * destination[name].element_size() for name in expected
        )
        with weights_path.open("rb") as stream:
            header_prefix = stream.read(8)
        if len(header_prefix) != 8:
            raise ValueError("artifact weights do not contain a complete SafeTensors header")
        header_bytes = int.from_bytes(header_prefix, byteorder="little", signed=False)
        if not 0 < header_bytes <= MAX_ARTIFACT_HEADER_BYTES:
            raise ValueError("artifact SafeTensors header exceeds the permitted size")
        expected_serialized_bytes = 8 + header_bytes + expected_payload_bytes
        if weights_bytes != expected_serialized_bytes:
            raise ValueError(
                "artifact serialized size does not match the declared adapter surface"
            )
        if _sha256(weights_path) != entry.get("sha256"):
            raise ValueError("artifact weights failed size or SHA-256 verification")
        # Materialize only after the closed inventory has imposed a tight serialized-size ceiling,
        # and stage on CPU so an untrusted artifact cannot allocate directly into accelerator RAM.
        state = load_file(str(weights_path), device="cpu")
        if set(state) != expected:
            raise ValueError("artifact tensor inventory does not match the declared adapter surface")
        for name, tensor in state.items():
            expected_tensor = destination[name]
            if tensor.shape != expected_tensor.shape or tensor.dtype != expected_tensor.dtype:
                raise ValueError(
                    f"artifact tensor {name} has shape/dtype {tuple(tensor.shape)}/{tensor.dtype}; "
                    f"expected {tuple(expected_tensor.shape)}/{expected_tensor.dtype}"
                )
            if not bool(tensor.isfinite().all()):
                raise ValueError(f"artifact tensor {name} contains non-finite values")
        self.model.load_state_dict(state, strict=False)
        # A freshly constructed base model is fully trainable. Restore the declared adapter surface
        # so a verified artifact can be fine-tuned again without exposing the backbone.
        self.freeze_for_adaptation()
        self.model.to(self.device).eval()
        self.adaptation_config = dict(adaptation)
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> ZoeDepthMetricPipeline:
        """Construct a fresh pinned base model and attach a verified adapter."""
        pipeline = cls.from_pretrained(
            device=device, weights_dir=weights_dir, allow_download=allow_download
        )
        pipeline.load_artifact(artifact_dir)
        return pipeline

    def predict(self, image: Image.Image, *, flip_augmentation: bool = FLIP_AUGMENTATION) -> dict[str, Any]:
        """Return metric depth in metres as a float32 H x W array at the input resolution."""
        rgb = validate_image(image)
        flip = _check_flip(flip_augmentation)
        depth = np.asarray(self._runner(rgb, flip), dtype=np.float32)
        if depth.shape != (rgb.height, rgb.width):
            raise RuntimeError(f"backend returned shape {depth.shape}, expected {(rgb.height, rgb.width)}")
        if not np.all(np.isfinite(depth)) or depth.min() <= 0:
            raise RuntimeError("backend returned non-finite or non-positive depth values")
        return {
            "depth": depth,
            "depth_kind": DEPTH_KIND,
            "depth_unit": DEPTH_UNIT,
            "depth_min": float(depth.min()),
            "depth_max": float(depth.max()),
            "depth_median": float(np.median(depth)),
            "flip_augmentation": flip,
            "height": rgb.height,
            "width": rgb.width,
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }
