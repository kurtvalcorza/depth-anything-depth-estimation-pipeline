"""Monocular relative depth estimation over the pinned Depth Anything V2 Small checkpoint, plus a bounded
supervised-adaptation contract.

Inference (`predict`) is unchanged: one PIL image in, relative inverse depth at the input resolution out.
The adaptation contract (`evaluate`, `adapt`, `save_artifact`, `from_artifact`) scores the model on a
validated `{id, image, depth, mask}` dataset by affine-aligned AbsRel and δ1 (`metrics.py`), fine-tunes the
DPT neck and head on the frozen backbone (the **frozen policy**) and optionally the last transformer blocks
with them (the **unfrozen policy**) under a scale-and-shift-invariant loss, keeps the epoch with the lowest
validation AbsRel — where epoch 0 is the untouched checkpoint, the **zero-shot policy**, competing on equal
terms — and exports the trained tensors as a safetensors adapter bound to the pinned base weights.
"""

from __future__ import annotations

import hashlib
import json
import math
import random
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
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
WEIGHTS_FILE = "model.safetensors"
WEIGHT_SHA256 = (
    "3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70"  # manifest digest of WEIGHTS_FILE
)
PARAMETER_COUNT = 24_785_089  # backbone 22,056,576 + neck 2,700,768 + head 27,745
TRANSFORMER_BLOCKS = 12  # DINOv2 ViT-S/14 backbone depth
DEFAULT_TRAINABLE_BLOCKS = (
    2  # the unfrozen policy trains the last two blocks (3,550,464 parameters) with neck + head
)
NECK_HEAD_PARAMETERS = 2_728_513  # the DPT neck and head, trained under every adapted policy
ADAPTER_PREFIXES = ("neck.", "head.")
BLOCK_PREFIX = "backbone.encoder.layer."
POLICY_ZERO_SHOT = "zero-shot (no adaptation)"
POLICY_FROZEN = "frozen backbone + DPT neck and head"
MAX_EVAL_RECORDS = 2_000
MIN_SCORED_RECORDS = 20  # below this a scored dataset is labelled a small sample
ARTIFACT_FORMAT = "org.valcorza.depth-anything-v2-small.adapter.v1"
ARTIFACT_FORMAT_VERSION = "1.0"
ARTIFACT_WEIGHTS_NAME = "adapter.safetensors"
ARTIFACT_MANIFEST_NAME = "manifest.json"


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
    (``ref_depth`` inside ``metrics.EVAL_DEPTH_RANGE_M``, 0.6..350 m), inverted to depth with the
    aligned inverse depth floored at ``1 / 350 m`` (the far cap of relative-depth evaluation, so a pixel
    pushed past it counts as 350 m rather than as an unbounded error), and scored as
    ``mean(|est - ref| / ref)``. With ``align=False`` the arrays are compared as given, which is only
    meaningful for metric input. `metrics.aligned_abs_rel` is the same computation with a mask argument.
    """
    from .metrics import EVAL_DEPTH_RANGE_M

    pred = np.asarray(pred, dtype=np.float64)
    ref_depth = np.asarray(ref_depth, dtype=np.float64)
    if pred.shape != ref_depth.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs ref {ref_depth.shape}")
    low, high = EVAL_DEPTH_RANGE_M
    valid = np.isfinite(ref_depth) & (ref_depth >= low) & (ref_depth <= high) & np.isfinite(pred)
    if valid.sum() < 2:
        raise ValueError(f"need at least 2 valid reference pixels ({low} <= ref_depth <= {high})")
    if align:
        target = 1.0 / ref_depth[valid]
        design = np.stack([pred[valid], np.ones(int(valid.sum()))], axis=1)
        (scale, shift), *_ = np.linalg.lstsq(design, target, rcond=None)
        estimate = 1.0 / np.clip(scale * pred[valid] + shift, 1.0 / high, None)
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


INPUT_SCHEMA: dict[str, Any] = {
    "input": "PIL.Image.Image, or a sequence of them for the validation stage; any mode, converted to RGB",
    "short_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "long_side_px": [MIN_IMAGE_SIDE, MAX_IMAGE_SIDE],
    "aspect_ratio": [1.0, MAX_ASPECT_RATIO],
    "output": f"{DEPTH_KIND} inverse depth, float32 H x W at the input resolution (larger = nearer)",
    "preprocessing": (
        "convert to RGB; the DPT processor rescales the sides to multiples of 14 near 518 px and the "
        "raw prediction is interpolated back (bicubic) to the input resolution"
    ),
}


def validate_inputs(images: Any, *, names: Sequence[str] | None = None) -> dict[str, Any]:
    """Validation stage: return the input manifest (schema, per-input observations, verdict).

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

    With ``reference_depth`` (metric depth in metres, same H x W as the prediction) the report carries
    ``abs_rel`` computed by the repository's own helper after least-squares affine alignment in inverse
    depth, as sample-sanity evidence. Without it the verdict is ``not-measurable`` and the report says
    what ground truth would make the task measurable: relative inverse depth has no intrinsic score.
    """
    depth = np.asarray(result["depth"])
    base = {
        "task": "monocular relative depth estimation",
        "score_semantics": (
            f"{result.get('depth_kind', DEPTH_KIND)} inverse depth with unknown per-image scale and shift: "
            "larger is nearer, values are not metres, carry no confidence, and no threshold is shipped"
        ),
        "sample_kind": sample_kind,
        "n_images": 1,
        "n_pixels": int(depth.size),
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
                "sensor, LiDAR, or an RGB-D benchmark, scored with abs_rel(pred, ref_depth, align=True) "
                "against a constant-depth or vertical-gradient prior as the trivial baseline"
            ),
        }
    from .metrics import EVAL_DEPTH_RANGE_M

    ref = np.asarray(reference_depth, dtype=np.float64)
    valid = int((np.isfinite(ref) & (ref >= EVAL_DEPTH_RANGE_M[0]) & (ref <= EVAL_DEPTH_RANGE_M[1])).sum())
    return {
        **base,
        "metrics": [
            {
                "id": "abs_rel",
                "value": abs_rel(depth, ref, align=True),
                "align": True,
                "n_valid_pixels": valid,
                "estimation": (
                    "single image, least-squares affine alignment in inverse depth, no dispersion estimate"
                ),
            }
        ],
        "verdict": "sample-sanity",
        "reason": "one image with caller-supplied metric depth from the tutorial sample; not a benchmark",
        "needs": "a held-out set of metric depth maps from the deployment domain for any generalisable claim",
    }


@dataclass
class DepthAnythingPipeline:
    """Monocular relative depth estimation over the pinned Depth Anything V2 Small checkpoint."""

    _runner: Callable[[Image.Image], np.ndarray]
    device: str
    source: str = "injected"
    _model: Any = field(default=None, repr=False)
    _processor: Any = field(default=None, repr=False)
    adapter: dict[str, Any] | None = field(default=None, repr=False)

    @classmethod
    def from_pretrained(
        cls,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> DepthAnythingPipeline:
        root = Path(weights_dir) if weights_dir is not None else DEFAULT_WEIGHTS_DIR
        if (root / MANIFEST_NAME).is_file():
            stage_missing_files(root, allow_download=allow_download)
            verify_snapshot(root)
            source, kwargs, origin = str(root), {"local_files_only": True}, "local-snapshot"
        elif allow_download:
            source, kwargs, origin = MODEL_ID, {}, "hub"
        else:
            raise FileNotFoundError(
                f"no verified snapshot at {root} and allow_download=False; "
                f"stage {MODEL_ID}@{MODEL_REVISION} under weights/{MODEL_KEY}"
            )
        # Refuse invalid snapshots before importing model libraries.
        import torch
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        resolved_device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
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

        return cls(runner, resolved_device, origin, _model=model, _processor=processor)

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

    # ---- adaptation ----
    def _require_model(self) -> tuple[Any, Any]:
        if self._model is None or self._processor is None:
            raise ValueError(
                "this operation needs a pipeline built with from_pretrained() or from_artifact()"
            )
        return self._model, self._processor

    def _trainable_names(self, trainable_blocks: int) -> list[str]:
        """`neck.*` and `head.*` always; plus the last `trainable_blocks` backbone blocks."""
        if isinstance(trainable_blocks, bool) or not isinstance(trainable_blocks, int):
            raise ValueError(f"trainable_blocks must be an int in 0..{TRANSFORMER_BLOCKS}")
        if not 0 <= trainable_blocks <= TRANSFORMER_BLOCKS:
            raise ValueError(f"trainable_blocks must be an int in 0..{TRANSFORMER_BLOCKS}")
        model, _ = self._require_model()
        first = TRANSFORMER_BLOCKS - trainable_blocks
        names = []
        for name, _p in model.named_parameters():
            is_block = name.startswith(BLOCK_PREFIX) and int(name[len(BLOCK_PREFIX) :].split(".")[0]) >= first
            if name.startswith(ADAPTER_PREFIXES) or is_block:
                names.append(name)
        return names

    def evaluate(self, records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Score `predict` on validated `{id, image, depth, mask}` records: affine-aligned AbsRel and δ1."""
        from .metrics import depth_metrics
        from .samples import validate_dataset

        checked = validate_dataset(records, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
        started = time.perf_counter()
        preds = [self.predict(r["image"])["depth"] for r in checked]
        metrics = depth_metrics(preds, checked)
        return {
            **metrics,
            "policy": self.adapter["policy"] if self.adapter else POLICY_ZERO_SHOT,
            "adapted": self.adapter is not None,
            "verdict": "measured" if len(checked) >= MIN_SCORED_RECORDS else "measured-small-sample",
            "seconds": round(time.perf_counter() - started, 3),
            "model_id": MODEL_ID,
            "model_revision": MODEL_REVISION,
        }

    @staticmethod
    def _targets(record: Mapping[str, Any], size: tuple[int, int], device: str) -> tuple[Any, Any]:
        """Inverse-depth target and validity mask at the model's working resolution (nearest resampling)."""
        import torch

        depth = torch.from_numpy(np.ascontiguousarray(record["depth"], dtype=np.float32))[None, None]
        mask = torch.from_numpy(np.ascontiguousarray(record["mask"], dtype=np.float32))[None, None]
        depth = torch.nn.functional.interpolate(depth, size=size, mode="nearest")[0, 0]
        valid = torch.nn.functional.interpolate(mask, size=size, mode="nearest")[0, 0] > 0.5
        valid &= depth > 0
        return (1.0 / depth.clamp_min(1e-6)).to(device), valid.to(device)

    @staticmethod
    def _ssi_loss(pred: Any, target: Any, valid: Any) -> Any:
        """Scale-and-shift-invariant squared error: the prediction is affinely aligned to the inverse-depth
        target over valid pixels in closed form (gradients flow through the fit), residuals normalised by
        the target's mean magnitude so images at different depths weigh alike."""
        p = pred[valid]
        t = target[valid]
        n = float(p.numel())
        sp, st, spp, spt = p.sum(), t.sum(), (p * p).sum(), (p * t).sum()
        det = (n * spp - sp * sp).clamp_min(1e-12)
        scale = (n * spt - sp * st) / det
        shift = (st - scale * sp) / n
        residual = (scale * p + shift - t) / (t.abs().mean() + 1e-6)
        return (residual * residual).mean()

    def adapt(
        self,
        train: Sequence[Mapping[str, Any]],
        val: Sequence[Mapping[str, Any]] | None = None,
        *,
        trainable_blocks: int = DEFAULT_TRAINABLE_BLOCKS,
        head_epochs: int = 2,
        epochs: int = 3,
        lr: float = 1e-5,
        seed: int = 0,
        progress: Callable[[dict[str, Any]], None] | None = None,
    ) -> dict[str, Any]:
        """Bounded supervised adaptation under an explicit policy ladder, selected on validation.

        Epoch 0 is the untouched checkpoint (the **zero-shot policy**). Stage A (the **frozen policy**) trains
        the DPT neck and head on the frozen backbone for `head_epochs` epochs. Stage B (the **unfrozen
        policy**, when `trainable_blocks` > 0 and `epochs` > 0) continues with the last `trainable_blocks`
        transformer blocks unfrozen for `epochs` more epochs. Every epoch trains one image at a time with
        AdamW (`lr`, weight decay 0.01, gradient clipping 1.0, seeded order, no augmentation) under the
        scale-and-shift-invariant loss and is scored on `val` by `evaluate`; the epoch with the **lowest
        validation AbsRel** is kept and its tensors restored — so the outcome can be "do not adapt".
        Without `val` the final epoch is kept.
        """
        if isinstance(head_epochs, bool) or not isinstance(head_epochs, int) or not 0 <= head_epochs <= 20:
            raise ValueError("head_epochs must be an int in 0..20")
        if isinstance(epochs, bool) or not isinstance(epochs, int) or not 0 <= epochs <= 20:
            raise ValueError("epochs must be an int in 0..20")
        if not isinstance(lr, int | float) or not 0.0 < float(lr) <= 1e-3:
            raise ValueError("lr must be in (0, 1e-3]")
        if isinstance(trainable_blocks, bool) or not isinstance(trainable_blocks, int):
            raise ValueError(f"trainable_blocks must be an int in 0..{TRANSFORMER_BLOCKS}")
        if not 0 <= trainable_blocks <= TRANSFORMER_BLOCKS:
            raise ValueError(f"trainable_blocks must be an int in 0..{TRANSFORMER_BLOCKS}")
        model, processor = self._require_model()
        import torch

        from .samples import validate_dataset

        names_b = self._trainable_names(trainable_blocks)
        names_a = [n for n in names_b if n.startswith(ADAPTER_PREFIXES)]
        train_records = validate_dataset(train)["records"]
        val_records = (
            validate_dataset(val, min_records=1, max_records=MAX_EVAL_RECORDS)["records"]
            if val is not None
            else None
        )
        params = dict(model.named_parameters())
        started = time.perf_counter()

        def brief(metrics: Mapping[str, Any] | None) -> dict[str, float] | None:
            if metrics is None:
                return None
            return {
                "n": metrics["n"],
                "abs_rel": round(metrics["abs_rel"], 6),
                "delta1": round(metrics["delta1"], 6),
            }

        def score() -> dict[str, float] | None:
            model.eval()
            return brief(self.evaluate(val_records)) if val_records is not None else None

        history: list[dict[str, Any]] = [
            {"epoch": 0, "stage": POLICY_ZERO_SHOT, "train_loss": None, "val": score()}
        ]
        best_epoch, best_score = 0, (history[0]["val"]["abs_rel"] if history[0]["val"] else math.inf)
        best_state = {n: params[n].detach().clone() for n in names_b}
        stages: list[tuple[str, list[str], int]] = [(POLICY_FROZEN, names_a, head_epochs)]
        if trainable_blocks > 0 and epochs > 0:
            stages.append((f"unfrozen last {trainable_blocks} blocks + DPT neck and head", names_b, epochs))
        torch.manual_seed(seed)
        rng = random.Random(seed)
        epoch = 0
        for stage, names, n_epochs in stages:
            if n_epochs == 0:
                continue
            for p in model.parameters():
                p.requires_grad_(False)
            for n in names:
                params[n].requires_grad_(True)
            optimiser = torch.optim.AdamW([params[n] for n in names], lr=float(lr), weight_decay=0.01)
            for _ in range(n_epochs):
                epoch += 1
                model.train()
                order = list(range(len(train_records)))
                rng.shuffle(order)
                losses = []
                for index in order:
                    record = train_records[index]
                    pixel_values = processor(images=record["image"], return_tensors="pt")["pixel_values"]
                    pred = model(pixel_values=pixel_values.to(self.device)).predicted_depth[0]
                    target, valid = self._targets(record, tuple(pred.shape), self.device)
                    loss = self._ssi_loss(pred.float(), target, valid)
                    optimiser.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_([params[n] for n in names], 1.0)
                    optimiser.step()
                    losses.append(float(loss.detach()))
                val_metrics = score()
                entry = {
                    "epoch": epoch,
                    "stage": stage,
                    "train_loss": float(np.mean(losses)),
                    "val": val_metrics,
                }
                history.append(entry)
                if progress is not None:
                    progress(entry)
                current = val_metrics["abs_rel"] if val_metrics else -epoch  # no val: the last epoch wins
                if current < best_score:
                    best_epoch, best_score = epoch, current
                    best_state = {n: params[n].detach().clone() for n in names_b}
        with torch.no_grad():
            for n, value in best_state.items():
                params[n].copy_(value)
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
        policy = history[best_epoch]["stage"]
        self.adapter = {
            "policy": policy,
            "trainable_blocks": trainable_blocks,
            "head_epochs": head_epochs,
            "epochs": epochs,
            "lr": float(lr),
            "seed": seed,
            "best_epoch": best_epoch,
            "selection": "lowest validation AbsRel (epoch 0 = zero-shot checkpoint)"
            if val_records is not None
            else "final epoch (no validation split)",
            "n_trainable_head": sum(params[n].numel() for n in names_a),
            "n_trainable_blocks": sum(
                params[n].numel() for n in names_b if not n.startswith(ADAPTER_PREFIXES)
            ),
            "n_total": sum(p.numel() for p in model.parameters()),
            "n_train": len(train_records),
            "n_val": len(val_records) if val_records is not None else 0,
            "history": history,
            "trainable_names": names_b if policy.startswith("unfrozen") else names_a,
            "seconds": round(time.perf_counter() - started, 3),
        }
        return dict(self.adapter)

    def save_artifact(self, output_dir: str | Path, metadata: Mapping[str, Any] | None = None) -> Path:
        """Write the DPT neck and head (and any trained block tensors) as safetensors with a manifest naming
        the base; the backbone blocks travel only when the unfrozen policy was selected."""
        if self.adapter is None:
            raise ValueError("nothing to save: call adapt() first")
        model, _ = self._require_model()
        from safetensors.torch import save_file

        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        names = set(self.adapter.get("trainable_names", []))
        tensors = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items() if k in names}
        weights_path = out / ARTIFACT_WEIGHTS_NAME
        save_file(tensors, str(weights_path), metadata={"format": "pt"})
        manifest = {
            "format": ARTIFACT_FORMAT,
            "format_version": ARTIFACT_FORMAT_VERSION,
            "base_model": {
                "id": MODEL_ID,
                "revision": MODEL_REVISION,
                "key": MODEL_KEY,
                "weight_file": WEIGHTS_FILE,
                "weight_sha256": WEIGHT_SHA256,
            },
            "adapter": {k: v for k, v in self.adapter.items() if k not in ("history", "trainable_names")},
            "history": self.adapter.get("history", []),
            "tensors": sorted(tensors),
            "files": [
                {
                    "path": ARTIFACT_WEIGHTS_NAME,
                    "bytes": weights_path.stat().st_size,
                    "sha256": _sha256(weights_path),
                }
            ],
            "metadata": dict(metadata or {}),
        }
        (out / ARTIFACT_MANIFEST_NAME).write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        return out

    def load_artifact(self, artifact_dir: str | Path) -> dict[str, Any]:
        """Verify an adapter's manifest and digest **before** deserialising, then overlay its tensors."""
        root = Path(artifact_dir)
        manifest = json.loads((root / ARTIFACT_MANIFEST_NAME).read_text(encoding="utf-8"))
        if manifest.get("format") != ARTIFACT_FORMAT:
            raise ValueError(f"artifact format {manifest.get('format')!r} != {ARTIFACT_FORMAT!r}")
        base = manifest.get("base_model", {})
        if (base.get("id"), base.get("revision"), base.get("weight_sha256")) != (
            MODEL_ID,
            MODEL_REVISION,
            WEIGHT_SHA256,
        ):
            raise ValueError("artifact was adapted from a different base model, revision or weight file")
        entry = manifest["files"][0]
        weights_path = root / entry["path"]
        if not weights_path.is_file():
            raise FileNotFoundError(f"artifact weights missing: {weights_path}")
        if _sha256(weights_path) != entry["sha256"] or weights_path.stat().st_size != entry["bytes"]:
            raise ValueError(f"{entry['path']}: digest or size mismatch; refusing to load")
        model, _ = self._require_model()
        import torch
        from safetensors.torch import load_file

        tensors = load_file(str(weights_path))
        if sorted(tensors) != manifest["tensors"]:
            raise ValueError("artifact tensor names differ from its manifest")
        state = model.state_dict()
        for key, value in tensors.items():
            if key not in state or not key.startswith((*ADAPTER_PREFIXES, BLOCK_PREFIX)):
                raise ValueError(
                    f"artifact tensor {key} is not an adaptable neck, head or transformer-block tensor"
                )
            if tuple(value.shape) != tuple(state[key].shape):
                raise ValueError(
                    f"artifact tensor {key}: shape {tuple(value.shape)} != {tuple(state[key].shape)}"
                )
        with torch.no_grad():
            params = dict(model.named_parameters())
            for key, value in tensors.items():
                params[key].copy_(value.to(params[key].dtype))
        model.eval()
        self.adapter = {
            **manifest["adapter"],
            "trainable_names": sorted(tensors),
            "history": manifest.get("history", []),
        }
        return manifest

    @classmethod
    def from_artifact(
        cls,
        artifact_dir: str | Path,
        *,
        device: str | None = None,
        weights_dir: str | Path | None = None,
        allow_download: bool = False,
    ) -> DepthAnythingPipeline:
        pipeline = cls.from_pretrained(device=device, weights_dir=weights_dir, allow_download=allow_download)
        pipeline.load_artifact(artifact_dir)
        return pipeline
