---
license: apache-2.0
model_card_spec: "1.1"
pipeline_tag: depth-estimation
base_model: depth-anything/Depth-Anything-V2-Small-hf
date_published: "2024-06-18"
date_published_source: "Hugging Face Hub repository creation date of the exact hosted checkpoint (`createdAt`, https://huggingface.co/api/models/depth-anything/Depth-Anything-V2-Small-hf)"
---

# Depth Anything V2 Small (DIMER package v0.1.0) — Monocular Relative Depth Estimation

[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-depth--anything%2FDepth--Anything--V2--Small--hf-ffcc4d?style=flat)](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
[![Upstream GitHub](https://img.shields.io/badge/Upstream%20GitHub-DepthAnything%2FDepth--Anything--V2-181717?style=flat&logo=github&logoColor=white)](https://github.com/DepthAnything/Depth-Anything-V2)
[![arXiv Paper](https://img.shields.io/badge/arXiv-2406.09414-b31b1b.svg)](https://arxiv.org/abs/2406.09414)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

> [!WARNING]
> ⚠️ **Provided for research, training, and evaluation purposes only.** Model weights are redistributed unmodified under their upstream license, which controls your use, including any commercial use or redistribution; the accompanying code and notebooks are released under this repository's license. All of it is supplied **"as is"**, without warranty of any kind, and has not been validated for production, clinical, or safety-critical use. Running the notebooks downloads third-party weights and datasets governed by their own licenses and consumes compute on your own Colab/Kaggle account. To the maximum extent permitted by law, the maintainers of this repository and the DIMER platform accept no liability for any damages arising from their use. Hosting implies no affiliation with or endorsement by the original authors.

---

## Interactive Colab Tutorials

This pipeline provides a ready-to-run interactive Google Colab notebook that exercises the repository's public API end to end — bootstrap a fresh runtime, resolve and verify the pinned upstream revision, validate an input, run the task, and inspect and export the outputs:

- **Task Inference Tutorial**:  
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/depth-anything-depth-estimation-pipeline/blob/main/tutorials/depth_anything_depth_estimation_colab.ipynb) [`depth_anything_depth_estimation_colab.ipynb`](https://github.com/kurtvalcorza/depth-anything-depth-estimation-pipeline/blob/main/tutorials/depth_anything_depth_estimation_colab.ipynb)  
  *Relative inverse-depth estimation from one RGB image through `DepthAnythingPipeline` with the pinned Depth Anything V2 Small weights — larger values are nearer, scale and shift are unknown, values are not metric; no adaptation occurs.*

---

#### Description

Depth Anything V2 Small is the smallest Transformers-format checkpoint of the Depth Anything V2 family, published as `depth-anything/Depth-Anything-V2-Small-hf` and pinned here to revision `5426e4f0f36572d16453bbda7a8389317b1bef99`. The snapshot `config.json` declares a `DepthAnythingForDepthEstimation` architecture: a DINOv2 ViT-S backbone (hidden size 384, patch size 14, four tapped stages) feeding a DPT-style reassemble/fusion neck (hidden sizes 48/96/192/384, fusion width 64) and a convolutional depth head. At inference the model reads one RGB image and emits a single-channel map of *relative inverse depth* — larger values are nearer — with no metric scale, and no adaptation happens at inference time. This repository adds nothing to the weights: it contributes `verify_snapshot` (manifest digest checking), `DepthAnythingPipeline.from_pretrained` (verified local loading with `trust_remote_code=False`), `predict` (input validation, resizing of the output back to the input resolution), the `abs_rel` helper for caller-supplied ground truth, and this card.

#### Intended Use and Limitations

The uses below are the ones the package was built to support; everything else is either out of scope (§Out-of-scope use cases) or prohibited (§Use cases).

###### Primary Intended Uses

The task is monocular relative depth estimation: input is one RGB still image (`PIL.Image.Image`, any mode, converted to RGB), output is a float32 array of shape `H x W` at the input resolution holding relative inverse depth, plus `depth_min`, `depth_max`, and `depth_kind: "relative"`. Envisioned applications are per-image depth ordering for photo editing effects (relighting, bokeh, parallax), coarse scene layout for robotics or AR prototyping where a downstream stage supplies scale, foreground/background separation in content tooling, and as a feature input to other models that consume depth maps. Within DIMER the package is an inference component and a zero-configuration baseline for relative depth, not a calibrated measurement service.

###### Primary Intended Users

Intended users are machine-learning engineers, computer-vision researchers, and application developers integrating a depth prior into a larger system, in research or internal enterprise settings. A user is expected to understand that inverse depth is not distance, that the map has arbitrary scale and shift and so cannot be compared across images without alignment, what `abs_rel` measures and why it requires an affine alignment step, and the ceilings the pipeline enforces (`MAX_IMAGE_SIDE`, `MIN_IMAGE_SIDE`, `MAX_ASPECT_RATIO`). Users who need metric depth, camera intrinsics handling, or temporal consistency across video frames are expected to know that none of those is provided here.

###### Out-of-scope use cases

1. **Capability boundary:** metric (absolute) depth is not produced — the output is relative inverse depth with unknown scale and shift, and the upstream metric-depth fine-tunes are not packaged. Video depth with temporal consistency, stereo or multi-view fusion, surface normals, and 3D reconstruction are not implemented.
2. **Input boundary:** `predict` rejects anything that is not a `PIL.Image.Image` (`TypeError`), images whose shorter side is under `MIN_IMAGE_SIDE = 14` px, images whose longer side exceeds `MAX_IMAGE_SIDE = 4096` px, and aspect ratios above `MAX_ASPECT_RATIO = 4.0` (`ValueError`). Batches are not accepted; call once per image.
3. **Input boundary:** non-photographic inputs — line drawings, medical scans, thermal or depth images, screenshots — fall outside the training distribution; the pipeline does not detect them and the output on them is undefined.
4. **Decision boundary:** not for autonomous safety-relevant decisions (vehicle braking, obstacle avoidance, fall detection) or any distance measurement with legal, medical, or financial consequence without an independent metric sensor and human oversight.

#### Factors

###### Groups

This pipeline is not human-centric: it predicts scene geometry, not attributes of people, and its output does not classify or identify anyone. Photographs supplied to it may nonetheless contain people, and the upstream corpus (595K synthetic labelled images plus 62M+ real unlabelled images, per the snapshot README) is not group-audited by the upstream authors or by this repository, so any difference in depth quality across skin tones, ages, body shapes, mobility aids, or clothing is unknown rather than known to be absent. The downstream operator who deploys the model on images of people is responsible for a fairness audit on their own data: stratify a labelled or human-judged sample by the relevant groups and compare `abs_rel` or ordinal agreement per stratum before relying on the output.

###### Instrumentation

The upstream training data comes from two instrument classes: rendered synthetic scenes with exact depth from the renderer, and real photographs of unspecified provenance whose pseudo-labels were produced by a larger teacher model rather than by any depth sensor. Inference images arrive from whatever camera the operator uses; sensor resolution, lens distortion, rolling shutter, HDR tone mapping, JPEG compression, and motion blur all change the pixel evidence the backbone sees. The pipeline resizes every input so that the sides are multiples of 14 near 518 px (`preprocessor_config.json`: `keep_aspect_ratio`, `ensure_multiple_of: 14`, bicubic resampling, ImageNet mean/std normalisation) and interpolates the output back, so fine detail below that scale is lost regardless of the source resolution. The pipeline cannot detect a miscalibrated or defective camera; it only validates type, size, and aspect ratio.

###### Environment

Operating environment: Python 3.12 with `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`. Inference runs in float32 on CUDA when available and on CPU otherwise; both paths were executed for this card (see Runtime). On an RTX 5070 Ti the 320x240 smoke image took 0.67 s cold and 0.024 s warm; peak allocated CUDA memory was 812 MiB for a 4096x1024 input and 298 MiB for 4096x4096, which is why `MAX_ASPECT_RATIO` is a ceiling: elongated inputs cost more backbone tokens than large square ones. Data environment: the model assumes a single ordinary photograph of a scene with perspective structure similar to the upstream synthetic and web-photo mix; outputs on fog, night, reflective or transparent surfaces, repetitive textures, and extreme close-ups are expected to degrade, and the pipeline reports no signal when they do.

#### Metrics

###### Performance Measures

The only measure the repository reports is `abs_rel(pred, ref_depth, align=True)`: absolute relative error, `mean(|est - ref| / ref)` over pixels with `ref_depth > 0`, after the prediction has been affinely aligned to `1 / ref_depth` by least squares and inverted to depth. It captures reconstruction error of the ordinal/affine depth structure and is the standard first metric in the monocular depth literature, which is why it was chosen over pixel-wise RMSE (dominated by far pixels) or threshold accuracies (which need a second convention). It requires caller-supplied metric ground truth; on unlabelled images the pipeline reports nothing, and it does not compute the upstream paper's benchmark numbers, which are not reproduced or claimed here. The public `evaluation_report(result, reference_depth)` helper is the only reporting path: it emits a machine-readable report whose verdict is `sample-sanity` with `abs_rel` when a metric reference depth is supplied, and `not-measurable` otherwise, stating in that case what ground truth would make the task measurable.

###### Decision thresholds

No decision threshold is applied. `predict` returns the raw float32 map together with its minimum and maximum; it does not binarise, quantise, or convert values to distances, and no `argmax` or acceptance rule exists in the code path. The `abs_rel` helper likewise returns a number without judging it. A deployment that needs a pass/fail rule — for example a maximum `abs_rel` on a held-out labelled set before a model update is accepted — must set it against its own data, weighing the cost of a wrong depth ordering (a false near/far assignment) against the cost of rejecting usable images, and owns recalibrating it when the camera or scene distribution changes.

###### Approaches to uncertainty and variability

This repository reports no central metric value and therefore no dispersion: the smoke run records timings and output ranges, not accuracy. Sources of run-to-run variability are floating-point kernel selection (CUDA versus CPU produced `depth_max` 2.969 versus 2.970 on the smoke image), bicubic interpolation of the output, and hardware; there is no sampling and no seed to set, so a fixed input on fixed hardware is repeatable but not guaranteed bitwise-identical across devices. The model emits no confidence map, and its values are not probabilities. A caller who needs an uncertainty estimate must supply labelled data and compute `abs_rel` over multiple images or bootstrap resamples themselves.

#### Ethical considerations and biases

No external ethics board, red-team, or population-specific clearance reviewed this repository or, to our knowledge, the upstream checkpoint; nothing below should be read as implying one.

###### Data

The snapshot README states the model was trained on about 595K synthetic labelled images and 62M+ real unlabelled images pseudo-labelled by a teacher; the upstream disclosure stops there, with no enumeration of the real-image sources, licences, or consent status. Whether personal data (faces, licence plates, private interiors) is present in that corpus is therefore unknown, not ruled out. This repository distributes code, tests, and documentation; it does not distribute the 99,173,660-byte `model.safetensors`, which is staged locally under `weights/depth-anything-v2-small/` and git-ignored, and it ships no sample images. The operator must audit the images they submit for personal, proprietary, or otherwise restricted content; the pipeline performs no such check.

###### Human Life

This pipeline is not intended for decisions in health, safety, criminal justice, employment, credit, or housing, and it has not been validated or certified for any of them by this repository, the upstream authors, or any regulator. Foreseeable but unintended sensitive uses — assistive navigation for visually impaired users, collision warning, clinical measurement from photographs — would be admissible only with an independent metric sensor providing scale, human oversight of every consequential output, domain validation on the deployment's own data, and whatever regulatory clearance the domain requires.

###### Mitigations

- **Supply-chain integrity:** `MODEL_REVISION` is a 40-hex commit; `verify_snapshot` compares the manifest's `modelId`/`revision` to the module constants and every listed file's byte size and SHA-256 to the manifest before any load; `from_pretrained` loads only from a verified local directory with `local_files_only=True`, falls back to the Hub only when `allow_download=True` is passed explicitly (still at `revision=MODEL_REVISION`), and always passes `trust_remote_code=False`. A test flips one hex digit of a manifest digest and asserts the loader refuses.
- **Input integrity:** `validate_image` rejects non-PIL inputs, sub-14-px sides, sides over 4096 px, and aspect ratios over 4.0 before the model runs; `predict` raises if the backend returns a map whose shape differs from the input. The public `validate_inputs(images, names=...)` stage routes every image through that same `validate_image`, so it raises exactly what `predict` raises while returning a machine-readable input manifest of the schema, ceilings, per-input observations and verdict.
- **Reproducibility:** exact `==` pins in `pyproject.toml`; every result carries `model_id` and `model_revision`.
- **Refusals:** no metric conversion, no video path, and no download without the explicit flag; a snapshot with a mismatched revision raises rather than loading.
- No statistical mitigation (class balancing, subsampling) applies: the model is a dense regressor with no classes.

###### Risks and harms

- **Wrong depth ordering** on out-of-distribution scenes (mirrors, glass, textureless walls, night): the operator and any downstream system bear the harm; likely under normal use on such content; magnitude ranges from a cosmetic artefact to a navigation error if the output is misused for control.
- **Scale misreading:** a user treats inverse-depth values as distances; harm falls on whoever acts on the number; the pipeline mitigates by labelling `depth_kind: "relative"` but cannot prevent it.
- **Automation bias:** a smooth, plausible map invites trust it has not earned; reviewers may stop checking.
- **Privacy exposure:** images of people or private spaces submitted for depth are processed without any content check; the data subject bears the harm.
- **Bias amplification:** any under-representation in the 62M web images (regions, building styles, skin tones) is reproduced as lower quality for those inputs, undetected because no per-group evaluation exists.
- **Resource failure:** near-ceiling inputs (4096 px, aspect 4) reached 812 MiB on the smoke GPU; smaller accelerators may fail out of memory.

###### Use cases

Prohibited even where the model would work: covert surveillance or tracking of people via depth-based presence or gait cues, biometric or demographic profiling, social scoring, and any use that discriminates unlawfully in employment, housing, credit, insurance, education, or healthcare access. Also prohibited are deceptive uses — fabricating "measured" depth or 3D evidence presented as sensor data — and any use that violates the upstream Apache-2.0 licence terms, the DIMER deployment terms, or the consent and data-protection obligations attached to the images processed. High-consequence physical control based on this output alone is prohibited by the intended-use contract above.

## Immutable provenance

- Model: `depth-anything/Depth-Anything-V2-Small-hf`
- Revision: `5426e4f0f36572d16453bbda7a8389317b1bef99`
- Manifest: `weights/depth-anything-v2-small/dimer-base-manifest.json`, 4 files, `totalBytes` 99179785
- `model.safetensors` (99,173,660 bytes) SHA-256: `3152477ce0d8d6978d76b995120de97cb5b928701fd0f817769f59e249a16b70`
- `config.json` (950 bytes) SHA-256: `c56698d3643dde1f83ea2212759e6b31a22b8f827246a36dd007ee8a22b3ff75`
- Weight format: SafeTensors; loader trust boundary `trust_remote_code=False`

## Input/output contract

- `DepthAnythingPipeline.from_pretrained(device=None, weights_dir=None, allow_download=False)` — verifies and loads the local snapshot; `device` defaults to `cuda:0` when available, else `cpu`.
- `predict(image: PIL.Image.Image) -> dict` with keys `depth` (float32 `H x W`, relative inverse depth, larger = nearer), `depth_kind` (`"relative"`), `depth_min`, `depth_max`, `height`, `width`, `model_id`, `model_revision`.
- Ceilings: `MIN_IMAGE_SIDE = 14`, `MAX_IMAGE_SIDE = 4096`, `MAX_ASPECT_RATIO = 4.0`.
- `abs_rel(pred, ref_depth, align=True) -> float` — caller supplies metric depth of the same shape; zeros and non-finite reference pixels are ignored.
- `verify_snapshot(path=None) -> dict` — raises `FileNotFoundError`/`ValueError` naming the first mismatch.

## Runtime

- Pins: `torch==2.14.0`, `transformers==4.57.6`, `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0`; Python 3.12.
- Measured 2026-09-12 in the Windows venv (`torch 2.14.0+cu130`, RTX 5070 Ti 16 GB, float32): `verify_snapshot` 0.09 s; load 5.70 s; `predict` on a synthetic 320x240 gradient image 0.668 s cold / 0.024 s warm, output `(240, 320)` float32, range 1.420–2.969; 4096x1024 input 0.31 s at 812 MiB peak allocated; 4096x4096 input 0.25 s at 298 MiB peak.
- CPU path executed on the same machine: load 5.43 s, 320x240 predict 0.339 s cold / 0.246 s warm, range 1.420–2.970.
- Not executed: Linux venv, half precision, batch inference, any accuracy measurement against ground truth.

## References

- Yang et al., *Depth Anything V2*, arXiv:2406.09414 — https://arxiv.org/abs/2406.09414
- Yang et al., *Depth Anything: Unleashing the Power of Large-Scale Unlabeled Data*, arXiv:2401.10891 — https://arxiv.org/abs/2401.10891
- Upstream model page: https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf
- Upstream code: https://github.com/DepthAnything/Depth-Anything-V2
- Transformers `DepthAnything` documentation: https://huggingface.co/docs/transformers/model_doc/depth_anything
