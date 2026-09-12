# Tutorials

[![GitHub](https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white)](https://github.com/kurtvalcorza/depth-anything-depth-estimation-pipeline)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kurtvalcorza/depth-anything-depth-estimation-pipeline/blob/main/tutorials/depth_anything_depth_estimation_colab.ipynb)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-depth--anything%2FDepth--Anything--V2--Small--hf-ffcc4d?style=flat)](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
[![Upstream](https://img.shields.io/badge/Upstream-DepthAnything%2FDepth--Anything--V2-181717?style=flat&logo=github&logoColor=white)](https://github.com/DepthAnything/Depth-Anything-V2)
[![arXiv](https://img.shields.io/badge/arXiv-2406.09414-b31b1b.svg)](https://arxiv.org/abs/2406.09414)

Notebook specification: **DIMER Notebook Specification 1.0**

| Notebook | Profile | Capability | Default runtime | BYOD | Release status |
|---|---|---|---|---|---|
| `depth_anything_depth_estimation_colab.ipynb` | `TASK-INFERENCE` | Depth Anything V2 Small monocular relative (inverse) depth estimation on a synthetic in-code sample; `abs_rel` scored only when the caller supplies metric ground truth | CPU (CUDA used automatically when available) | single image file, gated off by default | **Candidate** — static checks pass; the clean-runtime execution row in `../docs/release-verification.md` is pending and must be recorded for the exact notebook revision before promotion |

## Conformance notes

- The notebook exercises `DepthAnythingPipeline` from the repository public API rather than reimplementing model loading; the pipeline pins the immutable upstream revision, loads only from a digest-verified local snapshot (`verify_snapshot`), and refuses remote model code.
- The default sample is synthetic (a 320 x 240 luminance ramp with two drawn shapes, generated in code); it has no ground-truth depth, so no metric is reported on the default path and the run is plumbing/sanity evidence only. The repository's `abs_rel` helper is applied only when the user supplies a metric depth map for a BYOD image.
- The output is relative inverse depth (larger = nearer, unknown scale and shift), never metric depth; the notebook states this before, during, and after inference and surfaces `MIN_IMAGE_SIDE`, `MAX_IMAGE_SIDE`, and `MAX_ASPECT_RATIO` before the model runs.
- `USE_BYOD` defaults to `False` so the sample path never opens an upload dialog.
- Recorded `SHOULD` deviation: EVAL11 (no trivial baseline — one cannot be scored without ground truth, which the default sample lacks). Missing snapshot files are staged through the package's `stage_missing_files(..., allow_download=True)`, never by calling `huggingface_hub` from the notebook.
- `tools/validate_release_assets.py` performs source validation only. It does not satisfy the
  clean-runtime execution requirement; a release review must confirm that a recorded clean run in
  `docs/release-verification.md` matches the notebook revision under review before the status is
  promoted to `Release-grade`.
