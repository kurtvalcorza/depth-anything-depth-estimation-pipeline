"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 1.1 §3.6 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
module, and the model pin/stage/verify cells are produced by the generator from repository
sources so they cannot drift from the package.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "depth_anything_depth_estimation_pipeline",
    "repo_name": "depth-anything-depth-estimation-pipeline",
    "stem": "depth_anything_depth_estimation",
    "notebook_name": "depth_anything_depth_estimation_colab.ipynb",
    "profile": "TASK-INFERENCE",
    "pipeline_class": "DepthAnythingPipeline",
    "weights_key": "depth-anything-v2-small",
    "runtime_imports": ["torch", "transformers"],
    "title": "Depth Anything V2 Small — DIMER relative depth estimation tutorial (standalone)",
    "badges": [
        (
            "GitHub",
            "https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/kurtvalcorza/depth-anything-depth-estimation-pipeline",
        ),
        (
            "Open In Colab",
            "https://colab.research.google.com/assets/colab-badge.svg",
            "https://colab.research.google.com/github/kurtvalcorza/depth-anything-depth-estimation-pipeline/blob/main/tutorials/depth_anything_depth_estimation_colab.ipynb",
        ),
        (
            "Hugging Face",
            "https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-depth--anything%2FDepth--Anything--V2--Small--hf-ffcc4d?style=flat",
            "https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf",
        ),
        (
            "Upstream",
            "https://img.shields.io/badge/Upstream-DepthAnything%2FDepth--Anything--V2-181717?style=flat&logo=github&logoColor=white",
            "https://github.com/DepthAnything/Depth-Anything-V2",
        ),
        ("arXiv", "https://img.shields.io/badge/arXiv-2406.09414-b31b1b.svg", "https://arxiv.org/abs/2406.09414"),
    ],
    "capability": "monocular relative depth estimation from one RGB still image using the pinned Depth Anything V2 Small weights",
    "intro": (
        "The output is a per-pixel map of **relative inverse depth** — larger values are nearer, the scale and shift are "
        "unknown, and the values are **not metric**: they are not distances in metres and cannot be compared across "
        "images without an alignment step. **No adaptation occurs:** no training, fine-tuning, in-context conditioning, "
        "or preprocessing fitting happens in this notebook — the pinned checkpoint is used as published, and the carried "
        "pipeline module adds manifest verification, input validation, resizing of the output back to the input "
        "resolution, and the `abs_rel`, `validate_inputs` and `evaluation_report` helpers. The default sample is a "
        "synthetic image drawn in code; its depth map is demonstration (plumbing) evidence, not a production-quality or "
        "benchmark claim."
    ),
    "learning_objectives": (
        "install the pinned runtime, read what the carried pipeline module guarantees, resolve and digest-verify the "
        "immutable upstream model revision, draw a synthetic default input and validate it into an input manifest, run "
        "the supported task through the public API, interpret the relative-depth output and its sanity checks, "
        "understand when the shipped `abs_rel` metric applies and why the evaluation report is `not-measurable` without "
        "caller-supplied metric depth, and export machine-readable outputs plus provenance."
    ),
    "exclusions": (
        "metric (absolute) depth, video or temporal depth, stereo or multi-view fusion, surface normals, 3D "
        "reconstruction, or batch inference. The carried module does not provide metric depth, video, or batched paths, "
        "and this notebook must not be read as implying them."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; inference is float32 on both. The model is about 99 MB and the default sample is 320 x 240 px, so a CPU runtime completes the default path (the model card records a 320 x 240 CPU prediction at 0.34 s cold on the card's workstation; a hosted CPU runtime may be slower and no figure is claimed for it).",
        "- **Knowledge:** basic Python and NumPy, and the difference between inverse depth (larger = nearer, arbitrary scale and shift) and metric distance.",
        "- **Data:** the default sample is a 320 x 240 RGB image drawn in this notebook, so nothing is downloaded and no private data is needed. Optional BYOD upload is gated off by default so the sample path can run top-to-bottom without interaction. Expected BYOD input: one image file decodable by Pillow (PNG/JPEG/WebP and similar), any colour mode, shorter side at least 14 px, longer side at most 4096 px, aspect ratio at most 4.0. Do not upload confidential or restricted data to a hosted notebook environment unless you are authorized to do so. Uploaded inputs remain in the notebook runtime; this pipeline does not send them to a third-party inference API.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Draw the synthetic sample or optional BYOD\n\n"
                "The default sample is **synthetic**: a 320 x 240 RGB image drawn in this cell — a left-to-right "
                "luminance ramp with a bright square and a dark disc — the same kind of input the repository's smoke run "
                "used. It is generated deterministically from code (no randomness, so no seed is involved), its pixel "
                "digest is recorded in the export so a rerun can prove it saw the same input, and it needs no download "
                "and contains no personal data. It ships **no ground-truth depth**, and as a non-photographic image it "
                "lies outside the model's training distribution, so the depth map it produces is sanity evidence of the "
                "code path only — it says nothing about depth quality on real photographs and is not benchmark evidence. "
                "BYOD is optional and disabled by default. If you also hold a metric depth map for your image (a float "
                "array in metres with the same height and width, from a depth sensor or a benchmark), assign it to "
                "`reference_depth` after the upload and Section 7 will score it with the repository's `abs_rel` helper; "
                "leave it as `None` otherwise."
            ),
            "code": (
                "import hashlib\n"
                "import io\n\n"
                "import numpy as np\n"
                "from PIL import Image, ImageDraw\n\n"
                "USE_BYOD = False  # @param {{type:\"boolean\"}}\n"
                "SAMPLE_WIDTH = 320\n"
                "SAMPLE_HEIGHT = 240\n\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    sample_name = next(iter(uploaded))\n"
                "    image = Image.open(io.BytesIO(uploaded[sample_name]))\n"
                "    image.load()\n"
                "    sample_kind = 'BYOD upload'\n"
                "else:\n"
                "    # Deterministic synthetic scene: a left-to-right luminance ramp (40 -> 220) with a bright\n"
                "    # square and a dark disc. Drawn from code, so it is reproducible without any download.\n"
                "    ramp = np.linspace(40, 220, SAMPLE_WIDTH, dtype=np.float32)\n"
                "    rgb = np.repeat(np.repeat(ramp[None, :, None], SAMPLE_HEIGHT, axis=0), 3, axis=2).astype(np.uint8)\n"
                "    image = Image.fromarray(rgb, mode='RGB')\n"
                "    draw = ImageDraw.Draw(image)\n"
                "    draw.rectangle([200, 60, 280, 140], fill=(245, 245, 245))\n"
                "    draw.ellipse([40, 120, 130, 210], fill=(20, 20, 20))\n"
                "    sample_name = f'synthetic_ramp_{{SAMPLE_WIDTH}}x{{SAMPLE_HEIGHT}}'\n"
                "    sample_kind = 'synthetic'\n"
                "# Metric ground-truth depth in metres with the same H x W as the image, or None. The synthetic\n"
                "# sample has none, so the evaluation report is not-measurable for it.\n"
                "reference_depth = None\n"
                "sample_sha256 = hashlib.sha256(np.asarray(image.convert('RGB')).tobytes()).hexdigest()\n"
                "print({{'sample': sample_name, 'sample_kind': sample_kind, 'mode': image.mode, 'size': image.size, 'has_reference_depth': reference_depth is not None, 'pixel_sha256': sample_sha256}})"
            ),
        },
        {
            "md": (
                "## 5. Validate the input → input manifest\n\n"
                "`validate_inputs` is the pipeline's public validation stage: each image is routed through the same "
                "`validate_image` that `predict` itself calls, so the checks — PIL type, shorter side at least "
                "`MIN_IMAGE_SIDE`, longer side at most `MAX_IMAGE_SIDE`, aspect ratio at most `MAX_ASPECT_RATIO` — "
                "cannot diverge between the two. It returns an **input manifest** naming the schema and ceilings, each "
                "input's identifier, observed mode, size and aspect ratio, and the verdict, written to "
                "`outputs/{stem}_input_manifest.json`. The ceilings are printed first, before any model work. To show "
                "what rejection looks like, the cell also validates a deliberately over-wide image and records the "
                "pipeline's own error message as a finding. The notebook does not crop, resize, or subsample the input; "
                "inside the pipeline the image processor rescales it so its sides are multiples of 14 near 518 px and "
                "the raw prediction is interpolated back to the input resolution, so detail finer than that internal "
                "scale is lost regardless of the source resolution. The model card records that peak accelerator memory "
                "grows with aspect ratio (812 MiB at 4096 x 1024 versus 298 MiB at 4096 x 4096 on the card's GPU), "
                "which is why an aspect ceiling exists."
            ),
            "code": (
                "import json\n"
                "import os\n\n"
                "os.makedirs('outputs', exist_ok=True)\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_ASPECT_RATIO': MAX_ASPECT_RATIO}}}})\n"
                "input_manifest = validate_inputs(image, names=[sample_name])\n"
                "# Demonstrate rejection on an input that breaks a ceiling; the finding is recorded, not swallowed.\n"
                "try:\n"
                "    validate_inputs(Image.new('RGB', (MIN_IMAGE_SIDE, int(MIN_IMAGE_SIDE * (MAX_ASPECT_RATIO + 1)))))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'over-wide-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(input_manifest, indent=2))"
            ),
        },
        {
            "md": (
                "## 6. Run relative depth estimation\n\n"
                "`predict()` returns a dictionary: `depth` (float32 `H x W` at the input resolution), `depth_kind` "
                "(`\"relative\"`), `depth_min`, `depth_max`, `height`, `width`, `model_id`, `model_revision`. "
                "**Score semantics:** each value is relative inverse depth — larger means nearer — with an unknown "
                "per-image scale and shift. The values are not distances, not probabilities, and carry no confidence "
                "map; the pipeline applies no decision threshold, no binarisation, and no unit conversion, and any "
                "near/far cut-off is owned by the downstream caller. The checks below are falsifiable plumbing checks "
                "(shape equals the input, float32, finite, non-degenerate range) and the cell raises if any fails; they "
                "are not a quality measure. The timing is measured on the runtime identified in Section 1 for this one "
                "image."
            ),
            "code": (
                "import time\n\n"
                "started = time.perf_counter()\n"
                "result = pipe.predict(image)\n"
                "elapsed = time.perf_counter() - started\n"
                "depth = result['depth']\n"
                "checks = {{\n"
                "    'shape_matches_input': depth.shape == (result['height'], result['width']) == (image.height, image.width),\n"
                "    'dtype_float32': depth.dtype == np.float32,\n"
                "    'all_finite': bool(np.isfinite(depth).all()),\n"
                "    'non_degenerate_range': result['depth_max'] > result['depth_min'],\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'depth map failed a sanity check: {{checks}}')\n"
                "print({{key: value for key, value in result.items() if key != 'depth'}})\n"
                "print({{'depth_shape': depth.shape, 'seconds': round(elapsed, 3), 'checks': checks}})"
            ),
        },
        {
            "md": (
                "## 7. Evaluate → evaluation report\n\n"
                "`evaluation_report` is the pipeline's public evaluation stage and always produces a report. The only "
                "metric the repository ships is `abs_rel(pred, ref_depth, align=True)` — absolute relative error "
                "`mean(|est - ref| / ref)` over pixels with `ref_depth > 0`, computed after the prediction is affinely "
                "aligned to `1 / ref_depth` by least squares and inverted to depth. It applies only when the caller "
                "supplies metric ground-truth depth of the same height and width, so with `reference_depth` set the "
                "verdict is `sample-sanity` and the metric is a single-image tutorial figure with no dispersion "
                "estimate. The synthetic sample has none, so the verdict is `not-measurable` and the report states what "
                "would make the task measurable: a metric depth map in metres from a depth sensor, LiDAR, or an RGB-D "
                "benchmark, scored against a constant-depth or vertical-gradient prior as the trivial baseline. The "
                "report is written to `outputs/{stem}_evaluation_report.json`."
            ),
            "code": (
                "report = evaluation_report(result, reference_depth, sample_kind=sample_kind)\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(report, handle, indent=2, ensure_ascii=False)\n"
                "print(json.dumps(report, indent=2))\n"
                "if report['verdict'] == 'not-measurable':\n"
                "    print('No metric reference depth was supplied, so abs_rel is not computed; the depth map above is sanity evidence only.')"
            ),
        },
        {
            "md": (
                "## 8. Visualize the relative depth map\n\n"
                "The preview is a per-image min–max stretch of the relative inverse depth to 8-bit grey, shown beside "
                "the input: brighter means nearer (a larger value). The grey levels are a visual aid only — they are "
                "not distances, the stretch discards the model's scale, and two previews cannot be compared with each "
                "other. The machine-readable arrays exported in the next section, not this picture, are the outputs "
                "intended for downstream use. On the synthetic sample expect a plausible but meaningless map: the input "
                "is not a photograph."
            ),
            "code": (
                "lo, hi = result['depth_min'], result['depth_max']\n"
                "depth_u8 = np.round((depth - lo) / (hi - lo) * 255.0).astype(np.uint8)\n"
                "depth_preview = Image.fromarray(depth_u8).convert('RGB')\n"
                "side_by_side = Image.new('RGB', (image.width * 2, image.height))\n"
                "side_by_side.paste(image.convert('RGB'), (0, 0))\n"
                "side_by_side.paste(depth_preview, (image.width, 0))\n"
                "try:\n"
                "    from IPython.display import display\n"
                "    display(side_by_side)\n"
                "except ImportError:\n"
                "    print({{'preview': 'IPython display unavailable; the preview PNG is written in the next section'}})"
            ),
        },
        {
            "md": (
                "## 9. Export outputs and provenance\n\n"
                "Four files are written under `outputs/` alongside the two role-stage artefacts: the full-resolution "
                "float32 depth array (`.npy`, the array intended for downstream use), the side-by-side preview PNG, and "
                "a JSON record that ties them to the sample identity (name, kind, pixel digest, size), the depth "
                "summary (`depth_kind`, min, max, shape, dtype, measured seconds), the input manifest, the evaluation "
                "report, the sanity checks, the notebook's source (repository, revision, embedded module digest, "
                "generator), the model identifier, the immutable model revision, the model licence, and the runtime "
                "identity (Python, PyTorch, Transformers, device). No credentials are involved in any step, so none can "
                "reach the export."
            ),
            "code": (
                "np.save('outputs/{stem}_depth.npy', depth)\n"
                "side_by_side.save('outputs/{stem}_preview.png')\n"
                "payload = {{\n"
                "    'sample': {{'name': sample_name, 'kind': sample_kind, 'pixel_sha256': sample_sha256, 'width': image.width, 'height': image.height, 'has_reference_depth': reference_depth is not None}},\n"
                "    'prediction': {{\n"
                "        'depth_file': 'outputs/{stem}_depth.npy',\n"
                "        'preview_file': 'outputs/{stem}_preview.png',\n"
                "        'depth_kind': result['depth_kind'],\n"
                "        'depth_min': result['depth_min'],\n"
                "        'depth_max': result['depth_max'],\n"
                "        'shape': list(depth.shape),\n"
                "        'dtype': str(depth.dtype),\n"
                "        'seconds': round(elapsed, 3),\n"
                "    }},\n"
                "    'input_manifest': input_manifest,\n"
                "    'evaluation_report': report,\n"
                "    'sanity_checks': checks,\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'runtime': {{\n"
                "        'python': platform.python_version(),\n"
                "        'torch': torch.__version__,\n"
                "        'transformers': transformers.__version__,\n"
                "        'device': pipe.device,\n"
                "        'precision': 'float32',\n"
                "    }},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The depth map is a model-generated map of relative inverse depth: larger is nearer, the scale and shift are "
        "unknown, and the values are not metres, not probabilities, and not comparable across images without alignment. "
        "On the synthetic default sample the map has no ground truth and the input is not a photograph, so the run "
        "demonstrates the code path and its outputs, not depth quality; the evaluation report is `not-measurable` "
        "because `abs_rel` cannot be computed without caller-supplied metric depth, and where it is computed on your "
        "own image it is a single-image tutorial figure with no dispersion estimate. The pipeline provides no confidence "
        "map, no calibrated threshold, and no metric conversion; it does not provide metric depth, video or temporal "
        "consistency, stereo/multi-view fusion, normals, 3D reconstruction, or batching. Depth quality on mirrors, "
        "glass, textureless surfaces, night scenes, fog, and non-photographic inputs is expected to degrade and is not "
        "signalled. Run-to-run variability after the deterministic sample comes from floating-point kernel selection "
        "across devices (the model card records `depth_max` 2.969 on CUDA versus 2.970 on CPU for its smoke image) and "
        "from bicubic interpolation; results on fixed hardware are repeatable but not guaranteed bitwise-identical "
        "across devices.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline module, carried in this notebook, "
        "can acquire and digest-verify the pinned model, validate the demonstrated input against the enforced ceilings, "
        "execute the public pipeline path, and emit the shown machine-readable outputs in the tested runtime — without "
        "the repository being reachable. It does **not** establish benchmark superiority, deployment calibration, "
        "accuracy on any real-image domain, safety for high-consequence decisions, or production fitness on an unseen "
        "domain.\n\n"
        "**Troubleshooting.** `RuntimeError: Core dependencies changed while older modules were loaded` in Section 1: "
        "the pinned install replaced a package the runtime had pre-imported — restart the runtime and rerun from the "
        "top. `FileNotFoundError: snapshot file missing` or a `sha256`/`size` `ValueError` in Section 3: a staged file "
        "is incomplete or altered — delete it from the working-directory `weights/depth-anything-v2-small/` and rerun "
        "Section 3. A `ValueError` naming `MIN_IMAGE_SIDE`, `MAX_IMAGE_SIDE` or `MAX_ASPECT_RATIO` in Section 5 or 6: "
        "the BYOD image is outside the ceilings — resize or crop it. An out-of-memory error on a near-ceiling BYOD "
        "image: the card measured 812 MiB peak at 4096 x 1024 on its GPU; use a smaller or squarer image or a CPU "
        "runtime.\n\n"
        "**Next experiments:** upload a real photograph with `USE_BYOD` enabled and inspect whether the near/far "
        "ordering matches the scene; if you hold a metric depth map for it, assign it to `reference_depth` and watch "
        "the report switch to `sample-sanity`, then compare `abs_rel` with `align=True` against `align=False` to see "
        "why affine alignment is required; compare the CUDA and CPU `depth_min`/`depth_max` on the same image to "
        "observe kernel-level variability.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/depth-anything-depth-estimation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/depth-anything-depth-estimation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/depth-anything-depth-estimation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/DepthAnything/Depth-Anything-V2\n"
        "- Depth Anything V2 paper: https://arxiv.org/abs/2406.09414\n"
        "- Depth Anything (V1) paper: https://arxiv.org/abs/2401.10891\n"
        "- Transformers `DepthAnything` documentation: https://huggingface.co/docs/transformers/model_doc/depth_anything"
    ),
}
