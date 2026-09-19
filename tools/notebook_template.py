"""Per-repository template for tools/build_notebook.py (NOTEBOOK_SPEC 2.0 §4 standalone carrier).

Only the task-specific prose and stage cells live here. Runtime install, the embedded pipeline
modules (pipeline.py, metrics.py, samples.py), and the model pin/stage/verify cells are produced by
the generator from repository sources so they cannot drift from the package.

This template configures an E2E supervised-adaptation workflow: the pinned Depth Anything V2 Small snapshot
is digest-verified and loaded, a digest-pinned real RGB-D corpus (40 DIODE validation views with metric
depth) is fetched, validated and split by scan, the inference contract is exercised on a real view with a
metric reference (the first `sample-sanity` evaluation report of this repository), the untouched checkpoint
is scored on the test split beside two trivial priors (the zero-shot policy), the DPT neck and head are
trained on the frozen backbone (the frozen policy) and then the last blocks with them (the unfrozen policy),
the policy ladder is selected on validation, the held-out split is scored, and the adapter is exported and
reloaded.
"""
# ruff: noqa: E501  -- markdown prose and code-cell text are kept on single lines for readable rendering

TEMPLATE = {
    "package": "depth_anything_depth_estimation_pipeline",
    "repo_name": "depth-anything-depth-estimation-pipeline",
    "stem": "depth_anything_depth_estimation",
    "notebook_name": "depth_anything_depth_estimation_colab.ipynb",
    "profile": "E2E",
    "mode": "GUIDED",
    "run_all": (
        "Selecting **Run all** in a fresh supported runtime installs the pinned dependencies, stages and digest-verifies the "
        "pinned Depth Anything V2 Small snapshot (safetensors, 99 MB), fetches 40 digest-pinned DIODE validation views — "
        "RGB, metric depth and validity mask, 312 MB in total — from the Hugging Face Hub mirror (no credential), validates "
        "them and draws 24 / 8 / 8 training, validation and test views by a seeded split of whole scans, runs one test view "
        "through the inference contract with an input manifest, a rejection probe and a `sample-sanity` evaluation report "
        "against its metric reference, scores the untouched checkpoint on the test split beside a constant prior and a "
        "vertical-gradient prior (the **zero-shot policy**), trains the DPT neck and head on the frozen backbone (the "
        "**frozen policy**) and then the last two transformer blocks with them (the **unfrozen policy**), selects the "
        "policy ladder's best epoch by validation AbsRel — where epoch 0 is the untouched checkpoint — scores the held-out "
        "split with the selected model, renders depth maps before and after, exports the trained tensors as safetensors "
        "with a manifest, and reloads that artifact into a fresh pipeline to verify parity. The default path needs no "
        "repository clone, no DIMER worker or service, no credential, no upload dialog and no configuration edit "
        "(NOTEBOOK_SPEC 2.0 §5). On CPU the whole path takes about four minutes of model time after the downloads; a CUDA "
        "runtime is used automatically when present."
    ),
    "byod": (
        "After the tutorial workflow completes, set `USE_BYOD = True` in Section 4 and re-run from that cell to supply your own "
        "RGB-D records as a `.zip` holding `records.csv` (columns `id`, `image`, `depth`, `group`, optional `mask`; `group` — the scan, session or device — must be non-empty on every row, or the loader refuses the set) beside "
        "the image files and `.npy` arrays (depth in metres, mask boolean) — files are decoded from the archive, never "
        "extracted to disk. They pass through the same validation, seeded group-disjoint split, priors, zero-shot scoring, "
        "policy ladder, held-out evaluation, preview, artifact export and reload-parity cells as the DIODE sample. The "
        "expected schema and the ceilings are stated in the Prerequisites and in Section 4, and uploaded files stay inside "
        "this runtime. BYOD is optional and never part of the default path."
    ),
    "pipeline_class": "DepthAnythingPipeline",
    "weights_key": "depth-anything-v2-small",
    "modules": ["pipeline.py", "metrics.py", "samples.py"],
    "entry_module": "pipeline.py",
    "identity_names": {},
    "runtime_imports": ["torch", "transformers"],
    "title": "Depth Anything V2 Small — DIMER E2E supervised adaptation tutorial: zero-shot vs neck/head vs bounded unfreeze on DIODE (standalone)",
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
    "capability": "monocular relative depth estimation from one RGB still image and bounded supervised adaptation — the DPT neck and head on the frozen backbone with an optional unfreeze of the last transformer blocks, selected on validation against the untouched checkpoint — measured by affine-aligned AbsRel and δ1 against metric depth, using the pinned Depth Anything V2 Small weights",
    "intro": (
        "At inference the image is rescaled so its sides are multiples of 14 near 518 px, passed through the DINOv2 "
        "ViT-S/14 backbone and the DPT neck and head, and the raw prediction is interpolated back to the input "
        "resolution: a per-pixel map of **relative inverse depth** — larger is nearer, the scale and shift are unknown, "
        "and the values are **not metres**. The carried pipeline module adds manifest verification, input validation "
        "with named ceilings, the resize back to the input resolution, and the `abs_rel`, `validate_inputs` and "
        "`evaluation_report` helpers; `evaluation_report` becomes `sample-sanity` only when a caller supplies metric "
        "reference depth — which this notebook, unlike its inference-only predecessor, does.\n\n"
        "What this notebook adds to inference is **supervised adaptation under an explicit policy ladder**. The "
        "dataset is real and carries metric ground truth: 40 views from the DIODE validation release (CC BY 4.0) — the "
        "first two views of each of its 20 laser-scanned scenes, 10 indoor and 10 outdoor — pinned per file by byte "
        "size and SHA-256 as served by the Marigold evaluation mirror on the Hugging Face Hub at an immutable commit, "
        "fetched at run time and refused on any mismatch. Views of one scan share a scene, so the sample is split by "
        "**scan**, never by view. The carried `metrics.py` scores a prediction by **AbsRel** and **δ1** after a "
        "per-image least-squares alignment of the prediction to inverse reference depth (the standard protocol for a "
        "relative-depth model, with the far cap at 350 m); a **constant prior** (the image's own median depth) and a "
        "**vertical-gradient prior** (nearer at the bottom) frame the numbers. The **zero-shot policy** is the "
        "checkpoint as published; the **frozen policy** trains the DPT neck and head (2.7 M parameters) on the frozen "
        "backbone under a scale-and-shift-invariant loss; the **unfrozen policy** continues by training the last two "
        "transformer blocks with them; the epoch with the lowest validation AbsRel is kept — epoch 0, the untouched "
        "checkpoint, competes on equal terms, so the honest outcome \"do not adapt\" is available. The adaptation "
        "question is whether 24 DIODE training views buy anything over a model trained on 62 million images. Nothing "
        "here is a quality claim about your images: it is one seeded split of one small corpus."
    ),
    "learning_objectives": (
        "install the pinned runtime; read what the carried pipeline, metrics and dataset modules guarantee; stage and "
        "digest-verify the immutable upstream snapshot; fetch a digest-pinned real RGB-D corpus and validate and split it "
        "by scan without leakage; run one view through the public inference API and read a `sample-sanity` evaluation "
        "report against metric depth; read AbsRel and δ1 beside two trivial priors and understand why a relative-depth "
        "prediction must be aligned before it is scored; run a three-policy adaptation ladder with explicit "
        "hyperparameters and validation-based selection in which the untouched checkpoint can win; evaluate on an "
        "independent scan-disjoint test split; compare depth maps before and after; and export a safetensors adapter "
        "(neck and head plus any trained blocks) that reloads against the pinned base with verified parity."
    ),
    "exclusions": (
        "metric (absolute) depth, video or temporal depth, stereo or multi-view fusion, surface normals, 3D "
        "reconstruction, batch inference, data augmentation, full-backbone or patch-embedding training, and any claim "
        "that 20 DIODE scenes stand in for your images. The repository exposes none of these; an adapted model still "
        "emits relative inverse depth."
    ),
    "prerequisites": [
        "- **Runtime:** a fresh supported runtime (Google Colab or Jupyter, Python 3.12). The default path runs on CPU and uses CUDA automatically when available; float32 on both. The build record measured about 0.4 s per 1024 × 768 view to predict on CPU (3.2 s for the 8-view test split) and about 0.8 s per view per training step on the last two blocks with the neck and head, so the five-epoch ladder over 24 views with six validation passes took about 150 s. The pinned `torch==2.14.0` install and the 99 MB checkpoint are the large downloads of the run, then the 312 MB of DIODE files.",
        "- **Knowledge:** basic Python and NumPy; the difference between inverse depth (larger = nearer, arbitrary scale and shift) and metric distance; why a least-squares alignment is needed before a relative prediction can be scored against metres; what AbsRel and δ1 measure; what validation-based selection between policies means.",
        "- **Data contract:** records are `{{id, image, depth, mask}}` — a PIL image (or a path to one) with sides 14..4,096 px and aspect ratio at most 4.0, a float H × W array of metric depth in metres (or a path to a `.npy`), an optional boolean H × W validity mask (missing means `depth > 0`) covering at least 5 % of the pixels, depth at most 1,000 m, ids matching `[A-Za-z0-9_.:-]{{1,64}}` and unique; a training set needs 4..2,000 records; images are de-duplicated by decoded-pixel digest and split by `scan` / `group` so views of one scene never straddle splits. Only reference pixels inside 0.6..350 m are scored. BYOD accepts a `.zip` (or a directory) holding `records.csv` and the files, and requires a non-empty `group` on every row — the notebook's automatic split is group-disjoint only because the loader refuses ungrouped rows (`load_byod_dataset(..., require_group=False)` is the explicit opt-out, without that guarantee).",
        "- **Validation is structural, not semantic:** nothing checks that a depth map belongs to its image or that its unit is metres beyond the 1,000 m ceiling — a mis-paired or mis-scaled set is trained on without complaint.",
        "- **Privacy:** Do not upload confidential or restricted data to a hosted runtime unless you are authorized to process it there — RGB-D captures of homes, workplaces or people are exactly that. The default path uploads nothing.",
        "- **External access (data):** besides the Hub snapshot, the default path fetches 120 pinned objects — for each of 40 DIODE views the RGB PNG, the `_depth.npy` and the `_depth_mask.npy`, 312,448,846 bytes in total, one SHA-256 each in the carried `SAMPLE_RECORDS` table — from `huggingface.co/datasets/obukhovai/marigold_depth_eval` at commit `30c5b061` over HTTPS, each refused on any byte-size or SHA-256 mismatch before it is decoded; the files are DIODE's own (CC BY 4.0, Vasiljevic et al. 2019) as served by that mirror.",
    ],
    "cells": [
        {
            "md": (
                "## 4. Referenced corpus, validation and scan-level split\n\n"
                "`fetch_corpus` downloads the 120 pinned files (or reads them from the cache), refuses a byte-size or "
                "SHA-256 mismatch per file before it is decoded, and `read_corpus` turns each view into a `{{id, image, "
                "depth, mask}}` record with its domain, scene, scan and source URL. `build_sample_dataset` draws whole "
                "scans per domain — 6 training, 2 validation and 2 test scans from each of the indoor and outdoor sets — "
                "by a seeded shuffle, so the 24 / 8 / 8 views never share a scene across splits; `validate_dataset` then "
                "checks every record against the contract, `check_split_disjoint` asserts no view (by decoded-pixel "
                "digest) and no scan appears in two splits, `scan_summary` reports scans and domains per split, and the "
                "training records table is written to `outputs/{stem}_train.csv` in the shape BYOD expects.\n\n"
                "Look for: 40 views, 20 scans of 2, splits 24 / 8 / 8 with 12 / 4 / 4 scans, three digests, median "
                "depths from about a metre indoors to tens of metres outdoors, and four refusal probes — a duplicate id, "
                "a depth map of the wrong shape, a mask with no valid pixels and a dataset too small to train on — each "
                "rejected before `torch` does anything. About a minute on the first run for the 312 MB download."
            ),
            "code": (
                "import hashlib\n"
                "import io\n"
                "import json\n"
                "import time\n\n"
                'USE_BYOD = False  # @param {{type:"boolean"}}\n'
                'SPLIT_SEED = 42  # @param {{type:"integer"}}\n\n'
                "os.makedirs('outputs', exist_ok=True)\n"
                "t0 = time.perf_counter()\n"
                "if USE_BYOD:\n"
                "    from google.colab import files\n"
                "    uploaded = files.upload()\n"
                "    file_name, payload = next(iter(uploaded.items()))\n"
                "    byod_path = Path('work') / file_name\n"
                "    byod_path.parent.mkdir(parents=True, exist_ok=True)\n"
                "    byod_path.write_bytes(payload)\n"
                "    records = load_byod_dataset(byod_path)\n"
                "    splits = split_dataset(records, seed=SPLIT_SEED)\n"
                "    data_source = 'BYOD (' + file_name + ')'\n"
                "    raw_count = {{'byod': len(records)}}\n"
                "else:\n"
                "    corpus = read_corpus(fetch_corpus(cache_dir='weights/diode-sample'))\n"
                "    raw_count = {{'views': len(corpus), 'scans': len({{(r['domain'], r['scan']) for r in corpus}}), 'domains': sorted({{r['domain'] for r in corpus}})}}\n"
                "    splits = build_sample_dataset(corpus, seed=SPLIT_SEED)\n"
                "    data_source = f'{{CORPUS_NAME}} ({{CORPUS_RELEASE}}; {{CORPUS_LICENSE}})'\n"
                "fetch_seconds = round(time.perf_counter() - t0, 1)\n"
                "train_records, val_records, test_records = splits['train'], splits['validation'], splits['test']\n"
                "dataset_manifests = {{name: validate_dataset(part, min_records=1) for name, part in splits.items()}}\n"
                "disjoint = check_split_disjoint(splits)\n"
                "summary = scan_summary(splits)\n"
                "write_dataset_csv(train_records, 'outputs/{stem}_train.csv')\n"
                "print({{'data_source': data_source, 'raw': raw_count, 'splits': disjoint, 'scan_summary': summary, 'fetch_seconds': fetch_seconds, 'corpus_bytes': CORPUS_BYTES}})\n"
                "for name, manifest in dataset_manifests.items():\n"
                "    print({{name: {{'n': manifest['n_records'], 'scans': manifest['scans'], 'domains': manifest['domain_counts'], 'image_side': manifest['image_side'], 'valid_fraction': manifest['valid_fraction'], 'median_depth_m': manifest['median_depth_m'], 'digest': manifest['digest'][:16] + '...'}}}})\n"
                "example = train_records[0]\n"
                "print({{'example': {{k: example[k] for k in ('id', 'domain', 'scene', 'scan', 'source_url') if k in example}}, 'size': example['image'].size, 'depth_dtype': str(example['depth'].dtype), 'valid_fraction': round(float(example['mask'].mean()), 4)}})\n\n"
                "probes = {{\n"
                "    'duplicate id': [{{**r, 'id': 'same'}} for r in train_records[:4]],\n"
                "    'depth shape': [{{**train_records[0], 'depth': train_records[0]['depth'][:-1]}}, *train_records[1:4]],\n"
                "    'no valid depth': [{{**train_records[0], 'mask': np.zeros_like(train_records[0]['mask'])}}, *train_records[1:4]],\n"
                "    'too small': train_records[:3],\n"
                "}}\n"
                "for name, probe in probes.items():\n"
                "    try:\n"
                "        validate_dataset(probe)\n"
                "        print({{'probe': name, 'verdict': 'accepted'}})\n"
                "    except (TypeError, ValueError) as exc:\n"
                "        print({{'probe': name, 'rejected': str(exc)[:110]}})"
            ),
        },
        {
            "md": (
                "## 5. Predict through the inference contract, with a metric reference\n\n"
                "Before any adaptation, the inference contract is exercised as it always was, on one test view. "
                "`validate_inputs` applies exactly the checks `predict` applies — type, sides 14..`MAX_IMAGE_SIDE` px, "
                "aspect ratio at most `MAX_ASPECT_RATIO` — and returns an input manifest; a deliberately over-wide image "
                "is validated too and its rejection recorded as a finding. `predict` returns relative inverse depth at "
                "the input resolution with four sanity checks. Because this view carries metric depth from a laser "
                "scanner, `evaluation_report` can for the first time return **`sample-sanity`**: it aligns the "
                "prediction to the reference by least squares in inverse depth (the pipeline's own `abs_rel`, invalid "
                "pixels zeroed out of the reference) and reports AbsRel for this one image — sanity evidence, not a "
                "benchmark. The build record's probe view — an outdoor scan — scored 0.12; indoor views scored 0.04–0.13 "
                "and outdoor views with sky and distant structure up to 0.40. A side-by-side preview (image | predicted inverse depth) is displayed."
            ),
            "code": (
                "probe_record = test_records[0]\n"
                "image = probe_record['image']\n"
                "print({{'ceilings': {{'MIN_IMAGE_SIDE': MIN_IMAGE_SIDE, 'MAX_IMAGE_SIDE': MAX_IMAGE_SIDE, 'MAX_ASPECT_RATIO': MAX_ASPECT_RATIO}}, 'contract': {{'DEPTH_KIND': DEPTH_KIND, 'TRANSFORMER_BLOCKS': TRANSFORMER_BLOCKS, 'PARAMETER_COUNT': PARAMETER_COUNT, 'EVAL_DEPTH_RANGE_M': EVAL_DEPTH_RANGE_M}}}})\n"
                "input_manifest = validate_inputs(image, names=[probe_record['id']])\n"
                "try:\n"
                "    validate_inputs(Image.new('RGB', (MIN_IMAGE_SIDE, int(MIN_IMAGE_SIDE * (MAX_ASPECT_RATIO + 1)))))\n"
                "except ValueError as exc:\n"
                "    input_manifest['findings'].append({{'input': 'over-wide-probe', 'verdict': 'rejected', 'message': str(exc)}})\n"
                "with open('outputs/{stem}_input_manifest.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(input_manifest, handle, indent=2, ensure_ascii=False)\n"
                "started = time.perf_counter()\n"
                "result = pipe.predict(image)\n"
                "predict_seconds = round(time.perf_counter() - started, 3)\n"
                "depth = result['depth']\n"
                "checks = {{\n"
                "    'shape_matches_input': depth.shape == (result['height'], result['width']) == (image.height, image.width),\n"
                "    'dtype_float32': depth.dtype == np.float32,\n"
                "    'all_finite': bool(np.isfinite(depth).all()),\n"
                "    'non_degenerate_range': result['depth_max'] > result['depth_min'],\n"
                "}}\n"
                "if not all(checks.values()):\n"
                "    raise RuntimeError(f'depth map failed a sanity check: {{checks}}')\n"
                "reference_depth = np.where(probe_record['mask'], probe_record['depth'], 0.0)\n"
                "report = evaluation_report(result, reference_depth, sample_kind=f\"DIODE {{probe_record.get('domain', '')}} view {{probe_record['id']}}\" if not USE_BYOD else 'BYOD test view')\n"
                "print({{'probe_id': probe_record['id'], 'domain': probe_record.get('domain'), 'seconds': predict_seconds, 'device': pipe.device, 'checks': checks, 'findings': len(input_manifest['findings'])}})\n"
                "print({{'verdict': report['verdict'], 'metrics': report['metrics'], 'reason': report['reason']}})\n"
                "assert report['verdict'] == 'sample-sanity' and report['metrics'][0]['id'] == 'abs_rel'\n\n"
                "def preview(image, maps):\n"
                "    tiles = [image.convert('RGB')]\n"
                "    for m in maps:\n"
                "        lo, hi = float(m.min()), float(m.max())\n"
                "        tiles.append(Image.fromarray(np.round((m - lo) / (hi - lo + 1e-9) * 255.0).astype(np.uint8)).convert('RGB'))\n"
                "    canvas = Image.new('RGB', (image.width * len(tiles), image.height))\n"
                "    for i, tile in enumerate(tiles):\n"
                "        canvas.paste(tile, (image.width * i, 0))\n"
                "    return canvas\n\n"
                "try:\n"
                "    from IPython.display import display\n"
                "    display(preview(image, [depth]).reduce(2))\n"
                "except ImportError:\n"
                "    print({{'preview': 'IPython display unavailable; the preview PNG is written in Section 9'}})"
            ),
        },
        {
            "md": (
                "## 6. The two priors and the zero-shot policy\n\n"
                "Three numbers frame the adaptation, all on the 8 test views. The **constant prior** puts every pixel at "
                "the image's own median reference depth — an oracle constant, the strongest flat guess. The "
                "**vertical-gradient prior** predicts inverse depth rising from the top row to the bottom row and is "
                "aligned exactly like a model prediction: what \"the ground is nearer\" alone buys. The **zero-shot "
                "policy** is `pipe.evaluate` on the untouched checkpoint: mean AbsRel and δ1 over views, per domain and "
                "per view. The build record saw the constant prior at AbsRel 0.33, the gradient prior at 0.26 and the "
                "zero-shot checkpoint at 0.18 (0.08 indoors, 0.28 outdoors) — read the per-domain split; outdoor DIODE "
                "views with sky, foliage and 100 m ranges are where a relative-depth model struggles. Two zero-shot depth "
                "maps are kept for the before/after comparison in Section 9. About ten seconds on CPU."
            ),
            "code": (
                "def brief(m):\n"
                "    return {{'abs_rel': round(m['abs_rel'], 4), 'delta1': round(m['delta1'], 4), 'n': m['n']}}\n\n"
                "priors = prior_baselines(test_records)\n"
                "print({{'constant_prior': brief(priors['constant_prior']), 'baseline': priors['constant_prior']['baseline']}})\n"
                "print({{'vertical_gradient_prior': brief(priors['vertical_gradient_prior']), 'baseline': priors['vertical_gradient_prior']['baseline']}})\n"
                "t0 = time.perf_counter()\n"
                "zero_shot_test = pipe.evaluate(test_records)\n"
                "print({{'zero_shot_policy': zero_shot_test['policy'], 'test': brief(zero_shot_test), 'per_domain': {{d: {{'abs_rel': round(v['abs_rel'], 4), 'delta1': round(v['delta1'], 4), 'n': v['n']}} for d, v in zero_shot_test['per_domain'].items()}}, 'verdict': zero_shot_test['verdict'], 'seconds': round(time.perf_counter() - t0, 1)}})\n"
                "print({{'per_view': [(r['id'], r['domain'], round(r['abs_rel'], 3), round(r['delta1'], 3)) for r in zero_shot_test['per_image']]}})\n"
                "print({{'definitions': zero_shot_test['definitions']}})\n"
                "show = test_records[:2]\n"
                "zero_shot_maps = {{r['id']: pipe.predict(r['image'])['depth'] for r in show}}\n"
                "assert zero_shot_test['abs_rel'] < priors['constant_prior']['abs_rel'] and zero_shot_test['abs_rel'] < priors['vertical_gradient_prior']['abs_rel']"
            ),
        },
        {
            "md": (
                "## 7. The policy ladder: neck and head, then a bounded unfreeze, selected against the checkpoint\n\n"
                "`pipe.adapt` scores the untouched checkpoint on the validation views first (epoch 0, the zero-shot "
                "policy), then trains the DPT neck and head — 2,728,513 parameters — on the frozen backbone for "
                "`HEAD_EPOCHS` epochs (the frozen policy), then unfreezes the last `TRAINABLE_BLOCKS` transformer blocks "
                "— two by default, 3,550,464 of 24,785,089 parameters; the patch embedding, the position embedding, the "
                "earlier blocks and the final norm stay frozen — and trains them with the neck and head for `EPOCHS` more "
                "epochs (the unfrozen policy). Every epoch trains one view at a time with AdamW at `LEARNING_RATE` "
                "(weight decay 0.01, gradient clipping 1.0, seeded order, no augmentation) under a scale-and-shift-"
                "invariant squared error on inverse depth — the prediction is affinely aligned to the reference in closed "
                "form before the residual is taken, because the model's scale and shift are free — and is scored on "
                "validation by `evaluate`. The epoch with the **lowest validation AbsRel** is kept and its tensors "
                "restored, so the selected policy can be the checkpoint itself.\n\n"
                "Watch the validation AbsRel: the build record's ladder at 1e-5 went 0.243 (zero-shot) → 0.220 → 0.194 "
                "(neck and head) → 0.139 → 0.133 → 0.130 (unfrozen, still falling at epoch 5, which was selected); at "
                "3e-5 the neck-and-head epoch 2 was selected instead, and at 1e-4 the selected unfreeze scored worse than "
                "the checkpoint on the held-out split. The validation views are four scans — one view is an eighth of "
                "the number."
            ),
            "code": (
                'HEAD_EPOCHS = 2  # @param {{type:"integer"}}\n'
                'EPOCHS = 3  # @param {{type:"integer"}}\n'
                'LEARNING_RATE = 1e-5  # @param {{type:"number"}}\n'
                'TRAINABLE_BLOCKS = 2  # @param {{type:"integer"}}\n\n'
                "def report_epoch(entry):\n"
                "    row = {{'epoch': entry['epoch'], 'stage': entry['stage'], 'train_loss': round(entry['train_loss'], 4) if entry['train_loss'] is not None else None}}\n"
                "    if entry.get('val'):\n"
                "        row['val_abs_rel'] = round(entry['val']['abs_rel'], 4)\n"
                "        row['val_delta1'] = round(entry['val']['delta1'], 4)\n"
                "    print(row)\n\n"
                "t0 = time.perf_counter()\n"
                "adapt_result = pipe.adapt(train_records, val_records, trainable_blocks=TRAINABLE_BLOCKS, head_epochs=HEAD_EPOCHS, epochs=EPOCHS, lr=LEARNING_RATE, progress=report_epoch)\n"
                "adapt_seconds = round(time.perf_counter() - t0, 1)\n"
                "report_epoch(adapt_result['history'][0])\n"
                "print({{'selected_policy': adapt_result['policy'], 'best_epoch': adapt_result['best_epoch'], 'selection': adapt_result['selection'], 'trainable_neck_head': adapt_result['n_trainable_head'], 'trainable_blocks': adapt_result['n_trainable_blocks'], 'total_parameters': adapt_result['n_total'], 'seconds': adapt_seconds}})"
            ),
        },
        {
            "md": (
                "## 8. Held-out evaluation\n\n"
                "The test split was never used for training or policy selection, and no scan in it appears in the "
                "training or validation splits. The selected model is scored exactly as the checkpoint was in Section 6, "
                "and the four rows are put side by side: constant prior, vertical-gradient prior, zero-shot policy, "
                "selected policy — with per-domain and per-view AbsRel. Read the policy first: if validation kept "
                "epoch 0, the last two rows are the same model; otherwise the delta is what the adaptation bought on "
                "8 views. The build record saw AbsRel 0.180 → 0.167 and δ1 0.764 → 0.765 at the default settings — a "
                "small gain concentrated outdoors, within the grain of an 8-view split. The cell asserts the selected "
                "model beats both priors; it does **not** assert a gain over the checkpoint, because that is the "
                "question, not the answer. Eight views from four scans of one seeded split give no dispersion estimate."
            ),
            "code": (
                "adapted_test = pipe.evaluate(test_records)\n"
                "adapted_val = pipe.evaluate(val_records)\n"
                "comparison = {{\n"
                "    metric: {{'constant_prior': round(priors['constant_prior'][metric], 4), 'vertical_gradient_prior': round(priors['vertical_gradient_prior'][metric], 4), 'zero_shot': round(zero_shot_test[metric], 4), 'selected_policy': round(adapted_test[metric], 4)}}\n"
                "    for metric in ('abs_rel', 'delta1')\n"
                "}}\n"
                "comparison['per_domain_abs_rel'] = {{d: {{'zero_shot': round(zero_shot_test['per_domain'][d]['abs_rel'], 4), 'selected': round(v['abs_rel'], 4)}} for d, v in adapted_test['per_domain'].items()}}\n"
                "comparison['per_view_abs_rel'] = {{z['id']: {{'zero_shot': round(z['abs_rel'], 4), 'selected': round(a['abs_rel'], 4)}} for z, a in zip(zero_shot_test['per_image'], adapted_test['per_image'])}}\n"
                "comparison['delta_vs_zero_shot'] = {{metric: round(adapted_test[metric] - zero_shot_test[metric], 4) for metric in ('abs_rel', 'delta1')}}\n"
                "comparison['selected_policy'] = adapt_result['policy']\n"
                "for metric, row in comparison.items():\n"
                "    print({{metric: row}})\n"
                "evaluation_report_payload = {{\n"
                "    'model': {{'id': MODEL_ID, 'revision': MODEL_REVISION, 'key': MODEL_KEY}},\n"
                "    'data_source': data_source,\n"
                "    'dataset_digests': {{name: manifest['digest'] for name, manifest in dataset_manifests.items()}},\n"
                "    'splits': disjoint,\n"
                "    'scan_summary': summary,\n"
                "    'single_view_report': report,\n"
                "    'priors': {{k: {{kk: vv for kk, vv in v.items() if kk != 'per_image'}} for k, v in priors.items()}},\n"
                "    'zero_shot_test': zero_shot_test,\n"
                "    'validation_metrics': adapted_val,\n"
                "    'test_metrics': adapted_test,\n"
                "    'comparison': comparison,\n"
                "    'adaptation': {{k: v for k, v in adapt_result.items() if k not in ('history', 'trainable_names')}},\n"
                "    'history': adapt_result['history'],\n"
                "    'adaptation_seconds': adapt_seconds,\n"
                "}}\n"
                "with open('outputs/{stem}_evaluation_report.json', 'w', encoding='utf-8') as f:\n"
                "    json.dump(evaluation_report_payload, f, indent=2, ensure_ascii=False)\n"
                "assert adapted_test['abs_rel'] < priors['constant_prior']['abs_rel'] and adapted_test['abs_rel'] < priors['vertical_gradient_prior']['abs_rel']\n"
                "print({{'report': 'outputs/{stem}_evaluation_report.json'}})"
            ),
        },
        {
            "md": (
                "## 9. Depth maps before and after, export the adapter and reload it\n\n"
                "Two test views are predicted by the selected model and rendered beside the zero-shot maps kept in "
                "Section 6 (image | zero-shot | selected, normalised per map, brighter = nearer) with each map's aligned "
                "AbsRel; `outputs/{stem}_preview.png` holds the render. A relative-depth map is a per-image ordering, so "
                "read the pair for structure — where the selected model places the ground plane, the walls, the far "
                "field — not for absolute brightness.\n\n"
                "`pipe.save_artifact` writes the DPT neck and head tensors and, when the unfrozen policy was selected, "
                "the trained block tensors — 10.9 MB for neck and head, 25.1 MB with two blocks — as "
                "`adapter.safetensors`, with a `manifest.json` recording the artifact format, the base model id and "
                "revision, the digest of the base `model.safetensors`, the selected policy, the tensor names, the file "
                "size and SHA-256, the training configuration and the epoch history (OUT8). "
                "`DepthAnythingPipeline.from_artifact` re-verifies the base snapshot, checks the artifact manifest and "
                "digest **before** deserialising, refuses any tensor that is not a neck, head or transformer-block tensor "
                "of the base, and overlays the tensors onto a freshly loaded base — a new object from files, not the "
                "in-memory model (VER2). The cell asserts an identical depth map on a test view and an identical test "
                "AbsRel (VER4)."
            ),
            "code": (
                "import shutil\n\n"
                "rows = []\n"
                "canvases = []\n"
                "for record in show:\n"
                "    after = pipe.predict(record['image'])['depth']\n"
                "    before = zero_shot_maps[record['id']]\n"
                "    rows.append({{'id': record['id'], 'domain': record.get('domain', ''), 'zero_shot_abs_rel': round(aligned_abs_rel(before, record['depth'], record['mask']), 4), 'selected_abs_rel': round(aligned_abs_rel(after, record['depth'], record['mask']), 4), 'source_url': record.get('source_url', '')}})\n"
                "    print(rows[-1])\n"
                "    canvases.append(preview(record['image'], [before, after]))\n"
                "sheet = Image.new('RGB', (max(c.width for c in canvases), sum(c.height for c in canvases)))\n"
                "y = 0\n"
                "for c in canvases:\n"
                "    sheet.paste(c, (0, y))\n"
                "    y += c.height\n"
                "sheet.save('outputs/{stem}_preview.png')\n"
                "try:\n"
                "    from IPython.display import display\n"
                "    display(sheet.reduce(3))\n"
                "except ImportError:\n"
                "    pass\n\n"
                "artifact_dir = Path('outputs/{stem}_adapter')\n"
                "shutil.rmtree(artifact_dir, ignore_errors=True)\n"
                "pipe.save_artifact(artifact_dir, metadata={{'tutorial': '{stem}', 'data_source': data_source}})\n"
                "artifact_manifest = json.loads((artifact_dir / 'manifest.json').read_text(encoding='utf-8'))\n"
                "print({{'artifact': str(artifact_dir), 'format': artifact_manifest['format'], 'policy': artifact_manifest['adapter']['policy'], 'tensors': len(artifact_manifest['tensors']), 'bytes': artifact_manifest['files'][0]['bytes'], 'sha256': artifact_manifest['files'][0]['sha256'][:16] + '...'}})\n\n"
                "reloaded = DepthAnythingPipeline.from_artifact(artifact_dir, weights_dir=WEIGHTS_DIR, device=pipe.device)\n"
                "reloaded_map = reloaded.predict(show[0]['image'])['depth']\n"
                "reloaded_test = reloaded.evaluate(test_records)\n"
                "parity = {{'depth_map_identical': bool(np.array_equal(reloaded_map, pipe.predict(show[0]['image'])['depth'])), 'abs_rel_in_memory': round(adapted_test['abs_rel'], 6), 'abs_rel_reloaded': round(reloaded_test['abs_rel'], 6)}}\n"
                "print({{'reload_parity': parity, 'reloaded_policy': reloaded.adapter['policy'], 'reloaded_best_epoch': reloaded.adapter['best_epoch']}})\n"
                "assert parity['depth_map_identical'] and abs(adapted_test['abs_rel'] - reloaded_test['abs_rel']) < 1e-9\n\n"
                "weight_entry = next(entry for entry in MANIFEST['files'] if entry['path'] == WEIGHTS_FILE)\n"
                "result_payload = {{\n"
                "    'notebook_source': NOTEBOOK_SOURCE,\n"
                "    'repository_revision': NOTEBOOK_SOURCE['repository_revision'],\n"
                "    'model_id': MODEL_ID,\n"
                "    'model_revision': MODEL_REVISION,\n"
                "    'model_license': MODEL_LICENSE,\n"
                "    'snapshot': {{'path': str(WEIGHTS_DIR), 'files': snapshot['files'], 'total_bytes': snapshot.get('total_bytes'), 'fetched_this_run': fetched, 'weight_file': WEIGHTS_FILE, 'weight_format': 'safetensors, digest-verified', 'weight_sha256': weight_entry['sha256']}},\n"
                "    'data_source': data_source,\n"
                "    'corpus': {{'name': CORPUS_NAME, 'release': CORPUS_RELEASE, 'base_url': CORPUS_BASE_URL, 'commit': CORPUS_COMMIT, 'views': len(SAMPLE_RECORDS), 'bytes': CORPUS_BYTES, 'license': CORPUS_LICENSE}},\n"
                "    'inference_contract': {{'input_manifest': input_manifest, 'sanity_checks': checks, 'probe_id': probe_record['id'], 'single_view_report': report, 'seconds': predict_seconds}},\n"
                "    'comparison': comparison,\n"
                "    'before_after': rows,\n"
                "    'preview_file': 'outputs/{stem}_preview.png',\n"
                "    'artifact': {{'dir': str(artifact_dir), 'sha256': artifact_manifest['files'][0]['sha256'], 'bytes': artifact_manifest['files'][0]['bytes'], 'tensors': len(artifact_manifest['tensors']), 'policy': artifact_manifest['adapter']['policy']}},\n"
                "    'reload_parity': parity,\n"
                "    'runtime': {{'python': platform.python_version(), 'torch': torch.__version__, 'transformers': transformers.__version__, 'device': pipe.device, 'dtype': 'float32', 'source': pipe.source}},\n"
                "}}\n"
                "with open('outputs/{stem}_result.json', 'w', encoding='utf-8') as handle:\n"
                "    json.dump(result_payload, handle, indent=2, ensure_ascii=False)\n"
                "print(sorted(os.listdir('outputs')))"
            ),
        },
    ],
    "closing": (
        "## Interpretation and limits\n\n"
        "The untouched Depth Anything V2 Small checkpoint already beats both priors by a wide margin on eight DIODE "
        "views (AbsRel 0.18 against 0.26 and 0.33), and a bounded adaptation on 24 training views — neck and head "
        "first, then the last two blocks, selected against the checkpoint by validation AbsRel — was selected at its "
        "last epoch and lowered held-out AbsRel from 0.180 to 0.167 in the build record, almost all of it outdoors. "
        "That is the claim and the finding: the adaptation contract runs a three-policy ladder end to end on a real "
        "RGB-D corpus, lets the checkpoint win when it should, and reports the answer against two priors rather than in "
        "isolation.\n\n"
        "The test split is eight views from four scans of one seeded split of one small corpus with no dispersion "
        "estimate — one view is an eighth of every mean, and two views of one scan are nearly the same scene. AbsRel "
        "and δ1 are computed after a per-image affine alignment inside 0.6..350 m, so they say how well the *ordering "
        "and relative spacing* of depth match the laser reference, not whether any metre is right; the model's output "
        "stays relative after adaptation. A gain of 0.013 AbsRel at 1e-5 turned into a loss at 1e-4 in the build "
        "sweep, and the neck-and-head-only policy won at 3e-5: the ladder's outcome is a property of this corpus and "
        "these hyperparameters, not of the method.\n\n"
        "Three things to carry to real data. **Priors first:** the constant and gradient priors on *your* depth maps "
        "are the numbers to read before any model's — a scene that a vertical gradient explains is not testing the "
        "model. **Leakage:** split by capture session, scan or device (the contract splits by `scan` / `group`, never by "
        "view). **Policy:** the untouched checkpoint is a policy too; keep it in the ladder and let validation decide, "
        "and keep the learning rate small — a 25 M-parameter model trained on 62 million images has little to learn "
        "from 24 views except your sensor's habits.\n\n"
        "Successful execution proves that the recorded repository revision's pipeline modules, carried in this standalone "
        "notebook, can acquire and digest-verify the pinned model snapshot, fetch and digest-verify a real RGB-D corpus, "
        "validate the demonstrated dataset contract without leakage, execute the inference contract with a "
        "`sample-sanity` metric against a laser reference, run the zero-shot, frozen and unfrozen policies with "
        "validation-based selection, evaluate by aligned AbsRel and δ1 against two priors on an independent split, and "
        "emit the shown machine-readable artifacts — without the repository being reachable. It does **not** establish "
        "benchmark superiority, metric accuracy, a usable acceptance threshold, or production fitness.\n\n"
        "**Optional experiments (they do not affect the default path):** set `LEARNING_RATE = 3e-5` and watch the "
        "neck-and-head policy win on validation (the build record: epoch 2, held-out 0.175); set `LEARNING_RATE = 1e-4` "
        "and read what a hot unfreeze does to the held-out split (selected on validation, 0.186 on test — worse than "
        "the checkpoint); set `TRAINABLE_BLOCKS = 4` (the build record: neck and head selected, 0.175); set `EPOCHS = "
        "6` and watch whether validation AbsRel keeps falling; or bring your own RGB-D records through BYOD and read "
        "the priors before either policy.\n\n"
        "## References\n\n"
        "- Repository README: https://github.com/kurtvalcorza/depth-anything-depth-estimation-pipeline/blob/main/README.md\n"
        "- Repository model card: https://github.com/kurtvalcorza/depth-anything-depth-estimation-pipeline/blob/main/MODEL_CARD.md\n"
        "- Weight provenance: https://github.com/kurtvalcorza/depth-anything-depth-estimation-pipeline/blob/main/docs/WEIGHTS.md\n"
        "- Upstream model: https://huggingface.co/{MODEL_ID}\n"
        "- Upstream code: https://github.com/DepthAnything/Depth-Anything-V2\n"
        "- Depth Anything V2 paper (Yang et al., 2024): https://arxiv.org/abs/2406.09414\n"
        "- DIODE: A Dense Indoor and Outdoor DEpth Dataset (Vasiljevic et al., 2019; CC BY 4.0): https://diode-dataset.org — https://arxiv.org/abs/1908.00463\n"
        "- Marigold evaluation mirror serving the pinned DIODE files: https://huggingface.co/datasets/obukhovai/marigold_depth_eval\n"
        "- Towards Robust Monocular Depth Estimation (MiDaS; the scale-and-shift-invariant loss and aligned evaluation, Ranftl et al., 2020): https://arxiv.org/abs/1907.01341\n"
        "- Transformers `DepthAnything` documentation: https://huggingface.co/docs/transformers/model_doc/depth_anything\n"
        "- DIMER Notebook Specification 2.0 and Model Card Specification 1.1 (fleet specs in the ml-worker repository)"
    ),
}
