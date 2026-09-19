# Release verification

`tutorials/depth_anything_depth_estimation_colab.ipynb` (`E2E`, **standalone** carrier) is a **release candidate** until
the exact notebook revision has executed top-to-bottom in a clean supported runtime. Unit tests, JSON validation,
code-cell compilation, the generator parity checks and `tools/validate_release_assets.py` are necessary checks but
are **not** runtime evidence under DIMER Notebook Specification 2.0 (REL8). This file is the durable release-gate
record for the notebook.

## Automatic coverage (static, every pull request)

CI runs `tools/validate_release_assets.py`, which checks:

- notebook JSON parses; every code cell compiles as plain Python (no `%`/`!` magics); no persisted outputs or
  execution counts; no unresolved placeholder markers; every code cell is preceded by an explanatory markdown cell;
- exactly one tutorial notebook, named in `tutorials/README.md` with its `E2E` profile, the notebook-spec version
  and the standalone carrier; `metadata.dimer` declares that profile, spec `2.0`, a §3.3 pedagogical mode,
  `standalone: true` and `generated_from` (repository, revision, module SHA-256, generator);
- the standalone carrier (ST1–ST8, PAR1–PAR4): no clone, repository install or repository import on the primary
  path; one cell per carried module (`pipeline.py`, `metrics.py`, `samples.py`), each equal to its source after the
  generator's documented rewrites; the inline `MANIFEST` equal to the committed 3-entry snapshot manifest and the
  inline `PINS` equal to the `pyproject.toml` runtime pins; the notebook byte-identical (on LF) to
  `tools/build_notebook.py` output for its recorded revision; the pinned-install cell with its
  restart-on-stale-import guard; `NOTEBOOK_SOURCE` recorded in exports;
- `MODEL_ID`/`MODEL_REVISION` bound only in the carried module cell (and repeated in the inline manifest, which the
  notebook asserts against the module before fetching), the revision a 40-hex immutable commit, and the same
  identity string in `README.md`, `MODEL_CARD.md` and `docs/WEIGHTS.md` with no stray revisions (the 120 DIODE file
  digests live in the carried `samples.py`, not in prose);
- the profile-specific public-API calls (`stage_missing_files`, `verify_snapshot`,
  `DepthAnythingPipeline.from_pretrained(weights_dir=...)`, `fetch_corpus` from the pinned cache path,
  `read_corpus` + `build_sample_dataset(seed=SPLIT_SEED)` / `load_byod_dataset`, `validate_dataset` per split,
  `class_names`, `check_split_disjoint`, `observer_overlap`, `write_dataset_csv`, `validate_inputs` with the
  over-wide-image refusal probe, `pipe.predict` with the sanity checks and the `sample-sanity` `evaluation_report`, `prior_baselines`,
  `pipe.evaluate` on the untouched checkpoint with the two-prior assertion, `pipe.adapt` with
  `trainable_blocks=TRAINABLE_BLOCKS`, `head_epochs=HEAD_EPOCHS` and `lr=LEARNING_RATE`, `pipe.evaluate` on the
  validation and test splits after the ladder with the two-prior assertion, `aligned_abs_rel` on the before/after
  maps, the preview PNG, `pipe.save_artifact`, `DepthAnythingPipeline.from_artifact` and the reload-parity
  assertion, and the provenance fields `weight_format`, `weight_sha256` and the `corpus` block), the
  six expected `outputs/` paths, the learner-facing statements (relative inverse depth, not metres, supervised
  adaptation under an explicit policy ladder, the constant and vertical-gradient priors, the zero-shot, frozen and
  unfrozen policies, lowest validation AbsRel, the `sample-sanity` report, no dispersion estimate, named
  exclusions, the CC BY 4.0 licence) and the gated-off BYOD default; forbidden patterns (credential-in-URL, any `git
  clone` / `github.com` / repository import on the primary path, a mutable `revision='main'`, direct
  `from transformers import` / `from huggingface_hub import` / `urllib.request` / `safetensors` /
  `torch.optim` / `.backward(` / `pipe._model` / `pipe._processor` / `predicted_depth` / `np.linalg.lstsq(` /
  `AutoModelForDepthEstimation` / `AutoImageProcessor` use **outside the carried
  module cells**, `trust_remote_code=True`, `pickle.load`, `torch.load(` without `weights_only=True`,
  `extractall(`);
- `STATUS.md`, `README.md` and `tutorials/README.md` agree on one release-status token and no document makes an
  unsupported release-grade, production-readiness or benchmark claim;
- `MODEL_CARD.md` front matter (`model_card_spec: "1.1"`), single H1, the 19 required headings in order, and the
  immutable provenance section.

CI installs `numpy`, `pillow`, `pytest` and `ruff` only (no torch or transformers), runs `ruff check src tests tools`,
`tools/build_notebook.py --check`, and the offline unit suite (`tests/test_pipeline.py`, `tests/test_adaptation.py`,
`tests/test_role_helpers.py`, `tests/test_import_boundary.py`, `tests/test_notebook_parity.py`; injected
vertical-gradient runner and corpus fetcher, synthetic RGB / depth / mask scenes, temporary manifests, no weights —
`tests/test_model_backed.py` is skipped without `transformers` and the snapshot). These are source/provenance and
unit checks. They are **not** execution evidence.

## Executor paths

| Path | Runtime | Role |
|---|---|---|
| Google Colab (supported user path) | Colab CPU runtime (CUDA used automatically when present; float32 either way) | The runtime the tutorial is written for; a clean top-to-bottom run here is promotion evidence |
| Kaggle CLI kernel or equivalent fresh container | Fresh CPU or GPU container, Python 3.12 image; the committed notebook executed verbatim in a fresh interpreter with a `google.colab` shim and **no repository checkout** (the notebook is standalone) | Reproducible clean-room executor of the same class; needed whenever the hosted kernel pre-imports a NumPy, Pillow or torch that differs from the `pyproject.toml` pins, because the tutorial's fail-closed stale-import guard correctly halts the in-kernel path after the pinned install |
| Local harness (pre-flight only) | Workstation, sequential cell executor with a `google.colab` shim, pre-staged pins, `CUDA_VISIBLE_DEVICES=-1` | Builder pre-flight to catch defects before spending cloud runs; **not** a supported runtime and **not** promotion evidence |

## Supported release verification procedure

Before changing the registry status from `Candidate` to `Release-grade`:

1. resolve the exact PR/commit head under review and confirm static CI is green;
2. open that exact notebook revision in a new CPU or CUDA runtime (Colab, or a fresh-container executor above) with
   **no repository checkout**, an empty Hugging Face cache, and no pre-staged files under the working-directory
   snapshot `weights/depth-anything-v2-small/` or the corpus cache `weights/diode-sample/` (the standalone path writes the
   manifest itself, stages the missing file from the Hub, and fetches the 120 pinned DIODE files from the
   Hugging Face Hub mirror, so neither directory may be seeded);
3. run the notebook top-to-bottom without editing implementation cells (form parameters at their defaults:
   `USE_BYOD = False`, `SPLIT_SEED = 42`, `HEAD_EPOCHS = 2`, `EPOCHS = 3`, `LEARNING_RATE = 1e-5`,
   `TRAINABLE_BLOCKS = 2`);
4. verify that Section 1 reports `NOTEBOOK_SOURCE.repository_revision` equal to the revision recorded in
   `metadata.dimer.generated_from` and that the installed core package versions equal the inline `PINS`
   (= `pyproject.toml`): `torch==2.14.0`, `torchvision==0.29.0`, `transformers==4.57.6`, `huggingface-hub==0.36.2`,
   `safetensors==0.8.0`, `numpy==2.5.3`, `pillow==11.3.0` (an interpreter restart after the install is expected
   where the runtime's preinstalled torch, numpy or Pillow differ from the pins);
5. verify every default-path stage completes:
   - pinned runtime installed from the inline `PINS` with no GitHub access;
   - the three carried module cells execute (defining `DepthAnythingPipeline`, `verify_snapshot`,
     `stage_missing_files`, `validate_inputs`, `evaluation_report`, `abs_rel`, `align_inverse_depth`,
     `aligned_abs_rel`, `depth_metrics`, `prior_baselines`, `SAMPLE_RECORDS`, `fetch_corpus`, `read_corpus`,
     `build_sample_dataset`, `validate_dataset`, `check_split_disjoint`, `scan_summary`, `split_dataset`,
     `load_byod_dataset`, `write_dataset_csv` and the ceilings) with no import of the repository package;
   - the inline manifest asserted against the module's constants, then `stage_missing_files(WEIGHTS_DIR,
     allow_download=True)` reporting `['model.safetensors']` (and any other absent entry) fetched from
     `depth-anything/Depth-Anything-V2-Small-hf` at the immutable revision, and `verify_snapshot` returning its dict
     (4 files); `from_pretrained(weights_dir=WEIGHTS_DIR)` loading from the verified directory with `source`
     `local-snapshot`;
   - Section 4: `fetch_corpus` fetching the 120 pinned files (312,448,846 bytes) from the Hub mirror into
     `weights/diode-sample/`, 40 views of 20 scans read, and the seeded draw of 6 / 2 / 2 whole scans per domain into
     24 / 8 / 8 views with `check_split_disjoint` reporting no shared view and no shared scan, `scan_summary`
     printed and the three dataset digests `e7709c20…` / `ea60095b…` / `8f3bddbd…`; `outputs/…_train.csv`
     written; the four dataset refusal probes each raising `ValueError`;
   - Section 5: the ceilings (`MIN_IMAGE_SIDE` 14, `MAX_IMAGE_SIDE` 4096, `MAX_ASPECT_RATIO` 4.0) and the contract
     (`DEPTH_KIND` `relative`, `TRANSFORMER_BLOCKS` 12, `PARAMETER_COUNT` 24,785,089, `EVAL_DEPTH_RANGE_M`
     (0.6, 350.0)) surfaced; `validate_inputs` writing `outputs/…_input_manifest.json` (verdict `accepted`, one
     recorded rejection finding from the over-wide probe); `predict` on one test view with all four sanity checks
     `True`; `evaluation_report` against the view's masked metric reference with verdict **`sample-sanity`** and an
     `abs_rel` metric (≈ 0.12 on the recorded outdoor probe view); the side-by-side preview displayed;
   - Section 6: the constant prior (≈ 0.335 AbsRel) and the vertical-gradient prior (≈ 0.260) on the 8 test views,
     then `pipe.evaluate` on the untouched checkpoint (≈ 0.180 AbsRel, ≈ 0.764 δ1; indoor ≈ 0.075, outdoor
     ≈ 0.285) with per-view rows, two zero-shot maps kept, and the cell's assertion that the checkpoint beats both
     priors;
   - Section 7: `pipe.adapt` printing epoch 0 as the zero-shot checkpoint on validation (AbsRel ≈ 0.243), then
     2 neck-and-head epochs (2,728,513 trainable parameters) and 3 unfreeze epochs of the last two blocks
     (+ 3,550,464 of 24,785,089 parameters) with validation AbsRel / δ1 each epoch (0.243 → 0.220 → 0.194 →
     0.139 → 0.133 → 0.130 in the recorded run) and the selected policy `unfrozen last 2 blocks + DPT neck and
     head` (`best_epoch` 5);
   - Section 8: `pipe.evaluate` on the validation and test splits with the four-way comparison, per-domain and
     per-view AbsRel, and `outputs/…_evaluation_report.json` written (the cell asserts the selected model beats
     both priors — on the sample ≈ 0.167 versus 0.335 and 0.260; the delta over the checkpoint, −0.013 AbsRel, is
     reported, not asserted);
   - Section 9: two test views predicted by the selected model and rendered beside their zero-shot maps with
     per-view `aligned_abs_rel`, `outputs/…_preview.png` written; `pipe.save_artifact` writing
     `outputs/…_adapter/{adapter.safetensors, manifest.json}` (100 tensors, about 25.1 MB with two trained blocks —
     78 tensors, 10.9 MB when the neck-and-head policy is selected; `policy` recorded) and
     `DepthAnythingPipeline.from_artifact` reloading it with an identical depth map on a test view and an identical
     test AbsRel (the cell asserts both); `outputs/…_result.json` written with `NOTEBOOK_SOURCE`, the model identity
     and licence, the snapshot block (`weight_format`, `weight_sha256`), the `corpus` block, the inference-contract
     items with the single-view report, the comparison, the before/after rows, the artifact digest and policy, the
     reload parity, the runtime versions and device;
6. verify the exports exist and the interpretation section matches the observed path;
7. record the notebook Git blob id, commit, runtime (platform, Python, PyTorch, Transformers, device), the model identifier
   and immutable revision, whether the model cache, the weights directory and the corpus cache were clean, outcome,
   produced outputs, the observed metrics and the selected policy (as observations, not a benchmark) and any warning
   or applicable `SHOULD` deviation in the tables below;
8. record no access tokens or other secrets.

A known-failing default path in the supported runtime blocks release (REL11).

## Manual clean-runtime evidence

| Notebook | Commit / notebook blob | Date (UTC) | Executor | Outcome |
|---|---|---|---|---|
| `depth_anything_depth_estimation_colab.ipynb` (`E2E`) | `e777e2c` / `f3809066` | 2026-09-19 | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-depth-anything-depth-estimation` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | **PASSED** — 11/11 code cells ok (1 restart after install cell); 130 files, 412 MB staged from the Hub into a clean cache; comparison {abs_rel: {constant_prior: 0.3346, vertical_gradient_prior: 0.2597, zero_shot: 0.18, selected_policy: 0.1669}, delta1: {constant_prior: 0.5564, vertical_gradient_prior: 0.5828, zero_shot: 0.7644, selected_policy: 0.7652}, per_domain_abs_rel: {indoors: {zero_shot: 0.075, selected: 0.0705}, outdoor: {zero_shot: 0.285, selected: 0.2633}}, per_view_abs_rel: {test-000: {zero_shot: 0.1212, selected: 0.164}, test-001: {zero_shot: 0.3444, selected: 0.3064}, test-002: {zero_shot: 0.4037, selected: 0.3594}, test-003: {zero_shot: 0.1326, selected: 0.0988}, test-004: {zero_shot: 0.2705, selected: 0.2236}, test-005: {zero_shot: 0.0575, selected: 0.076}, test-006: {zero_shot: 0.071, selected: 0.0734}, test-007: {zero_shot: 0.0391, selected: 0.0339}}, delta_vs_zero_shot: {abs_rel: -0.0131, delta1: 0.0008}, selected_policy: unfrozen last 2 blocks + DPT neck and head}; reload parity {depth_map_identical: True, abs_rel_in_memory: 0.1669, abs_rel_reloaded: 0.1669}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-depth-anything-depth-estimation/v2/evidence/` in the workspace |
| `depth_anything_depth_estimation_colab.ipynb` (`E2E`) | `d6d8ba6` / `a9fd58ef` | 2026-09-19 | Local pre-flight harness (Windows, CPython 3.12.10, CPU, `google.colab` shim, pins pre-installed) | PASS — pre-flight only, **not** promotion evidence |
| `depth_anything_depth_estimation_colab.ipynb` (`TASK-INFERENCE`, superseded) | `a1b81b4` / `3ed3ea0` | 2026-09-13 | Kaggle Tesla T4 GPU (`kurtvalcorza/tut-depth-anything-verify` v11) | PASSED — `ok: True`, 9/9 code cells, 306 s (cells 39.8 s), live weights fetch, 5 artifacts verified (`evidence/release-verification-kaggle.json`); evidence for the earlier inference-only notebook, which it promoted to Release-grade — not for the `E2E` blob |

## Recorded executions

Notebook identity is the Git blob id of `tutorials/depth_anything_depth_estimation_colab.ipynb` (verify with
`git rev-parse <commit>:tutorials/depth_anything_depth_estimation_colab.ipynb`). Wall times are the sum of per-cell times
reported by the executor and include the model download where it occurred; they are measurements for the stated
runtime, not general estimates.

| Date (UTC) | Commit / notebook blob | Executor | Path exercised | Wall | Outcome |
|---|---|---|---|---|---|
| 2026-09-19 | `e777e2c` / `f3809066` | Kaggle Tesla T4 (`kurtvalcorza/dimer-nb2-depth-anything-depth-estimation` v2; image `torch 2.10.0+cu128` / `transformers 5.0.0` before the pinned install, `torch 2.14.0+cu130` / `transformers 4.57.6` after, Python 3.12.13, `cuda:0`) | Default sample path, `Run all` from a fresh interpreter with an empty Hugging Face cache and no repository checkout (blob SHA-1 verified against GitHub before execution) | 312.3 s | **PASSED** — 11/11 code cells ok (1 restart after install cell); 130 files, 412 MB staged from the Hub into a clean cache; comparison {abs_rel: {constant_prior: 0.3346, vertical_gradient_prior: 0.2597, zero_shot: 0.18, selected_policy: 0.1669}, delta1: {constant_prior: 0.5564, vertical_gradient_prior: 0.5828, zero_shot: 0.7644, selected_policy: 0.7652}, per_domain_abs_rel: {indoors: {zero_shot: 0.075, selected: 0.0705}, outdoor: {zero_shot: 0.285, selected: 0.2633}}, per_view_abs_rel: {test-000: {zero_shot: 0.1212, selected: 0.164}, test-001: {zero_shot: 0.3444, selected: 0.3064}, test-002: {zero_shot: 0.4037, selected: 0.3594}, test-003: {zero_shot: 0.1326, selected: 0.0988}, test-004: {zero_shot: 0.2705, selected: 0.2236}, test-005: {zero_shot: 0.0575, selected: 0.076}, test-006: {zero_shot: 0.071, selected: 0.0734}, test-007: {zero_shot: 0.0391, selected: 0.0339}}, delta_vs_zero_shot: {abs_rel: -0.0131, delta1: 0.0008}, selected_policy: unfrozen last 2 blocks + DPT neck and head}; reload parity {depth_map_identical: True, abs_rel_in_memory: 0.1669, abs_rel_reloaded: 0.1669}; run summary and executed notebook archived under `.agent/backups/kaggle-e2e-2026-09-19/out/dimer-nb2-depth-anything-depth-estimation/v2/evidence/` in the workspace |
| 2026-09-19 | `d6d8ba6` / `a9fd58ef` | Local pre-flight harness (Windows, CPython 3.12.10, CPU float32, `torch 2.14.0+cu130` with `CUDA_VISIBLE_DEVICES=-1`, `transformers 4.57.6`) | Default sample path (install skipped, pins pre-installed → three carried modules → inline manifest assert → `stage_missing_files` fetched 0 of 4 entries because the snapshot was pre-staged → `verify_snapshot` 4 files → `from_pretrained` on CPU → `fetch_corpus` served from the pre-staged cache after its 120 digest checks (1.0 s) → 40 views of 20 scans read, 24 / 8 / 8 drawn by whole scans with `check_split_disjoint` clean, 12 / 4 / 4 scans, digests `e7709c20…` / `ea60095b…` / `8f3bddbd…` → four dataset refusals → input manifest with the over-wide refusal → `predict` on `test-000` (outdoor, 0.37 s) with all four sanity checks `True` → `evaluation_report` **`sample-sanity`**, `abs_rel` 0.1212 over 703,835 reference pixels → constant prior → vertical-gradient prior → zero-shot `evaluate` → `adapt` ladder → validation + test evaluation → before/after maps + preview → adapter export → reload parity) | 118.3 s | **PASSED** — 11/11 code cells; constant prior 0.3346 / δ1 0.5564; vertical-gradient prior 0.2597 / 0.5828; zero-shot 0.1800 / 0.7644 (indoor 0.075, outdoor 0.285; 3.2 s); `adapt`: 2,728,513 neck-and-head + 3,550,464 block parameters, 5 epochs, 92.6 s, validation AbsRel 0.2426 (zero-shot) → 0.2201 → 0.1937 → 0.1394 → 0.1328 → 0.1295 with δ1 0.812 → 0.822 → 0.831 → 0.867 → 0.858 → 0.871, selected `unfrozen last 2 blocks + DPT neck and head` at `best_epoch` 5; **selected policy on the test split 0.1669 / δ1 0.7652 (Δ −0.0131 AbsRel, +0.0008 δ1 vs the checkpoint; indoor 0.0705, outdoor 0.2634)**; per view 0.121→0.164, 0.344→0.306, 0.404→0.359, 0.133→0.099, 0.271→0.224, 0.057→0.076, 0.071→0.073, 0.039→0.034 (three views worse); adapter 25,127,700 B / 100 tensors, SHA-256 `8545345c…`; reload parity exact (depth map identical, test AbsRel 0.166943 both ways); six exports written. Pre-flight; hosted clean-runtime run still required |
| 2026-09-13 | `a1b81b4` / `3ed3ea0` (`TASK-INFERENCE`, superseded) | Kaggle Tesla T4 GPU (`kurtvalcorza/tut-depth-anything-verify` v11) | Default sample path of the inference-only notebook: `synthetic_ramp_320x240`, `stage_missing_files` fetching the four manifest entries from the Hub, `verify_snapshot` over 4 files, `predict` with its sanity checks, `not-measurable` report, depth `.npy` + preview PNG + JSON exports | 306 s (cells 39.8 s) | **PASSED** — `ok: True`, 9/9 code cells, 5 artifacts verified (`evidence/release-verification-kaggle.json`); history only |

## Current status

**Release-grade.** The `E2E` notebook blob `f3809066` (committed at `e777e2c`) executed top-to-bottom in a clean Kaggle Tesla T4 runtime on 2026-09-19 (11/11 ok (1 restart after install cell), 312.3 s, 130 files, 412 MB fetched from the Hub and digest-verified inside the notebook) with no repository checkout — the REL1/REL10 supported-runtime evidence this file gates on. The local pre-flight rows above are what preceded it and remain history. Any later change to the carried modules or to the notebook produces a new blob, and the registry returns to **Candidate** until a clean run of that blob is recorded here.
