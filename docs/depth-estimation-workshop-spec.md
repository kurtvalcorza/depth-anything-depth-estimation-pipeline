# DIMER Multi-Model Depth Estimation — Notebook Specification

Status: **Implementation proposal. No notebook execution, model result or release qualification is claimed.**

Prepared: 2026-09-27. Normative basis: local `ml-worker/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md`, version 2.2. Only this design document is created by this task.

## 1. Learning unit

**DIMER Notebook: Image Depth Estimation — Relative Depth and Metric Distance**

Driving question: **Does a convincing depth map tell us how far away things really are?**

Position: **Level 2 — Applied tasks**, immediately after image segmentation. Basic Python and Colab familiarity; no prior depth-estimation knowledge. The unit is independent and self-paced. Completion records are optional personal notes, not required submissions.

| Field | Proposed decision |
|---|---|
| Host | `kurtvalcorza/depth-anything-depth-estimation-pipeline` |
| Artifact | `tutorials/DIMER_MultiModel_Depth_Estimation_Workshop.ipynb` |
| Profile / mode | `E2E` / `WORKSHOP`, standalone |
| Models | Depth Anything V2 Small; ZoeDepth NYU+KITTI |
| Adaptation | Depth Anything neck and head only; backbone frozen; ZoeDepth frozen |
| Default data | Existing pinned 40-view DIODE sample, regrouped by scene |
| Runtime | Fresh Colab T4-class GPU; isolated Python 3.12 worker; sequential model residency |
| Results | Raw depth arrays, labelled maps, errors, two separate metric protocols, frozen/adapted comparison, controlled brightness experiment, exported adapter and verified reload |
| Release state | Candidate until exact hosted notebook and required BYOD evidence are reviewed |

Catalogue description:

> Explore relative depth and distance estimates from a single photograph. Compare predictions with real RGB-D references, distinguish reference-assisted alignment from direct estimates in metres, and adapt a small depth model using separate training and validation scenes. Evaluate held-out scenes, inspect failure cases, change image brightness in a controlled activity, and export and reload the selected adapter.

This spec does not change the curriculum count or publish a Colab link before an artifact exists.

## 2. Verified source baseline

The following are inspected **local checkout** revisions, not a claim about current remote `main` or live DIMER service availability. Both working trees were clean during inspection.

| Repository | Inspected commit | Pinned model revision |
|---|---|---|
| Depth Anything | `07c4a9858d3d12f2f1924ffb8a23bad10cd992fb` | `depth-anything/Depth-Anything-V2-Small-hf` at `5426e4f0f36572d16453bbda7a8389317b1bef99` |
| ZoeDepth | `194fbbc6111e36873a84eb78404a687201aa1d73` | `Intel/zoedepth-nyu-kitti` at `f364d4c7936e91f465abba182208dd68142bf0ca` |

Depth Anything's `pipeline.py`, `metrics.py`, and `samples.py` implement relative inverse-depth inference, scale/shift-invariant adaptation, validation selection including epoch zero, pinned DIODE data, and adapter export/reload. Its current epoch-selection metric is rounded to six decimal places; preserve and disclose this, including earliest-epoch ties, unless a separately tested source change is made.

ZoeDepth's `pipeline.py` predicts metres and also supports metric-head fine-tuning. This unit deliberately uses its frozen capability as the metric reference. Its existing tutorial uses generated data, so those results are not real-DIODE evidence for this unit. Do not route DIODE through ZoeDepth's adaptation validator, which has a separate 80 m training-target ceiling.

Current source helpers can exclude invalid prediction pixels during metric computation. The workshop must instead reject nonfinite predictions on the fixed reference-valid support; ZoeDepth must also be positive there. Model-dependent pixel dropping must not improve a score silently. Record this stricter workshop preflight in provenance.

Primary references: [Depth Anything model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf), [ZoeDepth model card](https://huggingface.co/Intel/zoedepth-nyu-kitti), [DIMER notebook standard](https://github.com/kurtvalcorza/ml-worker/blob/main/integrations/dimer/fleet-specs/NOTEBOOK_SPEC.md). Cards identify Apache-2.0 and MIT respectively; retain pinned license files and dataset attribution in the built artifact.

## 3. Learning outcomes and instructional pattern

Learners should be able to:

1. Explain RGB input → model → one numerical depth value per pixel.
2. Distinguish depth, inverse depth, relative scale and metres; explain why a colour map is not a measurement.
3. Identify missing sensor depth and the exact pixels used for evaluation.
4. Explain AbsRel and delta-1, and distinguish an oracle-aligned result from direct metric inference.
5. Identify which scenes train, select and test the adapter.
6. Interpret gains, regressions and an epoch-zero selection without manufacturing improvement.
7. Export/reload an adapter and state what numerical parity does and does not prove.

Each substantive experiment follows: orient → Input → Model/System → Output → predict → run → notice → interpret → change one thing → compare → conclude with evidence and limitations. Include one worked metric example and collapsible interpretation hints. Do not require learners to inspect embedded implementation code to understand the activity.

## 4. Data and split contract

Reuse the existing 120-file RGB/depth/validity manifest: 40 views, 20 scans, **6 scenes**, 20 indoor and 20 outdoor views. Source is DIODE validation, read by HTTP Range requests at pinned byte offsets (`samples.ARCHIVE_MEMBERS`) from the uncompressed `diode_val.tar` of the Marigold evaluation dataset archive at ETH Zürich PRS (`https://share.phys.ethz.ch/~pf/bingkedata/marigold/evaluation_dataset/diode/diode_val.tar`, 6,400,440,320 bytes; a non-206 reply or a different `Content-Range` total is refused); existing declared payload is 312,448,846 bytes. (Corpus source change, 2026-10-03: the earlier Hugging Face Hub mirror `obukhovai/marigold_depth_eval` at `30c5b061` was withdrawn; the same 120 files have identical digests in the ETH archive, so the per-file pins are unchanged.) Every file has byte-size and SHA-256 checks. Cite DIODE (Vasiljevic et al., 2019), CC BY 4.0 as recorded in the source; verify pinned attribution at build time.

The source sample is selected independently of model performance. Freeze this scene assignment before model execution:

| Role | Indoor scene / views | Outdoor scene / views | Total |
|---|---|---|---:|
| Training | `scene_00020` / 8 | `scene_00023` / 6 | 14 |
| Validation | `scene_00019` / 2 | `scene_00024` / 4 | 6 |
| Test | `scene_00021` / 10 | `scene_00022` / 10 | 20 |

All views of a scene stay in one role. Use `(domain, scene)` group identities and retain scan IDs. Detect duplicate IDs, decoded RGB and RGB-D content across roles. No silent exclusions or model-dependent sampling. Freeze record IDs, source paths, scene/scan/domain, role, decoded-content hashes, shapes, units and validity counts in a sample manifest. The source's scan-level splitter must not be used unchanged.

This is a workshop reuse of a published validation release, **not an official DIODE benchmark split**. Twenty test views represent only two independent scenes. Report scene/domain results and small-sample limitations; do not infer population confidence from 20 correlated views or claim upstream pretraining contamination has been ruled out.

Canonical support: sensor-valid mask AND finite reference depth in inclusive `[0.6, 350]` metres. At least two valid pixels are required, and count/fraction must be shown. Do not interpolate invalid depth into valid observations or treat zero as a real depth. Reject invalid predictions on this support. Use original-image coordinates for final scoring; preserve each model's documented resize/postprocessing. Freeze any adaptation-target resampling separately.

## 5. Evaluation protocols — separate tables

**A. Relative structure, Depth Anything:** fit per-image least-squares scale and shift `1 / reference_depth = a * predicted_inverse_depth + b` on the fixed reference-valid pixels. Convert with `1 / max(a*p+b, 1/350)`. Export fitted coefficients, rank/degeneracy diagnostics, negative-scale flags and fraction reaching the far cap. Preserve the source least-squares semantics; do not silently add a positivity constraint on `a` or choose alignment based on test performance.

Label every such table **reference-assisted affine alignment**. It uses each evaluated image's reference depth, including test references, as part of a declared evaluation protocol. It is not an inference-time conversion to metres and cannot be exported as a calibration valid for a new image. A degenerate or reversed fit must be visible, not presented as proof of meaningful ordering.

**B. Metric distance, ZoeDepth:** score its raw positive predictions in metres, without fitted alignment, median scaling or clipping predictions to improve results. Use the same reference-valid support and reference range. Do not put its unaligned metrics and Depth Anything's aligned metrics in a single ranked leaderboard.

For both protocols:

- AbsRel = mean over valid pixels of `abs(predicted_metres - reference_metres) / reference_metres`; lower is better.
- Delta-1 = fraction with `max(pred/reference, reference/pred) < 1.25`; higher is better. The inequality is strict.
- Report per-image values, equal-image means, per-scene and per-domain means, valid counts and exclusions (expected zero). Do not label equal-image means as pooled-pixel scores.
- Undefined values are null with a reason, never zero. A default record that cannot be scored fails qualification rather than disappearing from the table.

Baselines:

1. **Training-only constant metric depth:** median of the 14 training-image valid-depth medians, fixed before validation/test. Score unaligned in metres beside ZoeDepth. The two-stage median is deliberate and gives training images equal weight.
2. **Oracle per-image median:** each evaluated image's reference-depth median, explicitly reference-assisted, beside aligned Depth Anything only.
3. **Vertical inverse-depth ramp:** fixed image-row gradient, aligned with the identical relative protocol; tests whether spatial structure exceeds a simple positional prior.

Do not reuse reference-derived baselines as deployable models.

## 6. Bounded adaptation and model selection

Default call semantics: `adapt(train, val, trainable_blocks=0, head_epochs=3, epochs=0, lr=1e-5, seed=42)`. This trains the DPT neck/head only, batch size one, AdamW weight decay 0.01, gradient clipping 1.0, no augmentation; retain the source scale-and-shift-invariant loss. Expected optimizer updates: **42**. ZoeDepth remains frozen.

Freeze these settings before evaluation. Capture untouched validation/test predictions before adaptation for the preregistered paired comparison, without using test results to change training or selection. Select epoch 0–3 by the source validation aligned AbsRel, earliest tie. No test selection, post-test hyperparameter search or switching to an unfreeze policy if results disappoint.

Record trainable tensor names/counts, frozen tensor hashes, actual optimizer steps, losses, per-epoch validation scores and weight deltas before restoring the selected epoch. Verify at least one real parameter update occurred. An epoch-zero winner is legitimate: distinguish observed training updates from the selected artifact's possible zero delta. Report whether adaptation improved, worsened or preserved each held-out image and scene.

Use float32 by default. Measure memory and synchronized elapsed time per model/stage; include data/bootstrap time separately. One model resident on GPU at a time. Target a bounded Colab session; publish measured time after qualification, not an invented estimate. Do not silently skip training or a model after OOM.

## 7. Controlled activity

After selection is frozen, use the **six validation images** for a descriptive input-brightness activity. Their reuse is disclosed; these are not additional held-out test observations and activity results cannot change the selected adapter.

Run the selected Depth Anything model with RGB factors `0.6`, `1.0`, `1.4`: `round(clip(float32(rgb) * factor, 0, 255)).astype(uint8)`. Geometry, references, masks, checkpoint and scoring protocol stay fixed. Record clipped-pixel fractions, input hashes and actual factors. This changes pixel brightness, not physically accurate exposure.

Compare paired aligned AbsRel/delta-1 deltas against factor 1.0; show improved/unchanged/worsened counts and per-scene results. Disclose that per-image alignment can hide scale changes. Do not label this a camera robustness benchmark or claim that brighter/darker images must perform worse.

## 8. Notebook sequence and visual outputs

1. Orientation, outcome, runtime/data requirements and optional learner prediction.
2. Anonymous downloads, isolated environment, immutable manifests and GPU preflight.
3. RGB/reference/valid-mask inspection, split table and contamination checks.
4. Tiny-array worked AbsRel/delta-1 example and near/far inverse-depth explanation.
5. Training-only and reference-assisted baselines with clear labels.
6. Frozen Depth Anything maps and aligned evaluation.
7. Frozen ZoeDepth maps and unaligned metric evaluation; explain separate protocols.
8. Predict whether adaptation will help; run three epochs; interpret validation selection.
9. Fresh-process selected-adapter reload and held-out paired comparison.
10. Predict brightness sensitivity; run the fixed activity and interpret counterexamples.
11. Export results; write an evidence-based conclusion with limitations.
12. Optional labelled BYOD full workflow; optional unlabelled single-image inference.

Show original RGB, reference depth, validity, predicted map and error map. Metric plots use shared documented metre ranges; error plots share a fixed scale. Relative colour maps are explicitly arbitrary units; any display-only per-image normalization is labelled and never used for scoring. Invalid reference pixels are neutral/transparent. Include a fixed example, best/worst adaptation deltas and both domains; state selection rules. Failure modes to investigate include boundaries, missing sensor returns, reflective surfaces and domain shift, without inventing observed failures.

## 9. Export, reload and provenance

Export the selected neck/head safetensors adapter with its compatible base-model manifest; no base-weight redistribution. Load it in a new Python process from verified local base files. Compare raw inverse-depth arrays on fixed validation probes and all test images, then independently recompute metrics. Proposed fixed float32 tolerance: `atol=1e-5`, `rtol=1e-4` for predictions, `1e-6` absolute for aggregate metrics; log maximum errors and fail excess drift. Any tolerance revision needs documented numerical evidence and a new qualification run.

Bundle an explicit allowlist: sample/split manifest, protocol/config, source identities, model/data/dependency pins, training history, selection, update evidence, per-image and aggregate metrics, activity deltas, reload checks, runtime/hardware, figure selection, adapter files, run summary and checksums. Preserve raw numeric outputs separately from display images. Reload exported tables/arrays and verify ZIP checksums. Record actual generating source hashes in addition to repository revision labels.

Invalidate downstream receipts when a prerequisite stage reruns. A failed model, changed manifest or stale output must prevent a completed-current-run claim. Terminal output names counts, selected epoch, protocols, parity and qualification state; a completed execution does not automatically confer release approval.

## 10. BYOD and runtime contract

The optional labelled path accepts a directory with `manifest.json` records: `id`, relative RGB `image`, float32 `.npy` `depth` in metres, boolean `.npy` `mask`, `group_id`, `role`, optional `domain`. Require explicit units and acquisition/source grouping. Use `allow_pickle=False`. Related camera sequences/scenes stay together. Positive reference support and both model image contracts must pass before GPU work.

Proposed notebook ceilings: 12–128 records, at least 6/2/4 train/validation/test; at least two distinct groups in each role; sides 32–2048 and aspect ratio at most 4; 32 MiB per file, 1 MiB JSON manifest, at most 64 million total image pixels. Validate headers, dimensions, allocation bounds and path containment before loading; refuse traversal, object arrays, mismatches, duplicate content, overlapping groups, invalid units and empty valid support. No automatic unit guessing or silent resizing to bypass limits.

Run the same baselines, bounded adaptation, validation selection, test evaluation, activity, export and fresh reload in a separate BYOD output directory. Real representative labelled BYOD execution is required for qualification; synthetic fixtures support validation tests only. Default Run all never prompts for uploads. Private images/depth arrays stay out of the shareable ZIP by default; IDs, metrics and labels may remain, so tell learners to review exports before sharing.

Optional unlabelled inference uses the selected adapter plus frozen ZoeDepth. Export a raw relative inverse-depth array and a separate predicted-metres array, each correctly labelled. With no reference, accuracy metrics are null/not-measurable; do not apply a previous image's alignment. No new adaptation claim.

Embed needed reference modules under distinct namespaces to prevent helper shadowing. Carry source/license/hash manifests and tested dependency locks. No runtime repository clone, raw Python-source fetch, installed DIMER package dependency, remote inference API or DIMER service. Isolated Python 3.12 must not replace Colab's kernel or require manual restart. Public default resources require no token; optional HF_TOKEN access uses Colab Secrets, never prints or exports credentials. Retry transient downloads only; integrity failures fail closed.

## 11. Acceptance checklist and build deliverables

- [ ] Notebook generator, embedded worker, immutable model/data manifests and hash-locked environment.
- [ ] Scene-disjoint frozen sample; actual role counts 14/6/20 and independent-scene counts 2/2/2.
- [ ] CPU tests for metrics, strict thresholds, fixed valid support, invalid prediction refusal, degenerate alignment, duplicate/group leakage and BYOD limits.
- [ ] Source-parity, notebook schema/code compilation, namespace isolation and stage invalidation tests.
- [ ] Real default data preparation verifies all 120 files and rendered reference maps.
- [ ] Fresh hosted GPU Run all completes all models, 42 updates, validation selection, activity and exports without manual interventions.
- [ ] Fresh-process prediction/metric parity and artifact schema/refusal checks pass.
- [ ] Real labelled BYOD completes the full E2E path; negative input cases fail before model execution.
- [ ] Actual scores, errors, timing and memory are preserved even when adaptation fails to improve.
- [ ] README/tutorial registry and release-evidence record identify exact notebook SHA, execution environment and pending gates.
- [ ] AI assistance disclosure, citations, dataset/model licenses and optional-completion note included.

AI Assistance Disclosure: This notebook's code and explanations were developed with generative AI assistance under maintainer direction. The maintainer is responsible for review, result validation and release decisions. AI assistance does not constitute independent verification, provider endorsement or release approval.

Implementation risks to resolve with evidence: T4 peak memory during head training; source-compatible adaptation target masks; least-squares degenerate/reversed fits; exact fresh-process float32 parity; anonymous availability of all pinned files. These are build/qualification checks, not confirmed failures.

Scope ends at this specification. Implementation, hosted execution, commit, push, PR and merge are separate actions.
