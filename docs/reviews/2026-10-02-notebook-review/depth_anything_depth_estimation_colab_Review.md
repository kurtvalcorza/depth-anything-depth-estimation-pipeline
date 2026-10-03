# Depth Anything V2 Small E2E Notebook — Review

**Verdict: Needs revision**  
**Review date:** 3 October 2026 (relay batch of 2 October 2026)  
**Repository:** `kurtvalcorza/depth-anything-depth-estimation-pipeline`  
**Notebook:** `tutorials/depth_anything_depth_estimation_colab.ipynb`  
**Reviewed commit:** `5f6b516c0ca532a6fefa379a280e9e127cec5d37` (`main`, confirmed with `gh api repos/kurtvalcorza/depth-anything-depth-estimation-pipeline/commits/main`)  
**Notebook Git blob:** `f38090667c3b514c2dfb9eb00694478fde18b1d1`. This is the blob of the recorded Kaggle Tesla T4 run of 2026-09-19 (commit `e777e2c`); no commit since `e777e2c` touches the notebook. `tools/build_notebook.py --check` and `tools/validate_release_assets.py` both exit 0 at the reviewed commit.  
**Finding prefix:** `DAC` (the workshop notebook in this repository uses `DEP`)  
**Framework:** Notebook Review Framework v1. **Requirements baseline:** NOTEBOOK_SPEC 2.2 (2026-09-26), `ml-worker` `origin/main` (`b1cfe13`).

## Executive assessment

The default path is technically strong and reproducible. The notebook carries the three package modules verbatim, stages and re-hashes the pinned 4-file snapshot, fetches 120 digest-pinned DIODE files, splits 40 views by whole scan (24 / 8 / 8, no shared view or scan), runs the inference contract with an input manifest, a refusal probe and a `sample-sanity` report against laser depth, scores two priors and the untouched checkpoint, runs a three-policy ladder in which the checkpoint competes as epoch 0, evaluates on the untouched test split per domain and per view, renders before/after maps, and exports a safetensors adapter that reloads to an identical map. The prose about alignment, priors, leakage and the size of the evidence is careful and honest.

This review re-ran the whole default path on CPU with the exact pins and reproduced the recorded figures to four decimals:

| Measure | This review (CPU, 24 threads, pins) | Kaggle T4 record (blob `f3809066`) |
|---|---|---|
| Code cells | 11/11 ok (install skipped, see below) | 11/11 ok **after one restart** |
| Dataset digests | `e7709c20…` / `ea60095b…` / `8f3bddbd…` | same |
| Probe view `test-000` AbsRel | 0.1212 (703,835 pixels) | ≈ 0.12 |
| Priors / zero-shot test AbsRel | 0.3346 / 0.2597 / 0.1800 | 0.3346 / 0.2597 / 0.18 |
| Validation ladder | 0.2426 → 0.2201 → 0.1937 → 0.1394 → 0.1328 → 0.1295, epoch 5 | same |
| Selected test AbsRel / δ1 | 0.1669 / 0.7652 (Δ −0.0131) | 0.1669 / 0.7652 |
| Reload parity | identical map; 0.166943 both ways | identical |
| Cell time | 249.7 s (ladder 204.1 s) | 155.1 s (pass 2) |

Four problems stand in the way of `Ready for intended use`:

1. **No one-pass `Run all` (DAC-M1).** The only hosted run stopped at the install cell's stale-module guard and passed after a restart; the record calls it PASSED and the status Release-grade.
2. **Re-running from the documented cells reuses the adapted model (DAC-M2).** The BYOD instruction ("re-run from that cell") and the optional experiments both start from the already adapted weights while labelling them "zero-shot". Observed directly.
3. **Guided layer mostly absent (DAC-M3)** for a notebook declared `GUIDED`.
4. Minor: CPU-default runtime never measured on a hosted CPU and the timings are unlabelled (DAC-m1); BYOD needs at least 4 groups and fails opaquely below that or on a cancelled upload (DAC-m2); the Prerequisites print a wrong id pattern with doubled braces (DAC-m3); the closing interpretation is fixed text about the build run, not the learner's run (DAC-m4).

## 1. Review contract and evidence

| Item | Value |
|---|---|
| Declared profile / mode | `E2E` / `GUIDED` (metadata `dimer.notebook_profile` / `notebook_mode`, opening cell) |
| Declared spec | DIMER Notebook Specification **2.0** (metadata, opening cell, `NOTEBOOK_SOURCE`, References) |
| Spec baseline applied | NOTEBOOK_SPEC **2.2** |
| Intended audience | Not stated as such. Knowledge prerequisites: basic Python and NumPy; inverse depth versus metric distance; why alignment precedes scoring; AbsRel and δ1; validation-based selection |
| Supported runtime | "Google Colab or Jupyter, Python 3.12"; CPU float32 default, CUDA used when present. `tutorials/README.md`: default runtime "CPU float32" |
| Promised outcomes | Pinned install; carried package; digest-verified snapshot; 40 digest-pinned DIODE views validated and split by scan; inference contract with manifest, refusal and `sample-sanity` report; two priors and zero-shot scores; neck-and-head then bounded unfreeze, selected on validation against the checkpoint; held-out AbsRel/δ1 per domain and per view; before/after maps; safetensors adapter export and reload parity; BYOD zip through "the same validation, seeded group-disjoint split, priors, zero-shot scoring, policy ladder, held-out evaluation, preview, artifact export and reload-parity cells" |
| Generator | `tools/build_notebook.py` (`build_notebook.py/2`) + `tools/notebook_template.py`; recorded generating revision `63939be` |

### Evidence actually obtained

- **Source inspection.** All 25 cells (11 code; cells 5, 7, 9 carry `metrics.py`, `pipeline.py`, `samples.py`, 84,754 characters). Also read: `adapt`, `evaluate`, `save_artifact`, `load_artifact`, `split_dataset`, `validate_dataset`, `load_byod_dataset`; `tools/build_notebook.py`, `tools/notebook_template.py`; `README.md`, `STATUS.md`, `tutorials/README.md`, `docs/release-verification.md`. The repository has no `AGENTS.md` and no `docs/execution-evidence/` directory.
- **Documented execution evidence.** `docs/release-verification.md` and the archived executor output (`.agent/backups/kaggle-e2e-2026-09-19/.../v2/evidence/run_summary.json`, `executed-pass1.ipynb`): Kaggle Tesla T4, 2026-09-19, **the reviewed blob**, clean Hugging Face cache. Pass 1 (157.2 s) stopped in cell 3 with `RuntimeError: Core dependencies changed while older modules were loaded: cuda-bindings: loaded=12.9.4, installed=13.4.2; numpy: loaded=2.0.2, installed=2.5.3. Restart the runtime, then rerun from the top.`; pass 2 ran 11/11 (155.1 s). No Colab run, no CPU hosted run, no BYOD run and no optional-experiment run is recorded.
- **Direct execution (this review).**
  - **Environment:** `run_probes.py`, Windows 11, CPU only (`CUDA_VISIBLE_DEVICES=-1`), 24 threads, build venv `dimer-next16` whose versions equal the notebook's `PINS` (Python 3.12.10, torch 2.14.0+cu130, torchvision 0.29.0, transformers 4.57.6, numpy 2.5.3, pillow 11.3.0, safetensors 0.8.0, huggingface-hub 0.36.2). Nothing was installed.
  - **Install skipped:** cell 3 ran with `DIMER_NOTEBOOK_CI_PREINSTALLED=1`, so the restart behaviour was not exercised here.
  - **Not a clean runtime:** `model.safetensors` and the 120 DIODE files were hard-linked from the local clone into a scratch working directory; cell 11 wrote the manifest fresh and `verify_snapshot` and `fetch_corpus` re-hashed every file.
  - **Executed:** P1 default path, cells 3–23; P2 the documented experiment `LEARNING_RATE = 3e-5` (edit cell 19, re-run cells 19, 21, 23); P3 cell 13's BYOD branch through a `google.colab` upload shim with a valid 40-view grouped zip followed by Section 6 (cell 17), plus three invalid uploads; P4 (second invocation, no model) BYOD zips of 8 views in 1, 3 and 4 groups. Wall 7 min 46 s plus 37 s. Two earlier attempts in a different conda env (torch 2.13.0+cpu) died with a native `numpy.linalg.lstsq` DLL fault in cell 15 and were discarded as environment faults.
- **Learner observation:** none. No claim here is about measured learning effectiveness.

## 2. Separate judgments

- **Technical correctness:** strong on the default path (reproduced exactly on CPU; T4 11/11 after a restart). Defects: the install pattern forces a restart (DAC-M1); re-runs of the documented cells continue from adapted weights and mislabel them (DAC-M2); BYOD failures below 4 groups and on a cancelled upload are not actionable (DAC-m2).
- **Promise fulfilment:** every default-path stage is implemented and was observed. The BYOD promise ("the same … priors, zero-shot scoring, policy ladder …") does not hold after the default run, because Section 6's "zero-shot policy" is then the adapted model (DAC-M2). The optional experiments' stated outcomes are not what a learner following the notebook gets (DAC-M2, DAC-m4).
- **Scientific validity:** good on the default path. Scan-disjoint split by pixel digest and scan; validation selects the epoch with the checkpoint competing; test used once; two meaningful priors; per-domain and per-view results; explicit "one seeded split, no dispersion". Re-runs break the zero-shot reference and the meaning of the recorded training configuration (DAC-M2).
- **Learner experience:** clear stage prose, a displayed depth preview, a before/after sheet, "Look for" and build-record notes. Missing: audience, how-to-use, roadmap, glossary, predictions, checkpoints, troubleshooting, conclusion scaffold, infrastructure labels (DAC-M3); CPU duration planning (DAC-m1).
- **Spec conformance:** applicable `MUST`s unmet: RUN1, RUN10, ENV6 and REL2/REL11 (DAC-M1); DAT13/DAT14 on the documented BYOD re-run (DAC-M2); DAT19 for the cancelled upload and the unnamed-split refusal (DAC-m2); REL12 has no recorded BYOD verification (DAC-m2); UX12 for unlabelled timing claims (DAC-m1). SHOULD gaps: GDL1–4, 6, 7, 9, 11–14, UX8, EXE2.

## 3. Promise and objective tracing

| Claim (cell) | Implementation | Observable result | Learner interpretation | Status |
|---|---|---|---|---|
| Run all completes in a fresh runtime (0) | install cell 3 with stale-import guard | T4 pass 1 stopped in cell 3; pass 2 11/11 | told only in Section 1 that the cell "stops with a restart instruction" | **Not met in one pass** (DAC-M1) |
| Digest-verified snapshot (10–11) | manifest assert → `stage_missing_files` → `verify_snapshot` → `from_pretrained` | 4 files verified, `source` local-snapshot (P1) | identity printed | Met |
| 40 pinned views split by scan (12–13) | `fetch_corpus`, `build_sample_dataset`, `check_split_disjoint` | 24/8/8, 12/4/4 scans, three digests, four refusals (P1) | "Look for" note matches | Met |
| Inference contract, `sample-sanity` (14–15) | `validate_inputs`, `predict`, `evaluation_report` | 4 checks True, AbsRel 0.1212, preview displayed (P1) | explained as sanity, not benchmark | Met |
| Priors and zero-shot (16–17) | `prior_baselines`, `pipe.evaluate` | 0.3346 / 0.2597 / 0.1800; indoor 0.075, outdoor 0.285 (P1) | per-domain reading prompted | Met on default path; mislabelled on re-run (DAC-M2) |
| Ladder selected on validation, checkpoint competes (18–19) | `pipe.adapt` | epoch 5 selected, 204.1 s (P1) | build-record trajectory given | Met on default path; epoch 0 is not the checkpoint on re-run (DAC-M2) |
| Held-out comparison (20–21) | `pipe.evaluate` ×2, comparison dict | Δ −0.0131 AbsRel; 3 of 8 views worse (P1) | "read the policy first"; no gain asserted | Met |
| Before/after maps, export, reload parity (22–23) | `save_artifact`, `from_artifact` | 100 tensors, 25,127,700 B, identical map (P1) | VER items explained | Met |
| Optional experiments (24) | edit cell 19 and re-run (no instructions) | lr 3e-5: unfrozen epoch 5 selected, test 0.1716 (P2) vs stated "neck-and-head, epoch 2, 0.175" | outcome stated, not asked | **Not delivered as described** (DAC-M2, DAC-m4) |
| BYOD through the same cells (0, 13) | `USE_BYOD` → `files.upload` → `load_byod_dataset` → `split_dataset` | valid zip accepted; Section 6 then prints `zero_shot_policy: 'unfrozen last 2 blocks + DPT neck and head'`, `adapted: True` (P3) | — | **Not met after the default run** (DAC-M2); limits understated (DAC-m2) |

| Objective (opening) | Learner activity | Evidence it was exercised |
|---|---|---|
| Install the pinned runtime | run cell 3 | requires a restart in hosted runtimes (DAC-M1) |
| Read what the carried modules guarantee | none; 84,754 characters of code, unlabelled | not exercised (DAC-M3) |
| Validate and split by scan without leakage | read printed splits, refusals | printed; no prompt asks the learner to check them |
| Read a `sample-sanity` report | read cell 15 output | printed and explained |
| Read AbsRel/δ1 beside priors; why alignment | read cell 17 output | explained; no prediction or checkpoint (DAC-M3) |
| Run the ladder; let the checkpoint win | read cell 19 output; optional experiments | default run observed; experiments do not behave as described on re-run (DAC-M2) |
| Evaluate on independent test split; compare maps | read cells 21, 23 | printed and rendered |
| Export and reload with parity | read cell 23 | asserted |

## 4. Journeys

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection, all 25 cells | Thorough prose and two visuals; no audience, how-to-use, roadmap, glossary, prediction, checkpoint, troubleshooting or conclusion template; carried modules not labelled infrastructure; restart only mentioned inside Section 1 (DAC-M1, DAC-M3); wrong id pattern in Prerequisites (DAC-m3); no CPU duration for the stated default (DAC-m1) |
| Clean default | Documented (Kaggle T4, reviewed blob, restart) + direct (CPU, pins, install skipped, files pre-staged) | Passed after one restart on T4 (DAC-M1). Direct CPU: 11/11, figures equal the record. No Colab run and no hosted CPU run recorded |
| Active learning | Direct (P2) | `LEARNING_RATE = 3e-5` re-run: epoch 0 printed as "zero-shot (no adaptation)" with val AbsRel 0.1295, the previous run's selected value (the checkpoint's is 0.2426); ladder selected unfrozen epoch 5, test 0.1716, Δ −0.0084 AbsRel / −0.0092 δ1. The documented outcome (neck-and-head at epoch 2, 0.175) is not what the learner gets (DAC-M2). `TRAINABLE_BLOCKS = 4` and `EPOCHS = 6` not run |
| Reuse and recovery | Direct (P3, P4) | Valid grouped zip accepted; Section 6 re-run reports the adapted model as zero-shot (DAC-M2). Ungrouped `records.csv`: clear refusal naming the fix. Not a zip: clear refusal. Cancelled/empty upload: bare `StopIteration`. 1 group: "split leaves 0 training records"; 3 groups: "0 records; 1..2000 are required" (split unnamed); 4 groups: accepted 4/2/2 (DAC-m2). Real Colab upload dialog not verified |

## 5. Findings

### Blockers

None.

### Major

#### DAC-M1 — `Run all` needs a manual restart after the in-kernel install

- **Cell/section:** Section 1, cell 3 (pinned `pip install` into the running kernel with a stale-module guard); `docs/release-verification.md` procedure step 4 and both evidence tables; `README.md`, `STATUS.md`, `tutorials/README.md` release status.
- **Observed issue:** the install replaces distributions the hosted kernel has already imported (numpy 2.0.2 → 2.5.3, cuda-bindings 12.9.4 → 13.4.2 on Kaggle; Colab also preloads a numpy that differs from the pin), so the guard raises and the learner must restart and run all again. The release procedure calls the restart "expected", and the run is recorded as **PASSED** with "11/11 code cells ok (1 restart after install cell)", which is the basis of the Release-grade status.
- **Consequence:** the first `Run all` a learner starts always stops after about 2.5 minutes with an exception; the release status rests on a restart-dependent run.
- **Evidence:** documented execution evidence: `run_summary.json` pass 1 `ok: false`, the `RuntimeError` above, `restarted_after_install_cell: true`; `executed-pass1.ipynb`. Source inspection: `tools/build_notebook.py` lines 45–71 (install cell). Direct execution did not exercise the install (skipped).
- **Spec:** RUN1, RUN10, ENV6 (MUST); REL2, REL11.
- **Recommended correction:** Adopt the fleet's **uv isolated-environment pattern**, which is how the capstone and newer workshop notebooks already run in one pass: the setup cell bootstraps uv, creates an isolated managed interpreter (`uv venv --managed-python --python 3.12.12 <ROOT>/env`), installs a hash-locked `requirements.txt` compiled with `uv pip compile` (`uv pip install --require-hashes --only-binary :all:`), and runs the pinned stages in that environment, so the kernel's preloaded NumPy/torch are never replaced and no restart can be required. Reference implementations on `main`: `ast-audio-classification-pipeline/tutorials/DIMER_Sound_Event_Classification_Workshop.ipynb` and `bioclip2-biodiversity-pipeline/tutorials/DIMER_Philippine_Biodiversity_Field_Survey_Capstone.ipynb` (this repository's own `DIMER_MultiModel_Depth_Estimation_Workshop.ipynb` already uses it). Do not add another in-kernel install guard or loosen pins to dodge the restart. Implement it in the repository's notebook generator (`tools/build_notebook.py` install cell, `tools/notebook_template.py`), regenerate, re-qualify with a one-pass hosted Run all, and correct the release record so a restart-dependent run is not reported as a `Run all` PASS.
- **Acceptance check:** a fresh Colab (or Kaggle) runtime executes the regenerated notebook with a single `Run all`, no restart, all code cells ok; the record states "no restart"; `docs/release-verification.md` no longer describes a restart as expected; the status documents do not cite the 2026-09-19 restart run as Release-grade evidence.

#### DAC-M2 — Re-running from the documented cells reuses the adapted model and mislabels it as zero-shot

- **Cell/section:** opening BYOD paragraph and Section 4 (cell 13: "set `USE_BYOD = True` … and re-run from that cell"); Section 6 (cell 17, `zero_shot_test`, `zero_shot_maps`); Section 7 (cell 19, `pipe.adapt`); "Optional experiments" in the closing cell.
- **Observed issue:** `pipe.adapt` trains whatever weights `pipe` currently holds and always labels its epoch 0 "zero-shot (no adaptation)"; `pipe.evaluate` reports `self.adapter['policy']`. Nothing reloads the base between runs.
  - BYOD after the default run (P3): Section 6 prints `'zero_shot_policy': 'unfrozen last 2 blocks + DPT neck and head'` with `adapted: True` — the "zero-shot" row, the epoch-0 rung and the "before" maps are the DIODE-adapted model.
  - Documented experiment `LEARNING_RATE = 3e-5` (P2): epoch 0 "zero-shot (no adaptation)" scored val AbsRel 0.1295 (the previous selected model; the checkpoint scores 0.2426); five further epochs; unfrozen epoch 5 selected, test 0.1716. The notebook says the learner will "watch the neck-and-head policy win (epoch 2, held-out 0.175)". The exported manifest records `lr` 3e-5, `head_epochs` 2, `epochs` 3, `best_epoch` 5, while the weights carry ten epochs from two learning rates.
  - Inferred from source, not run: if a re-run selects epoch 0, `save_artifact` writes a `no_op` "zero-shot" artifact whose tensors differ from the base and `load_artifact` refuses it (`pipeline.py` lines 732–735), so cell 23 fails; `TRAINABLE_BLOCKS = 4` after a 2-block run that then selects neck-and-head would export without the two blocks still modified in memory, failing the parity assertion.
- **Consequence:** the BYOD comparison against the untouched checkpoint — the notebook's central question — is silently wrong for the documented BYOD route; the experiments do not show what the notebook says they show, and the adapter's provenance misdescribes its training.
- **Evidence:** direct execution P2, P3 (`results.json`); source inspection of `adapt` and `evaluate` (`pipeline.py` lines 398–607).
- **Spec:** DAT13, DAT14 (MUST); GDL10, UX5, OUT8, EVAL14.
- **Recommended correction:** make every ladder start from the pinned base: in `tools/notebook_template.py`, reload the base (`pipe = DepthAnythingPipeline.from_pretrained(weights_dir=WEIGHTS_DIR)`) at the start of Section 6 (template around line 279) and Section 7 (around lines 304–315), or have `adapt` restore the base tensors before epoch 0 and refuse when `self.adapter` is set; state exact re-run instructions for BYOD ("re-run Sections 4–9") and for each experiment ("edit cell 19, then re-run Sections 6–9"); regenerate.
- **Acceptance check:** after a complete default run, (a) setting `LEARNING_RATE = 3e-5` and following the notebook's re-run instruction prints epoch 0 val AbsRel 0.2426 and selects the policy the closing text names; (b) setting `USE_BYOD = True` and re-running as instructed prints `zero_shot_policy: 'zero-shot (no adaptation)'` and `adapted: False` in Section 6; (c) a re-run that selects epoch 0 completes cell 23 with reload parity.

#### DAC-M3 — Declared `GUIDED`, but the guided layer is mostly absent

- **Cell/section:** opening (cells 0–1), every section boundary, the carried-module cells 4–9, the closing cell 24.
- **Observed issue:** no intended-audience statement, no "How to use this notebook" (Run all, form fields, which cells are infrastructure), no roadmap, no Input → Model → Output contract box, no glossary for inverse depth, alignment, AbsRel, δ1, DPT neck/head, policy, epoch 0; no prediction prompts before Sections 6–8; no interpretation checkpoints or sample answers; no troubleshooting section (restart, download, memory, digest mismatch, BYOD refusals); no conclusion template; 84,754 characters of carried code without an **Infrastructure** label or collapsed form (P0: no `cellView: form`, no "infrastructure" text, no "troubleshoot" text, no prediction prompt).
- **Consequence:** a learner new to depth estimation gets accurate prose but no structured practice; the stated objectives (read, compare, understand) are exercised only by reading printed dictionaries.
- **Evidence:** source inspection; P0 static scan.
- **Spec:** GDL1–GDL4, GDL6, GDL7, GDL9, GDL11–GDL14, UX8 (SHOULD).
- **Recommended correction:** add these elements in `tools/notebook_template.py` (opening and section markdown) and `tools/build_notebook.py` (label the carried-module cells as infrastructure and emit `cellView: form`); keep build-record figures as "Expected result" notes with the variation stated.
- **Acceptance check:** the regenerated notebook contains an audience line, a how-to-use block, a roadmap, a glossary covering the listed terms, at least one predict-before-run prompt per Sections 6–8 with collapsible sample answers, a troubleshooting section covering the five listed failure types, a conclusion template, and an "Infrastructure" label on each carried-module cell.

### Minor

#### DAC-m1 — CPU is the stated default, but the CPU runtime is unmeasured on a hosted runtime and the timings are unlabelled

- **Cell/section:** opening ("about four minutes of model time"), Prerequisites ("about 0.4 s per view … about 150 s"), `tutorials/README.md` "~four min … on CPU"; Section 7 progress.
- **Observed issue:** the timing claims do not name the environment and are not labelled as estimates; the only hosted record is a T4 GPU. On this review's 24-thread CPU the default cells took 249.7 s (ladder 204.1 s), against 118.3 s in the 2026-09-19 pre-flight; a 2-vCPU Colab CPU runtime is unmeasured and likely several times slower (estimate). Progress inside the ladder is one line per epoch (about 40 s apart on 24 threads).
- **Consequence:** a learner on the default Colab CPU runtime cannot plan the session and sees long silent stretches.
- **Evidence:** direct execution P1 cell timings; documented evidence table; source inspection.
- **Spec:** UX12 (MUST), ENV4, RUN12.
- **Recommended correction:** state measured times with their environment (T4, workstation CPU with thread count) and label any Colab-CPU figure as an estimate or record a hosted CPU run; print per-view progress inside `adapt` epochs (template Section 7).
- **Acceptance check:** every timing sentence names its environment or says "estimate"; a hosted CPU duration is recorded or the default runtime is changed to GPU; the ladder prints progress at least every 8 training views.

#### DAC-m2 — BYOD limits understated; three failure modes are not actionable; no BYOD evidence

- **Cell/section:** Prerequisites "Data contract"; opening BYOD paragraph; Section 4, cell 13 BYOD branch.
- **Observed issue:** the contract states record limits (4..2,000) but not that at least **4 groups** are needed: 1 group → "split leaves 0 training records; at least 4 are required"; 3 groups → "0 records; 1..2000 are required" with no split named (P4). A cancelled upload raises a bare `StopIteration` (P3). The Prerequisites say BYOD accepts "a `.zip` (or a directory)" and the runtime is "Colab or Jupyter", but the cell only offers `google.colab.files.upload()` with no path field (source; non-Colab use fails on the import, inferred). No BYOD run is recorded.
- **Consequence:** a learner with one capture session, or outside Colab, hits errors that do not say what to change.
- **Evidence:** direct execution P3, P4; source inspection of `split_dataset` (`samples.py` lines 866–905) and template lines 151–160.
- **Spec:** DAT12, DAT19 (MUST), REL12 (MUST), EXE2, UX10.
- **Recommended correction:** state "at least 4 groups" (or the formula) in the contract; check the group count before splitting and name the split in refusals; catch an empty upload with a message; add a `BYOD_PATH` form field that accepts a zip or directory outside Colab; record one representative BYOD run.
- **Acceptance check:** 1- and 3-group zips are refused with a message naming the group count needed; an empty upload prints a re-upload instruction; `BYOD_PATH` set to a directory runs Sections 4–9 in plain Jupyter; `docs/release-verification.md` records a BYOD run with one accepted and one refused input.

#### DAC-m3 — Prerequisites print doubled braces, including a wrong id pattern

- **Cell/section:** Prerequisites (cell 1), "Data contract".
- **Observed issue:** the cell reads `{{id, image, depth, mask}}` and ids matching `[A-Za-z0-9_.:-]{{1,64}}`. The second is a different regular expression from the enforced `{1,64}`. Section 4 renders the same record shape correctly with single braces.
- **Consequence:** a learner who copies the pattern to pre-check BYOD ids uses a wrong rule; the record shape looks like template syntax.
- **Evidence:** source inspection; P0 static scan. Root cause: `tools/notebook_template.py` line 124 escapes braces for `.format`, but the Prerequisites list is emitted unformatted (`tools/build_notebook.py` line 443).
- **Spec:** DAT12.
- **Recommended correction:** use single braces in that template string (or pass it through the same formatting as the other cells) and regenerate.
- **Acceptance check:** no markdown cell of the regenerated notebook contains `{{` or `}}`; the Prerequisites show `[A-Za-z0-9_.:-]{1,64}`.

#### DAC-m4 — The learner's own run is never interpreted; experiments state outcomes instead of asking

- **Cell/section:** Sections 6–8 notes and "Interpretation and limits" (cell 24).
- **Observed issue:** the closing text asserts the build run's outcome ("was selected at its last epoch and lowered held-out AbsRel from 0.180 to 0.167"), and each experiment names its result in advance. On P2 the stated outcome did not occur (partly because of DAC-M2), and on BYOD or a different device the closing paragraph would describe a run the learner did not have. The per-view mix (3 of 8 test views worse after adaptation on the default run) is not called out for the learner to examine.
- **Consequence:** the learner reads a conclusion rather than drawing one from the printed comparison.
- **Evidence:** source inspection; direct execution P1, P2.
- **Spec:** GDL7, GDL8, GDL14.
- **Recommended correction:** compute a short run-specific summary in cell 21 or 23 (selected policy, Δ vs checkpoint, views improved / worsened) and turn the closing paragraph and experiments into prompts with collapsible build-record answers.
- **Acceptance check:** the final cell output prints the run's policy, Δ AbsRel and improved/worsened view counts; the closing markdown phrases build-record figures as reference answers, not as the learner's result.

### Suggestions

- **DAC-S1:** declare `notebook_spec` 2.2 instead of 2.0 once the notebook meets it.
- **DAC-S2:** record per-stage wall times, the torch thread count and the device in `outputs/…_result.json` so learners and reviewers can compare runtimes.

## 6. Readiness

**Needs revision.** Remaining gates:

1. DAC-M1: one-pass `Run all` via the uv isolated environment, re-qualified on a hosted runtime, and the release record corrected.
2. DAC-M2: every ladder and the BYOD path start from the pinned base, with exact re-run instructions.
3. DAC-M3: the guided layer.
4. Applicable `MUST`s from DAC-m1 (UX12) and DAC-m2 (DAT19, REL12).

The Release-grade label in `README.md`, `STATUS.md` and `tutorials/README.md` rests on a restart-dependent run and should not stand until gate 1 is met.

## 7. Verified versus inferred

- **Verified by direct execution (CPU, pins, install skipped, files pre-staged):** the full default path and its figures; the experiment re-run behaviour at `LEARNING_RATE = 3e-5`; the BYOD-then-Section-6 mislabel; the four BYOD refusal messages and the group-count boundary.
- **Verified from documented evidence:** the restart requirement on Kaggle T4 for this blob.
- **Inferred, not run:** the restart on Colab (Colab preloads a numpy that differs from the pin); the epoch-0-selected and `TRAINABLE_BLOCKS = 4` re-run failures; non-Colab BYOD failing on the import; Colab-CPU duration.
- **Only Kurt can confirm:** a fresh Colab run of a regenerated notebook, and the real Colab upload dialog.
- **Most likely to be wrong:** the inferred cell-23 failure in DAC-M2 when a re-run selects epoch 0. It depends on the validation curve of a second ladder; P2 selected a later epoch, so that path was not observed. DAC-M2 stands as Major without it, on the observed zero-shot mislabel.

## Probe ZIP

`depth_anything_depth_estimation_colab_Review_Probes.zip` holds `run_probes.py`, `results.json` (P0–P4; contains local scratch paths in tracebacks) and `source_manifest.json` (SHA-256 of every inspected file, the spec and the archived Kaggle evidence).
