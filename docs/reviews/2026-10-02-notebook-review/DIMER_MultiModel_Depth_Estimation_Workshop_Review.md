# DIMER Notebook: Image Depth Estimation — Review

Review of `tutorials/DIMER_MultiModel_Depth_Estimation_Workshop.ipynb` against the Notebook Review Framework v1 and
DIMER Notebook Specification 2.2.

**Reviewed revision:** `kurtvalcorza/depth-anything-depth-estimation-pipeline@5f6b516c0ca532a6fefa379a280e9e127cec5d37`
(`main` on 2026-10-02, confirmed through the GitHub API). Notebook blob `aa6a35259d6f555fd2353ab29d8443762a2f57e0`,
the same blob as `068d681`, which the 2026-09-27 hosted run executed.
**Probe bundle:** `DIMER_MultiModel_Depth_Estimation_Workshop_Review_Probes.zip` (`run_probes.py`, `results.json`,
`source_manifest.json`).

## Executive assessment

The default path is carefully built and has a recorded passing hosted run. It uses a scene-disjoint split, a fixed
evaluation support, two evaluation protocols that it keeps apart and labels honestly, real and counted optimizer updates,
validation-only selection, and a reload in a new process. **Readiness: Needs revision.** One Major and seven Minor findings
remain open.

- **DEP-M1 (Major):** the optional labelled BYOD route enforces several rules that §10 never states. When it refuses
  data, the message either omits the failing record or file, or names the wrong one: a coverage failure on the 8th
  record reports `records[0]`. This fails the `MUST` requirements DAT12 and DAT19, so a learner with 12–128 records
  cannot tell what to fix.
- The Minor findings concern:
  - infrastructure labelling;
  - scoring code that a learner can read only inside an escaped string;
  - training evidence the cell does not display;
  - the download route and resource measurements;
  - the unlabelled-image path;
  - terminology and a missing glossary;
  - an activity in which the learner cannot change anything.

## 1. Review contract and evidence

| Field | Value |
|---|---|
| Repository / notebook | `depth-anything-depth-estimation-pipeline` / `tutorials/DIMER_MultiModel_Depth_Estimation_Workshop.ipynb` (26 cells, 13 code) |
| Revision | `5f6b516` (blob `aa6a3525`) |
| Specification | NOTEBOOK_SPEC 2.2 (`ml-worker` `origin/main`, last changed in `50b7eb4`) |
| Profile / mode | `E2E` / `WORKSHOP`, standalone (carried source, hash-locked `uv` environment) |
| Audience | Level 2 "Applied tasks"; basic Python and Colab; "no prior ML or depth-estimation experience" |
| Runtime | Fresh Colab T4 GPU (required; cell 2 refuses to run on CPU) |
| Promised outcomes | inspect sensor references; calculate AbsRel/delta-1; distinguish two evaluation protocols; adapt Depth Anything's neck/head; select on validation scenes; verify a saved adapter in a new process; change input brightness in a controlled experiment; optional labelled BYOD (full workflow) and unlabelled single image |

**Execution evidence that covers this revision.**
- `docs/release-verification.md` records a fresh **Colab T4** default Run all of blob `aa6a3525`, dated 2026-09-27.
  All 13 code cells ran, with no errors, in 403 s. Epoch 0 was selected. Reload parity passed. Both optional branches
  were left off. Peak GPU memory and disk use were not captured.
- The CPU builder pre-flight recorded in the same file is not a supported runtime.
- Neither the BYOD branch nor the unlabelled branch has ever been executed in a hosted runtime (REL12 is open).

**Direct execution in this review** (CPU only; Windows; anaconda Python 3.13; NumPy 2.3.5, Pillow 11.3.0; no model
weights, no GPU):
- the carried worker's `load_byod` and `prepare` stage on synthetic 48×64 RGB-D BYOD data (12 records), which passed;
- five invalid BYOD variants;
- `bounded_image` on a 4032×3024 photo and on an image with EXIF orientation 6;
- IPython `FileLink` on an absolute path.

No model stage (`depth`, `reload`, `zoe`, `activity`, `infer`) was executed.

| Journey | Evidence basis | Result |
|---|---|---|
| First-time learner | Source inspection | Mostly well scaffolded (Predict prompts, Notice notes, hints, a worked metric example, a conclusion template). Gaps: m1, m2, m3, m6 |
| Clean default | Documented execution evidence (Colab T4, 2026-09-27, this blob) | **Passed** for the default stages; m4 concerns what the run could not record |
| Active learning | Source inspection; documented run for the fixed activity | The brightness comparison runs, but no learner change reaches the computation (m7) |
| Reuse and recovery | Direct execution (CPU, synthetic) for BYOD validation and `prepare`; source inspection for the rest | BYOD `prepare` works on valid data, but refusals are not actionable (M1). Unlabelled path: limits unstated, no preview (m5). Model stages of both branches **not verified** |

## 2. Separate judgments

- **Technical correctness:** sound on the default path:
  - stage receipts invalidate downstream work when an upstream input changes;
  - every carried file is hash-checked;
  - prediction support is strict and fixed, with no pixels dropped depending on the model;
  - frozen tensors are hash-checked, and real optimizer steps are counted through a hook;
  - reload runs in a new process, with tolerances on the predictions and on the metrics.

  The defects are in recovery: BYOD error attribution (M1) and the unlabelled-input checks (m5).
- **Promise fulfilment:** delivered on the default path, with two qualifications:
  - "change input brightness" is a demonstration in which the learner changes nothing (m7);
  - the BYOD route is implemented but its contract is not fully stated (M1).

  The adaptation is genuine (42 updates), but epoch 0 wins. The notebook presents this honestly, though it does not
  display the evidence it tells the learner to inspect (m3).
- **Learner experience:**
  - Strengths: good prediction and interpretation scaffolding.
  - Problems:
    - a 290,653-character carried-source cell is shown uncollapsed and unlabelled (m1);
    - the core scoring code is readable only as an escaped string (m2);
    - jargon such as neck, head, backbone, AdamW and scale-and-shift-invariant loss is never defined, which matters for
      a "no prior ML" audience (m6);
    - the download links in Colab do not work (m4, inferred).
- **Spec conformance:**
  - **MUST gaps:** DAT12 (partial: four enforced rules unstated) and DAT19 (partial: refusals do not identify the record
    or file).
  - **Release gates:** REL12, a BYOD hosted run, has not been done.
  - **SHOULD gaps:** GDL6, GDL10, GDL11 and GDL15.
  - **Met:** RUN1–RUN14, ST1–ST8, ENV, MOD, EVAL, SPL, FT and VER, on source inspection plus the hosted run. ST5 is
    met in letter: the source is carried in the notebook (see m2).

### Promise → evidence

| Claim | Implementation | Observable result | Learner interpretation |
|---|---|---|---|
| Scene-disjoint split, 14/6/20 | `SCENES`, `validate_roles` (cell 3 worker) | `prepare.json` counts; hosted run 14/6/20 | md-05 table + Predict + hint ✓ |
| Two protocols kept apart | `score(aligned=…)`, `show_metric_tables` per protocol | separate tables (cells 10, 12, 16); hosted values | md-09/11 ✓ |
| Adapt neck/head, select on validation | `depth()` with counted AdamW hook, frozen-hash check | 42 steps; epoch 0 selected (hosted) | md-13 says epoch 0 is valid, but per-epoch weight deltas are not displayed (m3) |
| Reload in a new process | `reload()` PID check, `verify_parity` | parity on 26 images, max error 0.0 (hosted) | md-15 ✓ |
| ZoeDepth metres, unaligned | `zoe()` | table + panel (hosted) | md-17 ✓ |
| "Change input brightness" | fixed `(0.6, 1.0, 1.4)` in the worker; `brightness()` rejects other factors | paired table (hosted) | learner cannot change it (m7) |
| Export + checksummed ZIP | `report()` | `depth_results.zip` (hosted) | link not usable in Colab (m4, inferred) |
| Own labelled data, full workflow | `byod_run` → six child stages | `prepare` passes on synthetic data (direct); model stages not verified | contract incomplete, refusals misattributed (M1) |
| One unlabelled image | `infer()` | not executed | limits unstated, no visual output (m5) |

### Objective → activity → evidence

| Objective | Activity | Evidence it was exercised |
|---|---|---|
| Calculate AbsRel/delta-1 | cell 8 + worked answer | ✓ |
| Distinguish protocols | Predict in md-09, tables, hint | ✓ |
| Identify the selecting scenes | md-05 table, self-check | ✓ |
| Interpret epoch-0 selection | md-13 Notice | partial: the evidence named is not shown (m3) |
| Explain parity vs accuracy | md-13 Predict, md-15 | ✓ |
| Controlled change | md-17 Predict; fixed factors | observation only (m7) |
| Transfer to own data | §10/§11 | blocked by an unclear contract (M1, m5) |

## 3. Findings

### DEP-M1 — Major: BYOD contract incomplete; refusals do not identify the failing record or file (DAT12, DAT19)

**Cell/section:** §10 markdown (`md-21`), BYOD cell (`code-22`), and the carried worker's `load_byod`, `safe_file`,
`bounded_image`, `bounded_array`, `validate_roles`, and `prepare` → `samples.validate_dataset([r], min_records=1)`.

**Observed issue:**
- *Refusals that omit or misname the failing record.* `prepare` validates each record by calling
  `validate_dataset` on a one-element list, so every per-record refusal is labelled `records[0]`. The other refusals
  name neither the record nor the file:
  - "missing file or file exceeds 32 MiB";
  - "array shape/dtype mismatch before allocation";
  - "explicit group_id required";
  - "duplicate or unsafe record id";
  - "group crosses roles".
- *Rules that are enforced but never stated before data is supplied:*
  - ids must match `[A-Za-z0-9_-]{1,64}`;
  - at least 5% of pixels must carry valid depth;
  - masked depth must not exceed 1000 m, and at least two reference pixels must lie in [0.6, 350] m;
  - EXIF orientation must be 1;
  - `units` must be exactly `"metres"`;
  - depth arrays must be exactly float32.
- *Missing guidance:* there is no example `manifest.json`. Nothing says how to put a directory into the Colab runtime
  (Files pane or Drive), or whether the data leaves the runtime (DAT17).
- *Results not shown:* the BYOD cell shows only a link and the log tail, not the run's result tables. The BYOD run's
  `summary.md` also states, unconditionally, that "Default test views cover only two scenes".

**Consequence:** a learner bringing 12–128 records gets a refusal that points at the wrong record, or at none. The
transfer route the notebook promises ("same full baselines → adaptation → … → fresh reload") is hard to complete.

**Evidence (direct execution, CPU, synthetic):** probe P01. Valid synthetic BYOD passed `load_byod` + `prepare`. The
five invalid variants were refused with:
- `records[0]: only 2.0% of pixels carry valid depth…`, although the failing record was index 7 (`va01`);
- `missing file or file exceeds 32 MiB`;
- `array shape/dtype mismatch before allocation`;
- `explicit group_id required`;
- `group crosses roles`.

None named the failing record or group. Probe P02 (source inspection) found none of the eight contract items in §10.

**Recommended correction:**
- Prefix every BYOD refusal with the record id, or its index when the id is missing, and the file name.
- Report the expected versus the found shape and dtype.
- Run the per-record source contract in `load_byod`, before any child stage starts.
- State the full contract in §10: the id pattern, `units`, dtype, coverage, depth range, EXIF, an example manifest, how
  to supply the directory in Colab, and data locality.
- Show the BYOD summary tables in the BYOD cell.
- Make the summary's sample statement describe the BYOD split.

**Acceptance check:** P01 and P02 clear:
- valid synthetic BYOD still passes `prepare`;
- each of the five invalid variants is refused, before any model stage, with a message containing the failing record
  id (or the crossing group);
- the coverage refusal no longer says `records[0]`;
- §10 states all eight items;
- the BYOD cell displays the BYOD `summary.md`;
- the BYOD summary describes its own groups.

### DEP-m1 — Minor: infrastructure cells unlabelled and the 290 KB carried cell uncollapsed (GDL11, GDL12)

**Cell/section:** `code-02`, `code-03`, `code-04`.

**Observed issue:** `code-03` is a single 290,653-character dict literal and has no `cellView`. `code-04` is
`cellView: form` but has no `# @title`. None of the three cells is labelled Infrastructure. `md-01` says only that
"setup cells may be collapsed".

**Consequence:** the second thing a new learner sees is a wall of escaped source, with no sign that it is not lesson
content.

**Evidence:** source inspection (probe P03).

**Recommended correction:** start each of the three cells with `# @title Infrastructure: …`, set `cellView: form`, and
say in `md-01` that they may be run without study.

**Acceptance check:** P03 clears. Cells 2–4 begin with `# @title Infrastructure` and carry `cellView: form`, and the
validator still finds `CARRIED_FILES`.

### DEP-m2 — Minor: the scoring and alignment code is only readable as an escaped string (UX2; ST5 readability)

**Cell/section:** §3–§4. `support`, `score` and `aligned_values` live inside the `CARRIED_FILES['workshop.py']` string.

**Observed issue:** the prose describes AbsRel, delta-1, the fixed support and the affine fit precisely. A learner who
wants to trace a claim to code has to decode a string literal.

**Consequence:** "Claim → implementation" cannot be followed by the intended learner, so the metric lesson rests on
prose alone.

**Evidence:** source inspection (P04).

**Recommended correction:** add an optional learner cell that extracts these three functions from the carried worker
and prints them, with a sentence saying which line implements which rule.

**Acceptance check:** P04 clears. A non-infrastructure code cell prints `def support`, `def score` and
`def aligned_values` from `ROOT / 'workshop.py'`.

### DEP-m3 — Minor: the training evidence the notebook points to is not displayed; figure labels misdescribe a no-change run

**Cell/section:** `code-12` / `md-13`; the worker's `report()` figure selection.

**Observed issue:**
- `md-13` asks the learner to notice "pre-restoration parameter deltas". The cell prints only
  `selected_max_weight_delta`, which is 0.0 when epoch 0 is restored. The hosted run selected epoch 0.
- `report()` labels figures "best adaptation delta" and "worst adaptation delta" even when every test image is
  unchanged (hosted run: 20 of 20 unchanged). "Best" and "worst" then reduce to ordering by id.

**Consequence:** the learner sees 42 steps alongside a weight change of 0.0 and cannot confirm the notebook's statement
that training genuinely updated parameters. The figure-selection record also claims a ranking that does not exist.

**Evidence:** source inspection (P05). Documented execution: release-verification records the per-epoch delta
(3.7e-4) only in the CPU pre-flight notes.

**Recommended correction:**
- Add a per-epoch "max weight change before selection" column and a "steps so far" column to the training table.
- Factor figure selection into a function that labels zero-change runs honestly.

**Acceptance check:** P05 clears. The depth cell reads `epoch_observations[*].max_weight_delta_before_selection`.
`select_figures` returns no "best" or "worst" label when all deltas are within 1e-12, and keeps the same figure set.

### DEP-m4 — Minor: no usable download route; resource measurements never shown (RUN8, UX12)

**Cell/section:** `code-20`, `code-22`, `code-24`; `md-00`.

**Observed issue:**
- The cells use `FileLink(<absolute path>)`. IPython renders the absolute filesystem path as the link `href` (direct
  execution). In Colab, `/content/...` is not served, so the link does not download (inferred, not verified in Colab).
- The output folder is never named in prose.
- Peak GPU memory is recorded per stage in `stage_receipts.json` but is never displayed, and disk use is never
  measured. The release table asks for both, and the 2026-09-27 run could not capture them.
- `session_summary.json` is written after the ZIP is built.

**Consequence:** the learner's completion artifact is hard to retrieve. The release gate's resource measurements cannot
be read from a Run all.

**Evidence:** direct execution (FileLink), source inspection (P06), and documented execution ("peak memory and disk not
captured").

**Recommended correction:**
- Print absolute paths together with Colab Files-pane instructions.
- In the final cell, print per-stage seconds and peak GPU MiB from the receipts, plus the output tree's size and free
  disk.
- Save these values in `session_summary.json`.
- State the measured hosted wall time in the opening, labelled with its environment.

**Acceptance check:** P06 clears. The final cell shows `peak_gpu_bytes` and the disk figures, and gives a Files-pane
instruction next to each download.

### DEP-m5 — Minor: unlabelled-image path — limits unstated, misleading refusals, no prerequisite check, no visual output

**Cell/section:** `md-23`, `code-24`; the worker's `infer`, `bounded_image`.

**Observed issue:**
- §11 states no limits.
- A 4032×3024 phone photo is refused with "image must have sides 32..2048 and aspect <=4", which names neither the
  file nor its size.
- An EXIF-rotated photo is refused with "normalize EXIF orientation before providing references", although this path
  takes no references.
- Running it before §5 fails inside `from_artifact`.
- Only `.npy` files are produced, so no map is shown, even though the display loop includes `.png`.

**Consequence:** the most likely first attempt, a phone photo, fails without a clear remedy. A success shows nothing a
learner can look at.

**Evidence:** direct execution of `bounded_image` (P07) and source inspection.

**Recommended correction:**
- State the limits and remedies in §11.
- Make the messages name the file, its size and the fix.
- Check that the adapter exists, with a message telling the learner to run §5 first.
- Write and display a two-panel preview: relative inverse depth on a display scale, and metres on a log colour scale.

**Acceptance check:** P07 clears.

### DEP-m6 — Minor: "workshop" used for the notebook; ML jargon undefined, no glossary (GDL6, GDL15)

**Cell/section:** `md-05` ("a workshop regrouping"); `summary.md` heading "Depth workshop measured results"; `md-00`,
`md-11` (neck/head, backbone, adapter, AdamW, scale-and-shift-invariant loss).

**Observed issue:** the stated audience has no prior ML experience, but these terms are never defined, and the notebook
calls itself a workshop.

**Consequence:** the adaptation section reads as expert shorthand.

**Evidence:** source inspection (P08).

**Recommended correction:** add a "Key terms" glossary and replace "workshop" in learner-facing text.

**Acceptance check:** P08 clears. No learner-facing "workshop" remains apart from the mode token, and a glossary
defines neck, head, backbone, adapter, inverse depth, AbsRel, delta-1, affine alignment, AdamW, scale-and-shift-invariant
loss and parity.

### DEP-m7 — Minor: the controlled activity has no learner-chosen change (GDL10; promise "change input brightness")

**Cell/section:** §8 (`md-17`, `code-18`); the worker's `brightness()` and `activity()`.

**Observed issue:** the only form fields are the BYOD and unlabelled switches. The worker hard-codes
`(0.6, 1.0, 1.4)`, and `brightness()` raises on any other factor.

**Consequence:** "Predict → change one thing → run → observe → explain" becomes "observe a change someone else chose".
The opening says "You will … change input brightness".

**Evidence:** direct execution (`brightness(…, 0.8)` raises) and source inspection (P09).

**Recommended correction:** add an optional, off-by-default "choose your own factor" cell, bounded to [0.2, 2.0]. It
writes to its own directory, compares against the canonical factor-1.0 rows, and leaves canonical outputs and receipts
untouched. Update the opening wording to match.

**Acceptance check:**
- P09 clears.
- The canonical activity still uses exactly 0.6, 1.0 and 1.4.
- An out-of-range factor is refused before any model loads.
- The optional stage writes nothing under the canonical `outputs/` products.

### Suggestions (not release requirements)

- **DEP-S1:** an optional higher learning-rate or unfreeze exercise, so learners can see a non-zero adaptation change
  and a different selected epoch. The default deterministically keeps epoch 0.
- **DEP-S2:** a per-image error table for the worst test view, next to its figure.
- **DEP-S3:** record each stage's disk footprint in `stage_receipts.json`, not only in the session summary.

## 4. Readiness

**Needs revision.** DEP-M1 is open, and DAT12 and DAT19 (`MUST`) are partially unmet. The remaining gates after the fixes
are:
1. a fresh Colab T4 Run all of the fixed head;
2. a labelled BYOD run on representative real data with at least one clear refusal (REL12);
3. the unlabelled branch on a real photo;
4. reviewer confirmation before an integrator promotes the notebook.

The status stays **Candidate**.

**Verified versus inferred:**
- *Verified by direct CPU execution:* the BYOD `prepare` stage on synthetic data, the BYOD refusal texts, the
  `bounded_image` refusals and the FileLink href.
- *Verified from documented hosted evidence:* the default path (same blob).
- *Inferred:* that Colab does not serve FileLink absolute paths.
- *Not verified:* any model stage of the BYOD and unlabelled branches.

**The finding most likely to be wrong:** the m4 claim that the FileLink download does not work in Colab. It rests on
how IPython renders the link, not on a Colab observation.
