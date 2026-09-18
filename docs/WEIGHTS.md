# Weight provenance and DIMER hosting

- Upstream: `depth-anything/Depth-Anything-V2-Small-hf`
- Immutable revision: `5426e4f0f36572d16453bbda7a8389317b1bef99`
- Weight format: SafeTensors (`model.safetensors`, 99,173,660 bytes)
- Manifest: `weights/depth-anything-v2-small/dimer-base-manifest.json` (4 files, 99,179,785 bytes total, per-file SHA-256)
- Upstream weight license: Apache-2.0
- DIMER hosting: Apache-2.0 permits use, modification, distribution, and commercial use subject to preservation of the license and notices. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror the pinned snapshot in its model store under the upstream license.
- Loader trust boundary: Transformers `AutoModelForDepthEstimation` / `AutoImageProcessor` with `trust_remote_code=False`, `local_files_only=True` from the verified directory; `verify_snapshot()` must pass before any load.

## Adaptation corpus (tutorial data, not weights)

- Corpus: 40 views from the DIODE validation release (Vasiljevic et al., 2019; CC BY 4.0) — the first two views in file-name order of each of its 20 laser-scanned scans (10 indoor, 10 outdoor), selected a priori on 2026-09-19 by the fetcher recorded in the build ledger.
- Pins: `samples.SAMPLE_RECORDS` holds, per view, the record id, domain, scene, scan, file stem, and the byte size and SHA-256 of each of three files — the RGB PNG (1024 × 768), the depth `.npy` (float32 metres) and the validity-mask `.npy` — as served by the Marigold evaluation mirror `https://huggingface.co/datasets/obukhovai/marigold_depth_eval` at commit `30c5b061` (the full 40-hex commit is pinned in `samples.CORPUS_COMMIT`) (312,448,846 bytes in total, no credential). `samples.fetch_corpus` downloads each file at run time into the git-ignored `weights/diode-sample/` cache and refuses any byte or digest mismatch; `read_corpus` decodes them without re-encoding. The repository redistributes none of the files; every record keeps its source URL. Byte-identity with the original DIODE tarball has not been checked — the pin is to the served mirror files.
- Sample: `build_sample_dataset(seed=42)` draws 6 / 2 / 2 whole scans per domain (24 / 8 / 8 views) by a seeded shuffle; `check_split_disjoint` asserts no view (by decoded-pixel digest) and no scan is in two splits, `scan_summary` reports scans and domains per split. DIMER hosting of the weights is unaffected.
- Adapter artifacts written by the tutorial (`outputs/depth_anything_depth_estimation_adapter/`, `org.valcorza.depth-anything-v2-small.adapter.v1`, 10.9 MB for the DPT neck and head, about 25.1 MB with two trained blocks) carry the trained tensors with a manifest naming the base `model.safetensors` digest and the selected policy; they are outputs, not hosted weights.
