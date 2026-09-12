# Weight provenance and DIMER hosting

- Upstream: `depth-anything/Depth-Anything-V2-Small-hf`
- Immutable revision: `5426e4f0f36572d16453bbda7a8389317b1bef99`
- Weight format: SafeTensors (`model.safetensors`, 99,173,660 bytes)
- Manifest: `weights/depth-anything-v2-small/dimer-base-manifest.json` (4 files, 99,179,785 bytes total, per-file SHA-256)
- Upstream weight license: Apache-2.0
- DIMER hosting: Apache-2.0 permits use, modification, distribution, and commercial use subject to preservation of the license and notices. The Git repository does not vendor the checkpoint (`weights/**/*.safetensors` is git-ignored); DIMER may mirror the pinned snapshot in its model store under the upstream license.
- Loader trust boundary: Transformers `AutoModelForDepthEstimation` / `AutoImageProcessor` with `trust_remote_code=False`, `local_files_only=True` from the verified directory; `verify_snapshot()` must pass before any load.
