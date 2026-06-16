# FoundDiff

Official implementation of "FoundDiff: Foundational Diffusion Model for Generalizable Low-Dose CT Denoising".

## Cursor Cloud specific instructions

### What this repo is
A research codebase for low-dose CT denoising with a Mamba-based residual diffusion model (`train.py` → `src/DADiff.py` → `src/emamba2.py`). It also ships a CPU-only data pipeline: NIfTI/DICOM → 2D `.npy` slices → model → reconstruct back to NIfTI.

### GPU requirement (important)
Training and inference (`train.py`, `run_external_pipeline.py`, `run_custom_nifti.sh`) require a **CUDA GPU**: `src/emamba2.py` does `import selective_scan_cuda` (mamba-ssm), which has no CPU fallback and must be compiled with `nvcc` (see `ENV-REPRODUCE.txt` / `setup_new_machine.sh`). They also need downloaded weights `src/DA-CLIP.pth` and `checkpoints/FoundDiff/sample/model-400.pt` (Google Drive link in `README.md`). The default Cloud VM is **CPU-only**, so the model itself cannot run here; the codebase imports cleanly up to that `selective_scan_cuda` line. Do not assume a GPU is present.

### Dev environment
- A Python venv lives at `.venv` (the update script creates/refreshes it). Activate with `. .venv/bin/activate`. It holds the CPU-installable stack (CPU PyTorch + numpy/scipy/nibabel/pydicom/opencv/skimage/etc. and the model's import deps). It intentionally does **not** include `mamba-ssm`/`causal-conv1d` (GPU-only).
- `python3.12-venv` (system apt package) is required to create the venv.

### What runs on CPU (use this for verification)
The data pipeline runs without a GPU:
- Preprocess: `python Preprocess_nifti.py` (pairs `*_full.nii.gz` + `*_low.nii.gz` under `data/external/nifti/` → `data/external/external_2d/{test,train512}/{full_1mm,quarter_1mm}/lung-*.npy` + `slice_manifest.json`).
- Reconstruct: `python reconstruct_denoised_nifti.py --manifest ... --input-nii ... --denoised-dir ... --output ...` (stacks denoised `.npy` back into a `.nii.gz` using the manifest's z-index map).
- Visualize: `python view_results.py --data-root <dir> ...`.

### Gotchas
- `Preprocess_nifti.py` has **no `if __name__ == "__main__"` guard**, so running it directly is a no-op. Invoke it via `run_external_pipeline.py`, or call its `main()` (it is wired up correctly inside `run_external_pipeline.py`).
- `Preprocess_custom_nifti.py` does `from data.paths import CUSTOM_2D, CUSTOM_NIFTI`, but those symbols only exist in the **root** `paths.py`, not in `data/paths.py` → `ImportError`. Prefer the external pipeline (`Preprocess_nifti.py` / `run_external_pipeline.py`), which is self-contained.
- `data/__init__.py` imports `torch` and `ipdb` at package-import time, so any `from data...` import needs the venv active.
- The full training requirements file (`requirements-founddiff-pip.txt`) pins `torch==2.11.0+cu128` and other GPU/CUDA wheels; do not install it on the CPU VM.
