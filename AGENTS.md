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

### Running the model (train.py test/inference) on the CPU VM
Inference can be made to run on this CPU-only VM (slow: ~60-70s per 512x512 slice). This was verified end-to-end producing a denoised 3D NIfTI. Requirements:
1. Weights (downloaded from the README Google Drive folder via `gdown --folder <url>`): place `DA-CLIP.pth` at `src/DA-CLIP.pth` and `model-400.pt` at `checkpoints/FoundDiff/sample/model-400.pt`. These are gitignored (~2.5GB) and persist in the VM snapshot.
2. A CPU fallback for the mamba kernel: a root-level `selective_scan_cuda.py` providing `fwd(u,delta,A,B,C,D,z,delta_bias,delta_softplus) -> (out, last_state)` implemented as mamba_ssm's reference `selective_scan_ref` (sequential SSM scan in pure PyTorch). `src/emamba2.py` imports it when the CUDA `selective_scan_vmamba`/`mamba_ssm` extensions are absent (it sets `SSMODE="mamba_ssm"`). This file is **gitignored on purpose**: committing it would shadow the real CUDA extension on GPU machines. It persists in the VM snapshot.
3. Run on CPU: `CUDA_VISIBLE_DEVICES="" python train.py --name FoundDiff --epoch 400 --dataset 2020_seen --data-mode external [--max-test N]`. This reads `data/external/external_2d/test/{quarter_1mm,full_1mm}/lung-*.npy`, writes denoised slices to `checkpoints/FoundDiff/test_final_npy/`, then reconstruct with `reconstruct_denoised_nifti.py` (see below).
4. `--data-mode external` requires the slice filenames to start with `lung-` (the dataset infers the NDCT list from the `lung` prefix); the external preprocessor already names them that way.

Caveat: the model normalizes HU into a clipped soft-tissue window (`norm = clip((HU-24)/3000, 0, 1)`), so air (~-1000 HU) maps to 0 and is not recovered by the inverse `denorm = norm*3000+24`. Evaluate denoising in this normalized window, not raw-HU mean. The CPU reference scan is numerically equivalent to the CUDA kernel but has no backward (inference only; training still needs a GPU).

### Gotchas
- `Preprocess_nifti.py` has **no `if __name__ == "__main__"` guard**, so running it directly is a no-op. Invoke it via `run_external_pipeline.py`, or call its `main()` (it is wired up correctly inside `run_external_pipeline.py`).
- `Preprocess_custom_nifti.py` does `from data.paths import CUSTOM_2D, CUSTOM_NIFTI`, but those symbols only exist in the **root** `paths.py`, not in `data/paths.py` → `ImportError`. Prefer the external pipeline (`Preprocess_nifti.py` / `run_external_pipeline.py`), which is self-contained.
- `data/__init__.py` imports `torch` and `ipdb` at package-import time, so any `from data...` import needs the venv active.
- The full training requirements file (`requirements-founddiff-pip.txt`) pins `torch==2.11.0+cu128` and other GPU/CUDA wheels; do not install it on the CPU VM.
