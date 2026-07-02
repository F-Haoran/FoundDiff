#!/usr/bin/env bash
# FoundDiff setup on a new machine (requires network). Usage: bash setup_new_machine.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "==> Python env: FoundDiff (create if missing)"
if ! conda env list | grep -q '^FoundDiff '; then
  conda create -n FoundDiff python=3.10.20 -y
fi
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FoundDiff

echo "==> PyTorch cu128 (adjust index-url if not RTX 50-series)"
pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

echo "==> numpy/scipy + pip deps"
pip install "numpy>=2.0,<2.3" scipy==1.15.3
pip install -r requirements-founddiff-pip.txt 2>/dev/null || pip install \
  accelerate einops ema-pytorch kornia lpips timm matplotlib opencv-python \
  pydicom nibabel ipdb tqdm pillow scikit-image

echo "==> CLIP"
pip install git+https://github.com/openai/CLIP.git

echo "==> nvcc for mamba compile"
conda install -y -c nvidia cuda-nvcc=12.8 || true

echo "==> mamba-ssm (set TORCH_CUDA_ARCH_LIST for your GPU: 12.0=RTX50, 8.0=A100)"
export CUDA_HOME="${CONDA_PREFIX}"
export PATH="${CONDA_PREFIX}/bin:${PATH}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-12.0}"
pip install ninja packaging
pip install "causal-conv1d>=1.2.0" --no-build-isolation
pip install mamba-ssm --no-build-isolation || echo "WARN: mamba compile failed; fix nvcc/arch and retry"

echo "==> Re-pin torch if mamba upgraded it"
pip install torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu128

mkdir -p data/custom/nifti data/custom/custom_2d
mkdir -p checkpoints/FoundDiff/sample src

echo "==> Verify (place weights before running train.py)"
python - <<'PY'
import torch
print("torch", torch.__version__, "cuda", torch.cuda.is_available())
try:
    import selective_scan_cuda
    print("selective_scan_cuda OK")
except Exception as e:
    print("selective_scan_cuda FAIL:", e)
from data.paths import CUSTOM_NIFTI, CUSTOM_2D
print("CUSTOM_NIFTI", CUSTOM_NIFTI)
print("CUSTOM_2D", CUSTOM_2D)
PY

echo ""
echo "Next:"
echo "  1) Put src/DA-CLIP.pth and checkpoints/FoundDiff/sample/model-400.pt"
echo "  2) Put nii.gz in data/custom/nifti/  (exp001_LDCT.nii.gz + exp001_CT.nii.gz)"
echo "  3) bash run_custom_nifti.sh"
