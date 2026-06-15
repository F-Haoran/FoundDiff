#!/usr/bin/env bash
# One-shot: your noisy .nii.gz -> FoundDiff pretrained inference (external mode).
#
# Works on stock FoundDiff (author/idc/external only — no "custom" needed).
# Your noisy volume is used as LDCT input (_low). _full is a copy for pairing only.
#
# Usage:
#   bash run_noisy_nifti.sh /path/to/your_scan.nii.gz
#   bash run_noisy_nifti.sh /path/to/your_scan.nii.gz mycase
#   bash run_noisy_nifti.sh /path/to/your_scan.nii.gz mycase full   # all slices, no --max-test
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

NIFTI="${1:?Usage: bash run_noisy_nifti.sh /path/to/noisy.nii.gz [case_name] [full]}"
CASE="${2:-case001}"
MODE="${3:-quick}"

if [[ ! -f "$NIFTI" ]]; then
  echo "File not found: $NIFTI" >&2
  exit 1
fi

# conda
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate FoundDiff
fi

EXT_NIFTI="$ROOT/data/external/nifti"
EXT_2D="$ROOT/data/external/external_2d"
mkdir -p "$EXT_NIFTI"

LOW="$EXT_NIFTI/${CASE}_low.nii.gz"
FULL="$EXT_NIFTI/${CASE}_full.nii.gz"

echo "==> Your noisy nii.gz -> external pair"
echo "    input:  $NIFTI"
echo "    LDCT:   $LOW"
echo "    placeholder full: $FULL"
cp -f "$NIFTI" "$LOW"
cp -f "$NIFTI" "$FULL"
ls -lh "$EXT_NIFTI"/${CASE}_*.nii.gz

echo "==> Preprocess nii.gz -> npy (external)"
if [[ "$MODE" == "full" ]]; then
  python Preprocess_nifti.py --nifti-dir "$EXT_NIFTI" --out-root "$EXT_2D" --stride 1
  MAX_TEST=""
else
  python Preprocess_nifti.py --nifti-dir "$EXT_NIFTI" --out-root "$EXT_2D" --stride 2 --max-slices 50
  MAX_TEST="--max-test 10"
fi

N_TEST="$(find "$EXT_2D/test/quarter_1mm" -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')"
N_TR512="$(find "$EXT_2D/train512/quarter_1mm" -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')"
echo "    test slices: $N_TEST  train512 slices: $N_TR512"
if [[ "$N_TEST" -eq 0 || "$N_TR512" -eq 0 ]]; then
  echo "Preprocess failed: empty test/ or train512/. Check nii.gz is 3D CT volume." >&2
  exit 1
fi

echo "==> Weights"
test -f src/DA-CLIP.pth || { echo "Missing src/DA-CLIP.pth" >&2; exit 1; }
test -f checkpoints/FoundDiff/sample/model-400.pt || { echo "Missing model-400.pt" >&2; exit 1; }

echo "==> Inference (--data-mode external)"
# Single line — no hidden Unicode spaces
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train.py --name FoundDiff --epoch 400 --dataset 2020_seen --data-mode external ${MAX_TEST}

echo ""
echo "Done."
echo "  Denoised npy:  checkpoints/FoundDiff/test_final_npy/"
echo "  Input LDCT npy: $EXT_2D/test/quarter_1mm/"
echo ""
echo "Quick view (if view_results.py exists):"
echo "  python view_results.py --data-root data/external/external_2d --save-hu-npy"
