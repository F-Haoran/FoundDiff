#!/usr/bin/env bash
# Noisy .nii.gz -> full-volume FoundDiff inference -> denoised .nii.gz
#
# Usage:
#   bash run_noisy_nifti.sh /path/to/noisy.nii.gz
#   bash run_noisy_nifti.sh /path/to/noisy.nii.gz case001
#   bash run_noisy_nifti.sh /path/to/noisy.nii.gz case001 quick   # subset only (10 slices infer)
#
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

NIFTI="${1:?Usage: bash run_noisy_nifti.sh /path/to/noisy.nii.gz [case_name] [quick]}"
CASE="${2:-case001}"
MODE="${3:-full}"

if [[ ! -f "$NIFTI" ]]; then
  echo "File not found: $NIFTI" >&2
  exit 1
fi

if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate FoundDiff
fi

EXT_NIFTI="$ROOT/data/external/nifti"
EXT_2D="$ROOT/data/external/external_2d"
MANIFEST="$EXT_2D/slice_manifest.json"
mkdir -p "$EXT_NIFTI"

LOW="$EXT_NIFTI/${CASE}_low.nii.gz"
FULL="$EXT_NIFTI/${CASE}_full.nii.gz"

echo "==> Input nii.gz"
echo "    $NIFTI"
cp -f "$NIFTI" "$LOW"
cp -f "$NIFTI" "$FULL"

if [[ "$MODE" == "quick" ]]; then
  PRE_STRIDE=2
  PRE_MAX=50
  MAX_TEST="--max-test 10"
  echo "==> QUICK mode: stride=2 max-slices=50 max-test=10"
else
  PRE_STRIDE=1
  PRE_MAX=0
  MAX_TEST=""
  echo "==> FULL mode: all axial slices, no max-test, reconstruct nii.gz"
fi

echo "==> Preprocess -> npy"
PRE_ARGS=(--nifti-dir "$EXT_NIFTI" --out-root "$EXT_2D" --stride "$PRE_STRIDE" --test-ratio 1.0 --clean)
if [[ "$PRE_MAX" -gt 0 ]]; then
  PRE_ARGS+=(--max-slices "$PRE_MAX")
fi
python Preprocess_nifti.py "${PRE_ARGS[@]}"

N_TEST="$(find "$EXT_2D/test/quarter_1mm" -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')"
N_TR512="$(find "$EXT_2D/train512/quarter_1mm" -name '*.npy' 2>/dev/null | wc -l | tr -d ' ')"
echo "    test slices: $N_TEST  train512: $N_TR512"
if [[ "$N_TEST" -eq 0 || "$N_TR512" -eq 0 ]]; then
  echo "Preprocess failed." >&2
  exit 1
fi

test -f src/DA-CLIP.pth || { echo "Missing src/DA-CLIP.pth" >&2; exit 1; }
test -f checkpoints/FoundDiff/sample/model-400.pt || { echo "Missing model-400.pt" >&2; exit 1; }

echo "==> Inference"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train.py --name FoundDiff --epoch 400 --dataset 2020_seen --data-mode external ${MAX_TEST}

OUT_NII="$ROOT/checkpoints/FoundDiff/${CASE}_denoised.nii.gz"
echo "==> Reconstruct nii.gz"
RECON_ARGS=(--manifest "$MANIFEST" --input-nii "$LOW" --denoised-dir "$ROOT/checkpoints/FoundDiff/test_final_npy" --output "$OUT_NII" --volume-name "$CASE")
python reconstruct_denoised_nifti.py "${RECON_ARGS[@]}"

echo ""
echo "Done."
echo "  2D denoised:  checkpoints/FoundDiff/test_final_npy/  ($N_TEST slices expected in full mode)"
echo "  3D denoised:  $OUT_NII"
echo "  View PNG:     python view_results.py --data-root data/external/external_2d --save-hu-npy"
