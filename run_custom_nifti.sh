#!/usr/bin/env bash
# Custom nii.gz: full volume preprocess + infer + reconstruct nii.gz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FoundDiff

MODE="${1:-full}"
CASE="${2:-case001}"

if [[ "$MODE" == "quick" ]]; then
  python Preprocess_custom_nifti.py --stride 2 --max-slices 30 --clean
  MAX_TEST="--max-test 5"
else
  python Preprocess_custom_nifti.py --stride 1 --test-ratio 1.0 --clean
  MAX_TEST=""
fi

test -f src/DA-CLIP.pth || { echo "Missing src/DA-CLIP.pth"; exit 1; }
test -f checkpoints/FoundDiff/sample/model-400.pt || { echo "Missing model-400.pt"; exit 1; }

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train.py \
  --name FoundDiff --epoch 400 --dataset 2020_seen \
  --data-mode custom ${MAX_TEST}

MANIFEST="$ROOT/data/custom/custom_2d/slice_manifest.json"
LDCT="$ROOT/data/custom/nifti/${CASE}_LDCT.nii.gz"
OUT="$ROOT/checkpoints/FoundDiff/${CASE}_denoised.nii.gz"
if [[ -f "$MANIFEST" && -f "$LDCT" ]]; then
  python reconstruct_denoised_nifti.py \
    --manifest "$MANIFEST" \
    --input-nii "$LDCT" \
    --denoised-dir "$ROOT/checkpoints/FoundDiff/test_final_npy" \
    --output "$OUT" \
    --volume-name "$CASE"
  echo "3D output: $OUT"
fi

echo "2D: checkpoints/FoundDiff/test_final_npy/"
echo "View: python view_results.py --data-root data/custom/custom_2d --save-hu-npy"
