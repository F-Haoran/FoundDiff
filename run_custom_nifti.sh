#!/usr/bin/env bash
# 自定义 nii.gz 预处理 + FoundDiff 推理
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate FoundDiff

MODE="${1:-quick}"   # quick = 试跑; full = 全部

echo "==> Preprocess nii.gz -> npy"
if [[ "$MODE" == "quick" ]]; then
  python Preprocess_custom_nifti.py --stride 2 --max-slices 30
  MAX_TEST="--max-test 5"
else
  python Preprocess_custom_nifti.py --stride 1
  MAX_TEST=""
fi

echo "==> Check weights"
test -f src/DA-CLIP.pth || { echo "Missing src/DA-CLIP.pth"; exit 1; }
test -f checkpoints/FoundDiff/sample/model-400.pt || { echo "Missing model-400.pt"; exit 1; }

echo "==> Inference"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python train.py \
  --name FoundDiff --epoch 400 --dataset 2020_seen \
  --data-mode custom $MAX_TEST

echo "==> Results: checkpoints/FoundDiff/test_final_npy/"
echo "==> View: python view_results.py --data-root data/custom/custom_2d"
