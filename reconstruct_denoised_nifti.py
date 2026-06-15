#!/usr/bin/env python3
"""
Stack FoundDiff 2D denoised .npy slices back into a 3D .nii.gz volume.

Must use the same --stride / --max-slices as Preprocess_nifti.py.
Unprocessed slices (stride gaps) keep values from the input nii.gz.

Example:
  python reconstruct_denoised_nifti.py \\
    --input-nii data/external/nifti/case001_low.nii.gz \\
    --denoised-dir checkpoints/FoundDiff/test_final_npy \\
    --output checkpoints/FoundDiff/case001_denoised.nii.gz \\
    --stride 1
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np


HU_MIN = -1000
HU_MAX = 2000
HU_RANGE = HU_MAX - HU_MIN


def parse_args():
    p = argparse.ArgumentParser(description="Reconstruct denoised npy slices -> nii.gz")
    p.add_argument("--input-nii", type=Path, required=True, help="Original noisy nii.gz (for shape/affine)")
    p.add_argument(
        "--denoised-dir",
        type=Path,
        default=Path("checkpoints/FoundDiff/test_final_npy"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--stride", type=int, default=1, help="Same as Preprocess_nifti.py --stride")
    p.add_argument("--max-slices", type=int, default=0, help="Same as Preprocess_nifti.py --max-slices (0=all)")
    p.add_argument(
        "--prefix",
        default="lung",
        help="Slice filename prefix (default lung from lung-00000.npy)",
    )
    return p.parse_args()


def denorm_to_hu(norm_0_1: np.ndarray) -> np.ndarray:
    return norm_0_1.astype(np.float32) * HU_RANGE + HU_MIN + 1024


def load_nifti_z_y_x(path: Path) -> tuple[np.ndarray, object]:
    import nibabel as nib

    img = nib.load(str(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    if vol.ndim == 3:
        vol = np.transpose(vol, (2, 1, 0))  # z, y, x
    return vol, img


def embed_denoised(slice_hu: np.ndarray, denoised_norm: np.ndarray) -> np.ndarray:
    """Inverse of Preprocess_nifti crop512: paste 512x512 denoised HU into slice."""
    den = denorm_to_hu(np.squeeze(denoised_norm))
    h, w = slice_hu.shape
    out = slice_hu.copy()
    sh, sw = min(h, 512), min(w, 512)
    y0 = max((512 - sh) // 2, 0)
    x0 = max((512 - sw) // 2, 0)
    sy0 = max((h - sh) // 2, 0)
    sx0 = max((w - sw) // 2, 0)
    out[sy0 : sy0 + sh, sx0 : sx0 + sw] = den[y0 : y0 + sh, x0 : x0 + sw]
    return out


def sorted_denoised_files(denoised_dir: Path, prefix: str) -> list[Path]:
    pat = re.compile(rf"^{re.escape(prefix)}-(\d+)\.npy$")
    files = []
    for p in denoised_dir.glob("*.npy"):
        m = pat.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


def main():
    args = parse_args()
    denoised = sorted_denoised_files(args.denoised_dir, args.prefix)
    if not denoised:
        raise SystemExit(f"No {args.prefix}-*.npy in {args.denoised_dir}")

    vol_zyx, ref_img = load_nifti_z_y_x(args.input_nii)
    nz = vol_zyx.shape[0]
    if args.max_slices:
        nz = min(nz, args.max_slices)

    out_vol = vol_zyx.copy()
    n_written = 0
    for idx, dpath in enumerate(denoised):
        z = idx * args.stride
        if z >= nz:
            break
        out_vol[z] = embed_denoised(vol_zyx[z], np.load(dpath))
        n_written += 1

    import nibabel as nib

    out_xyz = np.transpose(out_vol, (2, 1, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out_xyz.astype(np.float32), ref_img.affine, ref_img.header), str(args.output))

    print(f"Input volume Z={vol_zyx.shape[0]}  preprocess cap Z={nz}  stride={args.stride}")
    print(f"Denoised slices found: {len(denoised)}  written into volume: {n_written}")
    if n_written < (nz + args.stride - 1) // args.stride:
        print("Warning: fewer denoised npy than expected — run train.py without --max-test?")
    if args.stride > 1:
        print(f"Note: slices between denoised layers still show original noisy HU (stride={args.stride})")
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
