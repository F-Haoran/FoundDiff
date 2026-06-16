#!/usr/bin/env python3
"""
Stack FoundDiff 2D denoised .npy slices back into a 3D .nii.gz volume.

Prefer slice_manifest.json from Preprocess_nifti.py (exact z-index mapping).
Fallback: --stride + sequential lung-00000.npy order.

Example:
  python reconstruct_denoised_nifti.py \\
    --manifest data/external/external_2d/slice_manifest.json \\
    --input-nii data/external/nifti/case001_low.nii.gz \\
    --denoised-dir checkpoints/FoundDiff/test_final_npy \\
    --output checkpoints/FoundDiff/case001_denoised.nii.gz
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

HU_MIN = -1000
HU_MAX = 2000
HU_RANGE = HU_MAX - HU_MIN


def parse_args():
    p = argparse.ArgumentParser(description="Reconstruct denoised npy slices -> nii.gz")
    p.add_argument("--input-nii", type=Path, required=True, help="Original noisy nii.gz (shape/affine)")
    p.add_argument(
        "--denoised-dir",
        type=Path,
        default=Path("checkpoints/FoundDiff/test_final_npy"),
    )
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="slice_manifest.json from Preprocess_nifti.py (recommended)",
    )
    p.add_argument("--volume-name", type=str, default=None, help="Case name in manifest (default: auto from input-nii)")
    p.add_argument("--stride", type=int, default=1, help="Fallback if no manifest")
    p.add_argument("--max-slices", type=int, default=0, help="Fallback if no manifest")
    p.add_argument("--prefix", default="lung", help="Slice filename prefix")
    return p.parse_args()


def denorm_to_hu(norm_0_1: np.ndarray) -> np.ndarray:
    return norm_0_1.astype(np.float32) * HU_RANGE + HU_MIN + 1024


def load_nifti_z_y_x(path: Path) -> tuple[np.ndarray, object]:
    import nibabel as nib

    img = nib.load(str(path))
    vol = np.asarray(img.dataobj, dtype=np.float32)
    if vol.ndim == 3:
        vol = np.transpose(vol, (2, 1, 0))
    return vol, img


def embed_denoised(slice_hu: np.ndarray, denoised_norm: np.ndarray) -> np.ndarray:
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


def sorted_denoised_files(denoised_dir: Path, prefix: str) -> list[tuple[int, Path]]:
    pat = re.compile(rf"^{re.escape(prefix)}-(\d+)\.npy$")
    files = []
    for p in denoised_dir.glob("*.npy"):
        m = pat.match(p.name)
        if m:
            files.append((int(m.group(1)), p))
    files.sort(key=lambda x: x[0])
    return files


def load_manifest_mapping(args) -> list[tuple[str, int]] | None:
    if args.manifest is None or not args.manifest.is_file():
        return None
    with open(args.manifest, encoding="utf-8") as f:
        data = json.load(f)

    vol_name = args.volume_name
    if vol_name is None:
        stem = args.input_nii.name.replace("_low.nii.gz", "").replace(".nii.gz", "")
        vol_name = stem

    for vol in data.get("volumes", []):
        if vol.get("name") == vol_name:
            test_slices = [(s["fname"], int(s["z"])) for s in vol["slices"] if s.get("phase") == "test"]
            if test_slices:
                return test_slices
            return [(s["fname"], int(s["z"])) for s in vol["slices"]]
    raise SystemExit(f"Volume '{vol_name}' not found in manifest {args.manifest}")


def main():
    args = parse_args()
    vol_zyx, ref_img = load_nifti_z_y_x(args.input_nii)
    out_vol = vol_zyx.copy()

    mapping = load_manifest_mapping(args)
    if mapping:
        n_written = 0
        n_missing = 0
        for fname, z in mapping:
            dpath = args.denoised_dir / fname
            if not dpath.is_file():
                n_missing += 1
                continue
            if z < 0 or z >= out_vol.shape[0]:
                raise SystemExit(f"Manifest z={z} out of range [0, {out_vol.shape[0]-1}] for {fname}")
            out_vol[z] = embed_denoised(vol_zyx[z], np.load(dpath))
            n_written += 1
        print(f"Manifest slices: {len(mapping)}  denoised written: {n_written}  missing npy: {n_missing}")
        if n_missing:
            print("Warning: re-run train.py WITHOUT --max-test to denoise all slices")
    else:
        denoised = sorted_denoised_files(args.denoised_dir, args.prefix)
        if not denoised:
            raise SystemExit(f"No {args.prefix}-*.npy in {args.denoised_dir}")
        nz = vol_zyx.shape[0]
        if args.max_slices:
            nz = min(nz, args.max_slices)
        n_written = 0
        for idx, (_, dpath) in enumerate(denoised):
            z = idx * args.stride
            if z >= nz:
                break
            out_vol[z] = embed_denoised(vol_zyx[z], np.load(dpath))
            n_written += 1
        print(f"Fallback mode: stride={args.stride}  written: {n_written}/{len(denoised)}")

    import nibabel as nib

    out_xyz = np.transpose(out_vol, (2, 1, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out_xyz.astype(np.float32), ref_img.affine, ref_img.header), str(args.output))
    print(f"Input Z={vol_zyx.shape[0]}  output: {args.output}")


if __name__ == "__main__":
    main()
