#!/usr/bin/env python3
"""
Convert paired .nii.gz CT volumes to FoundDiff 2D .npy layout.

Expects files like:
  data/external/nifti/mayo_c002_full.nii.gz
  data/external/nifti/mayo_c002_low.nii.gz

Writes:
  data/external/external_2d/{test,train512}/{full_1mm,quarter_1mm}/lung-{idx:05d}.npy
  data/external/external_2d/slice_manifest.json   (for reconstruct_denoised_nifti.py)

Axial slices are center-cropped/padded to 512x512, shape (1,512,512) float32 HU.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

EXTERNAL_NIFTI = Path(__file__).resolve().parent / "data" / "external" / "nifti"
EXTERNAL_2D = Path(__file__).resolve().parent / "data" / "external" / "external_2d"
MANIFEST_NAME = "slice_manifest.json"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--nifti-dir", type=Path, default=EXTERNAL_NIFTI)
    p.add_argument("--out-root", type=Path, default=EXTERNAL_2D)
    p.add_argument("--max-slices", type=int, default=0, help="Limit slices per volume (0=all)")
    p.add_argument("--test-ratio", type=float, default=1.0, help="Fraction of cases -> test (default 1.0 = all cases test)")
    p.add_argument("--stride", type=int, default=1, help="Use every Nth axial slice (1=all slices)")
    p.add_argument(
        "--bootstrap-train512",
        action="store_true",
        default=True,
        help="Copy first test slices to train512 so Trainer init works (default on)",
    )
    p.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing npy + manifest under --out-root before writing",
    )
    return p.parse_args()


def load_nifti_hu(path: Path) -> np.ndarray:
    import nibabel as nib

    vol = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
    if vol.ndim == 3:
        vol = np.transpose(vol, (2, 1, 0))
    return vol


def crop512(slice2d: np.ndarray) -> np.ndarray:
    h, w = slice2d.shape
    out = np.zeros((512, 512), dtype=np.float32)
    sh, sw = min(h, 512), min(w, 512)
    y0 = max((512 - sh) // 2, 0)
    x0 = max((512 - sw) // 2, 0)
    sy0 = max((h - sh) // 2, 0)
    sx0 = max((w - sw) // 2, 0)
    out[y0 : y0 + sh, x0 : x0 + sw] = slice2d[sy0 : sy0 + sh, sx0 : sx0 + sw]
    return out[np.newaxis, ...]


def find_pairs(nifti_dir: Path):
    pairs = []
    for full in sorted(nifti_dir.glob("*_full.nii.gz")):
        low = full.with_name(full.name.replace("_full", "_low"))
        if low.is_file():
            name = full.name.replace("_full.nii.gz", "")
            pairs.append((name, full, low))
    return pairs


def main():
    args = parse_args()
    pairs = find_pairs(args.nifti_dir)
    if not pairs:
        raise SystemExit(
            f"No *_full.nii.gz / *_low.nii.gz pairs in {args.nifti_dir}\n"
            "Example: case001_low.nii.gz + case001_full.nii.gz"
        )

    if args.clean and args.out_root.exists():
        for sub in ("test", "train", "train512"):
            p = args.out_root / sub
            if p.is_dir():
                shutil.rmtree(p)
        manifest = args.out_root / MANIFEST_NAME
        if manifest.is_file():
            manifest.unlink()

    n_pairs = len(pairs)
    n_test = max(1, int(round(n_pairs * args.test_ratio)))
    test_names = {pairs[i][0] for i in range(n_pairs - n_test, n_pairs)}

    total = 0
    bootstrap_slices = []
    manifest = {"stride": args.stride, "max_slices": args.max_slices, "volumes": []}
    global_slice_idx = 0

    for name, full_path, low_path in pairs:
        phase = "test" if name in test_names else "train"
        phases = [phase, "train512"] if phase == "train" else [phase]

        full_vol = load_nifti_hu(full_path)
        low_vol = load_nifti_hu(low_path)
        n = min(full_vol.shape[0], low_vol.shape[0])
        if args.max_slices:
            n = min(n, args.max_slices)

        vol_entry = {
            "name": name,
            "low_nii": str(low_path.resolve()),
            "full_nii": str(full_path.resolve()),
            "num_slices": n,
            "slices": [],
        }

        slice_idx = 0
        for z in range(0, n, args.stride):
            fname = f"lung-{global_slice_idx:05d}.npy"
            global_slice_idx += 1
            slice_idx += 1
            full_sl = crop512(full_vol[z])
            low_sl = crop512(low_vol[z])
            vol_entry["slices"].append({"fname": fname, "z": int(z), "phase": phase})
            if phase == "test" and len(bootstrap_slices) < 2:
                bootstrap_slices.append((fname, full_sl, low_sl))
            for ph in phases:
                for sub, arr in (("full_1mm", full_sl), ("quarter_1mm", low_sl)):
                    out = args.out_root / ph / sub / fname
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out, arr)
                    total += 1
        manifest["volumes"].append(vol_entry)
        print(f"{name}: {len(vol_entry['slices'])} slices (z=0..{n-1}, stride={args.stride}) -> {phase}")

    if args.bootstrap_train512 and bootstrap_slices:
        for fname, full_sl, low_sl in bootstrap_slices:
            for sub, arr in (("full_1mm", full_sl), ("quarter_1mm", low_sl)):
                out = args.out_root / "train512" / sub / fname
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, arr)
                total += 1
        print(f"Bootstrap train512: {len(bootstrap_slices)} slice pairs")

    manifest_path = args.out_root / MANIFEST_NAME
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {manifest_path}")
    print(f"Wrote {total} npy files under {args.out_root}")


if __name__ == "__main__":
    main()
