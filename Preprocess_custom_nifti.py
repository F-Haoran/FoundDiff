#!/usr/bin/env python3
"""
Custom nii.gz -> FoundDiff npy.

Pair naming (default):
  {CODE}_LDCT.nii.gz  -> quarter_1mm (LDCT input)
  {CODE}_CT.nii.gz    -> full_1mm (NDCT reference)

Output: data/custom/custom_2d/{test,train512}/{quarter_1mm,full_1mm}/lung-XXXXX.npy
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from data.paths import CUSTOM_2D, CUSTOM_NIFTI


def parse_args():
    p = argparse.ArgumentParser(description="Custom CT nii.gz to FoundDiff npy")
    p.add_argument("--nifti-dir", type=Path, default=Path(CUSTOM_NIFTI))
    p.add_argument("--out-root", type=Path, default=Path(CUSTOM_2D))
    p.add_argument("--ldct-suffix", default="_LDCT", help="Low-dose: {CODE}{suffix}.nii.gz")
    p.add_argument("--ndct-suffix", default="_CT", help="Full-dose: {CODE}{suffix}.nii.gz")
    p.add_argument("--max-slices", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--test-ratio", type=float, default=0.2)
    p.add_argument(
        "--simulate-noise",
        type=float,
        default=0.0,
        help="If no LDCT file, add Gaussian noise (HU std) to NDCT as fake LDCT",
    )
    return p.parse_args()


def load_nifti_hu(path: Path) -> np.ndarray:
    import nibabel as nib

    vol = np.asarray(nib.load(str(path)).dataobj, dtype=np.float32)
    if vol.ndim == 3:
        vol = np.transpose(vol, (2, 1, 0))
    return vol


def crop512(slice2d: np.ndarray) -> np.ndarray:
    h, w = slice2d.shape
    out = np.zeros((512, 512), dtype=np.float32)
    sh, sw = min(h, 512), min(w, 512)
    y0, x0 = (512 - sh) // 2, (512 - sw) // 2
    sy0, sx0 = (h - sh) // 2, (w - sw) // 2
    out[y0 : y0 + sh, x0 : x0 + sw] = slice2d[sy0 : sy0 + sh, sx0 : sx0 + sw]
    return out[np.newaxis, ...]


def find_pairs(nifti_dir: Path, ldct_suffix: str, ndct_suffix: str, simulate_noise: float):
    pairs = []
    for ndct in sorted(nifti_dir.glob(f"*{ndct_suffix}.nii.gz")):
        code = ndct.name[: -len(ndct_suffix) - len(".nii.gz")]
        ldct = nifti_dir / f"{code}{ldct_suffix}.nii.gz"
        if ldct.is_file():
            pairs.append((code, ldct, ndct, False))
        elif simulate_noise > 0:
            pairs.append((code, None, ndct, True))
        else:
            print(f"skip {code}: missing {ldct.name} (use --simulate-noise if only full-dose)")
    return pairs


def main():
    args = parse_args()
    pairs = find_pairs(args.nifti_dir, args.ldct_suffix, args.ndct_suffix, args.simulate_noise)
    if not pairs:
        raise SystemExit(f"No pairs in {args.nifti_dir}")

    n_pairs = len(pairs)
    n_test = max(1, int(round(n_pairs * args.test_ratio)))
    bootstrap = []
    total = 0
    rng = np.random.default_rng(0)

    for pi, (code, ldct_path, ndct_path, simulate) in enumerate(pairs):
        phase = "test" if pi >= n_pairs - n_test else "train"
        phases = [phase, "train512"] if phase == "train" else [phase]

        full_vol = load_nifti_hu(ndct_path)
        if simulate:
            low_vol = full_vol + rng.normal(0, args.simulate_noise, full_vol.shape).astype(np.float32)
        else:
            low_vol = load_nifti_hu(ldct_path)

        n = min(full_vol.shape[0], low_vol.shape[0])
        if args.max_slices:
            n = min(n, args.max_slices)

        slice_idx = 0
        for i in range(0, n, args.stride):
            fname = f"lung-{slice_idx:05d}.npy"
            slice_idx += 1
            low_sl = crop512(low_vol[i])
            full_sl = crop512(full_vol[i])
            if phase == "test" and len(bootstrap) < 2:
                bootstrap.append((fname, low_sl, full_sl))
            for ph in phases:
                for sub, arr in (("quarter_1mm", low_sl), ("full_1mm", full_sl)):
                    out = args.out_root / ph / sub / fname
                    out.parent.mkdir(parents=True, exist_ok=True)
                    np.save(out, arr)
                    total += 1
        tag = "sim-LDCT" if simulate else "paired"
        print(f"{code}: {slice_idx} slices ({tag}) -> {phase}")

    for fname, low_sl, full_sl in bootstrap:
        for sub, arr in (("quarter_1mm", low_sl), ("full_1mm", full_sl)):
            out = args.out_root / "train512" / sub / fname
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, arr)
            total += 1

    print(f"Done: {total} npy under {args.out_root}")


if __name__ == "__main__":
    main()
