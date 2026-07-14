#!/usr/bin/env python3
"""Convert a NIfTI volume to 512x512 .npy slices for FoundDiff."""

import argparse
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from skimage.transform import resize


def load_volume(nii_path):
    image = sitk.ReadImage(str(nii_path))
    arr = sitk.GetArrayFromImage(image).astype(np.float32)
    return arr


def to_hu(arr):
    """Keep values as HU if they already look like CT; otherwise return as-is."""
    if arr.min() >= 0 and arr.max() <= 1.0:
        return arr * 3000.0 - 1000.0
    return arr


def resize_slice(slice_2d, size=512):
    return resize(
        slice_2d,
        (size, size),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def main():
    parser = argparse.ArgumentParser(description='Convert NIfTI to FoundDiff .npy slices')
    parser.add_argument('--nii', required=True, help='Input .nii or .nii.gz path')
    parser.add_argument('--out_dir', required=True, help='Output directory for .npy files')
    parser.add_argument('--prefix', default='slice', help='Output filename prefix')
    parser.add_argument('--size', type=int, default=512, help='Output spatial size')
    parser.add_argument(
        '--pick_best',
        action='store_true',
        help='Save only the slice with the highest mean HU (usually best anatomy)',
    )
    args = parser.parse_args()

    volume = to_hu(load_volume(args.nii))
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best_idx, best_mean = 0, volume[0].mean()
    for z in range(volume.shape[0]):
        sl = resize_slice(volume[z], size=args.size)
        mean_hu = float(sl.mean())
        print(f'z={z:04d}: mean={mean_hu:.2f}, min={sl.min():.2f}, max={sl.max():.2f}')
        if mean_hu > best_mean:
            best_idx, best_mean = z, mean_hu

        if not args.pick_best:
            out_path = out_dir / f'{args.prefix}-{z:04d}.npy'
            np.save(out_path, sl)

    if args.pick_best:
        sl = resize_slice(volume[best_idx], size=args.size)
        out_path = out_dir / f'{args.prefix}-best-{best_idx:04d}.npy'
        np.save(out_path, sl)
        print(f'Saved best slice z={best_idx} (mean={best_mean:.2f}) -> {out_path}')
    else:
        print(f'Saved {volume.shape[0]} slices to {out_dir}')


if __name__ == '__main__':
    main()
