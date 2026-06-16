#!/usr/bin/env python3
"""
CPU-safe denoising for 3D NIfTI .nii.gz volumes.

This utility is useful when you need a Python-only path from compressed CT
volumes to denoised compressed 3D outputs without running the CUDA FoundDiff
model. It preserves the input affine and NIfTI header geometry.

Examples:
  python denoise_nifti_gz.py data/external/nifti/case001_low.nii.gz
  python denoise_nifti_gz.py case001_low.nii.gz --output case001_denoised.nii.gz
  python denoise_nifti_gz.py *.nii.gz --output-dir denoised --method median
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable

import numpy as np


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Denoise 3D .nii/.nii.gz volumes and save compressed .nii.gz outputs."
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="Input .nii or .nii.gz files")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output .nii.gz path. Only valid with one input.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for denoised outputs. Defaults to each input's directory.",
    )
    parser.add_argument(
        "--suffix",
        default="_denoised",
        help="Suffix used when --output is omitted (default: _denoised).",
    )
    parser.add_argument(
        "--method",
        choices=("gaussian", "median", "nlmeans"),
        default="gaussian",
        help="Denoising method. gaussian/median use SciPy; nlmeans uses scikit-image.",
    )
    parser.add_argument(
        "--sigma",
        default="0.6,0.6,0.6",
        help="Gaussian sigma in voxels: one value or z,y,x/native-axis triple.",
    )
    parser.add_argument(
        "--median-size",
        default="3",
        help="Median filter size in voxels: one odd value or native-axis triple.",
    )
    parser.add_argument("--nlm-patch-size", type=int, default=3, help="nlmeans patch size")
    parser.add_argument("--nlm-patch-distance", type=int, default=5, help="nlmeans search distance")
    parser.add_argument(
        "--nlm-h",
        type=float,
        default=0.0,
        help="nlmeans filter strength. 0 estimates noise and uses 0.8 * sigma.",
    )
    parser.add_argument(
        "--clip-min",
        type=float,
        default=None,
        help="Optional lower intensity bound applied before and after denoising.",
    )
    parser.add_argument(
        "--clip-max",
        type=float,
        default=None,
        help="Optional upper intensity bound applied before and after denoising.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow replacing an existing output file.",
    )
    return parser.parse_args(argv)


def parse_numeric_tuple(raw: str, *, name: str, integer: bool = False) -> tuple[float, ...] | tuple[int, ...]:
    values = [part.strip() for part in raw.split(",") if part.strip()]
    if len(values) not in (1, 3):
        raise SystemExit(f"{name} must contain one value or three comma-separated values")

    try:
        if integer:
            parsed = tuple(int(v) for v in values)
        else:
            parsed = tuple(float(v) for v in values)
    except ValueError as exc:
        raise SystemExit(f"{name} contains a non-numeric value: {raw}") from exc

    if integer and any(v <= 0 or v % 2 == 0 for v in parsed):
        raise SystemExit(f"{name} values must be positive odd integers")
    if not integer and any(v < 0 or not math.isfinite(v) for v in parsed):
        raise SystemExit(f"{name} values must be finite non-negative numbers")
    return parsed


def expand_to_rank(values: tuple[float, ...] | tuple[int, ...], rank: int) -> tuple[float, ...] | tuple[int, ...]:
    if len(values) == 1:
        return values * rank
    if len(values) != rank:
        raise SystemExit(f"Expected one value or {rank} values, got {len(values)}")
    return values


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    if name.endswith(".gz"):
        return name[:-3]
    return path.stem


def default_output_path(input_path: Path, output_dir: Path | None, suffix: str) -> Path:
    directory = output_dir if output_dir is not None else input_path.parent
    return directory / f"{nifti_stem(input_path)}{suffix}.nii.gz"


def resolve_outputs(args: argparse.Namespace) -> list[tuple[Path, Path]]:
    if args.output is not None and len(args.inputs) != 1:
        raise SystemExit("--output can only be used with a single input")

    pairs: list[tuple[Path, Path]] = []
    for input_path in args.inputs:
        input_path = input_path.expanduser().resolve()
        if not input_path.is_file():
            raise SystemExit(f"Input file not found: {input_path}")

        if args.output is not None:
            output_path = args.output.expanduser().resolve()
        else:
            output_path = default_output_path(input_path, args.output_dir, args.suffix).expanduser().resolve()

        if input_path == output_path:
            raise SystemExit(f"Refusing to overwrite input in place: {input_path}")
        if output_path.exists() and not args.overwrite:
            raise SystemExit(f"Output exists, pass --overwrite to replace: {output_path}")
        pairs.append((input_path, output_path))
    return pairs


def finite_fill_value(volume: np.ndarray) -> float:
    finite = volume[np.isfinite(volume)]
    if finite.size == 0:
        return 0.0
    return float(np.median(finite))


def sanitize_volume(volume: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    finite_mask = np.isfinite(volume)
    if finite_mask.all():
        return volume.astype(np.float32, copy=False), finite_mask

    cleaned = volume.astype(np.float32, copy=True)
    cleaned[~finite_mask] = finite_fill_value(cleaned)
    return cleaned, finite_mask


def clip_volume(volume: np.ndarray, clip_min: float | None, clip_max: float | None) -> np.ndarray:
    if clip_min is None and clip_max is None:
        return volume
    return np.clip(volume, clip_min, clip_max).astype(np.float32, copy=False)


def denoise_gaussian(volume: np.ndarray, sigma: tuple[float, ...]) -> np.ndarray:
    from scipy.ndimage import gaussian_filter

    return gaussian_filter(volume, sigma=sigma, mode="nearest").astype(np.float32, copy=False)


def denoise_median(volume: np.ndarray, size: tuple[int, ...]) -> np.ndarray:
    from scipy.ndimage import median_filter

    return median_filter(volume, size=size, mode="nearest").astype(np.float32, copy=False)


def denoise_nlmeans(
    volume: np.ndarray,
    *,
    patch_size: int,
    patch_distance: int,
    h: float,
) -> np.ndarray:
    from skimage.restoration import denoise_nl_means, estimate_sigma

    if patch_size <= 0 or patch_size % 2 == 0:
        raise SystemExit("--nlm-patch-size must be a positive odd integer")
    if patch_distance <= 0:
        raise SystemExit("--nlm-patch-distance must be a positive integer")
    if h < 0 or not math.isfinite(h):
        raise SystemExit("--nlm-h must be a finite non-negative number")

    sigma_est = float(np.mean(estimate_sigma(volume, channel_axis=None)))
    strength = h if h > 0 else max(sigma_est * 0.8, 1e-6)
    return denoise_nl_means(
        volume,
        patch_size=patch_size,
        patch_distance=patch_distance,
        h=strength,
        sigma=sigma_est,
        fast_mode=True,
        preserve_range=True,
        channel_axis=None,
    ).astype(np.float32, copy=False)


def denoise_volume(volume: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    cleaned, finite_mask = sanitize_volume(volume)
    cleaned = clip_volume(cleaned, args.clip_min, args.clip_max)

    if args.method == "gaussian":
        sigma = expand_to_rank(parse_numeric_tuple(args.sigma, name="--sigma"), cleaned.ndim)
        denoised = denoise_gaussian(cleaned, sigma)
    elif args.method == "median":
        size = expand_to_rank(
            parse_numeric_tuple(args.median_size, name="--median-size", integer=True),
            cleaned.ndim,
        )
        denoised = denoise_median(cleaned, size)
    elif args.method == "nlmeans":
        denoised = denoise_nlmeans(
            cleaned,
            patch_size=args.nlm_patch_size,
            patch_distance=args.nlm_patch_distance,
            h=args.nlm_h,
        )
    else:
        raise SystemExit(f"Unsupported method: {args.method}")

    denoised = clip_volume(denoised, args.clip_min, args.clip_max)
    if not finite_mask.all():
        denoised = denoised.astype(np.float32, copy=True)
        denoised[~finite_mask] = np.nan
    return denoised


def save_nifti_like(data: np.ndarray, ref_img: object, output_path: Path) -> None:
    import nibabel as nib

    header = ref_img.header.copy()
    header.set_data_dtype(np.float32)
    finite = data[np.isfinite(data)]
    if finite.size:
        header["cal_min"] = float(finite.min())
        header["cal_max"] = float(finite.max())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_img = nib.Nifti1Image(data.astype(np.float32, copy=False), ref_img.affine, header)
    nib.save(out_img, str(output_path))


def describe_range(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "all non-finite"
    return f"min={finite.min():.3f} max={finite.max():.3f} mean={finite.mean():.3f}"


def denoise_file(input_path: Path, output_path: Path, args: argparse.Namespace) -> None:
    import nibabel as nib

    img = nib.load(str(input_path))
    volume = np.asarray(img.dataobj, dtype=np.float32)
    if volume.ndim != 3:
        raise SystemExit(f"Expected a 3D NIfTI volume, got shape {volume.shape}: {input_path}")

    denoised = denoise_volume(volume, args)
    save_nifti_like(denoised, img, output_path)
    print(
        f"{input_path} -> {output_path}\n"
        f"  shape={volume.shape} method={args.method}\n"
        f"  input:  {describe_range(volume)}\n"
        f"  output: {describe_range(denoised)}"
    )


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    pairs = resolve_outputs(args)
    for input_path, output_path in pairs:
        denoise_file(input_path, output_path, args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
