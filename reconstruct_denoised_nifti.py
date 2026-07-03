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
from dataclasses import dataclass
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
    p.add_argument(
        "--intensity-scale",
        choices=("preserve-original", "founddiff-hu", "slice-range", "identity", "unit"),
        default="preserve-original",
        help=(
            "How to convert FoundDiff [0,1] npy values before embedding. "
            "preserve-original (default): apply only the model's norm delta in HU space "
            "(HU_out = HU_orig + 3000*(norm_out-norm_in); exact when model is identity). "
            "founddiff-hu: replace with norm*3000+24 (clips negatives to ~24). "
            "slice-range: legacy per-slice linear map (inconsistent with model norm)."
        ),
    )
    p.add_argument(
        "--range-source",
        choices=("slice", "roi"),
        default="slice",
        help="For slice-range, use the full 2D slice or only the embedded ROI for min/max.",
    )
    p.add_argument(
        "--intensity-match",
        choices=("minmax", "mean-ratio", "none"),
        default="none",
        help=(
            "Optional post-scale ROI intensity match against the original slice. "
            "none (default with founddiff-hu) keeps FoundDiff HU inverse; "
            "minmax stretches ROI to filtered original min/max; mean-ratio matches mean."
        ),
    )
    p.add_argument(
        "--range-stats-min",
        type=float,
        default=None,
        help=(
            "When estimating intensity bounds, ignore reference voxels below this value. "
            "Default: no lower clip (only --range-ignore-at-or-below applies)."
        ),
    )
    p.add_argument(
        "--range-stats-max",
        type=float,
        default=None,
        help=(
            "When estimating intensity bounds, ignore reference voxels above this value. "
            "Default: no upper clip."
        ),
    )
    p.add_argument(
        "--range-percentile",
        default=None,
        help=(
            "Optional robust percentile pair for slice-range, e.g. 2,98. "
            "Applied after --range-stats-min/max filtering."
        ),
    )
    p.add_argument(
        "--range-ignore-at-or-below",
        type=float,
        default=-2500.0,
        help=(
            "Ignore reference voxels at or below this value when estimating slice-range "
            "(default: -2500 to drop common -3000 air/padding)."
        ),
    )
    p.add_argument(
        "--range-fixed-min",
        type=float,
        default=None,
        help="Optional fixed output mapping minimum, e.g. -1500.",
    )
    p.add_argument(
        "--range-fixed-max",
        type=float,
        default=None,
        help="Optional fixed output mapping maximum, e.g. 1500.",
    )
    return p.parse_args()


@dataclass(frozen=True)
class RangeOptions:
    source: str
    stats_min: float | None
    stats_max: float | None
    percentile: tuple[float, float] | None
    ignore_at_or_below: float | None
    fixed_min: float | None
    fixed_max: float | None


def parse_percentile_pair(raw: str | None) -> tuple[float, float] | None:
    if raw is None or not str(raw).strip():
        return None
    parts = [part.strip() for part in str(raw).split(",") if part.strip()]
    if len(parts) != 2:
        raise SystemExit("--range-percentile must be two comma-separated values, e.g. 2,98")
    try:
        low = float(parts[0])
        high = float(parts[1])
    except ValueError as exc:
        raise SystemExit(f"--range-percentile contains non-numeric values: {raw}") from exc
    if not (0.0 <= low < high <= 100.0):
        raise SystemExit("--range-percentile must satisfy 0 <= low < high <= 100")
    return low, high


def range_options_from_args(args: argparse.Namespace) -> RangeOptions:
    if (args.range_fixed_min is None) ^ (args.range_fixed_max is None):
        raise SystemExit("Set both --range-fixed-min and --range-fixed-max, or neither.")
    if args.range_fixed_min is not None and args.range_fixed_max <= args.range_fixed_min:
        raise SystemExit("--range-fixed-max must be greater than --range-fixed-min")
    return RangeOptions(
        source=args.range_source,
        stats_min=args.range_stats_min,
        stats_max=args.range_stats_max,
        percentile=parse_percentile_pair(args.range_percentile),
        ignore_at_or_below=args.range_ignore_at_or_below,
        fixed_min=args.range_fixed_min,
        fixed_max=args.range_fixed_max,
    )


def denorm_to_hu(norm_0_1: np.ndarray) -> np.ndarray:
    return norm_0_1.astype(np.float32) * HU_RANGE + HU_MIN + 1024


def load_nifti_z_y_x(path: Path) -> tuple[np.ndarray, object]:
    import nibabel as nib

    img = nib.load(str(path))
    # Use get_fdata() so scl_slope / scl_inter are applied (needed for signed HU values).
    vol = np.asarray(img.get_fdata(), dtype=np.float32)
    if vol.ndim == 3:
        vol = np.transpose(vol, (2, 1, 0))
    return vol, img


def save_nifti_like(data_xyz: np.ndarray, ref_img: object, output_path: Path) -> None:
    import nibabel as nib

    header = ref_img.header.copy()
    header.set_data_dtype(np.float32)
    header["scl_slope"] = np.float32(1.0)
    header["scl_inter"] = np.float32(0.0)
    finite = data_xyz[np.isfinite(data_xyz)]
    if finite.size:
        header["cal_min"] = float(finite.min())
        header["cal_max"] = float(finite.max())
    out_img = nib.Nifti1Image(data_xyz.astype(np.float32, copy=False), ref_img.affine, header)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(out_img, str(output_path))


def center_crop_box(h: int, w: int, *, roi_h: int = 512, roi_w: int = 512) -> dict[str, int]:
    sh, sw = min(h, roi_h), min(w, roi_w)
    dst_y = max((roi_h - sh) // 2, 0)
    dst_x = max((roi_w - sw) // 2, 0)
    src_y = max((h - sh) // 2, 0)
    src_x = max((w - sw) // 2, 0)
    return {
        "src_y": src_y,
        "src_x": src_x,
        "dst_y": dst_y,
        "dst_x": dst_x,
        "sh": sh,
        "sw": sw,
    }


def finite_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.mean())


def finite_abs_mean(values: np.ndarray) -> float | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(np.abs(finite).mean())


def finite_min_max(values: np.ndarray) -> tuple[float, float] | None:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return None
    return float(finite.min()), float(finite.max())


def reference_values_for_range(reference: np.ndarray, opts: RangeOptions) -> np.ndarray:
    if opts.fixed_min is not None and opts.fixed_max is not None:
        return reference[np.isfinite(reference)]

    values = reference[np.isfinite(reference)].astype(np.float64, copy=False)
    if values.size == 0:
        return values

    if opts.ignore_at_or_below is not None:
        values = values[values > opts.ignore_at_or_below]
    if opts.stats_min is not None:
        values = values[values >= opts.stats_min]
    if opts.stats_max is not None:
        values = values[values <= opts.stats_max]
    return values


def reference_intensity_bounds(reference: np.ndarray, opts: RangeOptions) -> tuple[float, float] | None:
    if opts.fixed_min is not None and opts.fixed_max is not None:
        return float(opts.fixed_min), float(opts.fixed_max)

    values = reference_values_for_range(reference, opts)
    if values.size == 0:
        return None

    if opts.percentile is not None:
        low, high = opts.percentile
        src_min, src_max = np.percentile(values, [low, high])
        return float(src_min), float(src_max)

    return float(values.min()), float(values.max())


def match_mean_ratio(
    denoised512: np.ndarray,
    original_roi: np.ndarray,
    denoised_roi: np.ndarray,
) -> np.ndarray:
    original_mean = finite_mean(original_roi)
    denoised_mean = finite_mean(denoised_roi)
    if original_mean is None or denoised_mean is None:
        return denoised512

    if abs(denoised_mean) < 1e-6:
        original_abs_mean = finite_abs_mean(original_roi)
        denoised_abs_mean = finite_abs_mean(denoised_roi)
        if original_abs_mean is not None and denoised_abs_mean is not None and denoised_abs_mean >= 1e-6:
            return denoised512 * np.float32(original_abs_mean / denoised_abs_mean)
        return denoised512 + np.float32(original_mean - denoised_mean)

    ratio = original_mean / denoised_mean
    if not np.isfinite(ratio) or ratio <= 0:
        original_abs_mean = finite_abs_mean(original_roi)
        denoised_abs_mean = finite_abs_mean(denoised_roi)
        if original_abs_mean is not None and denoised_abs_mean is not None and denoised_abs_mean >= 1e-6:
            return denoised512 * np.float32(original_abs_mean / denoised_abs_mean)
        return denoised512 + np.float32(original_mean - denoised_mean)
    return denoised512 * np.float32(ratio)


def match_intensity_to_original(
    denoised512: np.ndarray,
    original_slice: np.ndarray,
    crop: dict[str, int],
    mode: str,
    *,
    range_opts: RangeOptions | None = None,
) -> np.ndarray:
    if mode == "none":
        return denoised512
    if mode not in {"minmax", "mean-ratio"}:
        raise SystemExit(f"Unsupported --intensity-match: {mode}")

    sy, sx, dy, dx = crop["src_y"], crop["src_x"], crop["dst_y"], crop["dst_x"]
    sh, sw = crop["sh"], crop["sw"]
    original_roi = original_slice[sy : sy + sh, sx : sx + sw]
    denoised_roi = denoised512[dy : dy + sh, dx : dx + sw]

    if mode == "mean-ratio":
        return match_mean_ratio(denoised512, original_roi, denoised_roi)

    if range_opts is not None:
        original_bounds = reference_intensity_bounds(original_roi, range_opts)
        denoised_bounds = reference_intensity_bounds(denoised_roi, range_opts)
        if denoised_bounds is None:
            denoised_bounds = finite_min_max(denoised_roi)
    else:
        original_bounds = finite_min_max(original_roi)
        denoised_bounds = finite_min_max(denoised_roi)
    if original_bounds is None or denoised_bounds is None:
        return denoised512
    original_min, original_max = original_bounds
    denoised_min, denoised_max = denoised_bounds
    denoised_range = denoised_max - denoised_min
    original_range = original_max - original_min
    if denoised_range <= 1e-6 or original_range <= 1e-6:
        return match_mean_ratio(denoised512, original_roi, denoised_roi)

    return (denoised512 - np.float32(denoised_min)) * np.float32(original_range / denoised_range) + np.float32(original_min)


def scale_model_output(
    output: np.ndarray,
    scale: str,
    original_slice: np.ndarray,
    crop: dict[str, int],
    *,
    range_opts: RangeOptions,
) -> np.ndarray:
    arr = np.squeeze(output).astype(np.float32)
    if arr.ndim != 2:
        raise SystemExit(f"Expected a 2D denoised slice, got shape {arr.shape}")

    if scale == "founddiff-hu":
        return denorm_to_hu(arr)
    if scale == "preserve-original":
        raise SystemExit("preserve-original is handled in embed_denoised(), not scale_model_output()")
    if scale == "slice-range":
        sy, sx = crop["src_y"], crop["src_x"]
        sh, sw = crop["sh"], crop["sw"]
        original_roi = original_slice[sy : sy + sh, sx : sx + sw]
        reference = original_slice if range_opts.source == "slice" else original_roi
        bounds = reference_intensity_bounds(reference, range_opts)
        if bounds is None:
            return np.full_like(arr, 0.0)
        src_min, src_max = bounds
        if not np.isfinite(src_min) or not np.isfinite(src_max) or src_max <= src_min:
            return np.full_like(arr, src_min if np.isfinite(src_min) else 0.0)
        return np.clip(arr, 0.0, 1.0) * (src_max - src_min) + src_min
    if scale == "unit":
        return arr
    if scale == "identity":
        return arr
    raise SystemExit(f"Unsupported --intensity-scale: {scale}")


def hu_delta_from_norm_delta(norm_out: np.ndarray, norm_in: np.ndarray) -> np.ndarray:
    """Apply FoundDiff denorm to the norm change only (preserves sub-24 HU on identity)."""
    return (np.clip(norm_out, 0.0, 1.0) - np.clip(norm_in, 0.0, 1.0)) * np.float32(3000.0)


def preserve_original_intensity(
    norm_out512: np.ndarray,
    original_slice: np.ndarray,
    crop: dict[str, int],
) -> np.ndarray:
    """
    HU_out = HU_orig + 3000*(norm_out - norm_in).

    If the model returns the same norm as the input, HU_out equals HU_orig exactly
    (including negative HU such as -2048). Only the denoising delta is applied.
    """
    sy, sx, dy, dx = crop["src_y"], crop["src_x"], crop["dst_y"], crop["dst_x"]
    sh, sw = crop["sh"], crop["sw"]
    roi_orig = original_slice[sy : sy + sh, sx : sx + sw].astype(np.float32, copy=False)
    roi_norm_out = norm_out512[dy : dy + sh, dx : dx + sw]
    norm_in = (roi_orig - np.float32(24.0)) / np.float32(3000.0)
    roi_out = roi_orig + hu_delta_from_norm_delta(roi_norm_out, norm_in)
    out = norm_out512.astype(np.float32, copy=True)
    out[dy : dy + sh, dx : dx + sw] = roi_out
    return out


def resolve_reconstruction_options(
    intensity_scale: str,
    intensity_match: str,
    range_opts: RangeOptions,
) -> tuple[str, str, RangeOptions]:
    """preserve-original uses HU residual correction; skip extra minmax unless requested."""
    if intensity_scale == "preserve-original" and intensity_match == "none":
        return intensity_scale, "none", range_opts
    return intensity_scale, intensity_match, range_opts


def embed_denoised(
    slice_values: np.ndarray,
    denoised_norm: np.ndarray,
    *,
    intensity_scale: str,
    intensity_match: str,
    range_opts: RangeOptions,
) -> np.ndarray:
    h, w = slice_values.shape
    crop = center_crop_box(h, w)
    intensity_scale, intensity_match, range_opts = resolve_reconstruction_options(
        intensity_scale, intensity_match, range_opts
    )
    arr = np.squeeze(denoised_norm).astype(np.float32)
    if arr.ndim != 2:
        raise SystemExit(f"Expected a 2D denoised slice, got shape {arr.shape}")

    if intensity_scale == "preserve-original":
        denoised512 = preserve_original_intensity(arr, slice_values, crop)
    else:
        denoised512 = scale_model_output(
            arr,
            intensity_scale,
            slice_values,
            crop,
            range_opts=range_opts,
        )
        denoised512 = match_intensity_to_original(
            denoised512,
            slice_values,
            crop,
            intensity_match,
            range_opts=range_opts,
        )

    out = slice_values.copy()
    sy, sx, dy, dx = crop["src_y"], crop["src_x"], crop["dst_y"], crop["dst_x"]
    sh, sw = crop["sh"], crop["sw"]
    out[sy : sy + sh, sx : sx + sw] = denoised512[dy : dy + sh, dx : dx + sw]
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


def describe_range(values: np.ndarray) -> str:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return "all non-finite"
    return f"min={finite.min():.3f} max={finite.max():.3f} mean={finite.mean():.3f}"


def describe_range_options(opts: RangeOptions) -> str:
    parts = [f"source={opts.source}"]
    if opts.fixed_min is not None and opts.fixed_max is not None:
        parts.append(f"fixed=[{opts.fixed_min}, {opts.fixed_max}]")
    else:
        if opts.stats_min is not None or opts.stats_max is not None:
            parts.append(f"stats=[{opts.stats_min}, {opts.stats_max}]")
        if opts.percentile is not None:
            parts.append(f"percentile={opts.percentile[0]},{opts.percentile[1]}")
        if opts.ignore_at_or_below is not None:
            parts.append(f"ignore<={opts.ignore_at_or_below}")
    return " ".join(parts)


def main():
    args = parse_args()
    range_opts = range_options_from_args(args)
    vol_zyx, ref_img = load_nifti_z_y_x(args.input_nii)
    out_vol = vol_zyx.copy()
    print(f"Loaded input with get_fdata(): {describe_range(vol_zyx)}")
    print(f"Range options: {describe_range_options(range_opts)}")

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
            out_vol[z] = embed_denoised(
                vol_zyx[z],
                np.load(dpath),
                intensity_scale=args.intensity_scale,
                intensity_match=args.intensity_match,
                range_opts=range_opts,
            )
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
            out_vol[z] = embed_denoised(
                vol_zyx[z],
                np.load(dpath),
                intensity_scale=args.intensity_scale,
                intensity_match=args.intensity_match,
                range_opts=range_opts,
            )
            n_written += 1
        print(f"Fallback mode: stride={args.stride}  written: {n_written}/{len(denoised)}")

    out_xyz = np.transpose(out_vol, (2, 1, 0))
    save_nifti_like(out_xyz, ref_img, args.output)
    print(
        f"Input Z={vol_zyx.shape[0]}  scale={args.intensity_scale}  match={args.intensity_match}\n"
        f"  range: {describe_range_options(range_opts)}\n"
        f"  input:  {describe_range(vol_zyx)}\n"
        f"  output: {describe_range(out_vol)}\n"
        f"  saved:  {args.output}"
    )


if __name__ == "__main__":
    main()
