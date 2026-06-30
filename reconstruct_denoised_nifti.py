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
    p.add_argument(
        "--intensity-scale",
        choices=("slice-range", "founddiff-hu", "identity", "unit"),
        default="slice-range",
        help=(
            "How to convert FoundDiff [0,1] npy values before embedding. "
            "slice-range maps each slice back to the original ROI min/max and preserves negatives."
        ),
    )
    p.add_argument(
        "--intensity-match",
        choices=("minmax", "mean-ratio", "none"),
        default="minmax",
        help="Optional post-scale ROI intensity match against the original slice.",
    )
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
) -> np.ndarray:
    arr = np.squeeze(output).astype(np.float32)
    if arr.ndim != 2:
        raise SystemExit(f"Expected a 2D denoised slice, got shape {arr.shape}")

    if scale == "slice-range":
        sy, sx = crop["src_y"], crop["src_x"]
        sh, sw = crop["sh"], crop["sw"]
        original_roi = original_slice[sy : sy + sh, sx : sx + sw]
        src_min = float(np.nanmin(original_roi))
        src_max = float(np.nanmax(original_roi))
        if not np.isfinite(src_min) or not np.isfinite(src_max) or src_max <= src_min:
            return np.full_like(arr, src_min if np.isfinite(src_min) else 0.0)
        return np.clip(arr, 0.0, 1.0) * (src_max - src_min) + src_min
    if scale == "founddiff-hu":
        return denorm_to_hu(arr)
    if scale == "unit":
        return arr
    if scale == "identity":
        return arr
    raise SystemExit(f"Unsupported --intensity-scale: {scale}")


def embed_denoised(
    slice_values: np.ndarray,
    denoised_norm: np.ndarray,
    *,
    intensity_scale: str,
    intensity_match: str,
) -> np.ndarray:
    h, w = slice_values.shape
    crop = center_crop_box(h, w)
    denoised512 = scale_model_output(denoised_norm, intensity_scale, slice_values, crop)
    denoised512 = match_intensity_to_original(denoised512, slice_values, crop, intensity_match)

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
            out_vol[z] = embed_denoised(
                vol_zyx[z],
                np.load(dpath),
                intensity_scale=args.intensity_scale,
                intensity_match=args.intensity_match,
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
            )
            n_written += 1
        print(f"Fallback mode: stride={args.stride}  written: {n_written}/{len(denoised)}")

    import nibabel as nib

    out_xyz = np.transpose(out_vol, (2, 1, 0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out_xyz.astype(np.float32), ref_img.affine, ref_img.header), str(args.output))
    print(
        f"Input Z={vol_zyx.shape[0]}  scale={args.intensity_scale}  match={args.intensity_match}\n"
        f"  input:  {describe_range(vol_zyx)}\n"
        f"  output: {describe_range(out_vol)}\n"
        f"  saved:  {args.output}"
    )


if __name__ == "__main__":
    main()
