#!/usr/bin/env python3
"""
Batch FoundDiff denoising for a folder of .nii.gz files.

Each input is processed in isolation through run_external_pipeline.py so the
manifest, slice layout, and reconstruction step stay aligned with one case.

By default only low-dose inputs are processed:
  *_LDCT.nii.gz

Full-dose reference files such as *_CT.nii.gz are skipped unless --include-ct is set.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Batch FoundDiff denoising for .nii.gz folders")
    p.add_argument(
        "--input-dir",
        type=Path,
        default=ROOT / "data" / "custom" / "nifti",
        help="Directory containing input .nii.gz files",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "checkpoints" / "FoundDiff" / "custom_denoised_files",
        help="Directory for reconstructed denoised .nii.gz outputs",
    )
    p.add_argument(
        "--mode",
        choices=("full", "quick"),
        default="full",
        help="full=all slices; quick=subset for smoke testing",
    )
    p.add_argument("--gpu", default="0", help="CUDA device id passed to run_external_pipeline.py")
    p.add_argument(
        "--include-ct",
        action="store_true",
        help="Also process *_CT.nii.gz files (default: only *_LDCT.nii.gz)",
    )
    p.add_argument(
        "--pattern",
        default="*.nii.gz",
        help="Glob pattern under --input-dir (default: *.nii.gz)",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace existing output .nii.gz files",
    )
    p.add_argument(
        "--intensity-scale",
        choices=("slice-range", "founddiff-hu", "identity", "unit"),
        default="slice-range",
        help="Passed to reconstruct_denoised_nifti.py (default: slice-range).",
    )
    p.add_argument(
        "--intensity-match",
        choices=("minmax", "mean-ratio", "none"),
        default="minmax",
        help="Passed to reconstruct_denoised_nifti.py (default: minmax).",
    )
    p.add_argument(
        "--range-source",
        choices=("slice", "roi"),
        default="slice",
        help="Passed to reconstruct_denoised_nifti.py (default: slice).",
    )
    p.add_argument(
        "--range-stats-min",
        type=float,
        default=-2000.0,
        help="Ignore reference voxels below this value when estimating slice-range.",
    )
    p.add_argument(
        "--range-stats-max",
        type=float,
        default=2000.0,
        help="Ignore reference voxels above this value when estimating slice-range.",
    )
    p.add_argument(
        "--range-percentile",
        default=None,
        help="Optional robust percentile pair for slice-range, e.g. 2,98.",
    )
    p.add_argument(
        "--range-ignore-at-or-below",
        type=float,
        default=-2500.0,
        help="Ignore reference voxels at or below this value (default: -2500 for -3000 padding).",
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


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def should_process(path: Path, *, include_ct: bool) -> bool:
    stem = nifti_stem(path)
    upper = stem.upper()
    if upper.endswith("_CT"):
        return include_ct
    if upper.endswith("_LDCT"):
        return True
    if upper.endswith("_LOW"):
        return True
    if include_ct:
        return True
    # Generic custom names like osasd-2.nii.gz
    return not upper.endswith("_FULL")


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    input_dir = args.input_dir.expanduser().resolve()
    if not input_dir.is_dir():
        raise SystemExit(f"Input directory not found: {input_dir}")

    files = sorted(input_dir.glob(args.pattern))
    selected = [path for path in files if path.is_file() and should_process(path, include_ct=args.include_ct)]
    if not selected:
        raise SystemExit(f"No matching .nii.gz files found in {input_dir}")
    return selected


def main() -> int:
    args = parse_args()
    inputs = collect_inputs(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(inputs)} file(s) to process")
    print(f"Output dir: {output_dir}")
    print(f"Mode: {args.mode}")

    failures: list[str] = []
    for index, input_path in enumerate(inputs, start=1):
        case = nifti_stem(input_path)
        output_path = output_dir / f"{case}_denoised.nii.gz"
        if output_path.exists() and not args.overwrite:
            print(f"[{index}/{len(inputs)}] Skip existing: {output_path}")
            continue

        print(f"\n[{index}/{len(inputs)}] Processing {input_path.name} (case={case})")
        cmd = [
            sys.executable,
            str(ROOT / "run_external_pipeline.py"),
            str(input_path),
            case,
            args.mode,
            "--output-dir",
            str(output_dir),
            "--gpu",
            str(args.gpu),
            "--intensity-scale",
            args.intensity_scale,
            "--intensity-match",
            args.intensity_match,
            "--range-source",
            args.range_source,
            "--range-stats-min",
            str(args.range_stats_min),
            "--range-stats-max",
            str(args.range_stats_max),
            "--range-ignore-at-or-below",
            str(args.range_ignore_at_or_below),
        ]
        if args.range_percentile:
            cmd.extend(["--range-percentile", args.range_percentile])
        if args.range_fixed_min is not None:
            cmd.extend(["--range-fixed-min", str(args.range_fixed_min)])
        if args.range_fixed_max is not None:
            cmd.extend(["--range-fixed-max", str(args.range_fixed_max)])
        try:
            subprocess.run(cmd, cwd=ROOT, check=True)
        except subprocess.CalledProcessError:
            failures.append(case)
            print(f"FAILED: {case}", file=sys.stderr)

    print("\nBatch finished.")
    print(f"Successful outputs in: {output_dir}")
    if failures:
        print("Failed cases:", ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
