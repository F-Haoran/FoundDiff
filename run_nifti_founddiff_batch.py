#!/usr/bin/env python3
"""
Batch FoundDiff denoising for a folder of .nii.gz files.

Each input is processed in isolation through run_external_pipeline.py so the
manifest, slice layout, and reconstruction step stay aligned with one case.

Default naming (ct): denoise *_CT.nii.gz files — common when exports use CT suffix
for the low-dose volume. Use --naming ldct for Mayo-style *_LDCT inputs.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from nifti_naming import NAMING_CHOICES, NAMING_CT, collect_input_files, infer_case_name, nifti_stem


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
        "--naming",
        choices=NAMING_CHOICES,
        default=NAMING_CT,
        help=(
            "Input filename convention: ct=*_CT.nii.gz is noisy input (default); "
            "ldct=*_LDCT input and skip *_CT; any=skip only *_FULL."
        ),
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
        choices=("founddiff-hu", "preserve-original", "slice-range", "identity", "unit"),
        default="founddiff-hu",
        help="Passed to reconstruct_denoised_nifti.py (default: founddiff-hu).",
    )
    p.add_argument(
        "--intensity-match",
        choices=("minmax", "mean-ratio", "none"),
        default="none",
        help="Passed to reconstruct_denoised_nifti.py (default: none).",
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


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    input_dir = args.input_dir.expanduser().resolve()
    return collect_input_files(
        input_dir,
        pattern=args.pattern,
        naming=args.naming,
    )


def main() -> int:
    args = parse_args()
    inputs = collect_inputs(args)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Found {len(inputs)} file(s) to process (naming={args.naming})")
    print(f"Output dir: {output_dir}")
    print(f"Mode: {args.mode}")

    failures: list[str] = []
    for index, input_path in enumerate(inputs, start=1):
        case = infer_case_name(input_path)
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
            "--naming",
            args.naming,
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
