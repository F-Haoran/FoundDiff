#!/usr/bin/env python3
"""
Single-file FoundDiff denoising entry point — edit config() and run, no long CLI.

Usage:
  python run_one_nifti.py              # use config() below
  python run_one_nifti.py --show       # print current config
  python run_one_nifti.py --dry-run    # print commands without executing

Pipeline (mode=quick / full):
  input .nii.gz
    -> Preprocess_nifti.py   (3D -> 2D lung-*.npy + slice_manifest.json)
    -> train.py --data-mode external   (FoundDiff inference -> 2D denoised npy)
    -> reconstruct_denoised_nifti.py (2D npy -> 3D denoised .nii.gz)

mode=reconstruct_only skips the first two steps and only rebuilds 3D (for tuning intensity range).
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def config() -> dict[str, Any]:
    """Edit paths and options here, then run: python run_one_nifti.py"""

    # Local FoundDiff root (Maybach example)
    root = Path("/home/FrankFei/FoundDiff")

    return {
        # ----- input / output -----
        # Use LDCT (low-dose) input, not *_CT.nii.gz (full-dose reference)
        "input_nii": root / "data/custom/nifti/APNHC00002_LDCT.nii.gz",
        "case_name": None,  # None = infer from filename (e.g. APNHC00718_LDCT)
        "output_dir": root / "checkpoints/FoundDiff/custom_denoised_files",
        "output_suffix": "_denoised",

        # ----- run mode -----
        # quick            : fast trial (~50 preprocess slices + 10 inference slices)
        # full             : all slices (full 3D output)
        # reconstruct_only : rebuild 3D only when 2D npy already exist
        "mode": "quick",
        "gpu": "0",

        # ----- intensity mapping (reconstruction) -----
        "intensity_scale": "slice-range",  # slice-range | founddiff-hu | identity | unit
        "intensity_match": "minmax",       # minmax | mean-ratio | none
        "range_source": "slice",           # slice | roi
        "range_stats_min": -2000.0,
        "range_stats_max": 2000.0,
        "range_ignore_at_or_below": -2500.0,  # ignore -3000 padding
        "range_percentile": None,             # e.g. "2,98"; None = no percentile
        "range_fixed_min": -1500.0,           # None = auto estimate; or fix tissue window
        "range_fixed_max": 1500.0,

        # ----- internal paths (usually unchanged) -----
        "manifest": root / "data/external/external_2d/slice_manifest.json",
        "denoised_dir": root / "checkpoints/FoundDiff/test_final_npy",
    }


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def resolve_case(cfg: dict[str, Any]) -> str:
    if cfg.get("case_name"):
        return str(cfg["case_name"])
    return nifti_stem(Path(cfg["input_nii"]))


def build_pipeline_cmd(cfg: dict[str, Any]) -> list[str]:
    input_nii = Path(cfg["input_nii"]).expanduser().resolve()
    case = resolve_case(cfg)
    mode = cfg.get("mode", "quick")
    if mode not in {"quick", "full"}:
        raise SystemExit(f"build_pipeline_cmd requires mode=quick/full, got {mode!r}")

    cmd = [
        sys.executable,
        str(ROOT / "run_external_pipeline.py"),
        str(input_nii),
        case,
        mode,
        "--output-dir",
        str(Path(cfg["output_dir"]).expanduser().resolve()),
        "--output-suffix",
        str(cfg.get("output_suffix", "_denoised")),
        "--gpu",
        str(cfg.get("gpu", "0")),
        "--intensity-scale",
        str(cfg.get("intensity_scale", "slice-range")),
        "--intensity-match",
        str(cfg.get("intensity_match", "minmax")),
        "--range-source",
        str(cfg.get("range_source", "slice")),
        "--range-stats-min",
        str(cfg.get("range_stats_min", -2000.0)),
        "--range-stats-max",
        str(cfg.get("range_stats_max", 2000.0)),
        "--range-ignore-at-or-below",
        str(cfg.get("range_ignore_at_or_below", -2500.0)),
    ]
    if cfg.get("range_percentile"):
        cmd.extend(["--range-percentile", str(cfg["range_percentile"])])
    if cfg.get("range_fixed_min") is not None:
        cmd.extend(["--range-fixed-min", str(cfg["range_fixed_min"])])
    if cfg.get("range_fixed_max") is not None:
        cmd.extend(["--range-fixed-max", str(cfg["range_fixed_max"])])
    return cmd


def build_reconstruct_cmd(cfg: dict[str, Any]) -> list[str]:
    input_nii = Path(cfg["input_nii"]).expanduser().resolve()
    case = resolve_case(cfg)
    output_dir = Path(cfg["output_dir"]).expanduser().resolve()
    suffix = str(cfg.get("output_suffix", "_denoised"))
    out_nii = output_dir / f"{case}{suffix}.nii.gz"
    manifest = Path(cfg.get("manifest", ROOT / "data/external/external_2d/slice_manifest.json"))
    denoised_dir = Path(cfg.get("denoised_dir", ROOT / "checkpoints/FoundDiff/test_final_npy"))

    cmd = [
        sys.executable,
        str(ROOT / "reconstruct_denoised_nifti.py"),
        "--input-nii",
        str(input_nii),
        "--denoised-dir",
        str(denoised_dir),
        "--output",
        str(out_nii),
        "--volume-name",
        case,
        "--manifest",
        str(manifest),
        "--intensity-scale",
        str(cfg.get("intensity_scale", "slice-range")),
        "--intensity-match",
        str(cfg.get("intensity_match", "minmax")),
        "--range-source",
        str(cfg.get("range_source", "slice")),
        "--range-stats-min",
        str(cfg.get("range_stats_min", -2000.0)),
        "--range-stats-max",
        str(cfg.get("range_stats_max", 2000.0)),
        "--range-ignore-at-or-below",
        str(cfg.get("range_ignore_at_or_below", -2500.0)),
    ]
    if cfg.get("range_percentile"):
        cmd.extend(["--range-percentile", str(cfg["range_percentile"])])
    if cfg.get("range_fixed_min") is not None:
        cmd.extend(["--range-fixed-min", str(cfg["range_fixed_min"])])
    if cfg.get("range_fixed_max") is not None:
        cmd.extend(["--range-fixed-max", str(cfg["range_fixed_max"])])
    return cmd


def print_config(cfg: dict[str, Any]) -> None:
    case = resolve_case(cfg)
    print("=== run_one_nifti.py current config ===")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    print(f"  -> case_name (resolved): {case}")
    mode = cfg.get("mode", "quick")
    if mode == "reconstruct_only":
        out = Path(cfg["output_dir"]) / f"{case}{cfg.get('output_suffix', '_denoised')}.nii.gz"
        print(f"  -> output: {out}")


def validate(cfg: dict[str, Any]) -> None:
    input_nii = Path(cfg["input_nii"]).expanduser().resolve()
    if not input_nii.is_file():
        raise SystemExit(f"Input file not found: {input_nii}")

    fixed_min = cfg.get("range_fixed_min")
    fixed_max = cfg.get("range_fixed_max")
    if (fixed_min is None) ^ (fixed_max is None):
        raise SystemExit("Set both range_fixed_min and range_fixed_max, or neither")

    mode = cfg.get("mode", "quick")
    if mode not in {"quick", "full", "reconstruct_only"}:
        raise SystemExit(f"Unknown mode: {mode!r}; choose quick | full | reconstruct_only")


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"\n>> {printable}\n")
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Single-file FoundDiff denoising (config-driven)")
    p.add_argument("--show", action="store_true", help="Print config() contents")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = config()

    if args.show or args.dry_run:
        print_config(cfg)

    if not args.dry_run:
        validate(cfg)
    else:
        mode = cfg.get("mode", "quick")
        if mode not in {"quick", "full", "reconstruct_only"}:
            raise SystemExit(f"Unknown mode: {mode!r}; choose quick | full | reconstruct_only")

    mode = cfg.get("mode", "quick")
    if mode == "reconstruct_only":
        run_cmd(build_reconstruct_cmd(cfg), dry_run=args.dry_run)
    else:
        run_cmd(build_pipeline_cmd(cfg), dry_run=args.dry_run)

    if not args.dry_run:
        case = resolve_case(cfg)
        out = Path(cfg["output_dir"]) / f"{case}{cfg.get('output_suffix', '_denoised')}.nii.gz"
        print(f"\nDone. Output: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
