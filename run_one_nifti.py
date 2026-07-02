#!/usr/bin/env python3
"""
FoundDiff denoising entry point — edit config() and run.

Supports a single .nii.gz file OR a folder + glob pattern.

Usage:
  python run_one_nifti.py              # use config() below
  python run_one_nifti.py --show       # print current config
  python run_one_nifti.py --dry-run    # print commands without executing
  python run_one_nifti.py --list       # list matched inputs only

Pipeline (mode=quick / full):
  input .nii.gz
    -> Preprocess_nifti.py
    -> train.py --data-mode external
    -> reconstruct_denoised_nifti.py

mode=reconstruct_only skips inference and only rebuilds 3D.
"""

from __future__ import annotations

import argparse
import copy
import subprocess
import sys
from pathlib import Path
from typing import Any

from nifti_naming import NAMING_CT, collect_input_files, infer_case_name


ROOT = Path(__file__).resolve().parent


def config() -> dict[str, Any]:
    """Edit paths and options here, then run: python run_one_nifti.py"""

    root = Path("/home/FrankFei/FoundDiff")

    return {
        # ----- input discovery -----
        # Set input_path to ONE file OR a folder:
        #   file:   .../APNHC00002_CT.nii.gz
        #   folder: .../data/custom/nifti
        "input_path": root / "data/custom/nifti",
        # Glob pattern when input_path is a folder (fnmatch on filename):
        #   "*_CT.nii.gz" | "APNHC*.nii.gz" | "*.nii.gz"
        "pattern": "*_CT.nii.gz",
        # ct | ldct | any — suffix-based filter after glob (see nifti_naming.py)
        "naming": "ct",
        # folder mode: False = first match only; True = all matches
        "process_all": False,
        # optional per-file case override; None = infer from filename
        "case_name": None,
        # strip suffixes from stem when inferring case; empty () keeps full stem (APNHC00002_CT)
        "case_strip_suffixes": (),
        "overwrite": False,

        # ----- output -----
        "output_dir": root / "checkpoints/FoundDiff/custom_denoised_files",
        "output_suffix": "_denoised",

        # ----- run mode -----
        # quick | full | reconstruct_only
        "mode": "quick",
        "gpu": "0",

        # ----- intensity mapping (reconstruction) -----
        "intensity_scale": "slice-range",
        "intensity_match": "minmax",
        "range_source": "slice",
        "range_stats_min": -2000.0,
        "range_stats_max": 2000.0,
        "range_ignore_at_or_below": -2500.0,
        "range_percentile": None,
        "range_fixed_min": -1500.0,
        "range_fixed_max": 1500.0,

        # ----- internal paths (usually unchanged) -----
        "manifest": root / "data/external/external_2d/slice_manifest.json",
        "denoised_dir": root / "checkpoints/FoundDiff/test_final_npy",
    }


def resolve_case(cfg: dict[str, Any], input_nii: Path) -> str:
    strip = cfg.get("case_strip_suffixes") or None
    return infer_case_name(input_nii, cfg.get("case_name"), strip_suffixes=strip)


def resolve_inputs(cfg: dict[str, Any]) -> list[Path]:
    files = collect_input_files(
        Path(cfg["input_path"]),
        pattern=str(cfg.get("pattern", "*.nii.gz")),
        naming=str(cfg.get("naming", NAMING_CT)),
    )
    if not cfg.get("process_all", False):
        return files[:1]
    return files


def build_pipeline_cmd(cfg: dict[str, Any], input_nii: Path, case: str) -> list[str]:
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
        "--naming",
        str(cfg.get("naming", NAMING_CT)),
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


def build_reconstruct_cmd(cfg: dict[str, Any], input_nii: Path, case: str) -> list[str]:
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


def output_path(cfg: dict[str, Any], case: str) -> Path:
    return Path(cfg["output_dir"]) / f"{case}{cfg.get('output_suffix', '_denoised')}.nii.gz"


def print_config(cfg: dict[str, Any], inputs: list[Path]) -> None:
    print("=== run_one_nifti.py current config ===")
    for key, value in cfg.items():
        print(f"  {key}: {value}")
    print(f"  -> matched inputs ({len(inputs)}):")
    for path in inputs:
        case = resolve_case(cfg, path)
        print(f"       {path.name}  (case={case})")


def validate(cfg: dict[str, Any], inputs: list[Path]) -> None:
    fixed_min = cfg.get("range_fixed_min")
    fixed_max = cfg.get("range_fixed_max")
    if (fixed_min is None) ^ (fixed_max is None):
        raise SystemExit("Set both range_fixed_min and range_fixed_max, or neither")

    mode = cfg.get("mode", "quick")
    if mode not in {"quick", "full", "reconstruct_only"}:
        raise SystemExit(f"Unknown mode: {mode!r}; choose quick | full | reconstruct_only")

    if not inputs:
        raise SystemExit("No input files matched")


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"\n>> {printable}\n")
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    out = copy.deepcopy(cfg)
    if args.input_path is not None:
        out["input_path"] = Path(args.input_path)
    if args.pattern is not None:
        out["pattern"] = args.pattern
    if args.process_all:
        out["process_all"] = True
    if args.mode is not None:
        out["mode"] = args.mode
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FoundDiff denoising (config-driven; file or folder)")
    p.add_argument("--show", action="store_true", help="Print config and matched inputs")
    p.add_argument("--list", action="store_true", help="List matched inputs and exit")
    p.add_argument("--dry-run", action="store_true", help="Print commands without executing")
    p.add_argument("--input-path", type=Path, default=None, help="Override config input_path (file or folder)")
    p.add_argument("--pattern", default=None, help="Override glob pattern, e.g. '*_CT.nii.gz'")
    p.add_argument("--process-all", action="store_true", help="Process all folder matches (not just first)")
    p.add_argument("--mode", choices=("quick", "full", "reconstruct_only"), default=None)
    return p.parse_args()


def process_one(cfg: dict[str, Any], input_nii: Path, *, dry_run: bool) -> Path:
    case = resolve_case(cfg, input_nii)
    out = output_path(cfg, case)
    if out.exists() and not cfg.get("overwrite", False):
        print(f"Skip existing: {out}")
        return out

    print(f"Processing {input_nii.name} (case={case})")
    mode = cfg.get("mode", "quick")
    if mode == "reconstruct_only":
        run_cmd(build_reconstruct_cmd(cfg, input_nii, case), dry_run=dry_run)
    else:
        run_cmd(build_pipeline_cmd(cfg, input_nii, case), dry_run=dry_run)
    return out


def main() -> int:
    args = parse_args()
    cfg = apply_cli_overrides(config(), args)
    inputs = resolve_inputs(cfg)

    if args.show or args.dry_run or args.list:
        print_config(cfg, inputs)
    if args.list:
        return 0

    if not args.dry_run:
        validate(cfg, inputs)
    else:
        mode = cfg.get("mode", "quick")
        if mode not in {"quick", "full", "reconstruct_only"}:
            raise SystemExit(f"Unknown mode: {mode!r}; choose quick | full | reconstruct_only")

    outputs: list[Path] = []
    failures: list[str] = []
    for index, input_nii in enumerate(inputs, start=1):
        if len(inputs) > 1:
            print(f"\n[{index}/{len(inputs)}] {input_nii.name}")
        try:
            outputs.append(process_one(cfg, input_nii, dry_run=args.dry_run))
        except subprocess.CalledProcessError:
            failures.append(input_nii.name)

    if not args.dry_run:
        print("\nDone.")
        for out in outputs:
            print(f"  output: {out}")
    if failures:
        print("Failed:", ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
