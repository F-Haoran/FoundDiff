#!/usr/bin/env python3
"""Reconstruct denoised npy -> nii.gz (no hidden shell characters)."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("case", nargs="?", default="case001")
    p.add_argument("--stride", type=int, default=1)
    args = p.parse_args()

    case = args.case
    low = ROOT / "data" / "external" / "nifti" / f"{case}_low.nii.gz"
    manifest = ROOT / "data" / "external" / "external_2d" / "slice_manifest.json"
    out = ROOT / "checkpoints" / "FoundDiff" / f"{case}_denoised.nii.gz"
    den = ROOT / "checkpoints" / "FoundDiff" / "test_final_npy"

    cmd = [
        sys.executable,
        str(ROOT / "reconstruct_denoised_nifti.py"),
        "--input-nii",
        str(low),
        "--denoised-dir",
        str(den),
        "--output",
        str(out),
        "--volume-name",
        case,
        "--intensity-scale",
        "preserve-original",
        "--intensity-match",
        "none",
    ]
    if manifest.is_file():
        cmd.extend(["--manifest", str(manifest)])
    else:
        cmd.extend(["--stride", str(args.stride)])

    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, check=True)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
