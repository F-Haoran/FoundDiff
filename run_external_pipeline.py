#!/usr/bin/env python3
"""
FoundDiff external pipeline without bash (avoids hidden Unicode in copied commands).

Usage:
  python run_external_pipeline.py /path/to/noisy.nii.gz
  python run_external_pipeline.py /path/to/noisy.nii.gz case001
  python run_external_pipeline.py /path/to/noisy.nii.gz case001 quick

Steps: copy nii -> Preprocess_nifti -> train.py (external) -> reconstruct nii.gz
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(cmd: list[str], env: dict | None = None) -> None:
    printable = " ".join(cmd)
    print(f"\n>> {printable}\n")
    merged = os.environ.copy()
    if env:
        merged.update(env)
    subprocess.run(cmd, cwd=ROOT, env=merged, check=True)


def parse_args():
    p = argparse.ArgumentParser(description="External nii.gz full FoundDiff pipeline")
    p.add_argument("nifti", type=Path, help="Your noisy .nii.gz file")
    p.add_argument("case", nargs="?", default="case001", help="Case name (default case001)")
    p.add_argument(
        "mode",
        nargs="?",
        default="full",
        choices=("full", "quick"),
        help="full=all slices + nii.gz; quick=subset test",
    )
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    return p.parse_args()


def main():
    args = parse_args()
    nifti = args.nifti.expanduser().resolve()
    if not nifti.is_file():
        print(f"File not found: {nifti}", file=sys.stderr)
        sys.exit(1)

    case = args.case
    ext_nifti = ROOT / "data" / "external" / "nifti"
    ext_2d = ROOT / "data" / "external" / "external_2d"
    manifest = ext_2d / "slice_manifest.json"
    ext_nifti.mkdir(parents=True, exist_ok=True)

    low = ext_nifti / f"{case}_low.nii.gz"
    full = ext_nifti / f"{case}_full.nii.gz"
    shutil.copy2(nifti, low)
    shutil.copy2(nifti, full)
    print(f"Input -> {low}")

    if args.mode == "quick":
        stride, max_slices, max_test = 2, 50, 10
        print("Mode: QUICK (subset only)")
    else:
        stride, max_slices, max_test = 1, 0, 0
        print("Mode: FULL (all slices + reconstruct nii.gz)")

    pre_cmd = [
        sys.executable,
        str(ROOT / "Preprocess_nifti.py"),
        "--nifti-dir",
        str(ext_nifti),
        "--out-root",
        str(ext_2d),
        "--stride",
        str(stride),
        "--test-ratio",
        "1.0",
        "--clean",
    ]
    if max_slices > 0:
        pre_cmd.extend(["--max-slices", str(max_slices)])
    run(pre_cmd)

    n_test = len(list((ext_2d / "test" / "quarter_1mm").glob("*.npy")))
    n_tr = len(list((ext_2d / "train512" / "quarter_1mm").glob("*.npy")))
    print(f"Preprocessed: test={n_test} train512={n_tr}")
    if n_test == 0 or n_tr == 0:
        print("Preprocess failed: empty test/ or train512/", file=sys.stderr)
        sys.exit(1)

    for w in (ROOT / "src" / "DA-CLIP.pth", ROOT / "checkpoints" / "FoundDiff" / "sample" / "model-400.pt"):
        if not w.is_file():
            print(f"Missing weight: {w}", file=sys.stderr)
            sys.exit(1)

    train_cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--name",
        "FoundDiff",
        "--epoch",
        "400",
        "--dataset",
        "2020_seen",
        "--data-mode",
        "external",
    ]
    if max_test > 0:
        train_cmd.extend(["--max-test", str(max_test)])
    run(train_cmd, env={"CUDA_VISIBLE_DEVICES": str(args.gpu)})

    out_nii = ROOT / "checkpoints" / "FoundDiff" / f"{case}_denoised.nii.gz"
    recon_cmd = [
        sys.executable,
        str(ROOT / "reconstruct_denoised_nifti.py"),
        "--input-nii",
        str(low),
        "--denoised-dir",
        str(ROOT / "checkpoints" / "FoundDiff" / "test_final_npy"),
        "--output",
        str(out_nii),
        "--volume-name",
        case,
    ]
    if manifest.is_file():
        recon_cmd.extend(["--manifest", str(manifest)])
    else:
        recon_cmd.extend(["--stride", str(stride)])
    run(recon_cmd)

    verify = ROOT / "verify_denoise.py"
    if verify.is_file():
        subprocess.run(
            [
                sys.executable,
                str(verify),
                "--data-root",
                str(ext_2d),
                "--results-dir",
                str(ROOT / "checkpoints" / "FoundDiff" / "test_final_npy"),
                "--manifest",
                str(manifest),
            ],
            cwd=ROOT,
        )

    print("\nDone.")
    print(f"  2D: {ROOT / 'checkpoints' / 'FoundDiff' / 'test_final_npy'}")
    print(f"  3D: {out_nii}")
    print(f"  View: python view_results.py --data-root data/external/external_2d --save-hu-npy")


if __name__ == "__main__":
    main()
