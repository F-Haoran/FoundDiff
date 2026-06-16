#!/usr/bin/env python3
"""Run train.py external inference only (no hidden shell characters)."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max-test", type=int, default=0, help="0 = all test slices")
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    args = p.parse_args()

    cmd = [
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
    if args.max_test > 0:
        cmd.extend(["--max-test", str(args.max_test)])

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


if __name__ == "__main__":
    main()
