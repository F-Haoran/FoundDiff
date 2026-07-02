#!/usr/bin/env python3
"""
单文件 FoundDiff 去噪入口 —— 改 config() 后直接运行，无需长命令行。

用法:
  python run_one_nifti.py              # 使用下方 config()
  python run_one_nifti.py --show       # 打印当前配置
  python run_one_nifti.py --dry-run    # 只打印将要执行的命令

Pipeline 概览 (mode=quick / full):
  输入 .nii.gz
    -> Preprocess_nifti.py   (3D -> 2D lung-*.npy + slice_manifest.json)
    -> train.py --data-mode external   (FoundDiff 推理，输出 2D denoised npy)
    -> reconstruct_denoised_nifti.py (2D npy -> 3D 去噪 .nii.gz)

mode=reconstruct_only 时跳过前两步，只重建 3D（调强度范围时用）。
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent


def config() -> dict[str, Any]:
    """在这里改路径和参数，然后运行: python run_one_nifti.py"""

    # 本地 FoundDiff 根目录（Maybach 示例）
    root = Path("/home/FrankFei/FoundDiff")

    return {
        # ----- 输入 / 输出 -----
        # 必须用 LDCT（低剂量），不要用 *_CT.nii.gz（全剂量参考图）
        "input_nii": root / "data/custom/nifti/APNHC00002_LDCT.nii.gz",
        "case_name": None,  # None = 从文件名自动推断（如 APNHC00718_LDCT）
        "output_dir": root / "checkpoints/FoundDiff/custom_denoised_files",
        "output_suffix": "_denoised",

        # ----- 运行模式 -----
        # quick            : 快速试跑（约 50 层预处理 + 10 层推理，几分钟）
        # full             : 全层处理（完整 3D 输出）
        # reconstruct_only : 仅重建 3D（2D npy 已有时，调强度参数用）
        "mode": "quick",
        "gpu": "0",

        # ----- 强度映射（重建阶段）-----
        "intensity_scale": "slice-range",  # slice-range | founddiff-hu | identity | unit
        "intensity_match": "minmax",       # minmax | mean-ratio | none
        "range_source": "slice",           # slice | roi
        "range_stats_min": -2000.0,
        "range_stats_max": 2000.0,
        "range_ignore_at_or_below": -2500.0,  # 忽略 -3000 填充
        "range_percentile": None,             # 例如 "2,98"；None = 不用百分位
        "range_fixed_min": -1500.0,           # None = 自动估计；可固定组织窗
        "range_fixed_max": 1500.0,

        # ----- 内部路径（一般不用改）-----
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
        raise SystemExit(f"build_pipeline_cmd 需要 mode=quick/full，当前为 {mode!r}")

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
    print("=== run_one_nifti.py 当前配置 ===")
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
        raise SystemExit(f"输入文件不存在: {input_nii}")

    fixed_min = cfg.get("range_fixed_min")
    fixed_max = cfg.get("range_fixed_max")
    if (fixed_min is None) ^ (fixed_max is None):
        raise SystemExit("range_fixed_min 和 range_fixed_max 必须同时设置，或同时为 None")

    mode = cfg.get("mode", "quick")
    if mode not in {"quick", "full", "reconstruct_only"}:
        raise SystemExit(f"未知 mode: {mode!r}，可选 quick | full | reconstruct_only")


def run_cmd(cmd: list[str], *, dry_run: bool) -> None:
    printable = " ".join(cmd)
    print(f"\n>> {printable}\n")
    if dry_run:
        return
    subprocess.run(cmd, cwd=ROOT, check=True)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="单文件 FoundDiff 去噪（config 驱动）")
    p.add_argument("--show", action="store_true", help="打印 config() 内容")
    p.add_argument("--dry-run", action="store_true", help="只打印命令，不执行")
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
            raise SystemExit(f"未知 mode: {mode!r}，可选 quick | full | reconstruct_only")

    mode = cfg.get("mode", "quick")
    if mode == "reconstruct_only":
        run_cmd(build_reconstruct_cmd(cfg), dry_run=args.dry_run)
    else:
        run_cmd(build_pipeline_cmd(cfg), dry_run=args.dry_run)

    if not args.dry_run:
        case = resolve_case(cfg)
        out = Path(cfg["output_dir"]) / f"{case}{cfg.get('output_suffix', '_denoised')}.nii.gz"
        print(f"\n完成。输出: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
