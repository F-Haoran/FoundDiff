#!/usr/bin/env python3
"""View FoundDiff denoising outputs (.npy) with optional LDCT/NDCT comparison."""

import argparse
import glob
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

from data.paths import MAYO2020_HEAD

# Same as PDFDataset val transforms in data/pdf_dataset.py
HU_MIN = -1000
HU_MAX = 2000
HU_RANGE = HU_MAX - HU_MIN  # 3000
WINDOW_MIN = -1000
WINDOW_MAX = 400


def normalize_hu(hu: np.ndarray) -> np.ndarray:
    """Forward: raw HU -> [0,1], same as data/transforms.py Normalize."""
    m = hu.astype(np.float32) - 1024
    norm = (m - HU_MIN) / HU_RANGE
    return np.clip(norm, 0, 1)


def denorm_to_hu(norm_0_1: np.ndarray) -> np.ndarray:
    """Inverse: [0,1] -> HU.  HU = norm * 3000 + (-1000) + 1024 = norm * 3000 + 24"""
    return norm_0_1.astype(np.float32) * HU_RANGE + HU_MIN + 1024


def hu_via_model_scale(hu: np.ndarray) -> np.ndarray:
    """Raw HU through training normalize, then inverse scale back to HU."""
    return denorm_to_hu(normalize_hu(hu))


def load_slice(path: str) -> np.ndarray:
    arr = np.load(path)
    return np.squeeze(arr).astype(np.float32)


def find_pairs(result_path: str, data_root: str):
    """Map denoised file to LDCT / NDCT paths if they exist."""
    name = os.path.basename(result_path)
    stem = name[:-4] if name.endswith(".npy") else name
    parts = stem.split("-")
    if len(parts) >= 2 and parts[0] in ("head", "lung", "ab"):
        if len(parts) == 2:
            ldct = os.path.join(data_root, "test", "quarter_1mm", f"{stem}.npy")
            ndct = os.path.join(data_root, "test", "full_1mm", f"{stem}.npy")
            return (
                ldct if os.path.isfile(ldct) else None,
                ndct if os.path.isfile(ndct) else None,
            )
        dose = parts[1]
        ldct = os.path.join(data_root, "test", f"sim-{dose}", f"{stem}.npy")
        ndct = os.path.join(data_root, "test", "full_1mm", f"{stem}.npy")
        if not os.path.isfile(ldct):
            ldct = os.path.join(data_root, "test", "quarter_1mm", f"{stem}.npy")
        return ldct if os.path.isfile(ldct) else None, ndct if os.path.isfile(ndct) else None
    return None, None


def show_slice(ax, img, title, *, hu_window=False):
    if hu_window:
        ax.imshow(img, cmap="gray", vmin=WINDOW_MIN, vmax=WINDOW_MAX)
    else:
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")


def view_one(result_path, data_root, save_dir, show, display_mode, save_hu_npy):
    denoised_norm = load_slice(result_path)
    name = os.path.splitext(os.path.basename(result_path))[0]
    ldct_path, ndct_path = find_pairs(result_path, data_root)

    # Denoised: model outputs norm -> inverse scale to HU
    denoised_hu = denorm_to_hu(denoised_norm)

    ncols = 1
    if ldct_path:
        ncols += 1
    if ndct_path:
        ncols += 1

    fig, axes = plt.subplots(1, ncols, figsize=(4 * ncols, 4))
    if ncols == 1:
        axes = [axes]

    col = 0
    if ldct_path:
        ldct_raw = load_slice(ldct_path)
        if display_mode == "hu":
            ldct_show = hu_via_model_scale(ldct_raw)
            show_slice(axes[col], ldct_show, "LDCT (HU, model scale)", hu_window=True)
        else:
            show_slice(axes[col], normalize_hu(ldct_raw), "LDCT (norm)", hu_window=False)
        col += 1

    if display_mode == "hu":
        show_slice(axes[col], denoised_hu, "Denoised (HU, inverse scale)", hu_window=True)
    else:
        show_slice(axes[col], denoised_norm, "Denoised (norm)", hu_window=False)
    col += 1

    if ndct_path:
        ndct_raw = load_slice(ndct_path)
        if display_mode == "hu":
            ndct_show = hu_via_model_scale(ndct_raw)
            show_slice(axes[col], ndct_show, "NDCT (HU, model scale)", hu_window=True)
        else:
            show_slice(axes[col], normalize_hu(ndct_raw), "NDCT (norm)", hu_window=False)

    fig.suptitle(name, fontsize=11)
    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        out_png = os.path.join(save_dir, f"{name}_view.png")
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
        print(f"saved {out_png}")

    if save_hu_npy and save_dir:
        hu_dir = os.path.join(save_dir, "hu_npy")
        os.makedirs(hu_dir, exist_ok=True)
        hu_path = os.path.join(hu_dir, f"{name}_denoised_hu.npy")
        np.save(hu_path, denoised_hu.astype(np.float32))
        print(f"saved {hu_path}  HU range [{denoised_hu.min():.0f}, {denoised_hu.max():.0f}]")

    if show:
        plt.show()
    else:
        plt.close(fig)

    print(
        f"{name}: norm [{denoised_norm.min():.4f}, {denoised_norm.max():.4f}] "
        f"-> HU [{denoised_hu.min():.0f}, {denoised_hu.max():.0f}]"
    )
    if ndct_path:
        ref_norm = normalize_hu(load_slice(ndct_path))
        rmse_norm = np.sqrt(np.mean((denoised_norm - ref_norm) ** 2))
        ref_hu = hu_via_model_scale(load_slice(ndct_path))
        rmse_hu = np.sqrt(np.mean((denoised_hu - ref_hu) ** 2))
        print(f"  vs NDCT RMSE norm: {rmse_norm:.4f}  |  vs NDCT RMSE HU (model scale): {rmse_hu:.1f}")


def main():
    parser = argparse.ArgumentParser(description="View FoundDiff .npy denoising results.")
    parser.add_argument(
        "--results-dir",
        default="checkpoints/FoundDiff/test_final_npy",
        help="Folder with denoised output .npy files",
    )
    parser.add_argument(
        "--data-root",
        default=MAYO2020_HEAD,
        help="Mayo2020 data root for LDCT/NDCT pairing",
    )
    parser.add_argument(
        "--save-dir",
        default="checkpoints/FoundDiff/test_final_preview",
        help="Save PNG previews here",
    )
    parser.add_argument(
        "--show-only",
        action="store_true",
        help="Open matplotlib windows instead of saving PNGs",
    )
    parser.add_argument(
        "--norm",
        action="store_true",
        help="Show all panels in norm [0,1] (training space). Default: inverse-scale to HU",
    )
    parser.add_argument(
        "--save-hu-npy",
        action="store_true",
        help="Also save denoised inverse-scaled HU .npy under save-dir/hu_npy/",
    )
    parser.add_argument(
        "files",
        nargs="*",
        help="Specific .npy files (default: all in --results-dir)",
    )
    args = parser.parse_args()

    display_mode = "norm" if args.norm else "hu"

    if args.files:
        paths = args.files
    else:
        paths = sorted(glob.glob(os.path.join(args.results_dir, "*.npy")))
    if not paths:
        print(f"No .npy files found in {args.results_dir}", file=sys.stderr)
        sys.exit(1)

    save_dir = None if args.show_only else args.save_dir
    for path in paths:
        view_one(path, args.data_root, save_dir, args.show_only, display_mode, args.save_hu_npy)


if __name__ == "__main__":
    main()
