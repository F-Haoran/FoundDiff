#!/usr/bin/env python3
"""
FoundDiff 3D NIfTI LDCT denoising — explicit PyTorch inference in this file.

Pipeline per 2D axial slice:
  1. HU window -> [0,1] (matches data/transforms.py / PDFDataset)
  2. Stage 1 DA-CLIP (frozen, src/DA-CLIP.pth inside Unet): e_dose, e_anatomy
  3. Stage 2 DA-Diff (checkpoints/FoundDiff/sample/model-400.pt): 2-step DDIM
  4. Clean = Input - I_res  (ResidualDiffusion.sample, objective=pred_res)
  5. HU residual restore -> stack 3D -> save .nii.gz

Usage:
  python Test.py
  python Test.py --mode full --process-all
  python Test.py --list
"""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import torch
from ema_pytorch import EMA
from tqdm import tqdm

from nifti_naming import NAMING_CT, collect_input_files, infer_case_name
from src.DADiff import ResidualDiffusion, UnetRes, set_seed

ROOT = Path(__file__).resolve().parent
DACLIP_PATH = ROOT / "src" / "DA-CLIP.pth"
DEFAULT_CKPT = ROOT / "checkpoints" / "FoundDiff" / "sample" / "model-400.pt"

IMAGE_SIZE = 512
HU_MIN = -1000
HU_MAX = 2000
HU_OFFSET = 1024


def config() -> dict[str, Any]:
    root = Path("/home/FrankFei/FoundDiff")
    return {
        "input_path": root / "data/custom/nifti",
        "pattern": "*_CT.nii.gz",
        "naming": NAMING_CT,
        "process_all": False,
        "case_name": None,
        "case_strip_suffixes": (),
        "overwrite": False,
        "output_dir": root / "checkpoints/FoundDiff/custom_denoised_files",
        "output_suffix": "_denoised",
        "mode": "full",  # quick | full
        "gpu": "0",
        "checkpoint": DEFAULT_CKPT,
        "sampling_steps": 2,
        "batch_size": 1,
        "clip_hu": True,
        "seed": 10,
    }


# ---------------------------------------------------------------------------
# HU / tensor helpers
# ---------------------------------------------------------------------------

def clip_hu(hu: np.ndarray) -> np.ndarray:
    return np.clip(hu.astype(np.float32), HU_MIN, HU_MAX)


def hu_to_norm_01(hu: np.ndarray) -> np.ndarray:
    """Same as data/transforms.py Normalize: (HU-1024+1000)/3000 clipped to [0,1]."""
    m = hu.astype(np.float32) - np.float32(HU_OFFSET)
    norm = (m - np.float32(HU_MIN)) / np.float32(HU_MAX - HU_MIN)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def center_crop_box(h: int, w: int, *, roi_h: int = 512, roi_w: int = 512) -> dict[str, int]:
    sh, sw = min(h, roi_h), min(w, roi_w)
    dst_y = max((roi_h - sh) // 2, 0)
    dst_x = max((roi_w - sw) // 2, 0)
    src_y = max((h - sh) // 2, 0)
    src_x = max((w - sw) // 2, 0)
    return {"src_y": src_y, "src_x": src_x, "dst_y": dst_y, "dst_x": dst_x, "sh": sh, "sw": sw}


def hu_delta_from_norm_delta(norm_out: np.ndarray, norm_in: np.ndarray) -> np.ndarray:
    return (np.clip(norm_out, 0.0, 1.0) - np.clip(norm_in, 0.0, 1.0)) * np.float32(3000.0)


def preserve_original_intensity(
    norm_out512: np.ndarray,
    original_slice: np.ndarray,
    crop: dict[str, int],
) -> np.ndarray:
    """HU_out = HU_orig + 3000*(norm_out - norm_in); preserves negative HU on near-identity runs."""
    sy, sx, dy, dx = crop["src_y"], crop["src_x"], crop["dst_y"], crop["dst_x"]
    sh, sw = crop["sh"], crop["sw"]
    roi_orig = original_slice[sy : sy + sh, sx : sx + sw].astype(np.float32, copy=False)
    roi_norm_out = norm_out512[dy : dy + sh, dx : dx + sw]
    norm_in = (roi_orig - np.float32(24.0)) / np.float32(3000.0)
    roi_out = roi_orig + hu_delta_from_norm_delta(roi_norm_out, norm_in)
    out = norm_out512.astype(np.float32, copy=True)
    out[dy : dy + sh, dx : dx + sw] = roi_out
    return out


def center_crop_pad_512(slice2d: np.ndarray, fill: float = float(HU_MIN)) -> np.ndarray:
    h, w = slice2d.shape
    out = np.full((IMAGE_SIZE, IMAGE_SIZE), fill, dtype=np.float32)
    sh, sw = min(h, IMAGE_SIZE), min(w, IMAGE_SIZE)
    y0 = max((IMAGE_SIZE - sh) // 2, 0)
    x0 = max((IMAGE_SIZE - sw) // 2, 0)
    sy0 = max((h - sh) // 2, 0)
    sx0 = max((w - sw) // 2, 0)
    out[y0 : y0 + sh, x0 : x0 + sw] = slice2d[sy0 : sy0 + sh, sx0 : sx0 + sw]
    return out


def load_nifti_volume(path: Path) -> tuple[np.ndarray, nib.Nifti1Image]:
    img = nib.load(str(path))
    vol = np.asarray(img.get_fdata(), dtype=np.float32)
    if vol.ndim != 3:
        raise ValueError(f"Expected 3D NIfTI, got shape {vol.shape} for {path}")
    vol = np.transpose(vol, (2, 1, 0))  # Z, Y, X
    return vol, img


def save_nifti_volume(vol_zyx: np.ndarray, ref_img: nib.Nifti1Image, out_path: Path) -> None:
    header = ref_img.header.copy()
    header.set_data_dtype(np.float32)
    header["scl_slope"] = np.float32(1.0)
    header["scl_inter"] = np.float32(0.0)
    finite = vol_zyx[np.isfinite(vol_zyx)]
    if finite.size:
        header["cal_min"] = float(finite.min())
        header["cal_max"] = float(finite.max())
    out_xyz = np.transpose(vol_zyx, (2, 1, 0))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(out_xyz.astype(np.float32), ref_img.affine, header), str(out_path))


# ---------------------------------------------------------------------------
# FoundDiff model (train.py architecture)
# ---------------------------------------------------------------------------

def build_founddiff(sampling_timesteps: int) -> ResidualDiffusion:
    unet = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=1,
        condition=True,
        input_condition=False,
        objective="pred_res",
        test_res_or_noise="res",
    )
    return ResidualDiffusion(
        unet,
        image_size=IMAGE_SIZE,
        timesteps=1000,
        sampling_timesteps=sampling_timesteps,
        objective="pred_res",
        loss_type="l2",
        condition=True,
        sum_scale=0.01,
        input_condition=False,
        input_condition_mask=False,
        test_res_or_noise="res",
    )


def load_founddiff_ema(checkpoint: Path, sampling_steps: int, device: torch.device) -> ResidualDiffusion:
    if not DACLIP_PATH.is_file():
        raise FileNotFoundError(f"Missing DA-CLIP weights: {DACLIP_PATH}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing FoundDiff checkpoint: {checkpoint}")

    diffusion = build_founddiff(sampling_steps)
    ema = EMA(diffusion, beta=0.995, update_every=1)
    payload = torch.load(str(checkpoint), map_location="cpu")
    diffusion.load_state_dict(payload["model"])
    ema.load_state_dict(payload["ema"])
    ema.to(device)

    model = ema.ema_model
    model.to(device)
    model.eval()
    model.init()

    for p in model.parameters():
        p.requires_grad = False
    if hasattr(model.model, "unet0") and hasattr(model.model.unet0, "dose_encoder"):
        model.model.unet0.dose_encoder.eval()
    return model


@torch.no_grad()
def extract_daclip_embeddings(unet: UnetRes, ldct_bchw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stage 1 — DA-CLIP (frozen, weights in src/DA-CLIP.pth via Unet.dose_encoder).

    Returns:
        e_dose    (1024-d): modulates diffusion timestep embedding
        e_anatomy (256-d): cross-attention context in Mamba blocks
    """
    rgb = ldct_bchw[:, :1].repeat(1, 3, 1, 1)
    _, e_dose, e_anatomy = unet.dose_encoder(rgb)
    return e_dose, e_anatomy


@torch.no_grad()
def ddim_denoise_slice(
    model: ResidualDiffusion,
    unet: UnetRes,
    ldct_norm_01: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Stage 2 — DA-Diff DDIM (2 steps by default).

    ResidualDiffusion.sample():
      - concat(x_t, ldct) -> Unet -> pred_res
      - x_clean = ldct - pred_res   (in [-1,1], then mapped to [0,1])

    Returns:
        norm_out [B,1,H,W] in [0,1], e_dose, e_anatomy
    """
    ldct_norm_01 = ldct_norm_01.to(model.device)
    e_dose, e_anatomy = extract_daclip_embeddings(unet, ldct_norm_01)
    samples = model.sample(ldct_norm_01, batch_size=ldct_norm_01.shape[0], last=True)
    norm_out = samples[-1].clamp(0.0, 1.0)
    return norm_out, e_dose, e_anatomy


def denoise_hu_slice(
    model: ResidualDiffusion,
    unet: UnetRes,
    slice_hu: np.ndarray,
    *,
    clip_hu: bool,
) -> np.ndarray:
    """Run FoundDiff on one full-resolution 2D HU slice; return denoised HU slice."""
    original = clip_hu(slice_hu) if clip_hu else slice_hu.astype(np.float32)
    hu512 = center_crop_pad_512(original, fill=float(HU_MIN))
    norm_in = hu_to_norm_01(hu512)

    ldct = torch.from_numpy(norm_in).unsqueeze(0).unsqueeze(0).to(model.device)  # [1,1,512,512]
    norm_out, _, _ = ddim_denoise_slice(model, unet, ldct)
    norm_out_np = norm_out[0, 0].detach().cpu().numpy().astype(np.float32)

    crop = center_crop_box(original.shape[0], original.shape[1])
    denoised512 = preserve_original_intensity(norm_out_np, original, crop)

    out = original.copy()
    sy, sx = crop["src_y"], crop["src_x"]
    dy, dx = crop["dst_y"], crop["dst_x"]
    sh, sw = crop["sh"], crop["sw"]
    out[sy : sy + sh, sx : sx + sw] = denoised512[dy : dy + sh, dx : dx + sw]
    return out


def denoise_hu_batch_slices(
    model: ResidualDiffusion,
    unet: UnetRes,
    slices_hu: list[np.ndarray],
    *,
    clip_hu: bool,
) -> list[np.ndarray]:
    """Batch version when all slices share 512x512 after crop (same H,W)."""
    originals = [clip_hu(s) if clip_hu else s.astype(np.float32) for s in slices_hu]
    hu512_list = [center_crop_pad_512(s, fill=float(HU_MIN)) for s in originals]
    norms = np.stack([hu_to_norm_01(x) for x in hu512_list], axis=0)
    ldct = torch.from_numpy(norms).unsqueeze(1).to(model.device)  # [B,1,512,512]

    norm_out, _, _ = ddim_denoise_slice(model, unet, ldct)
    norm_out_np = norm_out[:, 0].detach().cpu().numpy()

    results: list[np.ndarray] = []
    for i, original in enumerate(originals):
        crop = center_crop_box(original.shape[0], original.shape[1])
        denoised512 = preserve_original_intensity(norm_out_np[i], original, crop)
        out = original.copy()
        sy, sx = crop["src_y"], crop["src_x"]
        dy, dx = crop["dst_y"], crop["dst_x"]
        sh, sw = crop["sh"], crop["sw"]
        out[sy : sy + sh, sx : sx + sw] = denoised512[dy : dy + sh, dx : dx + sw]
        results.append(out)
    return results


def slice_indices(nz: int, mode: str) -> range:
    if mode == "quick":
        return range(0, min(nz, 50), 2)
    if mode == "full":
        return range(nz)
    raise ValueError(f"Unknown mode {mode!r}; use quick or full")


def process_nifti_volume(
    input_nii: Path,
    output_nii: Path,
    model: ResidualDiffusion,
    unet: UnetRes,
    *,
    mode: str,
    batch_size: int,
    clip_hu: bool,
    overwrite: bool,
) -> Path:
    if output_nii.is_file() and not overwrite:
        print(f"Skip existing: {output_nii}")
        return output_nii

    vol, ref_img = load_nifti_volume(input_nii)
    nz, ny, nx = vol.shape
    out_vol = vol.copy()
    z_indices = list(slice_indices(nz, mode))

    print(f"Inference: {input_nii.name}  shape Z={nz} Y={ny} X={nx}  slices={len(z_indices)}  mode={mode}")

    batch: list[tuple[int, np.ndarray]] = []
    for z in tqdm(z_indices, desc="FoundDiff DDIM", unit="slice"):
        batch.append((z, vol[z]))
        if len(batch) < batch_size:
            continue

        zs = [item[0] for item in batch]
        sls = [item[1] for item in batch]
        if batch_size == 1:
            denoised = [denoise_hu_slice(model, unet, sls[0], clip_hu=clip_hu)]
        else:
            denoised = denoise_hu_batch_slices(model, unet, sls, clip_hu=clip_hu)
        for zi, sl_out in zip(zs, denoised):
            out_vol[zi] = sl_out
        batch.clear()

    if batch:
        zs = [item[0] for item in batch]
        sls = [item[1] for item in batch]
        if batch_size == 1:
            denoised = [denoise_hu_slice(model, unet, sls[0], clip_hu=clip_hu)]
        else:
            denoised = denoise_hu_batch_slices(model, unet, sls, clip_hu=clip_hu)
        for zi, sl_out in zip(zs, denoised):
            out_vol[zi] = sl_out

    save_nifti_volume(out_vol, ref_img, output_nii)
    fin = out_vol[np.isfinite(out_vol)]
    print(
        f"Saved {output_nii}\n"
        f"  input HU:  min={vol.min():.3f} max={vol.max():.3f} mean={vol.mean():.3f}\n"
        f"  output HU: min={fin.min():.3f} max={fin.max():.3f} mean={fin.mean():.3f}"
    )
    return output_nii


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def resolve_inputs(cfg: dict[str, Any]) -> list[Path]:
    files = collect_input_files(
        Path(cfg["input_path"]),
        pattern=str(cfg.get("pattern", "*.nii.gz")),
        naming=str(cfg.get("naming", NAMING_CT)),
    )
    if not cfg.get("process_all", False):
        return files[:1]
    return files


def output_path(cfg: dict[str, Any], case: str) -> Path:
    return Path(cfg["output_dir"]) / f"{case}{cfg.get('output_suffix', '_denoised')}.nii.gz"


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
    if args.gpu is not None:
        out["gpu"] = args.gpu
    if args.batch_size is not None:
        out["batch_size"] = args.batch_size
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FoundDiff 3D NIfTI denoising (explicit inference)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--show", action="store_true")
    p.add_argument("--input-path", type=Path, default=None)
    p.add_argument("--pattern", default=None)
    p.add_argument("--process-all", action="store_true")
    p.add_argument("--mode", choices=("quick", "full"), default=None)
    p.add_argument("--gpu", default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    cfg = apply_cli_overrides(config(), args)
    if args.overwrite:
        cfg["overwrite"] = True

    inputs = resolve_inputs(cfg)
    if args.show or args.list:
        print("=== Test.py config ===")
        for k, v in cfg.items():
            print(f"  {k}: {v}")
        print(f"Matched inputs ({len(inputs)}):")
        for p in inputs:
            print(f"  {p}")

    if args.list:
        return 0
    if not inputs:
        print("No input NIfTI matched.", file=sys.stderr)
        return 1

    if not torch.cuda.is_available():
        print("CUDA required (Mamba selective_scan_cuda).", file=sys.stderr)
        return 1

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(cfg.get("gpu", "0"))
    device = torch.device("cuda")
    set_seed(int(cfg.get("seed", 10)))

    ckpt = Path(cfg.get("checkpoint", DEFAULT_CKPT))
    steps = int(cfg.get("sampling_steps", 2))
    print(f"Loading FoundDiff EMA from {ckpt} (DDIM steps={steps}) on {device}")
    model = load_founddiff_ema(ckpt, steps, device)
    unet = model.model.unet0

    failures: list[str] = []
    outputs: list[Path] = []
    for i, nii_path in enumerate(inputs, start=1):
        case = infer_case_name(nii_path, cfg.get("case_name"), strip_suffixes=cfg.get("case_strip_suffixes") or None)
        out_nii = output_path(cfg, case)
        if len(inputs) > 1:
            print(f"\n[{i}/{len(inputs)}] {nii_path.name} -> {out_nii.name}")
        try:
            outputs.append(
                process_nifti_volume(
                    nii_path,
                    out_nii,
                    model,
                    unet,
                    mode=str(cfg.get("mode", "full")),
                    batch_size=max(1, int(cfg.get("batch_size", 1))),
                    clip_hu=bool(cfg.get("clip_hu", True)),
                    overwrite=bool(cfg.get("overwrite", False)),
                )
            )
        except Exception as exc:
            print(f"FAILED {nii_path.name}: {exc}", file=sys.stderr)
            failures.append(nii_path.name)

    print("\nDone.")
    for p in outputs:
        print(f"  {p}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
