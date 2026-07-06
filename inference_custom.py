#!/usr/bin/env python3
"""
FoundDiff custom LDCT denoising inference script.

Loads pre-trained weights:
  - src/DA-CLIP.pth          (frozen DA-CLIP inside Unet; e_dose / e_anatomy)
  - checkpoints/FoundDiff/sample/model-400.pt  (ResidualDiffusion + EMA)

Expected custom data directory: ./my_custom_data/
Supported inputs: .npy (HU), .nii/.nii.gz, .dcm, .png/.jpg/.tif

Normalization matches data/pdf_dataset.py + src/DADiff.py inference:
  HU -> [0, 1] via (HU - 24) / 3000 clipped, then DDIM uses [-1, 1] internally.

Usage:
  python inference_custom.py
  python inference_custom.py --data-dir ./my_custom_data --output-dir ./output_denoised
  python inference_custom.py --batch-size 2 --sampling-steps 2
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from ema_pytorch import EMA
from torch.utils.data import DataLoader, Dataset
from torchvision import utils

from src.DADiff import ResidualDiffusion, UnetRes, set_seed

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data" / "custom" / "nifti"
DEFAULT_OUTPUT = ROOT / "checkpoints" / "FoundDiff" / "custom_denoised_files"
DEFAULT_CKPT = ROOT / "checkpoints" / "FoundDiff" / "sample" / "model-400.pt"
DACLIP_PATH = ROOT / "src" / "DA-CLIP.pth"

IMAGE_SIZE = 512
HU_MIN = -1000
HU_MAX = 2000
HU_OFFSET = 1024  # same as data/transforms.py Normalize


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="FoundDiff custom LDCT inference")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA, help="Input folder")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT, help="Save denoised images here")
    p.add_argument("--checkpoint", type=Path, default=DEFAULT_CKPT, help="model-400.pt path")
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--sampling-steps", type=int, default=2, help="DDIM steps (T=1000 trained, 2 at inference)")
    p.add_argument("--device", default="cuda", help="cuda or cpu (cuda required for Mamba)")
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--save-hu-npy", action="store_true", help="Also save denoised HU .npy")
    p.add_argument("--extensions", nargs="+", default=[".npy", ".nii", ".nii.gz", ".dcm", ".png", ".jpg", ".jpeg", ".tif", ".tiff"])
    return p.parse_args()


# ---------------------------------------------------------------------------
# HU <-> model normalization (must match PDFDataset / FoundDiff training)
# ---------------------------------------------------------------------------

def hu_to_norm_01(hu: np.ndarray) -> np.ndarray:
    """Map HU to [0, 1] exactly like data/transforms.py Normalize."""
    m = hu.astype(np.float32) - np.float32(HU_OFFSET)
    norm = (m - np.float32(HU_MIN)) / np.float32(HU_MAX - HU_MIN)
    return np.clip(norm, 0.0, 1.0).astype(np.float32)


def norm_01_to_hu(norm: np.ndarray) -> np.ndarray:
    """Inverse of hu_to_norm_01: HU = norm * 3000 + 24."""
    return norm.astype(np.float32) * np.float32(3000.0) + np.float32(24.0)


def norm_01_to_hu_preserve_original(hu_orig: np.ndarray, norm_in: np.ndarray, norm_out: np.ndarray) -> np.ndarray:
    """HU_out = HU_orig + 3000*(norm_out - norm_in); keeps negative HU on near-identity runs."""
    return hu_orig.astype(np.float32) + (np.clip(norm_out, 0, 1) - np.clip(norm_in, 0, 1)) * np.float32(3000.0)


def center_crop_pad_512(arr2d: np.ndarray, fill: float = 0.0) -> np.ndarray:
    h, w = arr2d.shape
    out = np.full((IMAGE_SIZE, IMAGE_SIZE), fill, dtype=np.float32)
    sh, sw = min(h, IMAGE_SIZE), min(w, IMAGE_SIZE)
    y0 = max((IMAGE_SIZE - sh) // 2, 0)
    x0 = max((IMAGE_SIZE - sw) // 2, 0)
    sy0 = max((h - sh) // 2, 0)
    sx0 = max((w - sw) // 2, 0)
    out[y0 : y0 + sh, x0 : x0 + sw] = arr2d[sy0 : sy0 + sh, sx0 : sx0 + sw]
    return out


# ---------------------------------------------------------------------------
# Custom dataset
# ---------------------------------------------------------------------------

def collect_data_files(data_dir: Path, extensions: list[str]) -> list[Path]:
    files: list[Path] = []
    seen: set[Path] = set()
    for ext in extensions:
        ext_lower = ext.lower()
        for path in sorted(data_dir.rglob("*")):
            if not path.is_file():
                continue
            name = path.name.lower()
            if ext_lower == ".nii.gz" and name.endswith(".nii.gz"):
                key = path.resolve()
            elif ext_lower != ".nii.gz" and name.endswith(ext_lower):
                key = path.resolve()
            else:
                continue
            if key not in seen:
                seen.add(key)
                files.append(path)
    return sorted(files, key=lambda p: str(p).lower())


def load_hu_slice_from_file(path: Path) -> list[tuple[str, np.ndarray]]:
    """Return list of (slice_id, hu_2d) items. Volumes expand to multiple axial slices."""
    suffix = path.name.lower()
    if suffix.endswith(".npy"):
        arr = np.load(path)
        if arr.ndim == 2:
            return [(path.stem, arr.astype(np.float32))]
        if arr.ndim == 3:
            return [(f"{path.stem}_z{z:04d}", arr[z].astype(np.float32)) for z in range(arr.shape[0])]
        raise ValueError(f"Unsupported npy shape {arr.shape} in {path}")

    if suffix.endswith(".nii") or suffix.endswith(".nii.gz"):
        import nibabel as nib

        vol = np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)
        if vol.ndim == 2:
            return [(path.stem, vol)]
        if vol.ndim == 3:
            vol = np.transpose(vol, (2, 1, 0))
            return [(f"{path.stem}_z{z:04d}", vol[z]) for z in range(vol.shape[0])]
        raise ValueError(f"Unsupported nifti shape {vol.shape} in {path}")

    if suffix.endswith(".dcm"):
        import pydicom

        ds = pydicom.dcmread(str(path))
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        hu = arr * slope + intercept
        return [(path.stem, hu)]

    if suffix.endswith((".png", ".jpg", ".jpeg", ".tif", ".tiff")):
        from PIL import Image

        img = np.asarray(Image.open(path).convert("L"), dtype=np.float32)
        # Assume 8-bit PNG stores windowed HU mapped to 0..255 for [-1000, 2000]
        hu = img / 255.0 * (HU_MAX - HU_MIN) + HU_MIN
        return [(path.stem, hu)]

    raise ValueError(f"Unsupported file type: {path}")


class CustomLDCTDataset(Dataset):
    """
    Loads LDCT slices from ./my_custom_data/.

    Each item:
      hu_orig: (H, W) float32 HU
      norm_in: (1, 512, 512) float32 in [0, 1]
      meta: dict with source path and slice id
    """

    def __init__(self, data_dir: Path, extensions: list[str]):
        self.items: list[tuple[str, np.ndarray, Path]] = []
        if not data_dir.is_dir():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        files = collect_data_files(data_dir, extensions)
        if not files:
            raise FileNotFoundError(
                f"No supported files under {data_dir}. "
                f"Place .npy / .nii.gz / .dcm / .png files in my_custom_data/."
            )

        for fpath in files:
            for slice_id, hu2d in load_hu_slice_from_file(fpath):
                self.items.append((slice_id, hu2d, fpath))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict:
        slice_id, hu2d, src = self.items[index]
        hu512 = center_crop_pad_512(hu2d, fill=float(HU_MIN))
        norm = hu_to_norm_01(hu512)
        tensor = torch.from_numpy(norm).unsqueeze(0)  # (1, H, W)
        return {
            "ldct": tensor,
            "hu_orig": torch.from_numpy(hu512),
            "norm_in": torch.from_numpy(norm),
            "slice_id": slice_id,
            "source": str(src),
        }


# ---------------------------------------------------------------------------
# Model build / load (matches train.py)
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
    diffusion = ResidualDiffusion(
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
    return diffusion


def load_ema_model(checkpoint: Path, sampling_timesteps: int, device: torch.device) -> ResidualDiffusion:
    if not DACLIP_PATH.is_file():
        raise FileNotFoundError(f"Missing DA-CLIP weights: {DACLIP_PATH}")
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing FoundDiff checkpoint: {checkpoint}")

    diffusion = build_founddiff(sampling_timesteps)
    ema = EMA(diffusion, beta=0.995, update_every=1)

    data = torch.load(str(checkpoint), map_location="cpu")
    diffusion.load_state_dict(data["model"])
    ema.load_state_dict(data["ema"])
    ema.to(device)

    model = ema.ema_model
    model.to(device)
    model.eval()
    model.init()
    return model


@torch.no_grad()
def extract_daclip_embeddings(unet: UnetRes, ldct_norm_01: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Stage 1 (DA-CLIP): frozen encoder inside Unet.

    Returns:
      e_dose     (e_dose / dose_embedding, 1024-d) — modulates diffusion timestep
      e_anatomy  (e_anatomy / context_embedding, 256-d) — cross-attention in Mamba blocks
    """
    # Unet.forward uses channel 1 (LDCT condition). Replicate to pseudo-RGB for CLIP.
    rgb = ldct_norm_01.unsqueeze(1).repeat(1, 3, 1, 1)
    _, e_dose, e_anatomy = unet.dose_encoder(rgb)
    return e_dose, e_anatomy


@torch.no_grad()
def ddim_denoise_batch(model: ResidualDiffusion, ldct_norm_01: torch.Tensor) -> torch.Tensor:
    """
    Stage 2 (DA-Diff): 2-step DDIM conditioned on LDCT.

    Internally Unet concatenates (x_t, ldct), runs DA-CLIP on ldct, predicts pred_res,
    and returns clean estimate x_start = ldct - pred_res in [-1, 1], then mapped to [0, 1].
    """
    x_cond = ldct_norm_01.to(model.device)
    samples = model.sample(x_cond, batch_size=x_cond.shape[0], last=True)
    return samples[-1].clamp(0.0, 1.0)


def save_outputs(
    output_dir: Path,
    slice_ids: list[str],
    norm_in: torch.Tensor,
    norm_out: torch.Tensor,
    hu_orig: torch.Tensor,
    save_hu_npy: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for i, sid in enumerate(slice_ids):
        safe = sid.replace("/", "_")
        ni = norm_in[i, 0].cpu().numpy()
        no = norm_out[i, 0].cpu().numpy()
        hu = hu_orig[i].cpu().numpy()
        hu_denoised = norm_01_to_hu_preserve_original(hu, ni, no)

        np.save(output_dir / f"{safe}_denoised_norm.npy", no.astype(np.float32))
        if save_hu_npy:
            np.save(output_dir / f"{safe}_denoised_hu.npy", hu_denoised.astype(np.float32))

        # PNG preview: window [-1000, 2000] HU -> 0..1 for display
        preview = np.clip((hu_denoised - HU_MIN) / (HU_MAX - HU_MIN), 0.0, 1.0)
        utils.save_image(torch.from_numpy(preview).unsqueeze(0), str(output_dir / f"{safe}_denoised.png"))


def main() -> int:
    args = parse_args()
    set_seed(args.seed)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print("CUDA not available. FoundDiff requires GPU (Mamba selective_scan_cuda).", file=sys.stderr)
        return 1

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading FoundDiff checkpoint: {args.checkpoint}")
    model = load_ema_model(args.checkpoint, args.sampling_steps, device)

    dataset = CustomLDCTDataset(args.data_dir, args.extensions)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    print(f"Found {len(dataset)} slice(s) under {args.data_dir}")

    unet = model.model.unet0
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for batch in loader:
        ldct = batch["ldct"].to(device, non_blocking=True)  # (B, 1, 512, 512) in [0, 1]
        hu_orig = batch["hu_orig"]
        norm_in = batch["norm_in"]
        slice_ids = batch["slice_id"]

        # Stage 1: DA-CLIP embeddings (explicit call for clarity; also used inside Unet during DDIM)
        e_dose, e_anatomy = extract_daclip_embeddings(unet, ldct)
        _ = (e_dose, e_anatomy)  # consumed inside model.sample(); kept for debugging / logging

        # Stage 2: DDIM sampling -> denoised norm in [0, 1]
        norm_out = ddim_denoise_batch(model, ldct)

        save_outputs(
            args.output_dir,
            list(slice_ids),
            norm_in,
            norm_out,
            hu_orig,
            args.save_hu_npy,
        )
        print(f"Saved batch: {', '.join(slice_ids)}")

    print(f"\nDone. Denoised outputs -> {args.output_dir.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
