#!/usr/bin/env python3
"""
End-to-end FoundDiff denoising: NIfTI -> process in memory -> NIfTI.

Intermediate .npy files are NOT required.
Typical speed on GPU: ~20-30 sec/slice (batch_size=1). Use --batch_size 2 or 4 if VRAM allows.
"""

import argparse
import csv
import sys
import time
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from ema_pytorch import EMA
from skimage.transform import resize
from tqdm import tqdm

from intensity import (
    HU_MIN, HU_MAX, HU_RANGE, hu_to_model_norm, model_norm_to_hu, make_hu_nifti_header,
)
from src.DADiff import ResidualDiffusion, UnetRes, set_seed


def slice_stats(arr):
    mn = float(np.min(arr))
    mx = float(np.max(arr))
    return {
        'min': mn,
        'max': mx,
        'mean': float(np.mean(arr)),
        'range': mx - mn,
    }


def linear_match_intensity(out, target_min, target_max, eps=1e-6):
    """
    Linear map out domain [out_min, out_max] -> [target_min, target_max].
    matched = scale * out + offset
    """
    out = out.astype(np.float32)
    o_min = float(out.min())
    o_max = float(out.max())
    o_range = o_max - o_min
    t_range = float(target_max - target_min)
    if o_range < eps:
        mid = (target_min + target_max) * 0.5
        return np.full_like(out, mid, dtype=np.float32), 0.0, mid
    scale = t_range / o_range
    offset = target_min - o_min * scale
    matched = out * scale + offset
    return matched.astype(np.float32), scale, offset


def compute_global_range(volume, axis, z_start, z_end):
    """Min/max over all slices in [z_start, z_end)."""
    vmin, vmax = np.inf, -np.inf
    for z in range(z_start, z_end):
        sl = np.take(volume, z, axis=axis)
        vmin = min(vmin, float(sl.min()))
        vmax = max(vmax, float(sl.max()))
    return vmin, vmax


def build_model(checkpoint_path, device):
    model = UnetRes(
        dim=64,
        dim_mults=(1, 2, 4, 8),
        num_unet=1,
        condition=True,
        input_condition=False,
        objective='pred_res',
        test_res_or_noise='res',
    )
    diffusion = ResidualDiffusion(
        model,
        image_size=512,
        timesteps=1000,
        sampling_timesteps=2,
        objective='pred_res',
        loss_type='l2',
        condition=True,
        sum_scale=0.01,
        input_condition=False,
        input_condition_mask=False,
        test_res_or_noise='res',
    )

    print(f'Loading FoundDiff checkpoint: {checkpoint_path}')
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    ema = EMA(diffusion, beta=0.995, update_every=10)
    ema.load_state_dict(ckpt['ema'])
    ema.ema_model.init()
    ema.to(device)
    ema.ema_model.eval()
    print('FoundDiff EMA model loaded.')
    return ema.ema_model


def maybe_resize(hu_slice, size=512):
    if hu_slice.shape == (size, size):
        return hu_slice.astype(np.float32)
    return resize(
        hu_slice,
        (size, size),
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def preprocess_slice(hu_slice, size=512):
    resized = maybe_resize(hu_slice, size=size)
    norm = hu_to_model_norm(resized)
    tensor = torch.from_numpy(norm.astype(np.float32))
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    return tensor


def preprocess_batch(slices, size=512):
    tensors = [preprocess_slice(sl, size=size) for sl in slices]
    return torch.stack(tensors, dim=0)


@torch.no_grad()
def denoise_batch(diffusion, tensor, device):
    x_in = tensor.to(device, non_blocking=True)
    out = diffusion.sample([x_in], batch_size=x_in.shape[0], last=True)
    result = out[-1].cpu().numpy()
    if result.ndim == 4:
        return result[:, 0]
    return result


def resize_back(slice_2d, out_shape):
    if slice_2d.shape == out_shape:
        return slice_2d.astype(np.float32)
    return resize(
        slice_2d,
        out_shape,
        order=1,
        preserve_range=True,
        anti_aliasing=True,
    ).astype(np.float32)


def insert_slice(volume, axis, index, slice_2d):
    idx = [slice(None)] * volume.ndim
    idx[axis] = index
    volume[tuple(idx)] = slice_2d


def copy_volume_range(dst, src, axis, z_start, z_end):
    """Bulk-copy a contiguous slice range (much faster than per-slice loop)."""
    idx = [slice(None)] * dst.ndim
    idx[axis] = slice(z_start, z_end)
    dst[tuple(idx)] = src[tuple(idx)]


def log(msg):
    print(msg, flush=True)


def warmup_gpu(diffusion, device, size=512):
    dummy = torch.zeros(1, 1, size, size)
    denoise_batch(diffusion, dummy, device)
    if device.type == 'cuda':
        torch.cuda.synchronize()


def denoise_volume(
    in_nii,
    out_nii,
    diffusion,
    device,
    stats_csv=None,
    axis=2,
    z_start=None,
    z_end=None,
    skip_air=False,
    size=512,
    batch_size=1,
    subset_only=False,
    fast_save=False,
    match_intensity='none',
    out_nii_raw=None,
):
    """Denoise one NIfTI volume. Model must already be loaded."""
    nii_in = nib.load(in_nii)
    volume_hu = nii_in.get_fdata().astype(np.float32)
    affine = nii_in.affine.copy()
    log(f'Input volume shape: {volume_hu.shape}, dtype float32 HU')

    z_start = z_start if z_start is not None else 0
    z_end = z_end if z_end is not None else volume_hu.shape[axis]
    z_end = min(z_end, volume_hu.shape[axis])
    num_slices = z_end - z_start
    log(f'Processing slices [{z_start}, {z_end}) = {num_slices} slices')

    out_volume = np.zeros_like(volume_hu, dtype=np.float32)
    raw_volume = np.zeros_like(volume_hu, dtype=np.float32) if out_nii_raw else None

    stats_path = stats_csv or (str(out_nii) + '.stats.csv')
    rows = []
    t0 = time.time()
    processed = 0

    global_in_min, global_in_max = None, None
    if match_intensity == 'global':
        global_in_min, global_in_max = compute_global_range(volume_hu, axis, z_start, z_end)
        log(
            f'Global input HU range [{z_start}:{z_end}]: '
            f'min={global_in_min:.2f}, max={global_in_max:.2f}'
        )

    z_indices = list(range(z_start, z_end))
    pbar = tqdm(total=num_slices, desc='denoising slices', unit='slice')

    batch_z = []
    batch_sl = []

    def flush_batch():
        nonlocal processed
        if not batch_z:
            return
        tensor = preprocess_batch(batch_sl, size=size)
        denoised_norm = denoise_batch(diffusion, tensor, device)
        for i, z in enumerate(batch_z):
            sl_hu = batch_sl[i]
            in_stat = slice_stats(sl_hu)
            denoised_hu_raw = resize_back(model_norm_to_hu(denoised_norm[i]), sl_hu.shape)
            raw_stat = slice_stats(denoised_hu_raw)

            if raw_volume is not None:
                insert_slice(raw_volume, axis, z, denoised_hu_raw)

            if match_intensity == 'none':
                denoised_hu = denoised_hu_raw
                scale, offset = 1.0, 0.0
                target_min, target_max = raw_stat['min'], raw_stat['max']
            elif match_intensity == 'slice':
                target_min, target_max = in_stat['min'], in_stat['max']
                denoised_hu, scale, offset = linear_match_intensity(
                    denoised_hu_raw, target_min, target_max)
            else:
                target_min, target_max = global_in_min, global_in_max
                denoised_hu, scale, offset = linear_match_intensity(
                    denoised_hu_raw, target_min, target_max)

            insert_slice(out_volume, axis, z, denoised_hu)
            matched_stat = slice_stats(denoised_hu)
            in_range = in_stat['range']
            raw_range = raw_stat['range']
            range_ratio = (raw_range / in_range) if in_range > 1e-6 else float('nan')

            rows.append({
                'slice': z,
                'skipped': False,
                'in_min': in_stat['min'],
                'in_max': in_stat['max'],
                'in_mean': in_stat['mean'],
                'in_range': in_range,
                'out_raw_min': raw_stat['min'],
                'out_raw_max': raw_stat['max'],
                'out_raw_mean': raw_stat['mean'],
                'out_raw_range': raw_range,
                'out_matched_min': matched_stat['min'],
                'out_matched_max': matched_stat['max'],
                'out_matched_mean': matched_stat['mean'],
                'out_matched_range': matched_stat['range'],
                'target_min': target_min,
                'target_max': target_max,
                'match_scale': scale,
                'match_offset': offset,
                'range_ratio_raw_vs_in': range_ratio,
                'norm_min': float(denoised_norm[i].min()),
                'norm_max': float(denoised_norm[i].max()),
                'norm_mean': float(denoised_norm[i].mean()),
            })
            processed += 1
            pbar.update(1)
        batch_z.clear()
        batch_sl.clear()

    for z in z_indices:
        sl_hu = np.take(volume_hu, z, axis=axis)
        in_stat = slice_stats(sl_hu)

        if skip_air and in_stat['mean'] < -900:
            insert_slice(out_volume, axis, z, sl_hu)
            rows.append({
                'slice': z, 'skipped': True,
                'in_min': in_stat['min'], 'in_max': in_stat['max'],
                'in_mean': in_stat['mean'], 'in_range': in_stat['range'],
            })
            pbar.update(1)
            continue

        batch_z.append(z)
        batch_sl.append(sl_hu)
        if len(batch_z) >= batch_size:
            flush_batch()

    flush_batch()
    pbar.close()
    log('Denoising done. Assembling output volume...')

    if subset_only:
        idx = [slice(None)] * out_volume.ndim
        idx[axis] = slice(z_start, z_end)
        out_to_save = out_volume[tuple(idx)].copy()
        affine_out = affine.copy()
        log(f'Subset mode: saving {out_to_save.shape} (slices {z_start}:{z_end} only)')
    else:
        out_to_save = out_volume
        affine_out = affine
        if z_start > 0:
            log(f'Copying unchanged slices 0:{z_start}...')
            copy_volume_range(out_volume, volume_hu, axis, 0, z_start)
        if z_end < volume_hu.shape[axis]:
            log(f'Copying unchanged slices {z_end}:{volume_hu.shape[axis]}...')
            copy_volume_range(out_volume, volume_hu, axis, z_end, volume_hu.shape[axis])

    elapsed = time.time() - t0
    if processed > 0:
        sec_per = elapsed / processed
        log(
            f'Denoised {processed} slices in {elapsed/60:.1f} min '
            f'({sec_per:.1f} s/slice). '
            f'Estimate for {volume_hu.shape[axis]} slices: '
            f'{sec_per * volume_hu.shape[axis] / 3600:.1f} hours'
        )

    out_path = Path(out_nii)
    if fast_save and str(out_path).endswith('.gz'):
        out_path = Path(str(out_path)[:-3])
        log(f'fast_save: writing uncompressed {out_path}')
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mb = out_to_save.nbytes / (1024 * 1024)
    log(f'Saving NIfTI ({mb:.0f} MB raw) -> {out_path} ...')
    t_save = time.time()
    out_img = nib.Nifti1Image(
        out_to_save.astype(np.float32), affine_out, header=make_hu_nifti_header(nii_in.header),
    )
    nib.save(out_img, str(out_path))
    log(f'NIfTI saved in {time.time() - t_save:.1f} s')

    if raw_volume is not None:
        raw_path = Path(out_nii_raw)
        if fast_save and str(raw_path).endswith('.gz'):
            raw_path = Path(str(raw_path)[:-3])
        log(f'Saving raw NIfTI -> {raw_path} ...')
        if subset_only:
            idx = [slice(None)] * raw_volume.ndim
            idx[axis] = slice(z_start, z_end)
            raw_to_save = raw_volume[tuple(idx)].copy()
            raw_affine = affine.copy()
        else:
            if z_start > 0:
                copy_volume_range(raw_volume, volume_hu, axis, 0, z_start)
            if z_end < volume_hu.shape[axis]:
                copy_volume_range(raw_volume, volume_hu, axis, z_end, volume_hu.shape[axis])
            raw_to_save = raw_volume
            raw_affine = affine
        nib.save(
            nib.Nifti1Image(
                raw_to_save.astype(np.float32), raw_affine,
                header=make_hu_nifti_header(nii_in.header),
            ),
            str(raw_path),
        )

    with open(stats_path, 'w', newline='') as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

    log(f'Saved: {out_path}')
    log(f'Stats: {stats_path}')


def main():
    parser = argparse.ArgumentParser(
        description='FoundDiff end-to-end NIfTI denoising (no intermediate .npy needed)'
    )
    parser.add_argument('--in_nii', required=True, help='Input .nii or .nii.gz (HU values)')
    parser.add_argument('--out_nii', required=True, help='Output denoised .nii.gz')
    parser.add_argument('--stats_csv', default=None, help='Per-slice stats CSV')
    parser.add_argument('--checkpoint', default='checkpoints/FoundDiff/sample/model-400.pt')
    parser.add_argument('--axis', type=int, default=2, help='Slice axis (default: 2)')
    parser.add_argument('--z_start', type=int, default=None, help='First slice index (inclusive)')
    parser.add_argument('--z_end', type=int, default=None, help='Last slice index (exclusive)')
    parser.add_argument('--skip_air', action='store_true', help='Skip slices with mean HU < -900')
    parser.add_argument('--size', type=int, default=512, help='Model input size')
    parser.add_argument('--batch_size', type=int, default=1, help='Slices per GPU forward pass (try 2 or 4)')
    parser.add_argument(
        '--subset_only',
        action='store_true',
        help='Save only processed slices [z_start:z_end] (small file, fast for testing)',
    )
    parser.add_argument(
        '--fast_save',
        action='store_true',
        help='Save uncompressed .nii instead of .nii.gz (much faster write)',
    )
    parser.add_argument(
        '--match_intensity',
        choices=['none', 'slice', 'global'],
        default='none',
        help='Optional extra linear match to input min/max. Default none: use formula [0,1]->[-2000,1000] only',
    )
    parser.add_argument(
        '--out_nii_raw',
        default=None,
        help='Optional path to save denoised volume BEFORE intensity matching',
    )
    args = parser.parse_args()

    if not Path(args.in_nii).exists():
        raise FileNotFoundError(
            f'Input not found: {args.in_nii}\n'
            'Use your real path, e.g. data/mydata/APNHC00002_CT.nii.gz'
        )

    set_seed(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    log(f'Using device: {device}, batch_size={args.batch_size}')

    diffusion = build_model(args.checkpoint, device)
    if device.type == 'cuda':
        log('GPU warmup...')
        warmup_gpu(diffusion, device, size=args.size)

    denoise_volume(
        in_nii=args.in_nii,
        out_nii=args.out_nii,
        diffusion=diffusion,
        device=device,
        stats_csv=args.stats_csv,
        axis=args.axis,
        z_start=args.z_start,
        z_end=args.z_end,
        skip_air=args.skip_air,
        size=args.size,
        batch_size=args.batch_size,
        subset_only=args.subset_only,
        fast_save=args.fast_save,
        match_intensity=args.match_intensity,
        out_nii_raw=args.out_nii_raw,
    )
    log(
        f'Output HU domain: [{HU_MIN}, {HU_MAX}]. '
        f'NIfTI scl_slope=1, scl_inter=0 for ITK.'
    )
    log('All done.')


if __name__ == '__main__':
    main()
