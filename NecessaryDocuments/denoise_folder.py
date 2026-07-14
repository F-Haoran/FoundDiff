#!/usr/bin/env python3
"""Batch denoise all .nii/.nii.gz files in a folder (full volume, model loaded once)."""

import argparse
import sys
import time
from pathlib import Path


def log(msg):
    print(msg, flush=True)


log('denoise_folder.py starting...')

try:
    import torch
    from denoise_nii import build_model, denoise_volume, warmup_gpu
    from src.DADiff import set_seed
except ImportError as e:
    log(f'IMPORT ERROR: {e}')
    log('Activate conda env: conda activate FoundDiff2.0')
    log('Run from repo root: ~/FoundDiff2.0/FoundDiff')
    log('Sync scripts: bash FoundDiff-tools/sync.sh')
    sys.exit(1)


def find_nii_files(in_dir):
    in_dir = Path(in_dir)
    files = sorted(in_dir.glob('*.nii.gz')) + sorted(in_dir.glob('*.nii'))
    # Drop *.nii that are actually *.nii.gz stems (e.g. foo.nii when foo.nii.gz exists)
    gz_stems = {f.name[:-7] for f in in_dir.glob('*.nii.gz')}
    files = [f for f in files if f.suffix == '.gz' or f.stem not in gz_stems]
    return files


def output_name(in_path, out_dir, suffix='_denoised', use_nii=True):
    name = in_path.name
    if name.endswith('.nii.gz'):
        base = name[:-7]
    elif name.endswith('.nii'):
        base = name[:-4]
    else:
        base = in_path.stem
    ext = '.nii' if use_nii else '.nii.gz'
    return Path(out_dir) / f'{base}{suffix}{ext}'


def main():
    parser = argparse.ArgumentParser(
        description='Denoise all NIfTI files in a folder (full volume per file)'
    )
    parser.add_argument('--in_dir', required=True, help='Input folder with .nii/.nii.gz files')
    parser.add_argument('--out_dir', required=True, help='Output folder for denoised volumes')
    parser.add_argument('--checkpoint', default='checkpoints/FoundDiff/sample/model-400.pt')
    parser.add_argument('--axis', type=int, default=2)
    parser.add_argument('--skip_air', action='store_true', help='Skip air slices (mean HU < -900)')
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--fast_save', action='store_true', default=True,
                        help='Save uncompressed .nii (default on)')
    parser.add_argument('--gzip', action='store_true', help='Save as .nii.gz (slow, disables fast_save)')
    parser.add_argument('--match_intensity', choices=['none', 'slice', 'global'], default='none')
    parser.add_argument('--suffix', default='_denoised', help='Output filename suffix')
    parser.add_argument('--skip_existing', action='store_true', help='Skip if output already exists')
    args = parser.parse_args()

    in_dir = Path(args.in_dir)
    out_dir = Path(args.out_dir)
    if not in_dir.is_dir():
        raise FileNotFoundError(f'Input folder not found: {in_dir}')
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_nii_files(in_dir)
    if not files:
        raise FileNotFoundError(f'No .nii/.nii.gz files in {in_dir}')

    log(f'Found {len(files)} NIfTI file(s) in {in_dir}')
    for f in files:
        log(f'  - {f.name}')

    set_seed(10)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    log(f'Using device: {device}, batch_size={args.batch_size}')

    log('Loading model (once for all files)...')
    diffusion = build_model(args.checkpoint, device)
    if device.type == 'cuda':
        warmup_gpu(diffusion, device, size=args.size)

    use_nii = not args.gzip
    t_all = time.time()
    ok, fail = 0, 0

    for i, in_path in enumerate(files, 1):
        out_path = output_name(in_path, out_dir, suffix=args.suffix, use_nii=use_nii)
        stats_path = str(out_path) + '.stats.csv'

        if args.skip_existing and out_path.exists():
            log(f'[{i}/{len(files)}] SKIP (exists): {out_path.name}')
            ok += 1
            continue

        log(f'[{i}/{len(files)}] {in_path.name} -> {out_path.name}')
        t0 = time.time()
        try:
            denoise_volume(
                in_nii=str(in_path),
                out_nii=str(out_path),
                diffusion=diffusion,
                device=device,
                stats_csv=stats_path,
                axis=args.axis,
                z_start=None,
                z_end=None,
                skip_air=args.skip_air,
                size=args.size,
                batch_size=args.batch_size,
                subset_only=False,
                fast_save=(not args.gzip),
                match_intensity=args.match_intensity,
                out_nii_raw=None,
            )
            log(f'  Done in {(time.time() - t0) / 60:.1f} min')
            ok += 1
        except Exception as e:
            log(f'  FAILED: {e}')
            fail += 1

    log(f'Batch complete: {ok} ok, {fail} failed, total {(time.time() - t_all) / 3600:.2f} hours')
    log(f'Outputs in: {out_dir}')


if __name__ == '__main__':
    main()
