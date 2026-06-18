#!/usr/bin/env python3
"""
Run FoundDiff on CPR MetaImage volumes (.mhd/.raw).

The FoundDiff model consumes 2D .npy slices from the repository's external
dataset layout. This script converts CPR .mhd/.raw volumes to that layout,
optionally runs FoundDiff inference, and reconstructs denoised slices back to a
3D MetaImage volume in the same container format as the input.

Typical CUDA machine usage:
  python3 run_cpr_mhd_founddiff.py --slice-axis z

For CPR volumes stored as (z, angular, linear), use:
  --slice-axis z        denoise each angular-vs-linear CPR plane
  --slice-axis angular  denoise z-vs-linear planes
  --slice-axis linear   denoise z-vs-angular planes

If you only have a .raw file without a .mhd header, also pass:
  --raw-shape Z,Y,X --raw-dtype float32
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

ROOT = Path(__file__).resolve().parent
DEFAULT_INPUT_DIR = ROOT / "data" / "custom" / "CPR"
DEFAULT_WORK_ROOT = ROOT / "data" / "external" / "external_2d"
DEFAULT_OUTPUT_DIR = ROOT / "checkpoints" / "FoundDiff" / "cpr_denoised_mhd"
DEFAULT_RESULTS_DIR = ROOT / "checkpoints" / "FoundDiff" / "test_final_npy"
DEFAULT_MANIFEST = DEFAULT_WORK_ROOT / "slice_manifest.json"

HU_MIN = -1000.0
HU_MAX = 2000.0
HU_RANGE = HU_MAX - HU_MIN

MHD_TO_NUMPY = {
    "MET_CHAR": np.int8,
    "MET_UCHAR": np.uint8,
    "MET_SHORT": np.int16,
    "MET_USHORT": np.uint16,
    "MET_INT": np.int32,
    "MET_UINT": np.uint32,
    "MET_FLOAT": np.float32,
    "MET_DOUBLE": np.float64,
}

NUMPY_TO_MHD = {
    np.dtype(np.int8): "MET_CHAR",
    np.dtype(np.uint8): "MET_UCHAR",
    np.dtype(np.int16): "MET_SHORT",
    np.dtype(np.uint16): "MET_USHORT",
    np.dtype(np.int32): "MET_INT",
    np.dtype(np.uint32): "MET_UINT",
    np.dtype(np.float32): "MET_FLOAT",
    np.dtype(np.float64): "MET_DOUBLE",
}


@dataclass
class ImageVolume:
    name: str
    path: Path
    array: np.ndarray
    dtype: np.dtype
    spacing: tuple[float, ...] | None = None
    origin: tuple[float, ...] | None = None
    direction: tuple[float, ...] | None = None
    sitk_image: object | None = None


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Prepare/run/reconstruct FoundDiff for CPR .mhd/.raw volumes.")
    p.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help="Optional .mhd/.mha/.raw files. If .mhd and its .raw are both present, only .mhd is processed.",
    )
    p.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    p.add_argument("--work-root", type=Path, default=DEFAULT_WORK_ROOT, help="FoundDiff external_2d directory.")
    p.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    p.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR, help="FoundDiff denoised .npy directory.")
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--output-suffix", default="_founddiff_denoised")
    p.add_argument("--slice-axis", choices=("z", "y", "x", "axial", "angular", "linear"), default="z")
    p.add_argument("--stride", type=int, default=1, help="Use every Nth slice along --slice-axis.")
    p.add_argument("--max-slices", type=int, default=0, help="Maximum slices per volume; 0 means all.")
    p.add_argument("--prefix", default="lung", help="Slice filename prefix; keep 'lung' for FoundDiff labels.")
    p.add_argument("--clean", action="store_true", help="Remove previous prepared external_2d files first.")
    p.add_argument("--prepare-only", action="store_true", help="Only create FoundDiff 2D input layout.")
    p.add_argument(
        "--reconstruct-only",
        action="store_true",
        help="Only reconstruct MetaImage output from existing FoundDiff .npy outputs.",
    )
    p.add_argument("--skip-inference", action="store_true", help="Prepare and reconstruct, but do not call train.py.")
    p.add_argument("--name", default="FoundDiff", help="Checkpoint name under checkpoints/.")
    p.add_argument("--epoch", default="400", help="Checkpoint epoch to load, e.g. 400.")
    p.add_argument("--gpu", default=os.environ.get("CUDA_VISIBLE_DEVICES", "0"))
    p.add_argument("--max-test", type=int, default=0, help="Pass through to train.py; 0 means all prepared slices.")
    p.add_argument("--raw-shape", default=None, help="Required for raw-only input, in Z,Y,X or Y,X order.")
    p.add_argument("--raw-dtype", default="float32", help="NumPy dtype for raw-only input.")
    p.add_argument("--raw-spacing", default=None, help="Optional raw-only spacing in X,Y,Z order for output header.")
    p.add_argument(
        "--model-output-scale",
        choices=("founddiff-hu", "identity", "unit"),
        default="founddiff-hu",
        help="How to convert FoundDiff .npy values before writing .mhd.",
    )
    return p.parse_args(argv)


def axis_name(axis: str) -> str:
    aliases = {"axial": "z", "angular": "y", "linear": "x"}
    return aliases.get(axis, axis)


def parse_shape(raw: str | None, *, name: str) -> tuple[int, ...] | None:
    if raw is None:
        return None
    values = tuple(int(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) not in (2, 3) or any(v <= 0 for v in values):
        raise SystemExit(f"{name} must be Y,X or Z,Y,X positive integers")
    return values


def parse_float_tuple(raw: str | None, *, name: str) -> tuple[float, ...] | None:
    if raw is None:
        return None
    values = tuple(float(part.strip()) for part in raw.split(",") if part.strip())
    if len(values) not in (2, 3) or any(v <= 0 for v in values):
        raise SystemExit(f"{name} must be X,Y or X,Y,Z positive numbers")
    return values


def read_mhd_header(path: Path) -> dict[str, str]:
    header: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            header[key.strip()] = value.strip()
    return header


def suffix_lower(path: Path) -> str:
    return path.suffix.lower()


def referenced_raw_path(header_path: Path) -> Path | None:
    if suffix_lower(header_path) != ".mhd":
        return None
    try:
        header = read_mhd_header(header_path)
    except OSError:
        return None
    raw_name = header.get("ElementDataFile")
    if not raw_name or raw_name.upper() in {"LOCAL", "LIST"}:
        return None
    raw_path = Path(raw_name)
    if not raw_path.is_absolute():
        raw_path = header_path.parent / raw_path
    return raw_path.resolve()


def read_mhd_fallback(path: Path) -> ImageVolume:
    header = read_mhd_header(path)
    if "DimSize" not in header or "ElementType" not in header or "ElementDataFile" not in header:
        raise SystemExit(f"Invalid .mhd header: {path}")

    dims_xyz = tuple(int(v) for v in header["DimSize"].split())
    dtype = np.dtype(MHD_TO_NUMPY[header["ElementType"]])
    if header.get("ElementByteOrderMSB", "False").lower() == "true":
        dtype = dtype.newbyteorder(">")
    else:
        dtype = dtype.newbyteorder("<")

    raw_path = Path(header["ElementDataFile"])
    if not raw_path.is_absolute():
        raw_path = path.parent / raw_path
    data = np.fromfile(raw_path, dtype=dtype)
    expected = int(np.prod(dims_xyz))
    if data.size != expected:
        raise SystemExit(f"{raw_path} has {data.size} values; expected {expected} from {path}")
    native_dtype = dtype.newbyteorder("=")
    array = data.reshape(tuple(reversed(dims_xyz))).astype(np.float32, copy=False)
    spacing = tuple(float(v) for v in header.get("ElementSpacing", "").split()) or None
    origin = tuple(float(v) for v in header.get("Offset", header.get("Position", "")).split()) or None
    direction = tuple(float(v) for v in header.get("TransformMatrix", "").split()) or None
    return ImageVolume(path.stem, path, array, dtype=np.dtype(native_dtype), spacing=spacing, origin=origin, direction=direction)


def read_raw(path: Path, shape: tuple[int, ...], dtype_name: str, spacing: tuple[float, ...] | None) -> ImageVolume:
    dtype = np.dtype(dtype_name)
    data = np.fromfile(path, dtype=dtype)
    expected = int(np.prod(shape))
    if data.size != expected:
        raise SystemExit(f"{path} has {data.size} values; expected {expected} from --raw-shape")
    array = data.reshape(shape).astype(np.float32, copy=False)
    return ImageVolume(path.stem, path, array, dtype=np.dtype(dtype), spacing=spacing)


def read_image(path: Path, args: argparse.Namespace) -> ImageVolume:
    suffix = suffix_lower(path)
    if suffix == ".raw":
        raw_shape = parse_shape(args.raw_shape, name="--raw-shape")
        if raw_shape is None:
            raise SystemExit("--raw-shape is required for raw-only input")
        return read_raw(path, raw_shape, args.raw_dtype, parse_float_tuple(args.raw_spacing, name="--raw-spacing"))

    try:
        import SimpleITK as sitk

        img = sitk.ReadImage(str(path))
        original_array = sitk.GetArrayFromImage(img)
        array = original_array.astype(np.float32, copy=False)
        return ImageVolume(
            path.stem,
            path,
            array,
            dtype=np.dtype(original_array.dtype),
            spacing=tuple(img.GetSpacing()),
            origin=tuple(img.GetOrigin()),
            direction=tuple(img.GetDirection()),
            sitk_image=img,
        )
    except ImportError:
        if suffix != ".mhd":
            raise
        return read_mhd_fallback(path)


def image_files_in_dir(input_dir: Path) -> list[Path]:
    supported = {".mhd", ".mha", ".raw"}
    return sorted(p for p in input_dir.iterdir() if p.is_file() and suffix_lower(p) in supported)


def normalize_input_paths(paths: list[Path], *, allow_raw_only: bool) -> list[Path]:
    header_paths = [p for p in paths if suffix_lower(p) in {".mhd", ".mha"}]
    referenced_raws = {raw for p in header_paths if (raw := referenced_raw_path(p)) is not None}
    normalized = list(header_paths)

    for path in paths:
        if suffix_lower(path) != ".raw":
            continue
        resolved = path.resolve()
        if resolved in referenced_raws:
            print(f"Skip paired raw file because its .mhd header will read it: {path}")
            continue
        if allow_raw_only:
            normalized.append(path)
        else:
            print(f"Skip raw-only file without --raw-shape: {path}")

    if not normalized and paths:
        raise SystemExit("Only raw files were found. Pass --raw-shape Z,Y,X --raw-dtype <dtype> for raw-only input.")
    return normalized


def find_inputs(args: argparse.Namespace) -> list[Path]:
    if args.inputs:
        paths = [p.expanduser().resolve() for p in args.inputs]
    else:
        input_dir = args.input_dir.expanduser().resolve()
        if not input_dir.is_dir():
            raise SystemExit(f"Input directory not found: {input_dir}")
        paths = image_files_in_dir(input_dir)
    paths = normalize_input_paths(paths, allow_raw_only=args.raw_shape is not None)
    if not paths:
        raise SystemExit(f"No .mhd/.mha inputs found in {args.input_dir}")
    for path in paths:
        if not path.is_file():
            raise SystemExit(f"Input file not found: {path}")
    return paths


def clean_external_layout(work_root: Path) -> None:
    for phase in ("test", "train", "train512"):
        path = work_root / phase
        if path.is_dir():
            shutil.rmtree(path)
    manifest = work_root / "slice_manifest.json"
    if manifest.is_file():
        manifest.unlink()


def crop512(slice2d: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    h, w = slice2d.shape
    out = np.zeros((512, 512), dtype=np.float32)
    sh, sw = min(h, 512), min(w, 512)
    dst_y = max((512 - sh) // 2, 0)
    dst_x = max((512 - sw) // 2, 0)
    src_y = max((h - sh) // 2, 0)
    src_x = max((w - sw) // 2, 0)
    out[dst_y : dst_y + sh, dst_x : dst_x + sw] = slice2d[src_y : src_y + sh, src_x : src_x + sw]
    info = {"h": h, "w": w, "sh": sh, "sw": sw, "dst_y": dst_y, "dst_x": dst_x, "src_y": src_y, "src_x": src_x}
    return out[np.newaxis, ...], info


def slice_count(array: np.ndarray, axis: str) -> int:
    if array.ndim == 2:
        return 1
    if axis == "z":
        return array.shape[0]
    if axis == "y":
        return array.shape[1]
    if axis == "x":
        return array.shape[2]
    raise SystemExit(f"Unsupported slice axis: {axis}")


def get_slice(array: np.ndarray, axis: str, index: int) -> np.ndarray:
    if array.ndim == 2:
        return array
    if axis == "z":
        return array[index, :, :]
    if axis == "y":
        return array[:, index, :]
    if axis == "x":
        return array[:, :, index]
    raise SystemExit(f"Unsupported slice axis: {axis}")


def put_slice(array: np.ndarray, axis: str, index: int, value: np.ndarray) -> None:
    if array.ndim == 2:
        array[:, :] = value
    elif axis == "z":
        array[index, :, :] = value
    elif axis == "y":
        array[:, index, :] = value
    elif axis == "x":
        array[:, :, index] = value
    else:
        raise SystemExit(f"Unsupported slice axis: {axis}")


def prepare_external_layout(args: argparse.Namespace) -> Path:
    if args.clean:
        clean_external_layout(args.work_root)
    args.work_root.mkdir(parents=True, exist_ok=True)

    axis = axis_name(args.slice_axis)
    global_idx = 0
    bootstrap: list[tuple[str, np.ndarray]] = []
    manifest = {"source": "run_cpr_mhd_founddiff.py", "slice_axis": axis, "stride": args.stride, "volumes": []}

    for path in find_inputs(args):
        volume = read_image(path, args)
        if volume.array.ndim not in (2, 3):
            raise SystemExit(f"Expected 2D or 3D image, got shape {volume.array.shape}: {path}")
        n = slice_count(volume.array, axis)
        if args.max_slices:
            n = min(n, args.max_slices)

        vol_entry = {
            "name": volume.name,
            "input_path": str(path),
            "shape": list(volume.array.shape),
            "raw_dtype": args.raw_dtype if suffix_lower(path) == ".raw" else None,
            "spacing": list(volume.spacing) if volume.spacing else None,
            "origin": list(volume.origin) if volume.origin else None,
            "direction": list(volume.direction) if volume.direction else None,
            "slices": [],
        }
        written = 0
        for source_index in range(0, n, args.stride):
            slice2d = get_slice(volume.array, axis, source_index)
            cropped, crop_info = crop512(slice2d)
            fname = f"{args.prefix}-{global_idx:05d}.npy"
            global_idx += 1
            for sub in ("quarter_1mm", "full_1mm"):
                out = args.work_root / "test" / sub / fname
                out.parent.mkdir(parents=True, exist_ok=True)
                np.save(out, cropped)
            if len(bootstrap) < 2:
                bootstrap.append((fname, cropped))
            vol_entry["slices"].append(
                {"fname": fname, "index": int(source_index), "phase": "test", "crop": crop_info}
            )
            written += 1
        manifest["volumes"].append(vol_entry)
        print(f"Prepared {volume.name}: {written} slices from shape {volume.array.shape} axis={axis}")

    for fname, cropped in bootstrap:
        for sub in ("quarter_1mm", "full_1mm"):
            out = args.work_root / "train512" / sub / fname
            out.parent.mkdir(parents=True, exist_ok=True)
            np.save(out, cropped)

    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    with open(args.manifest, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest: {args.manifest}")
    print(f"FoundDiff input: {args.work_root}")
    return args.manifest


def run_founddiff(args: argparse.Namespace) -> None:
    for weight in (ROOT / "src" / "DA-CLIP.pth", ROOT / "checkpoints" / args.name / "sample" / f"model-{args.epoch}.pt"):
        if not weight.is_file():
            raise SystemExit(f"Missing FoundDiff weight: {weight}")

    cmd = [
        sys.executable,
        str(ROOT / "train.py"),
        "--name",
        args.name,
        "--epoch",
        str(args.epoch),
        "--dataset",
        "2020_seen",
        "--data-mode",
        "external",
    ]
    if args.max_test:
        cmd.extend(["--max-test", str(args.max_test)])
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    print(">> " + " ".join(cmd))
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def model_output_to_values(output: np.ndarray, scale: str) -> np.ndarray:
    arr = np.squeeze(output).astype(np.float32)
    if scale == "founddiff-hu":
        return arr * HU_RANGE + HU_MIN + 1024.0
    if scale == "unit":
        return arr
    if scale == "identity":
        return arr
    raise SystemExit(f"Unsupported --model-output-scale: {scale}")


def embed_slice(original: np.ndarray, denoised512: np.ndarray, crop: dict[str, int]) -> np.ndarray:
    out = original.copy()
    sy, sx, dy, dx = crop["src_y"], crop["src_x"], crop["dst_y"], crop["dst_x"]
    sh, sw = crop["sh"], crop["sw"]
    out[sy : sy + sh, sx : sx + sw] = denoised512[dy : dy + sh, dx : dx + sw]
    return out


def cast_to_dtype(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    dtype = np.dtype(dtype).newbyteorder("=")
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.clip(np.rint(array), info.min, info.max).astype(dtype, copy=False)
    return array.astype(dtype, copy=False)


def write_mhd_fallback(
    path: Path,
    array: np.ndarray,
    dtype: np.dtype,
    spacing: Iterable[float] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw_path = path.with_suffix(".raw")
    data = cast_to_dtype(array, dtype)
    data.tofile(raw_path)
    dims_xyz = " ".join(str(v) for v in reversed(data.shape))
    spacing_values = tuple(spacing) if spacing else tuple(1.0 for _ in range(data.ndim))
    if len(spacing_values) != data.ndim:
        spacing_values = tuple(1.0 for _ in range(data.ndim))
    spacing_text = " ".join(str(v) for v in spacing_values)
    with open(path, "w", encoding="utf-8") as f:
        f.write("ObjectType = Image\n")
        f.write(f"NDims = {data.ndim}\n")
        f.write("BinaryData = True\n")
        f.write("BinaryDataByteOrderMSB = False\n")
        f.write("CompressedData = False\n")
        f.write("TransformMatrix = " + " ".join("1" if i % (data.ndim + 1) == 0 else "0" for i in range(data.ndim**2)) + "\n")
        f.write("Offset = " + " ".join("0" for _ in range(data.ndim)) + "\n")
        f.write(f"ElementSpacing = {spacing_text}\n")
        f.write(f"DimSize = {dims_xyz}\n")
        f.write(f"ElementType = {NUMPY_TO_MHD.get(np.dtype(data.dtype), 'MET_FLOAT')}\n")
        f.write(f"ElementDataFile = {raw_path.name}\n")


def write_image(path: Path, array: np.ndarray, ref: ImageVolume) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = cast_to_dtype(array, ref.dtype)
    try:
        import SimpleITK as sitk

        out_img = sitk.GetImageFromArray(data)
        if ref.sitk_image is not None and tuple(ref.array.shape) == tuple(array.shape):
            out_img.CopyInformation(ref.sitk_image)
        elif ref.spacing and len(ref.spacing) == array.ndim:
            out_img.SetSpacing(tuple(float(v) for v in ref.spacing))
        sitk.WriteImage(out_img, str(path))
    except ImportError:
        write_mhd_fallback(path, array, ref.dtype, ref.spacing)


def output_path_for_volume(args: argparse.Namespace, vol: dict, ref: ImageVolume) -> Path:
    suffix = suffix_lower(ref.path)
    if suffix == ".mha":
        extension = ".mha"
    else:
        extension = ".mhd"
    return args.output_dir / f"{vol['name']}{args.output_suffix}{extension}"


def reconstruct_outputs(args: argparse.Namespace) -> None:
    if not args.manifest.is_file():
        raise SystemExit(f"Manifest not found: {args.manifest}")
    with open(args.manifest, encoding="utf-8") as f:
        manifest = json.load(f)
    axis = manifest.get("slice_axis", axis_name(args.slice_axis))

    for vol in manifest.get("volumes", []):
        input_path = Path(vol["input_path"])
        if suffix_lower(input_path) == ".raw" and args.raw_shape is None:
            ref = read_raw(
                input_path,
                tuple(int(v) for v in vol["shape"]),
                vol.get("raw_dtype") or args.raw_dtype,
                tuple(vol["spacing"]) if vol.get("spacing") else None,
            )
        else:
            ref = read_image(input_path, args)
        out_array = ref.array.copy()
        written = 0
        missing = 0
        for item in vol.get("slices", []):
            den_path = args.results_dir / item["fname"]
            if not den_path.is_file():
                missing += 1
                continue
            denoised = model_output_to_values(np.load(den_path), args.model_output_scale)
            original_slice = get_slice(ref.array, axis, int(item["index"]))
            restored = embed_slice(original_slice, denoised, item["crop"])
            put_slice(out_array, axis, int(item["index"]), restored)
            written += 1
        out_path = output_path_for_volume(args, vol, ref)
        write_image(out_path, out_array, ref)
        print(f"Reconstructed {out_path}  written={written} missing={missing}")
        if missing:
            print("Warning: missing denoised .npy files; rerun FoundDiff without --max-test if needed.")


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stride <= 0:
        raise SystemExit("--stride must be positive")
    if args.manifest == DEFAULT_MANIFEST and args.work_root != DEFAULT_WORK_ROOT:
        args.manifest = args.work_root / "slice_manifest.json"

    if not args.reconstruct_only:
        prepare_external_layout(args)
    if args.prepare_only:
        return 0
    if not args.skip_inference and not args.reconstruct_only:
        run_founddiff(args)
    reconstruct_outputs(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
