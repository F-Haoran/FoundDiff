#!/usr/bin/env python3
"""
Download / prepare open-source CT volumes for FoundDiff external testing.

Sources used (all open access):
  1. Existing Mayo IDC DICOM (already on disk) -> .nii.gz export
  2. Optional: IDC LIDC-IDRI chest CT (1 series, ~150 MB) -> DICOM then .nii.gz
  3. Optional: simulate LDCT from full-dose .nii.gz when no low-dose pair exists

Outputs under data/external/nifti/:
  {name}_full.nii.gz
  {name}_low.nii.gz   (real low-dose or noise-simulated)

Then run:
  python Preprocess_nifti.py
  CUDA_VISIBLE_DEVICES=0 python train.py --name FoundDiff --epoch 400 \\
      --dataset 2020_seen --data-mode external --max-test 20

Other open datasets (manual, mostly not .nii.gz natively):
  - Mayo LDCT IDC (DICOM): python Download_Mayo.py  [already integrated]
  - LoDoPaB-CT (HDF5): https://zenodo.org/record/3384092
  - CQ500 head CT: https://www.cq500.org/ (registration required)
  - NLST lung screening: https://cdas.cancer.gov/nlst/ (application)
  - LIDC-IDRI (DICOM): TCIA / IDC -> convert with dcm2niix or this script
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from data.paths import MAYO_DICOM_ROOT

NIFTI_OUT = Path(__file__).resolve().parent / "data" / "external" / "nifti"


def parse_args():
    p = argparse.ArgumentParser(description="Prepare open CT .nii.gz for FoundDiff")
    p.add_argument("--out-dir", type=Path, default=NIFTI_OUT)
    p.add_argument("--from-mayo-dicom", action="store_true", default=True,
                   help="Export paired Full/Low DICOM already downloaded (default)")
    p.add_argument("--mayo-patient", type=str, default="C002",
                   help="PatientID with both Full and Low series in local DICOM")
    p.add_argument("--download-lidc", action="store_true",
                   help="Download 1 LIDC-IDRI chest series from IDC (~150MB)")
    p.add_argument("--simulate-noise-std", type=float, default=0.0,
                   help="If only full-dose nii exists, add Gaussian noise (HU) for fake LDCT")
    return p.parse_args()


def dicom_series_to_volume(dcm_paths: list[Path]):
    import pydicom

    datasets = [pydicom.dcmread(str(p)) for p in dcm_paths]

    def sort_key(ds):
        inst = getattr(ds, "InstanceNumber", None)
        if inst is not None:
            return (0, int(inst))
        pos = getattr(ds, "ImagePositionPatient", None)
        if pos is not None:
            return (1, float(pos[2]))
        return (2, str(getattr(ds, "SOPInstanceUID", "")))

    datasets.sort(key=sort_key)
    slices = []
    for ds in datasets:
        arr = ds.pixel_array.astype(np.float32)
        slope = float(getattr(ds, "RescaleSlope", 1))
        intercept = float(getattr(ds, "RescaleIntercept", 0))
        slices.append(arr * slope + intercept)
    return np.stack(slices, axis=0)


def save_nifti(path: Path, volume: np.ndarray):
    import nibabel as nib

    path.parent.mkdir(parents=True, exist_ok=True)
    # RAS-like: z, y, x stored as nibabel expects (x,y,z) -> transpose
    vol = np.transpose(volume, (2, 1, 0)).astype(np.float32)
    nib.save(nib.Nifti1Image(vol, np.eye(4)), str(path))
    print(f"  saved {path} shape={volume.shape} HU [{volume.min():.0f}, {volume.max():.0f}]")


def collect_mayo_series(dicom_root: Path, patient: str, description: str) -> list[Path]:
    import pydicom

    paths = []
    for p in dicom_root.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(p), stop_before_pixels=True)
        except Exception:
            continue
        if str(getattr(ds, "PatientID", "")) != patient:
            continue
        if str(getattr(ds, "SeriesDescription", "")) != description:
            continue
        paths.append(p)
    return sorted(paths)


def export_mayo_pair(dicom_root: Path, out_dir: Path, patient: str):
    full_paths = collect_mayo_series(dicom_root, patient, "Full Dose Images")
    low_paths = collect_mayo_series(dicom_root, patient, "Low Dose Images")
    if not full_paths:
        print(f"No Full Dose series for patient {patient}", file=sys.stderr)
        return False
    name = f"mayo_{patient.lower()}"
    full_vol = dicom_series_to_volume(full_paths)
    save_nifti(out_dir / f"{name}_full.nii.gz", full_vol)
    if low_paths:
        low_vol = dicom_series_to_volume(low_paths)
        save_nifti(out_dir / f"{name}_low.nii.gz", low_vol)
    else:
        print(f"  no Low Dose for {patient}, skip low nifti")
    return True


def download_lidc_series(out_dir: Path, dicom_dir: Path):
    try:
        from idc_index import IDCClient
    except ImportError:
        print("pip install idc-index", file=sys.stderr)
        return False

    client = IDCClient()
    q = """
SELECT SeriesInstanceUID, PatientID, series_size_MB
FROM index
WHERE collection_id = 'LIDC-IDRI'
  AND Modality = 'CT'
  AND sop_class_name = 'CT Image Storage'
ORDER BY series_size_MB ASC
LIMIT 1
"""
    df = client.sql_query(q)
    if df.empty:
        print("No LIDC series found on IDC", file=sys.stderr)
        return False
    uid = df.iloc[0]["SeriesInstanceUID"]
    pid = df.iloc[0]["PatientID"]
    print(f"Downloading LIDC {pid} ({df.iloc[0]['series_size_MB']:.0f} MB)...")
    dicom_dir.mkdir(parents=True, exist_ok=True)
    client.download_dicom_series(seriesInstanceUID=[uid], downloadDir=str(dicom_dir))
    paths = list(dicom_dir.rglob("*.dcm"))
    if not paths:
        return False
    vol = dicom_series_to_volume(paths)
    name = f"lidc_{pid}"
    save_nifti(out_dir / f"{name}_full.nii.gz", vol)
    return True


def simulate_low_from_full(out_dir: Path, noise_std: float):
    import nibabel as nib

    for full in out_dir.glob("*_full.nii.gz"):
        low_path = full.with_name(full.name.replace("_full", "_low"))
        if low_path.exists():
            continue
        img = nib.load(str(full))
        vol = np.asarray(img.dataobj, dtype=np.float32)
        vol = np.transpose(vol, (2, 1, 0))  # back to z,y,x
        noise = np.random.default_rng(0).normal(0, noise_std, vol.shape).astype(np.float32)
        save_nifti(low_path, vol + noise)
        print(f"  simulated LDCT -> {low_path.name} (noise_std={noise_std})")


def main():
    args = parse_args()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    dicom_root = Path(MAYO_DICOM_ROOT)

    print(f"Output: {out_dir}\n")

    if args.from_mayo_dicom and dicom_root.is_dir():
        print(f"[1] Export Mayo DICOM patient {args.mayo_patient} -> .nii.gz")
        export_mayo_pair(dicom_root, out_dir, args.mayo_patient)

    if args.download_lidc:
        print("\n[2] Download LIDC-IDRI sample from IDC")
        lidc_dicom = out_dir.parent / "lidc_dicom"
        download_lidc_series(out_dir, lidc_dicom)

    if args.simulate_noise_std > 0:
        print(f"\n[3] Simulate LDCT from full-dose (noise_std={args.simulate_noise_std})")
        simulate_low_from_full(out_dir, args.simulate_noise_std)

    nii = list(out_dir.glob("*.nii.gz"))
    print(f"\nDone. {len(nii)} .nii.gz files in {out_dir}")
    if not nii:
        print("Nothing created. Ensure data/mayo/dicom has paired series or use --download-lidc")
        sys.exit(1)


if __name__ == "__main__":
    main()
