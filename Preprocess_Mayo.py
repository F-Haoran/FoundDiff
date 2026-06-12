#!/usr/bin/env python3
"""
Convert IDC Mayo DICOM (Full / Low Dose) to FoundDiff .npy layout.

IDC ldct_and_projection_data provides:
  - CHEST    -> Mayo2020_lung_2d
  - ABDOMEN  -> Mayo2020_ab_2d
  (no head series on IDC)

Low-dose reconstructed series (~25%) are written to quarter_1mm/.
Full-dose series go to full_1mm/.

Output slice shape: (1, H, W) float32 HU.

Example:
  python Preprocess_Mayo.py --dicom-root data/mayo/dicom
  python Preprocess_Mayo.py --dicom-root data/mayo/dicom --limit-patients 2
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pydicom
from tqdm import tqdm

from data.paths import MAYO2020_AB, MAYO2020_LUNG, MAYO_DICOM_ROOT

BODY_TO_ROOT = {
    "CHEST": MAYO2020_LUNG,
    "ABDOMEN": MAYO2020_AB,
}

SERIES_FULL = "Full Dose Images"
SERIES_LOW = "Low Dose Images"
LOW_DOSE_FOLDER = "quarter_1mm"
FULL_DOSE_FOLDER = "full_1mm"
PHASES = ("train", "test", "train512")


def parse_args():
    p = argparse.ArgumentParser(description="DICOM -> FoundDiff Mayo2020 .npy")
    p.add_argument("--dicom-root", type=Path, default=Path(MAYO_DICOM_ROOT))
    p.add_argument("--test-ratio", type=float, default=0.15, help="Fraction of patients for test")
    p.add_argument("--train512-patients", type=int, default=10, help="Patients copied into train512")
    p.add_argument("--limit-patients", type=int, default=0, help="Only process N patients (0=all)")
    p.add_argument("--skip-existing", action="store_true", help="Skip slices that already exist")
    p.add_argument("--manifest", type=Path, default=None, help="Write JSON manifest of pairs")
    return p.parse_args()


def hu_array(ds: pydicom.Dataset) -> np.ndarray:
    arr = ds.pixel_array.astype(np.float32)
    slope = float(getattr(ds, "RescaleSlope", 1))
    intercept = float(getattr(ds, "RescaleIntercept", 0))
    return arr * slope + intercept


def slice_sort_key(ds: pydicom.Dataset):
    inst = getattr(ds, "InstanceNumber", None)
    if inst is not None:
        return (0, int(inst))
    pos = getattr(ds, "ImagePositionPatient", None)
    if pos is not None and len(pos) >= 3:
        return (1, float(pos[2]))
    return (2, getattr(ds, "SOPInstanceUID", ""))


def collect_series(dicom_root: Path):
    """Group DICOM files by (patient, body, series description)."""
    groups = defaultdict(list)
    for dcm_path in dicom_root.rglob("*.dcm"):
        try:
            ds = pydicom.dcmread(str(dcm_path), stop_before_pixels=True)
        except Exception:
            continue
        if getattr(ds, "Modality", "") != "CT":
            continue
        body = str(getattr(ds, "BodyPartExamined", "") or "").upper()
        if body not in BODY_TO_ROOT:
            continue
        desc = str(getattr(ds, "SeriesDescription", "") or "")
        if desc not in (SERIES_FULL, SERIES_LOW):
            continue
        pid = str(getattr(ds, "PatientID", "unknown"))
        groups[(pid, body, desc)].append(dcm_path)
    return groups


def load_volume(paths: list[Path]) -> list[np.ndarray]:
    slices = []
    meta = []
    for p in paths:
        ds = pydicom.dcmread(str(p))
        meta.append(ds)
    meta.sort(key=slice_sort_key)
    for ds in meta:
        hu = hu_array(ds)
        if hu.ndim == 2:
            hu = hu[np.newaxis, ...]
        slices.append(hu.astype(np.float32))
    return slices


def region_prefix(body: str) -> str:
    return "lung" if body == "CHEST" else "ab"


def assign_phases(patient_ids: list[str], test_ratio: float) -> dict[str, str]:
    patient_ids = sorted(set(patient_ids))
    n_test = max(1, int(round(len(patient_ids) * test_ratio)))
    test_set = set(patient_ids[-n_test:])
    phases = {}
    for pid in patient_ids:
        phases[pid] = "test" if pid in test_set else "train"
    return phases


def save_slice(path: Path, arr: np.ndarray, skip_existing: bool):
    path.parent.mkdir(parents=True, exist_ok=True)
    if skip_existing and path.exists():
        return False
    np.save(path, arr.astype(np.float32))
    return True


def main():
    args = parse_args()
    dicom_root = args.dicom_root.resolve()
    if not dicom_root.is_dir():
        print(f"DICOM root not found: {dicom_root}", file=sys.stderr)
        print("Run: python Download_Mayo.py", file=sys.stderr)
        sys.exit(1)

    print(f"Scanning DICOM under {dicom_root} ...")
    groups = collect_series(dicom_root)
    patients = sorted({k[0] for k in groups})
    if args.limit_patients:
        patients = patients[: args.limit_patients]

    phase_map = assign_phases(patients, args.test_ratio)
    train512_set = set(sorted([p for p in patients if phase_map[p] == "train"])[: args.train512_patients])

    stats = {"pairs": 0, "slices": 0, "skipped": 0, "patients": 0}
    manifest = []

    for pid in tqdm(patients, desc="patients"):
        for body, out_root in BODY_TO_ROOT.items():
            full_key = (pid, body, SERIES_FULL)
            low_key = (pid, body, SERIES_LOW)
            if full_key not in groups or low_key not in groups:
                continue

            full_slices = load_volume(groups[full_key])
            low_slices = load_volume(groups[low_key])
            n = min(len(full_slices), len(low_slices))
            if n == 0:
                continue

            region = region_prefix(body)
            phase = phase_map[pid]
            phases_to_write = [phase]
            if pid in train512_set:
                phases_to_write.append("train512")

            stats["patients"] += 1
            stats["pairs"] += 1

            for i in range(n):
                fname = f"{region}-{i:05d}.npy"
                for ph in phases_to_write:
                    full_path = Path(out_root) / ph / FULL_DOSE_FOLDER / fname
                    low_path = Path(out_root) / ph / LOW_DOSE_FOLDER / fname
                    if save_slice(full_path, full_slices[i], args.skip_existing):
                        stats["slices"] += 1
                    else:
                        stats["skipped"] += 1
                    if save_slice(low_path, low_slices[i], args.skip_existing):
                        stats["slices"] += 1
                    else:
                        stats["skipped"] += 1

            manifest.append(
                {
                    "patient": pid,
                    "body": body,
                    "region": region,
                    "phase": phase,
                    "slices": n,
                    "out_root": out_root,
                }
            )

    print("\nDone.")
    print(f"  patients with pairs : {stats['patients']}")
    print(f"  slice files written : {stats['slices']} (+ skipped {stats['skipped']})")
    print(f"  train512 patients   : {len(train512_set)}")

    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(manifest, indent=2))
        print(f"  manifest            : {args.manifest}")

    for root in (MAYO2020_LUNG, MAYO2020_AB):
        for ph in PHASES:
            for sub in (FULL_DOSE_FOLDER, LOW_DOSE_FOLDER):
                n = len(list(Path(root).glob(f"{ph}/{sub}/*.npy")))
                if n:
                    print(f"  {root}/{ph}/{sub}: {n} files")


if __name__ == "__main__":
    main()
