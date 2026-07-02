#!/usr/bin/env python3
"""Filename conventions for custom NIfTI denoising pipelines."""

from __future__ import annotations

from pathlib import Path

# ct:   *_CT.nii.gz is the noisy volume to denoise (common custom export naming)
# ldct: *_LDCT.nii.gz is input; *_CT.nii.gz treated as full-dose reference (Mayo-style)
# any:  process all .nii.gz except obvious reference / output names
NAMING_CT = "ct"
NAMING_LDCT = "ldct"
NAMING_ANY = "any"
NAMING_CHOICES = (NAMING_CT, NAMING_LDCT, NAMING_ANY)


def nifti_stem(path: Path) -> str:
    name = path.name
    if name.endswith(".nii.gz"):
        return name[:-7]
    if name.endswith(".nii"):
        return name[:-4]
    return path.stem


def infer_case_name(path: Path, case: str | None = None) -> str:
    """Case id for manifest / staging; defaults to the input filename stem."""
    if case:
        return case
    return nifti_stem(path)


def is_output_artifact(path: Path) -> bool:
    upper = path.name.upper()
    return upper.endswith("_DENOISED.NII.GZ") or upper.endswith("_DENOISED.NII")


def should_denoise(path: Path, *, naming: str = NAMING_CT) -> bool:
    """Return True if this file should be treated as a denoising input."""
    if not path.is_file() or is_output_artifact(path):
        return False

    stem = nifti_stem(path).upper()
    if naming == NAMING_CT:
        if stem.endswith("_LDCT"):
            return False
        if stem.endswith("_CT"):
            return True
        if stem.endswith("_LOW"):
            return True
        return not stem.endswith("_FULL")

    if naming == NAMING_LDCT:
        if stem.endswith("_CT"):
            return False
        if stem.endswith("_LDCT"):
            return True
        if stem.endswith("_LOW"):
            return True
        return not stem.endswith("_FULL")

    if naming == NAMING_ANY:
        return not stem.endswith("_FULL")

    raise ValueError(f"Unknown naming mode: {naming!r}; choose {NAMING_CHOICES}")
