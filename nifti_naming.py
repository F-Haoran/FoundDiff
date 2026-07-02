#!/usr/bin/env python3
"""Filename conventions and input discovery for custom NIfTI denoising."""

from __future__ import annotations

import fnmatch
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


def infer_case_name(
    path: Path,
    case: str | None = None,
    *,
    strip_suffixes: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Case id for manifest / staging; defaults to filename stem (optionally stripped)."""
    if case:
        return case
    stem = nifti_stem(path)
    if strip_suffixes:
        upper = stem.upper()
        for suffix in strip_suffixes:
            token = str(suffix).upper()
            if upper.endswith(token):
                return stem[: -len(suffix)]
    return stem


def is_output_artifact(path: Path) -> bool:
    upper = path.name.upper()
    return upper.endswith("_DENOISED.NII.GZ") or upper.endswith("_DENOISED.NII")


def matches_pattern(path: Path, pattern: str) -> bool:
    """Match filename against a glob pattern (e.g. '*_CT.nii.gz', 'APNHC*.nii.gz')."""
    if not pattern or pattern in {"*", "*.*"}:
        return True
    return fnmatch.fnmatch(path.name, pattern)


def should_denoise(path: Path, *, naming: str = NAMING_CT, pattern: str | None = None) -> bool:
    """Return True if this file should be treated as a denoising input."""
    if not path.is_file() or is_output_artifact(path):
        return False
    if pattern and not matches_pattern(path, pattern):
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


def collect_input_files(
    input_path: Path,
    *,
    pattern: str = "*.nii.gz",
    naming: str = NAMING_CT,
) -> list[Path]:
    """
    Resolve input_path to one or more NIfTI files.

    - If input_path is a file: return [file] when it passes filters.
    - If input_path is a directory: glob pattern inside the folder, then filter.
    """
    path = input_path.expanduser().resolve()
    if path.is_file():
        if should_denoise(path, naming=naming, pattern=pattern):
            return [path]
        raise SystemExit(
            f"Input file does not match pattern/naming filters: {path}\n"
            f"  pattern={pattern!r}  naming={naming!r}"
        )

    if not path.is_dir():
        raise SystemExit(f"Input path not found: {path}")

    candidates = sorted(path.glob(pattern))
    selected = [p for p in candidates if should_denoise(p, naming=naming, pattern=pattern)]
    if not selected:
        raise SystemExit(
            f"No matching inputs under {path}\n"
            f"  pattern={pattern!r}  naming={naming!r}"
        )
    return selected
