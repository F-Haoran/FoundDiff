#!/usr/bin/env python3
"""
Download the AAPM-Mayo low-dose CT dataset from NCI Imaging Data Commons (IDC).

Collection: ldct_and_projection_data
  - 2020 Mayo LDCT (head/chest/abdomen; IDC index currently lists chest + abdomen images)
  - 2016 AAPM Grand Challenge cases (merged into the same TCIA/IDC collection)

Data is saved as DICOM under:
  <FoundDiff>/data/mayo/dicom/

Requires:
  pip install idc-index

Examples:
  # Preview what would be downloaded (~37 GB for all reconstructed CT images)
  python Download_Mayo.py --list-only

  # Download a small sample for testing
  python Download_Mayo.py --limit 2

  # Download all reconstructed CT images (recommended for FoundDiff preprocessing)
  python Download_Mayo.py --dose all

  # Download only abdominal full-dose images
  python Download_Mayo.py --body-part ABDOMEN --dose full

  # Download chest low-dose images only
  python Download_Mayo.py --body-part CHEST --dose low
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from data.paths import MAYO_DICOM_ROOT

COLLECTION_ID = "ldct_and_projection_data"
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DOWNLOAD_DIR = Path(MAYO_DICOM_ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download AAPM-Mayo LDCT data from IDC into the FoundDiff folder."
    )
    parser.add_argument(
        "--download-dir",
        type=Path,
        default=DEFAULT_DOWNLOAD_DIR,
        help=f"Output directory for DICOM files (default: {DEFAULT_DOWNLOAD_DIR})",
    )
    parser.add_argument(
        "--body-part",
        choices=["ABDOMEN", "CHEST", "ALL"],
        default="ALL",
        help="Anatomical region filter (default: ALL)",
    )
    parser.add_argument(
        "--dose",
        choices=["full", "low", "all"],
        default="all",
        help="Dose level filter based on series description (default: all)",
    )
    parser.add_argument(
        "--include-projections",
        action="store_true",
        help="Include raw projection data (~1 TB+). Default downloads reconstructed CT images only.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Max number of series to download (0 = no limit)",
    )
    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Query and print matching series without downloading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Pass dry_run=True to idc-index (no files written)",
    )
    return parser.parse_args()


def build_query(args: argparse.Namespace) -> str:
    conditions = [f"collection_id = '{COLLECTION_ID}'"]

    if args.include_projections:
        conditions.append("Modality = 'CT'")
    else:
        conditions.append("sop_class_name = 'CT Image Storage'")

    if args.body_part != "ALL":
        conditions.append(f"BodyPartExamined = '{args.body_part}'")

    dose_map = {
        "full": "SeriesDescription = 'Full Dose Images'",
        "low": "SeriesDescription = 'Low Dose Images'",
    }
    if args.dose in dose_map:
        conditions.append(dose_map[args.dose])

    where_clause = " AND ".join(conditions)
    limit_clause = f"\nLIMIT {args.limit}" if args.limit > 0 else ""

    return f"""
SELECT
    collection_id,
    PatientID,
    BodyPartExamined,
    SeriesDescription,
    SeriesInstanceUID,
    series_size_MB,
    instanceCount
FROM index
WHERE {where_clause}
ORDER BY PatientID, SeriesNumber
{limit_clause}
""".strip()


def ensure_idc_index():
    try:
        from idc_index import IDCClient
    except ImportError:
        print(
            "Error: idc-index is not installed.\n"
            "Install it with:  pip install idc-index",
            file=sys.stderr,
        )
        sys.exit(1)
    return IDCClient()


def print_summary(results) -> None:
    total_series = len(results)
    total_mb = results["series_size_MB"].sum() if total_series else 0.0
    total_gb = total_mb / 1024.0

    print(f"Matching series : {total_series}")
    print(f"Estimated size  : {total_gb:.2f} GB ({total_mb:.1f} MB)")

    if total_series == 0:
        return

    if "BodyPartExamined" in results.columns and "SeriesDescription" in results.columns:
        summary = (
            results.groupby(["BodyPartExamined", "SeriesDescription"])
            .agg(series=("SeriesInstanceUID", "count"), size_gb=("series_size_MB", lambda s: s.sum() / 1024))
            .reset_index()
        )
        print("\nBreakdown:")
        for _, row in summary.iterrows():
            print(
                f"  {row['BodyPartExamined']:8s} | {str(row['SeriesDescription']):20s} | "
                f"{int(row['series']):4d} series | {row['size_gb']:.2f} GB"
            )


def main() -> None:
    args = parse_args()
    download_dir = args.download_dir.resolve()
    download_dir.mkdir(parents=True, exist_ok=True)

    client = ensure_idc_index()
    query = build_query(args)

    print("Connecting to IDC...")
    print(f"Collection      : {COLLECTION_ID}")
    print(f"Download dir    : {download_dir}")
    print(f"Query filters   : body_part={args.body_part}, dose={args.dose}, "
          f"images_only={not args.include_projections}")
    if args.limit:
        print(f"Limit           : {args.limit} series")
    print()

    results = client.sql_query(query)
    print_summary(results)
    print()

    if len(results) == 0:
        print("No series matched the query. Nothing to download.")
        return

    if args.list_only:
        print("Series list (--list-only):")
        display_cols = [
            "PatientID",
            "BodyPartExamined",
            "SeriesDescription",
            "instanceCount",
            "series_size_MB",
            "SeriesInstanceUID",
        ]
        print(results[display_cols].to_string(index=False))
        return

    series_uids = results["SeriesInstanceUID"].tolist()
    print(f"Starting download of {len(series_uids)} series...")
    print("(This may take a long time for the full dataset. Use --limit for a quick test.)\n")

    client.download_dicom_series(
        seriesInstanceUID=series_uids,
        downloadDir=str(download_dir),
        dry_run=args.dry_run,
        quiet=False,
        show_progress_bar=True,
    )

    if args.dry_run:
        print("\nDry run complete. No files were written.")
    else:
        print(f"\nDownload complete. DICOM files are in:\n  {download_dir}")
        print(
            "\nNext step — convert DICOM to .npy:\n"
            "  python Preprocess_Mayo.py --dicom-root data/mayo/dicom\n"
            "Then run inference (IDC chest/abdomen data):\n"
            "  CUDA_VISIBLE_DEVICES=0 python train.py --name FoundDiff --epoch 400 "
            "--dataset 2020_seen --data-mode idc"
        )


if __name__ == "__main__":
    main()
