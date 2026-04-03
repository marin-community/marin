#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Emit per-region STS prefix inputs and backup bucket suggestions.

Storage Transfer Service manifests require concrete object names, not prefixes.
This script groups the already-resolved direct prefixes into per-region CSVs that
can be used to configure STS jobs, emits companion listing-based regional CSVs
for wildcard families that still need one listing pass, and emits a companion
plan for temporary same-region backup buckets with soft delete disabled.
"""

import argparse
import csv
from collections import defaultdict
from pathlib import Path

BUCKET_LOCATIONS = {
    "marin-eu-west4": "EUROPE-WEST4",
    "marin-us-central1": "US-CENTRAL1",
    "marin-us-central2": "US-CENTRAL2",
    "marin-us-east1": "US-EAST1",
    "marin-us-east5": "US-EAST5",
    "marin-us-west4": "US-WEST4",
}


def _region_from_bucket(bucket: str) -> str:
    prefix = "marin-"
    if bucket.startswith(prefix):
        return bucket.removeprefix(prefix)
    return bucket


def _location_for_bucket(bucket: str) -> str:
    return BUCKET_LOCATIONS.get(bucket, _region_from_bucket(bucket).upper())


def _suggest_backup_bucket(bucket: str, suffix: str) -> str:
    region = _region_from_bucket(bucket)
    return f"marin-tmp-backup-{region}-{suffix}"


def emit_region_inputs(
    input_csv: Path,
    classified_input_csv: Path,
    output_dir: Path,
    via_listing_output_dir: Path,
    plan_csv: Path,
    backup_suffix: str,
) -> dict[str, int]:
    with input_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))
    with classified_input_csv.open(newline="") as f:
        classified_rows = list(csv.DictReader(f))

    rows_by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        rows_by_bucket[row["bucket"]].append(row)

    via_listing_rows_by_bucket: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in classified_rows:
        if row["classification"] != "sts_prefix_via_listing":
            continue
        via_listing_rows_by_bucket[row["bucket"]].append(row)

    output_dir.mkdir(parents=True, exist_ok=True)
    via_listing_output_dir.mkdir(parents=True, exist_ok=True)

    counts: dict[str, int] = {}
    plan_rows: list[dict[str, str]] = []

    for bucket in sorted(rows_by_bucket):
        bucket_rows = sorted(rows_by_bucket[bucket], key=lambda row: row["sts_prefix"])
        region = _region_from_bucket(bucket)
        location = _location_for_bucket(bucket)
        counts[region] = len(bucket_rows)

        region_csv = output_dir / f"sts_direct_prefixes_{region}.csv"
        with region_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "sts_prefix",
                    "bucket",
                    "owners",
                    "reasons",
                    "sources",
                    "artifact_kinds",
                    "priority_max",
                    "normalized_glob",
                ],
            )
            writer.writeheader()
            writer.writerows(bucket_rows)

        via_listing_csv = via_listing_output_dir / f"sts_via_listing_{region}.csv"
        via_listing_rows = sorted(via_listing_rows_by_bucket.get(bucket, []), key=lambda row: row["normalized_glob"])
        with via_listing_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "listing_prefix",
                    "concrete_prefix_hint",
                    "bucket",
                    "owners",
                    "reasons",
                    "sources",
                    "artifact_kinds",
                    "priority_max",
                    "normalized_glob",
                ],
            )
            writer.writeheader()
            writer.writerows(
                {
                    "listing_prefix": row["listing_prefix"],
                    "concrete_prefix_hint": row["concrete_prefix_hint"],
                    "bucket": row["bucket"],
                    "owners": row["owners"],
                    "reasons": row["reasons"],
                    "sources": row["sources"],
                    "artifact_kinds": row["artifact_kinds"],
                    "priority_max": row["priority_max"],
                    "normalized_glob": row["normalized_glob"],
                }
                for row in via_listing_rows
            )

        backup_bucket = _suggest_backup_bucket(bucket, backup_suffix)
        plan_rows.append(
            {
                "region": region,
                "location": location,
                "source_bucket": bucket,
                "sts_prefix_csv": str(region_csv),
                "direct_prefix_count": str(len(bucket_rows)),
                "sts_via_listing_csv": str(via_listing_csv),
                "via_listing_count": str(len(via_listing_rows)),
                "suggested_backup_bucket": backup_bucket,
                "create_bucket_command": " ".join(
                    [
                        f"gcloud storage buckets create gs://{backup_bucket}",
                        f"--location={location}",
                        "--soft-delete-duration=0",
                        "--no-enable-autoclass",
                    ]
                ),
                "disable_soft_delete_command": f"gcloud storage buckets update --clear-soft-delete gs://{backup_bucket}",
            }
        )

    plan_csv.parent.mkdir(parents=True, exist_ok=True)
    with plan_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "region",
                "location",
                "source_bucket",
                "sts_prefix_csv",
                "direct_prefix_count",
                "sts_via_listing_csv",
                "via_listing_count",
                "suggested_backup_bucket",
                "create_bucket_command",
                "disable_soft_delete_command",
            ],
        )
        writer.writeheader()
        writer.writerows(plan_rows)

    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Emit per-region STS prefix inputs.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("scripts/storage/protect_manifest_sts_direct.csv"),
        help="STS-direct prefix CSV.",
    )
    parser.add_argument(
        "--classified-input-csv",
        type=Path,
        default=Path("scripts/storage/protect_manifest_classified.csv"),
        help="Classified protect manifest used to extract listing-based rows.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("scripts/storage/sts_direct_regions"),
        help="Directory for per-region STS prefix CSVs.",
    )
    parser.add_argument(
        "--via-listing-output-dir",
        type=Path,
        default=Path("scripts/storage/sts_via_listing_regions"),
        help="Directory for per-region listing-based STS CSVs.",
    )
    parser.add_argument(
        "--plan-csv",
        type=Path,
        default=Path("scripts/storage/sts_region_backup_plan.csv"),
        help="Output CSV with regional backup bucket suggestions.",
    )
    parser.add_argument(
        "--backup-suffix",
        default="purge-tmp-20260323",
        help="Suffix to use when suggesting temporary backup bucket names.",
    )
    args = parser.parse_args()

    counts = emit_region_inputs(
        input_csv=args.input_csv,
        classified_input_csv=args.classified_input_csv,
        output_dir=args.output_dir,
        via_listing_output_dir=args.via_listing_output_dir,
        plan_csv=args.plan_csv,
        backup_suffix=args.backup_suffix,
    )

    print(f"Wrote {len(counts)} per-region STS prefix CSVs to {args.output_dir}.")
    for region, count in sorted(counts.items()):
        print(f"{region}: {count}")
    print(f"Wrote per-region listing-based STS CSVs to {args.via_listing_output_dir}.")
    print(f"Wrote regional backup bucket plan to {args.plan_csv}.")


if __name__ == "__main__":
    main()
