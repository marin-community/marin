#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Extract STS-ready direct prefixes from the classified protect manifest."""

import argparse
import csv
from pathlib import Path


def extract_sts_direct_prefixes(input_csv: Path, output_csv: Path) -> int:
    with input_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    output_rows: list[dict[str, str]] = []
    for row in rows:
        if row["classification"] != "sts_prefix_direct":
            continue

        output_rows.append(
            {
                "sts_prefix": row["listing_prefix"],
                "bucket": row["bucket"],
                "owners": row["owners"],
                "reasons": row["reasons"],
                "sources": row["sources"],
                "artifact_kinds": row["artifact_kinds"],
                "priority_max": row["priority_max"],
                "normalized_glob": row["normalized_glob"],
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
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
        writer.writerows(output_rows)

    return len(output_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract STS-direct prefixes from the classified protect manifest.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("scripts/storage/protect_manifest_classified.csv"),
        help="Classified protect manifest.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("scripts/storage/protect_manifest_sts_direct.csv"),
        help="Output CSV of STS-direct prefixes.",
    )
    args = parser.parse_args()

    total = extract_sts_direct_prefixes(args.input_csv, args.output_csv)
    print(f"Wrote {total} STS-direct prefixes to {args.output_csv}.")


if __name__ == "__main__":
    main()
