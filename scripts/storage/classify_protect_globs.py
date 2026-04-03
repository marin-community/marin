#!/usr/bin/env -S uv run
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Classify normalized protect globs by how they can be backed up.

This is local-only. It does not touch GCS.

The goal is to separate rows into:
- STS-compatible direct prefixes
- STS-compatible after listing a parent directory and resolving concrete prefixes
- rows that need explicit object-manifest expansion or manual review
"""

import argparse
import csv
from pathlib import Path

WILDCARD_CHARS = set("*?[]")


def _bucket_from_gs_url(url: str) -> str:
    return url.removeprefix("gs://").split("/", 1)[0]


def _object_path_from_gs_url(url: str) -> str:
    parts = url.removeprefix("gs://").split("/", 1)
    return parts[1] if len(parts) == 2 else ""


def _segments(path: str) -> list[str]:
    return [segment for segment in path.split("/") if segment != ""]


def _has_wildcard(segment: str) -> bool:
    return any(ch in segment for ch in WILDCARD_CHARS)


def _classify_glob(url: str) -> tuple[str, str, str]:
    """
    Returns:
        classification,
        listing_prefix,
        concrete_prefix_hint
    """
    bucket = _bucket_from_gs_url(url)
    object_path = _object_path_from_gs_url(url)
    segs = _segments(object_path)

    if not segs:
        return "object_manifest_or_manual", f"gs://{bucket}/", ""

    # Direct recursive prefix: gs://bucket/foo/bar/**
    if object_path.endswith("/**"):
        base = object_path[:-3].rstrip("/")
        if "*" not in base and "?" not in base and "[" not in base and "]" not in base:
            return "sts_prefix_direct", f"gs://{bucket}/{base}/", f"gs://{bucket}/{base}/"

    wildcard_indices = [i for i, segment in enumerate(segs) if _has_wildcard(segment)]
    if not wildcard_indices:
        # Exact object or exact prefix; if no trailing /** survived normalization,
        # treat it as a concrete prefix candidate.
        prefix = object_path.rstrip("/")
        return "sts_prefix_direct", f"gs://{bucket}/{prefix}/", f"gs://{bucket}/{prefix}/"

    first_wildcard = wildcard_indices[0]
    last_wildcard = wildcard_indices[-1]

    # If wildcards appear in more than one segment, or before the final segment,
    # we will likely need object-manifest expansion.
    if first_wildcard != last_wildcard or first_wildcard < len(segs) - 1:
        listing_base = "/".join(segs[:first_wildcard]).rstrip("/")
        listing_prefix = f"gs://{bucket}/{listing_base}/" if listing_base else f"gs://{bucket}/"
        return "object_manifest_or_manual", listing_prefix, ""

    prefix_segments = segs[:first_wildcard]
    wildcard_segment = segs[first_wildcard]
    suffix_segments = segs[first_wildcard + 1 :]

    listing_base = "/".join(prefix_segments).rstrip("/")
    listing_prefix = f"gs://{bucket}/{listing_base}/" if listing_base else f"gs://{bucket}/"

    # Best case for listing-based STS resolution:
    # - exactly one wildcard segment
    # - all remaining suffix segments are literal
    # - most common case is matching directories under one parent
    if suffix_segments and any(_has_wildcard(segment) for segment in suffix_segments):
        return "object_manifest_or_manual", listing_prefix, ""

    concrete_hint = f"{listing_prefix}{wildcard_segment.rstrip('*')}"

    return "sts_prefix_via_listing", listing_prefix, concrete_hint


def classify_manifest(input_csv: Path, output_csv: Path) -> tuple[int, dict[str, int]]:
    with input_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    output_rows: list[dict[str, str]] = []
    counts: dict[str, int] = {
        "sts_prefix_direct": 0,
        "sts_prefix_via_listing": 0,
        "object_manifest_or_manual": 0,
    }

    for row in rows:
        normalized_glob = row["normalized_glob"]
        classification, listing_prefix, concrete_prefix_hint = _classify_glob(normalized_glob)
        counts[classification] += 1
        output_rows.append(
            {
                **row,
                "classification": classification,
                "listing_prefix": listing_prefix,
                "concrete_prefix_hint": concrete_prefix_hint,
            }
        )

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                *rows[0].keys(),
                "classification",
                "listing_prefix",
                "concrete_prefix_hint",
            ],
        )
        writer.writeheader()
        writer.writerows(output_rows)

    return len(output_rows), counts


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify protect globs by backup strategy.")
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=Path("scripts/storage/protect_manifest_deduped.csv"),
        help="Normalized deduped protect manifest.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("scripts/storage/protect_manifest_classified.csv"),
        help="Output classified manifest.",
    )
    args = parser.parse_args()

    total, counts = classify_manifest(args.input_csv, args.output_csv)
    print(f"Wrote {total} classified protect rows to {args.output_csv}.")
    for key, value in counts.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
