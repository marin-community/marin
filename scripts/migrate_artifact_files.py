# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Migrate legacy artifact files on GCS to the canonical ``.artifact.json`` name.

Reads the GCS object inventory at
``gs://marin-us-central2/tmp/storage-scan/deduped/objects-{shard:05d}.parquet``
(one parquet per inventory shard; columns: ``bucket``, ``name``, ``size_bytes``,
``storage_class_id``, ``created``, ``updated``), filters to objects whose
basename is ``.artifact`` or ``artifact.json``, and renames each to
``.artifact.json`` in the same prefix.

Two modes:

* ``--dry-run`` (default): logs each planned migration, performs no GCS writes.
* ``--no-dry-run``: performs the rename as a server-side copy followed by a
  delete. Skips (and counts) any case where the target object already exists
  to avoid clobbering.

Submit on iris (us-central2, CPU, interactive):

    uv run iris --cluster=marin job run --region us-central2 --extra=cpu \\
        --priority interactive \\
        -- python scripts/migrate_artifact_files.py --no-dry-run

Run locally for inspection:

    uv run python scripts/migrate_artifact_files.py
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable

import gcsfs
import pyarrow.compute as pc
import pyarrow.parquet as pq
from fray import ResourceConfig
from marin.utils import fsspec_glob
from rigging.log_setup import configure_logging
from zephyr import Dataset, ZephyrContext

logger = logging.getLogger(__name__)


LISTING_GLOB = "gs://marin-us-central2/tmp/storage-scan/deduped/objects-*.parquet"
TARGET_BASENAME = ".artifact.json"
# Matches an object whose basename (segment after last '/', or the whole name
# if no '/') equals '.artifact' or 'artifact.json'. Object names never start
# with '/', so a leading '/' would not match either branch.
BASENAME_REGEX = r"(?:^|/)(?:\.artifact|artifact\.json)$"

Counts = dict[str, int]
COUNT_KEYS = (
    "scanned",
    "matched",
    "migrated",
    "skipped_target_exists",
    "skipped_src_missing",
    "errors",
)


def _target_name(name: str) -> str:
    """Return ``name`` with its basename replaced by ``.artifact.json``."""
    if "/" in name:
        return name.rsplit("/", 1)[0] + "/" + TARGET_BASENAME
    return TARGET_BASENAME


def _migrate_one(path: str, *, dry_run: bool) -> Counts:
    """Read one inventory parquet, find matches, and migrate (or log) each."""
    counts: Counts = {k: 0 for k in COUNT_KEYS}
    fs = gcsfs.GCSFileSystem()

    with fs.open(path, "rb") as f:
        table = pq.read_table(f, columns=["bucket", "name"])
    counts["scanned"] = table.num_rows

    mask = pc.match_substring_regex(table.column("name"), BASENAME_REGEX)
    filtered = table.filter(mask)
    counts["matched"] = filtered.num_rows
    if filtered.num_rows == 0:
        return counts

    for row in filtered.to_pylist():
        bucket = row["bucket"]
        src_name = row["name"]
        tgt_name = _target_name(src_name)
        src_uri = f"gs://{bucket}/{src_name}"
        tgt_uri = f"gs://{bucket}/{tgt_name}"

        if src_name == tgt_name:
            # Cannot happen given the regex, but guard anyway.
            continue

        if dry_run:
            logger.info("DRY-RUN rename: %s -> %s", src_uri, tgt_uri)
            continue

        try:
            if fs.exists(tgt_uri):
                logger.warning("SKIP target exists: %s (src=%s)", tgt_uri, src_uri)
                counts["skipped_target_exists"] += 1
                continue
            # gcsfs.mv is server-side copy + delete (GCS has no atomic rename).
            fs.mv(src_uri, tgt_uri)
            counts["migrated"] += 1
            logger.info("MIGRATED: %s -> %s", src_uri, tgt_uri)
        except FileNotFoundError:
            logger.warning("SKIP src missing: %s", src_uri)
            counts["skipped_src_missing"] += 1
        except Exception:
            logger.exception("ERROR migrating %s -> %s", src_uri, tgt_uri)
            counts["errors"] += 1

    return counts


def _merge_counts(per_shard: Iterable[Counts]) -> Counts:
    merged: Counts = {k: 0 for k in COUNT_KEYS}
    for c in per_shard:
        for k, v in c.items():
            merged[k] = merged.get(k, 0) + v
    return merged


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--listing-glob", default=LISTING_GLOB)
    parser.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="If true (default), log planned renames but do not write to GCS.",
    )
    parser.add_argument("--max-workers", type=int, default=64)
    args = parser.parse_args()

    configure_logging(logging.INFO)

    logger.info("Listing inventory shards: %s", args.listing_glob)
    shards = sorted(fsspec_glob(args.listing_glob))
    if not shards:
        raise RuntimeError(f"No parquet shards under {args.listing_glob}")
    logger.info("Found %d parquet shards (dry_run=%s)", len(shards), args.dry_run)

    num_workers = min(args.max_workers, len(shards))
    ctx = ZephyrContext(
        resources=ResourceConfig(cpu=1, ram="2g"),
        max_workers=num_workers,
        name="migrate-artifact-files",
    )

    dry_run = args.dry_run
    ds = Dataset.from_list(shards).map(lambda path: _migrate_one(path, dry_run=dry_run))
    outcome = ctx.execute(ds, verbose=True)
    totals: Counts = _merge_counts(outcome.results)

    logger.info("=" * 60)
    logger.info("MODE: %s", "DRY-RUN" if dry_run else "LIVE")
    for key in COUNT_KEYS:
        logger.info("  %-25s %d", key, totals[key])


if __name__ == "__main__":
    main()
