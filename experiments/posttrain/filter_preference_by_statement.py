#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Filter existing Bloom preference shards by statement_id and write per-statement datasets.

Downloads the full preference dataset from GCS once, filters by statement,
writes new shards locally, then uploads to GCS. No re-generation needed —
just filtering existing data.

Usage:
    uv run python experiments/posttrain/filter_preference_by_statement.py

Output paths on GCS:
    gs://marin-us-central1/preference/bloom_v2_singleton/{statement_id}/{train,val}/shard-00000.jsonl.gz
    gs://marin-us-central1/preference/bloom_v2_singleton/{combined_name}/{train,val}/shard-00000.jsonl.gz
"""

from __future__ import annotations

import gzip
import json
import logging
import subprocess
import sys
from collections import Counter
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Source dataset on GCS
SOURCE_PREFIX = "gs://marin-us-central1/preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite"

# Output prefix on GCS
OUTPUT_PREFIX = "gs://marin-us-central1/preference/bloom_v2_singleton"

# Local scratch
LOCAL_ROOT = Path.home() / "preference_filter_scratch"

# Individual statements to filter
SINGLETON_STATEMENTS = [
    "support_mental_health",
    "do_not_encourage_self_harm",
    "avoid_overstepping",
]

# Combined set name
COMBINED_NAME = "support_mental_health+do_not_encourage_self_harm+avoid_overstepping"
COMBINED_STATEMENTS = set(SINGLETON_STATEMENTS)

SHARD_SIZE = 5000


def download_source(split: str) -> list[dict]:
    """Download all shards for a split from GCS, return all records."""
    src = f"{SOURCE_PREFIX}/{split}/"
    local_dir = LOCAL_ROOT / "source" / split
    local_dir.mkdir(parents=True, exist_ok=True)

    logger.info("downloading %s to %s", src, local_dir)
    subprocess.run(
        ["gcloud", "storage", "cp", f"{src}*.jsonl.gz", str(local_dir) + "/"],
        check=True,
        capture_output=True,
    )

    records = []
    for shard in sorted(local_dir.glob("*.jsonl.gz")):
        with gzip.open(shard, "rt", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    logger.info("loaded %d records from %s (%d shards)", len(records), split, len(list(local_dir.glob("*.jsonl.gz"))))
    return records


def write_shards(records: list[dict], local_dir: Path, shard_size: int = SHARD_SIZE) -> int:
    """Write records as gzipped JSONL shards. Returns number of shards."""
    local_dir.mkdir(parents=True, exist_ok=True)
    shard_idx = 0
    for start in range(0, len(records), shard_size):
        batch = records[start : start + shard_size]
        shard_path = local_dir / f"shard-{shard_idx:05d}.jsonl.gz"
        with gzip.open(shard_path, "wt", encoding="utf-8") as f:
            for r in batch:
                f.write(json.dumps(r, ensure_ascii=True) + "\n")
        shard_idx += 1
    return shard_idx


def upload_to_gcs(local_dir: Path, gcs_prefix: str, dry_run: bool = False) -> None:
    if dry_run:
        logger.info("[dry-run] would upload %s -> %s", local_dir, gcs_prefix)
        return
    shards = list(local_dir.glob("*.jsonl.gz"))
    if not shards:
        logger.warning("no shards to upload in %s", local_dir)
        return
    cmd = ["gcloud", "storage", "cp"] + [str(s) for s in shards] + [gcs_prefix]
    logger.info("uploading %d shards to %s", len(shards), gcs_prefix)
    subprocess.run(cmd, check=True, capture_output=True)


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    # Download source data
    train_records = download_source("train")
    val_records = download_source("val_deduped")

    # Count source distribution
    train_dist = Counter(r["statement_id"] for r in train_records)
    val_dist = Counter(r["statement_id"] for r in val_records)
    logger.info("source train: %d records, %d statements", len(train_records), len(train_dist))
    logger.info("source val: %d records, %d statements", len(val_records), len(val_dist))

    # Filter and write each singleton
    for stmt in SINGLETON_STATEMENTS:
        logger.info("--- filtering: %s ---", stmt)
        train_filtered = [r for r in train_records if r["statement_id"] == stmt]
        val_filtered = [r for r in val_records if r["statement_id"] == stmt]

        train_dir = LOCAL_ROOT / "output" / stmt / "train"
        val_dir = LOCAL_ROOT / "output" / stmt / "val"

        n_train_shards = write_shards(train_filtered, train_dir)
        n_val_shards = write_shards(val_filtered, val_dir)

        # Count unique prompts
        train_prompts = len(set(r["prompt"] for r in train_filtered))
        val_prompts = len(set(r["prompt"] for r in val_filtered))

        logger.info(
            "  %s: train=%d pairs (%d prompts, %d shards), val=%d pairs (%d prompts, %d shards)",
            stmt,
            len(train_filtered),
            train_prompts,
            n_train_shards,
            len(val_filtered),
            val_prompts,
            n_val_shards,
        )

        upload_to_gcs(train_dir, f"{OUTPUT_PREFIX}/{stmt}/train/", dry_run)
        upload_to_gcs(val_dir, f"{OUTPUT_PREFIX}/{stmt}/val/", dry_run)

    # Filter and write combined set
    logger.info("--- filtering: %s ---", COMBINED_NAME)
    train_combined = [r for r in train_records if r["statement_id"] in COMBINED_STATEMENTS]
    val_combined = [r for r in val_records if r["statement_id"] in COMBINED_STATEMENTS]

    train_dir = LOCAL_ROOT / "output" / COMBINED_NAME / "train"
    val_dir = LOCAL_ROOT / "output" / COMBINED_NAME / "val"

    n_train_shards = write_shards(train_combined, train_dir)
    n_val_shards = write_shards(val_combined, val_dir)

    train_prompts = len(set(r["prompt"] for r in train_combined))
    val_prompts = len(set(r["prompt"] for r in val_combined))

    logger.info(
        "  %s: train=%d pairs (%d prompts, %d shards), val=%d pairs (%d prompts, %d shards)",
        COMBINED_NAME,
        len(train_combined),
        train_prompts,
        n_train_shards,
        len(val_combined),
        val_prompts,
        n_val_shards,
    )

    upload_to_gcs(train_dir, f"{OUTPUT_PREFIX}/{COMBINED_NAME}/train/", dry_run)
    upload_to_gcs(val_dir, f"{OUTPUT_PREFIX}/{COMBINED_NAME}/val/", dry_run)

    # Summary
    print()
    print("=" * 80)
    print(" Summary")
    print("=" * 80)
    for stmt in SINGLETON_STATEMENTS:
        train_n = sum(1 for r in train_records if r["statement_id"] == stmt)
        val_n = sum(1 for r in val_records if r["statement_id"] == stmt)
        train_p = len(set(r["prompt"] for r in train_records if r["statement_id"] == stmt))
        print(f"  {stmt:<45} train={train_n:>5} pairs ({train_p:>3} prompts)  val={val_n:>4}")
    train_n = len(train_combined)
    val_n = len(val_combined)
    print(f"  {COMBINED_NAME:<45} train={train_n:>5} pairs ({train_prompts:>3} prompts)  val={val_n:>4}")
    print()
    print("GCS output prefix:", OUTPUT_PREFIX)
    print("Local scratch:", LOCAL_ROOT / "output")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
