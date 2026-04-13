#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deduplicate Bloom SpecEval v2 validation preference JSONL from 9 pairs per prompt to 1.

The original val set has 23,454 rows across 5 shards. Each unique question_id has
9 rows (3 chosen x 3 rejected cross-product). This script keeps one representative
pair per question_id, preserving all unique prompts.

Usage:
    uv run python scripts/dedupe_val_prefs.py
"""

import gzip
import json
import subprocess
import tempfile
from collections import Counter

GCS_SRC = "gs://marin-us-central1/preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/val"
GCS_DST = "gs://marin-us-central1/preference/bloom_openai_model_spec_v2_gpt41_vs_mixtral_opposite/val_deduped"
NUM_SHARDS = 5


def read_all_shards() -> list[dict]:
    """Read all val shards from GCS and return as list of dicts."""
    all_rows = []
    for i in range(NUM_SHARDS):
        shard_path = f"{GCS_SRC}/shard-{i:05d}.jsonl.gz"
        print(f"Reading {shard_path}...")
        result = subprocess.run(
            ["gcloud", "storage", "cat", shard_path],
            capture_output=True,
            check=True,
        )
        raw = gzip.decompress(result.stdout)
        for line in raw.decode("utf-8").strip().split("\n"):
            if line:
                all_rows.append(json.loads(line))
    print(f"Read {len(all_rows)} total rows from {NUM_SHARDS} shards.")
    return all_rows


def dedupe(rows: list[dict]) -> list[dict]:
    """Keep first row per question_id."""
    seen = set()
    deduped = []
    for row in rows:
        qid = row["question_id"]
        if qid not in seen:
            seen.add(qid)
            deduped.append(row)
    return deduped


def verify(original: list[dict], deduped: list[dict]) -> None:
    """Assert correctness of deduplication."""
    orig_qids = {row["question_id"] for row in original}
    deduped_qids = [row["question_id"] for row in deduped]

    assert len(deduped) == len(orig_qids), f"Expected {len(orig_qids)} rows, got {len(deduped)}"
    assert len(deduped_qids) == len(set(deduped_qids)), "Duplicate question_ids in output"
    assert set(deduped_qids) == orig_qids, "Missing question_ids in output"

    # Coverage check
    orig_statements = Counter(row["statement_id"] for row in original)
    deduped_statements = Counter(row["statement_id"] for row in deduped)
    assert set(deduped_statements.keys()) == set(orig_statements.keys()), "Missing statement_ids in output"

    print("All verification checks passed.")
    print("\nPer-statement_id counts (deduped):")
    for sid, count in sorted(deduped_statements.items()):
        orig_count = orig_statements[sid]
        print(f"  {sid:45s}  {count:4d}  (was {orig_count})")


def upload(deduped: list[dict]) -> None:
    """Write deduped rows to a single gzipped JSONL shard on GCS."""
    dst_path = f"{GCS_DST}/shard-00000.jsonl.gz"
    print(f"\nWriting {len(deduped)} rows to {dst_path}...")

    with tempfile.NamedTemporaryFile(suffix=".jsonl.gz", delete=False) as tmp:
        tmp_path = tmp.name
        with gzip.open(tmp, "wt", encoding="utf-8") as gz:
            for row in deduped:
                gz.write(json.dumps(row, ensure_ascii=True) + "\n")

    subprocess.run(
        ["gcloud", "storage", "cp", tmp_path, dst_path],
        check=True,
    )
    print(f"Uploaded to {dst_path}")


def main():
    rows = read_all_shards()

    orig_qids = {row["question_id"] for row in rows}
    print(f"Unique question_ids: {len(orig_qids)}")
    print(f"Rows per question_id: {len(rows) / len(orig_qids):.1f} avg")

    deduped = dedupe(rows)
    print(f"\nAfter dedup: {len(deduped)} rows (was {len(rows)})")

    verify(rows, deduped)
    upload(deduped)

    print(f"\nDone. {len(rows)} -> {len(deduped)} rows ({len(rows) / len(deduped):.1f}x reduction)")


if __name__ == "__main__":
    main()
