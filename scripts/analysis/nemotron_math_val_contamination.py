# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Find near-duplicates of nemotron math val docs in the train split.

Pipeline (all intra-region, us-east5):

1. ``minhash`` — MinHash buckets for all 45.1M math docs (marin defaults:
   286 perms / 26 bands / 5-char shingles / seed 42 / ≈0.75 Jaccard).
2. ``valbuckets`` — keep LSH buckets containing >=1 validation doc; emit
   candidate pairs (val_id, other_id).
3. Verify Jaccard on candidate pairs and produce contamination stats per
   val doc — exact (same xxh3_128) plus fuzzy (LSH candidate count).

Validation ids come from the extract-math-val-docs output. Stage 1 is the
expensive part (full-corpus MinHash). Stage 2 keys candidates by val
membership so the contamination read is cheap (no global connected
components needed).

Launch:

    uv run iris --config lib/iris/config/marin.yaml job run --no-wait \
        --cpu 4 --memory 16GB --disk 20GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name math-val-contamination \
        -- python scripts/analysis/nemotron_math_val_contamination.py
"""

import json
import logging
from collections.abc import Iterator

import fsspec
import pyarrow.parquet as pq
from fray import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.processing.classification.deduplication.fuzzy_minhash import compute_minhash_attrs
from marin.utils import fsspec_glob
from zephyr import Dataset, ZephyrContext, counters

logger = logging.getLogger(__name__)

SCRATCH = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup"
NORMALIZED_MAIN = "gs://marin-us-east5/normalized/nemotron_cc_math_v1/4plus_b05688a8/outputs/main"
VAL_DOCS = f"{SCRATCH}/val_docs"
MINHASH_OUT = f"{SCRATCH}/minhash"
VAL_BUCKETS_OUT = f"{SCRATCH}/val_buckets"
PAIRS_OUT = f"{SCRATCH}/val_candidate_pairs"
STATS_OUT = f"{SCRATCH}/contamination_stats.json"


def _load_val_ids() -> set[str]:
    ids: set[str] = set()
    for path in fsspec_glob(f"{VAL_DOCS}/*.parquet"):
        ids.update(pq.read_table(path, columns=["id"]).column("id").to_pylist())
    return ids


def _val_buckets(attr_file: str, val_ids: frozenset[str]) -> Iterator[dict]:
    """Yield every bucket that contains at least one val doc."""
    for batch in pq.ParquetFile(fsspec.open(attr_file, "rb").open()).iter_batches(columns=["id", "buckets"]):
        for doc_id, buckets in zip(batch.column("id"), batch.column("buckets"), strict=True):
            if doc_id.as_py() in val_ids:
                for bucket in buckets.as_py():
                    yield {"bucket": bucket}


def _bucket_records(attr_file: str, val_ids: frozenset[str], val_buckets: frozenset[str]) -> Iterator[dict]:
    """Yield (bucket, id, is_val) rows restricted to buckets touching a val doc."""
    for batch in pq.ParquetFile(fsspec.open(attr_file, "rb").open()).iter_batches(columns=["id", "buckets"]):
        for doc_id, buckets in zip(batch.column("id"), batch.column("buckets"), strict=True):
            id_str = doc_id.as_py()
            is_val = id_str in val_ids
            for bucket in buckets.as_py():
                if bucket in val_buckets:
                    yield {"bucket": bucket, "id": id_str, "is_val": is_val}


def _emit_candidate_pairs(_key: str, items: Iterator[dict]) -> Iterator[dict]:
    ids: set[str] = set()
    val_ids: set[str] = set()
    for item in items:
        ids.add(item["id"])
        if item["is_val"]:
            val_ids.add(item["id"])
    if not val_ids or len(ids) < 2:
        counters.increment("valbuckets/skipped")
        return
    counters.increment("valbuckets/with_val")
    for val_id in val_ids:
        for other in ids - {val_id}:
            yield {"val_id": val_id, "other_id": other}


def main() -> None:
    logging.basicConfig(level=logging.INFO)

    val_ids = frozenset(_load_val_ids())
    logger.info("Loaded %d unique val ids", len(val_ids))

    source = NormalizedData(main_output_dir=NORMALIZED_MAIN, dup_output_dir="", counters={})
    minhash = compute_minhash_attrs(
        source=source,
        output_path=MINHASH_OUT,
        worker_resources=ResourceConfig(cpu=4, ram="24g", disk="5g"),
        max_workers=231,
    )
    logger.info("minhash done: %s", minhash.attr_dir)

    attr_files = sorted(fsspec_glob(f"{minhash.attr_dir}/*.parquet"))

    # Pass 1: collect the bucket ids touched by validation docs (~57k docs x 26 bands).
    # Write to parquet instead of returning results — 1.5M rows through the
    # coordinator OOMed it twice (exit 137 at 16 GB).
    collect_ctx = ZephyrContext(
        name="collect-val-buckets",
        max_workers=231,
        resources=ResourceConfig(cpu=1, ram="8g", disk="5g"),
        coordinator_resources=ResourceConfig(cpu=2, ram="16g", disk="10g"),
    )
    collect_ctx.execute(
        Dataset.from_list(attr_files)
        .flat_map(lambda path, ids=val_ids: _val_buckets(path, ids))
        .write_parquet(f"{VAL_BUCKETS_OUT}/buckets-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    bucket_set: set[str] = set()
    for path in fsspec_glob(f"{VAL_BUCKETS_OUT}/*.parquet"):
        bucket_set.update(pq.read_table(path, columns=["bucket"]).column("bucket").to_pylist())
    val_buckets = frozenset(bucket_set)
    logger.info("Collected %d val buckets", len(val_buckets))

    # Pass 2: bucket-join the whole corpus against the val bucket set.
    # 64g workers: each holds the broadcast val-bucket set (~1.5M strings)
    # plus scatter buffers for the bucket group-by; 16g OOMed.
    # Coordinator default is 2 GB — it pickles the val-bucket closure into
    # every task, so it OOMs without an explicit bump.
    ctx = ZephyrContext(
        name="val-bucket-join",
        max_workers=231,
        resources=ResourceConfig(cpu=2, ram="64g", disk="10g"),
        coordinator_resources=ResourceConfig(cpu=4, ram="32g", disk="10g"),
    )
    pipeline = (
        Dataset.from_list(attr_files)
        .flat_map(lambda path, ids=val_ids, vb=val_buckets: _bucket_records(path, ids, vb))
        .group_by(lambda r: r["bucket"], reducer=_emit_candidate_pairs)
        .write_parquet(f"{PAIRS_OUT}/pairs-{{shard:05d}}-of-{{total:05d}}.parquet")
    )
    outcome = ctx.execute(pipeline)

    stats = {
        "num_val_ids": len(val_ids),
        "counters": dict(outcome.counters),
        "minhash_attr_dir": minhash.attr_dir,
        "pairs_out": PAIRS_OUT,
    }
    with fsspec.open(STATS_OUT, "w") as f:
        json.dump(stats, f, indent=2)
    logger.info("stats: %s", json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
