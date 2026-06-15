# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Flag `3`-slice docs that share an LSH bucket with any `4plus` doc (band-level dedup).

A robust, single-pass alternative to the full connected-components dedup: a `3`
doc is "contaminated" iff it shares >=1 of its 284x71 MinHash band-buckets with
some `4plus` doc (band threshold ~0.31-0.35, the conservative reading of a ~0.4
cutoff). This is a 1-hop semi-join — it skips the transitive closure CC computes,
but at band level A~B~4plus almost always implies A~4plus, so the miss is
negligible, and it avoids the iterative CC's per-iteration dense-bucket reduces
(the full CC was ~23 min/iteration x up to 15).

Reuses the EXISTING 284x71 MinHash for both sides (the `3`-slice shards were
already staged by crossdedup_three_vs_fourplus.py). Two shuffles:
1. group every (bucket -> docs); for buckets holding >=1 `4plus` doc, emit the
   `3` doc ids in them.
2. distinct the flagged `3` ids -> the contaminated set to drop in stage 2.

Launch (us-east5, all preemptible — the standing rule for this cluster):

    uv run iris --controller-url=http://localhost:10000 --cluster=marin job run --no-wait \
        --cpu 8 --memory 64GB --disk 50GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name three-semijoin \
        -- python scripts/analysis/crossdedup_three_semijoin.py
"""

import argparse
import json
import logging
from collections.abc import Iterator

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fray import ResourceConfig
from marin.utils import fsspec_glob
from zephyr import Dataset, ZephyrContext, counters

logger = logging.getLogger(__name__)

THREE_SLICE_MINHASH = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/three_slice_10b/minhash/outputs"
FOURP_MINHASH = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/4plus_284x71/minhash/outputs"
SLICE_ROOT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/three_slice_10b"
FLAGGED_RAW = f"{SLICE_ROOT}/contaminated_raw"
CONTAMINATED = f"{SLICE_ROOT}/contaminated_ids"
THREE_ID_SCHEMA = pa.schema([("three_id", pa.string())])


def emit_bucket_membership(path_is_three: tuple[str, bool]) -> Iterator[dict]:
    """Emit {bucket, three_id} per (doc, bucket); three_id is None for `4plus` docs."""
    path, is_three = path_is_three
    with fsspec.open(path, "rb") as f:
        for batch in pq.ParquetFile(f).iter_batches(columns=["id", "buckets"]):
            for doc_id, buckets in zip(batch.column("id"), batch.column("buckets"), strict=True):
                tid = doc_id.as_py() if is_three else None
                for bucket in buckets.as_py():
                    yield {"bucket": bucket, "three_id": tid}


def flag_three_in_fourplus_buckets(_bucket: str, items: Iterator[dict]) -> Iterator[dict]:
    """If this bucket holds >=1 `4plus` doc, emit each distinct `3` doc id in it."""
    has_fourplus = False
    three_ids: set[str] = set()
    for item in items:
        if item["three_id"] is None:
            has_fourplus = True
        else:
            three_ids.add(item["three_id"])
    if not has_fourplus or not three_ids:
        counters.increment("bucket/skipped")
        return
    counters.increment("bucket/contaminating")
    for tid in three_ids:
        yield {"three_id": tid}


def distinct_three(three_id: str, _items: Iterator[dict]) -> Iterator[dict]:
    counters.increment("contaminated/distinct")
    yield {"three_id": three_id}


def _preemptible(cpu: int, ram: str) -> ResourceConfig:
    return ResourceConfig(cpu=cpu, ram=ram, disk="10g", preemptible=True, regions=("us-east5",))


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-workers", type=int, default=512)
    args = parser.parse_args()

    slice_mh = sorted(fsspec_glob(f"{THREE_SLICE_MINHASH}/*.parquet"))
    fourp_mh = sorted(fsspec_glob(f"{FOURP_MINHASH}/*.parquet"))
    if not slice_mh:
        raise FileNotFoundError(f"no `3`-slice minhash under {THREE_SLICE_MINHASH} (run the slice staging first)")
    inputs = [(p, True) for p in slice_mh] + [(p, False) for p in fourp_mh]
    logger.info("semi-join: %d `3`-slice + %d `4plus` minhash shards", len(slice_mh), len(fourp_mh))

    flag_ctx = ZephyrContext(
        name="three-semijoin-flag",
        max_workers=args.max_workers,
        resources=_preemptible(2, "32g"),
        coordinator_resources=_preemptible(2, "16g"),
    )
    flag_ctx.execute(
        Dataset.from_list(inputs)
        .flat_map(emit_bucket_membership)
        .group_by(lambda r: r["bucket"], reducer=flag_three_in_fourplus_buckets)
        .write_parquet(
            f"{FLAGGED_RAW}/flagged-{{shard:05d}}-of-{{total:05d}}.parquet", schema=THREE_ID_SCHEMA, skip_existing=True
        )
    )

    distinct_ctx = ZephyrContext(
        name="three-semijoin-distinct",
        max_workers=args.max_workers,
        resources=_preemptible(2, "24g"),
        coordinator_resources=_preemptible(2, "16g"),
    )
    outcome = distinct_ctx.execute(
        Dataset.from_files(f"{FLAGGED_RAW}/*.parquet")
        .load_parquet()
        .group_by(lambda r: r["three_id"], reducer=distinct_three)
        .write_parquet(
            f"{CONTAMINATED}/ids-{{shard:05d}}-of-{{total:05d}}.parquet", schema=THREE_ID_SCHEMA, skip_existing=True
        )
    )

    contaminated = sum(
        pq.read_metadata(fsspec.open(p, "rb").open()).num_rows for p in fsspec_glob(f"{CONTAMINATED}/*.parquet")
    )
    manifest = {
        "three_slice_minhash": THREE_SLICE_MINHASH,
        "fourplus_minhash": FOURP_MINHASH,
        "method": "1-hop band-level semi-join (>=1 shared 284x71 bucket with a 4plus doc)",
        "contaminated_three_docs": contaminated,
        "contaminated_ids": CONTAMINATED,
        "counters": dict(outcome.counters),
    }
    with fsspec.open(f"{SLICE_ROOT}/semijoin_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("done: %d contaminated `3` docs -> %s", contaminated, CONTAMINATED)


if __name__ == "__main__":
    main()
