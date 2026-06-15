# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Stage 3: materialize + tokenize the clean (deduped) `3` slice.

Filters the fixed 10B-token `3` slice (the 48 staged doc shards) by dropping the
contaminated ids found by crossdedup_three_clean.py (docs whose 286x26 cluster
contains a `4plus` doc), then tokenizes the survivors with the llama3 tokenizer
into a Levanter cache — a fresh, untrained ~8.6B-token math corpus to carve a
clean val set from.

    uv run iris --controller-url=http://localhost:10000 --cluster=marin job run --no-wait \
        --cpu 8 --memory 64GB --disk 50GB --priority interactive --extra cpu \
        --enable-extra-resources --preemptible --region us-east5 \
        --job-name three-clean-materialize \
        -- python scripts/analysis/crossdedup_three_materialize.py
"""

import argparse
import json
import logging
from collections.abc import Iterator

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq
from fray import ResourceConfig
from marin.processing.tokenize.tokenize import TokenizeConfig, tokenize
from marin.utils import fsspec_glob
from zephyr import Dataset, ZephyrContext, counters

logger = logging.getLogger(__name__)

SLICE_ROOT = "gs://marin-us-east5/scratch/ahmed/midtrain_dedup/three_slice_10b"
SLICE_DOCS = f"{SLICE_ROOT}/docs"
CONTAMINATED_IDS = f"{SLICE_ROOT}/contaminated_ids_286"
CLEAN_DOCS = f"{SLICE_ROOT}/clean_docs"
CLEAN_CACHE = "gs://marin-us-east5/tokenized/nemotron_math_3_clean_10b"
TOKENIZER = "meta-llama/Meta-Llama-3.1-8B"
CLEAN_SCHEMA = pa.schema([("id", pa.string()), ("text", pa.string())])
STATS_SCHEMA = pa.schema([("shard", pa.string()), ("kept", pa.int64()), ("dropped", pa.int64())])

_CONTAM: frozenset[str] | None = None


def contaminated_ids() -> frozenset[str]:
    """Per-worker cache of the contaminated `3` id set (loaded from GCS, not closures)."""
    global _CONTAM
    if _CONTAM is None:
        ids: set[str] = set()
        for path in fsspec_glob(f"{CONTAMINATED_IDS}/*.parquet"):
            with fsspec.open(path, "rb") as f:
                ids.update(pq.read_table(f, columns=["three_id"]).column("three_id").to_pylist())
        _CONTAM = frozenset(ids)
    return _CONTAM


def filter_shard(path: str) -> Iterator[dict]:
    """Drop contaminated docs from one slice shard; write the survivors."""
    contam = contaminated_ids()
    basename = path.rsplit("/", 1)[1]
    with fsspec.open(path, "rb") as f:
        table = pq.read_table(f, columns=["id", "text"])
    keep = [r for r in table.to_pylist() if r["id"] not in contam]
    with fsspec.open(f"{CLEAN_DOCS}/{basename}", "wb") as f:
        pq.write_table(pa.Table.from_pylist(keep, schema=CLEAN_SCHEMA), f)
    counters.increment("filter/kept", len(keep))
    counters.increment("filter/dropped", table.num_rows - len(keep))
    yield {"shard": basename, "kept": len(keep), "dropped": table.num_rows - len(keep)}


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    argparse.ArgumentParser().parse_args()

    shards = sorted(fsspec_glob(f"{SLICE_DOCS}/*.parquet"))
    logger.info("filtering %d slice shards (dropping %d contaminated ids)", len(shards), len(contaminated_ids()))
    ctx = ZephyrContext(
        name="three-clean-filter",
        max_workers=len(shards),
        resources=ResourceConfig(cpu=2, ram="16g", disk="10g", preemptible=True, regions=("us-east5",)),
        coordinator_resources=ResourceConfig(cpu=2, ram="12g", disk="10g", preemptible=True, regions=("us-east5",)),
    )
    outcome = ctx.execute(
        Dataset.from_list(shards)
        .flat_map(filter_shard)
        .write_parquet(
            f"{SLICE_ROOT}/clean_filter_stats/stats-{{shard:05d}}-of-{{total:05d}}.parquet",
            schema=STATS_SCHEMA,
            skip_existing=True,
        )
    )
    kept = outcome.counters.get("filter/kept", 0)
    dropped = outcome.counters.get("filter/dropped", 0)
    logger.info("filtered: kept %d, dropped %d -> tokenizing", kept, dropped)

    tokenize(
        TokenizeConfig(
            train_paths=[f"{CLEAN_DOCS}/*.parquet"],
            validation_paths=[],
            cache_path=CLEAN_CACHE,
            tokenizer=TOKENIZER,
        )
    )
    with fsspec.open(f"{CLEAN_CACHE}/train/.stats.json") as f:
        stats = json.load(f)
    manifest = {
        "clean_docs": CLEAN_DOCS,
        "clean_cache": CLEAN_CACHE,
        "kept_docs": kept,
        "dropped_docs": dropped,
        "cache_docs": stats.get("total_elements"),
        "cache_tokens": stats.get("total_tokens"),
        "tokenizer": TOKENIZER,
    }
    with fsspec.open(f"{SLICE_ROOT}/clean_cache_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    logger.info("done: %s", json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
