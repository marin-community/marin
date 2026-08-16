# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check whether a tokenization holds exactly one row per document, everywhere.

The store joins the dense attribute tables against the tokenize stream by
position, and uses ``chunk_index`` to tell "one document across three rows" from
"three documents". A leaf written before #8100 has no such column, so the store
refuses it outright rather than risk a silent misalignment after the first
oversized document.

That refusal is right for a leaf that *might* be chunked and cannot say so. It is
unnecessary for one that provably is not, and this decides which case a
tokenization is in: a document spanning several rows makes its shard hold more
rows than the source has documents, so comparing each shard's row count against
the decontamination side settles it. Sampling cannot -- a single chunked shard
anywhere in 162,535 is exactly the thing that hides.

Footer counts only, batched so that one task covers hundreds of shards rather
than one. Submit in the CoreWeave data region::

    uv run iris --cluster=marin job run --no-wait \\
        --target-cluster cw-us-east-02a --priority interactive \\
        --cpu 2 --memory 8g --enable-extra-resources \\
        -e MARIN_PREFIX s3://marin-us-east-02a/marin \\
        -- python -m experiments.datakit.scripts.audit_tokenize_rows
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import TypedDict

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.datakit.decon import DeconAttributes
from marin.execution.artifact import read_artifact
from marin.processing.tokenize.attributes import TokenizedAttrData
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.filesystem.storage_path import StoragePath
from rigging.log_setup import configure_logging
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

from experiments.datakit import hero_data
from experiments.datakit.reference_pipeline import SPLIT

logger = logging.getLogger(__name__)

SHARDS_PER_TASK = 400
READ_THREADS = 16


class _Shard(TypedDict):
    source: str
    tokenize: str
    decontam: str


def _rows(path: str) -> int:
    with StoragePath(path).open("rb") as handle:
        return pq.ParquetFile(handle).metadata.num_rows


def compare_batch(batch: list[_Shard]) -> list[dict]:
    """Return one entry per shard whose row counts disagree."""

    def one(shard: _Shard) -> dict | None:
        tokenize_rows = _rows(shard["tokenize"])
        decontam_rows = _rows(shard["decontam"])
        if tokenize_rows == decontam_rows:
            return None
        return {
            "source": shard["source"],
            "shard": shard["tokenize"].rsplit("/", 1)[-1],
            "tokenize_rows": tokenize_rows,
            "decontam_rows": decontam_rows,
        }

    with ThreadPoolExecutor(max_workers=READ_THREADS) as pool:
        return [row for row in pool.map(one, batch) if row]


def build_shards() -> list[_Shard]:
    """Pair every tokenize shard with the decontamination shard of the same name."""

    def per_source(source: str) -> list[_Shard]:
        tokenize = read_artifact(hero_data.tokenized(source).output_path, TokenizedAttrData)
        decontam = read_artifact(hero_data.decontam(source).output_path, DeconAttributes)
        tokenize_dir = tokenize.output_dirs[SPLIT].rstrip("/")
        decontam_dir = decontam.main_output_dir.rstrip("/")
        bases = sorted(str(m).rsplit("/", 1)[-1] for m in StoragePath(tokenize_dir + "/*.parquet").glob())
        return [
            {"source": source, "tokenize": f"{tokenize_dir}/{base}", "decontam": f"{decontam_dir}/{base}"}
            for base in bases
        ]

    with ThreadPoolExecutor(max_workers=32) as pool:
        return [shard for shards in pool.map(per_source, hero_data.source_names()) for shard in shards]


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--worker-cpu", type=float, default=4)
    parser.add_argument("--worker-ram", default="16g")
    args = parser.parse_args(argv)

    configure_logging(logging.INFO)
    configure_coreweave_s3()

    shards = build_shards()
    batches = [shards[i : i + SHARDS_PER_TASK] for i in range(0, len(shards), SHARDS_PER_TASK)]
    logger.info(
        "%d shards over %d sources -> %d tasks",
        len(shards),
        len({s["source"] for s in shards}),
        len(batches),
    )

    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk="8g")
    context = ZephyrContext(
        name="audit-tokenize-rows",
        resources=worker,
        max_workers=min(args.max_workers, len(batches)),
    )
    outcome = context.execute(Dataset.from_list(batches).map(compare_batch), verbose=True, map_task_resources=worker)

    mismatched = [row for result in outcome.results if result for row in result]
    by_source: dict[str, int] = {}
    for row in mismatched:
        by_source[row["source"]] = by_source.get(row["source"], 0) + 1

    logger.info("shards compared: %d", len(shards))
    logger.info("shards where tokenize rows != decontam rows: %d", len(mismatched))
    for source, count in sorted(by_source.items(), key=lambda kv: -kv[1]):
        rows = [r for r in mismatched if r["source"] == source]
        extra = sum(r["tokenize_rows"] - r["decontam_rows"] for r in rows)
        logger.info("  %-44s %d shards, %+d rows, e.g. %s", source, count, extra, rows[0]["shard"])
    if not mismatched:
        logger.info("every shard holds exactly one row per document; this leaf needs no chunk_index")


if __name__ == "__main__":
    main()
