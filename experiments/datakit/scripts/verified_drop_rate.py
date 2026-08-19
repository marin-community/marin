# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure exactly what share of the corpus a verified marker tree removes.

Counts rather than samples. Every earlier number in this investigation came
from a shard sample against partial trees, which forced a restriction to shards
both runs finished and left the absolute rate softer than the ratio. A complete
tree can be counted outright, and Parquet carries row counts in the footer, so
the whole corpus costs one metadata read per file rather than a scan.

Three denominators, because the same tree gives three different-looking answers
and they are all correct:

* Normalized documents: what share of the corpus the store will drop.
* Clustered members: what share of the documents the rule can act on at all,
  since a document in no cluster is never a candidate.
* Per source: where the removals land, which is what shifts the mixture.

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \
        --no-wait --priority interactive --cpu 4 --memory 16GB \
        -- python experiments/datakit/scripts/verified_drop_rate.py \
            --cluster-text s3://.../user/rav/dedup/cluster_text/v11 \
            --markers s3://.../user/rav/dedup/verified/v11-c075-restored \
            --out s3://.../user/rav/dedup/reports/drop_rate_c075.json
"""

import argparse
import collections
import json
import logging
from collections.abc import Iterator
from typing import Any

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from rigging.filesystem.storage_path import StoragePath, prefix_join
from zephyr import counters
from zephyr.dataset import Dataset
from zephyr.execution import ZephyrContext

logger = logging.getLogger(__name__)

COUNTER_PREFIX = "fuzzy/drop_rate"


def row_count(path: str) -> int | None:
    """Rows in a Parquet file, from its footer. ``None`` when the file is absent."""
    try:
        with StoragePath(path).open("rb") as handle:
            return pq.ParquetFile(handle).metadata.num_rows
    except FileNotFoundError:
        return None


def count_batch(spec: dict[str, Any]) -> Iterator[dict[str, Any]]:
    """Count documents, clustered members and markers for a batch of shards."""
    totals: collections.Counter = collections.Counter()
    per_source: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    missing_normalized = 0

    for shard in spec["shards"]:
        documents = row_count(shard["normalized_path"])
        if documents is None:
            missing_normalized += 1
            continue
        # A sparse attribute tree omits a shard with no duplicates, and a source
        # outside the candidate set has no cluster file. Absent means zero here,
        # never an error, which is why this counts rather than joins.
        clustered = row_count(shard["candidate_path"]) or 0
        markers = row_count(prefix_join(spec["markers"], f"outputs/{shard['source_tag']}/{shard['basename']}")) or 0

        totals["documents"] += documents
        totals["clustered"] += clustered
        totals["markers"] += markers
        totals["shards"] += 1
        entry = per_source[shard["source_key"]]
        entry["documents"] += documents
        entry["clustered"] += clustered
        entry["markers"] += markers

    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/shards", totals["shards"])
    counters.pipeline.update_counter(f"{COUNTER_PREFIX}/missing_normalized", missing_normalized)
    yield {
        "totals": dict(totals),
        "missing_normalized": missing_normalized,
        "per_source": {key: dict(value) for key, value in per_source.items()},
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-text", required=True, help="Root whose manifest names every normalized shard")
    parser.add_argument("--markers", required=True, help="Verified marker tree to measure")
    parser.add_argument("--out", required=True)
    parser.add_argument("--shards-per-task", type=int, default=200)
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--worker-cpu", type=float, default=8)
    parser.add_argument("--worker-ram", default="32g")
    parser.add_argument("--task-cpu", type=float, default=4)
    parser.add_argument("--task-ram", default="4g")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    manifest = json.loads(StoragePath(prefix_join(args.cluster_text, "manifest.json")).read_bytes())
    shards = manifest["shards"]
    batches = [
        {"shards": shards[start : start + args.shards_per_task], "markers": args.markers}
        for start in range(0, len(shards), args.shards_per_task)
    ]
    logger.info("Counting %d shards as %d tasks against %s", len(shards), len(batches), args.markers)

    worker = ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk="32g")
    task = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk="8g")
    context = ZephyrContext(name="verified-drop-rate", resources=worker, max_workers=args.max_workers)
    outcome = context.execute(Dataset.from_list(batches).flat_map(count_batch), verbose=True, map_task_resources=task)

    totals: collections.Counter = collections.Counter()
    per_source: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    missing = 0
    for result in outcome.results:
        if not isinstance(result, dict):
            continue
        totals.update(result["totals"])
        missing += result["missing_normalized"]
        for key, value in result["per_source"].items():
            per_source[key].update(value)

    documents, clustered, markers = totals["documents"], totals["clustered"], totals["markers"]
    ranked = sorted(per_source.items(), key=lambda kv: -kv[1]["markers"])
    payload = {
        "markers_tree": args.markers,
        "shards_counted": totals["shards"],
        "shards_missing_normalized": missing,
        "documents": documents,
        "clustered_members": clustered,
        "markers": markers,
        "drop_rate_of_corpus": markers / documents if documents else 0.0,
        "drop_rate_of_clustered": markers / clustered if clustered else 0.0,
        "clustered_share_of_corpus": clustered / documents if documents else 0.0,
        "per_source": {key: dict(value) for key, value in ranked},
    }
    StoragePath(args.out).write_bytes(json.dumps(payload, indent=1).encode())

    logger.info("shards counted %d (%d missing a normalized file)", totals["shards"], missing)
    logger.info("documents        %15d", documents)
    logger.info("clustered members%15d  (%.2f%% of corpus)", clustered, 100 * payload["clustered_share_of_corpus"])
    logger.info("markers          %15d", markers)
    logger.info(
        "drop rate: %.3f%% of the corpus, %.3f%% of clustered members",
        100 * payload["drop_rate_of_corpus"],
        100 * payload["drop_rate_of_clustered"],
    )
    for key, value in ranked[:10]:
        share = 100 * value["markers"] / value["documents"] if value["documents"] else 0
        logger.info("  %-58s %12d docs  %5.2f%% dropped", key[:58], value["documents"], share)
    logger.info("Wrote %s", args.out)


if __name__ == "__main__":
    main()
