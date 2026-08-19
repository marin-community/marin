# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Solve every materialized fuzzy-duplicate cluster and write duplicate markers.

A thin command line over
:func:`marin.processing.classification.deduplication.cluster_verify.verify_cluster_text`,
which holds the rule, the memory guards, and the output layout. This script
chooses the worker shape and writes the artifact record, so a hand-run output
is readable by the store exactly like a step output.

    uv run iris --cluster=cw-us-east-02a job run --no-wait --priority batch \
        --cpu 4 --memory 16GB -- python experiments/datakit/scripts/fuzzy_cluster_verify.py \
            --cluster-text s3://.../user/rav/dedup/cluster_text/v1 \
            --out s3://.../user/rav/dedup/verified/v1
"""

import argparse
import logging

from fray.types import ResourceConfig
from marin.execution.artifact import write_artifact
from marin.processing.classification.deduplication.cluster_dedup import ClusterDedupParams
from marin.processing.classification.deduplication.cluster_verify import (
    COUNTER_PREFIX,
    DEFAULT_FILES_PER_TASK,
    DEFAULT_MAX_SHARD_FAILURES,
    DEFAULT_REDUCE_SHARDS,
    verify_cluster_text,
)

logger = logging.getLogger(__name__)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cluster-text", required=True, help="Root written by fuzzy_cluster_text.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--minimum-containment", type=float, default=0.60)
    parser.add_argument("--probe-ngrams", type=int, default=32)
    parser.add_argument("--maximum-candidates", type=int, default=32)
    parser.add_argument("--exact-scan-maximum", type=int, default=256)
    parser.add_argument("--limit-files", type=int, default=0, help="Solve only the first N text files")
    parser.add_argument(
        "--files-per-task", type=int, default=DEFAULT_FILES_PER_TASK, help="Grouped text files solved by one map task"
    )
    parser.add_argument("--reduce-shards", type=int, default=DEFAULT_REDUCE_SHARDS, help="Reduce tasks")
    parser.add_argument(
        "--max-shard-failures",
        type=int,
        default=DEFAULT_MAX_SHARD_FAILURES,
        help="Attempts one shard gets before the pipeline aborts",
    )
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--worker-cpu", type=float, default=32)
    parser.add_argument("--worker-ram", default="192g")
    parser.add_argument("--worker-disk", default="256g")
    parser.add_argument("--task-cpu", type=float, default=1)
    parser.add_argument("--task-ram", default="12g")
    parser.add_argument("--task-disk", default="48g")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    params = ClusterDedupParams(
        minimum_containment=args.minimum_containment,
        probe_ngrams=args.probe_ngrams,
        maximum_candidates=args.maximum_candidates,
        exact_scan_maximum=args.exact_scan_maximum,
    )
    task = ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk=args.task_disk)
    result = verify_cluster_text(
        cluster_text=args.cluster_text,
        output_path=args.out,
        params=params,
        max_workers=args.max_workers,
        worker_resources=ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk),
        map_task_resources=task,
        reduce_task_resources=task,
        text_file_limit=args.limit_files or None,
        files_per_task=args.files_per_task,
        reduce_shards=args.reduce_shards,
        max_shard_failures=args.max_shard_failures,
    )
    write_artifact(result, args.out)
    logger.info("Wrote %s duplicate markers to %s", result.counters[f"{COUNTER_PREFIX}/markers"], args.out)


if __name__ == "__main__":
    main()
