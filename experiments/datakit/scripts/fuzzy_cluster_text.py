# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Materialize candidate text with the production cluster split policy."""

import argparse
import logging

from fray.types import ResourceConfig
from marin.datakit.copartitioned import CopartitionedSource
from marin.execution.artifact import read_artifact, write_artifact
from marin.processing.classification.deduplication.cluster_text import ClusterTextParams, resolve_data_path
from marin.processing.classification.deduplication.fuzzy_dups import FuzzyDupsAttrData
from marin.processing.classification.deduplication.large_clusters import LargeClusterPlan
from marin.processing.classification.deduplication.materialize_cluster_text import (
    DEFAULT_MAX_SHARD_FAILURES,
    materialize_cluster_text,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--large-clusters", required=True, help="Artifact root from fuzzy_large_clusters.py")
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-cluster-size", type=int, default=ClusterTextParams().max_cluster_size)
    parser.add_argument("--output-shards", type=int, default=ClusterTextParams().output_shards, help="Reduce tasks")
    parser.add_argument(
        "--groups-per-shard",
        type=int,
        default=ClusterTextParams().groups_per_shard,
        help="Output files assigned to each reduce task",
    )
    parser.add_argument("--shards-per-task", type=int, default=8, help="Input shards joined by one map task")
    parser.add_argument("--max-workers", type=int, default=64)
    parser.add_argument("--worker-cpu", type=float, default=32)
    parser.add_argument("--worker-ram", default="128g")
    parser.add_argument("--worker-disk", default="512g")
    parser.add_argument("--task-cpu", type=float, default=1)
    parser.add_argument("--task-ram", default="12g", help="Map task memory")
    parser.add_argument("--task-disk", default="48g")
    parser.add_argument("--reduce-task-ram", default="26g", help="Reduce holds a partition and its sort spill")
    parser.add_argument("--max-shard-failures", type=int, default=DEFAULT_MAX_SHARD_FAILURES)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    candidates = read_artifact(resolve_data_path(args.prefix, args.candidates), FuzzyDupsAttrData)
    result = materialize_cluster_text(
        prefix=args.prefix,
        candidates=args.candidates,
        normalized_sources=[
            CopartitionedSource(source_key=key, input_dir=resolve_data_path(args.prefix, key))
            for key in sorted(candidates.sources)
        ],
        plan=read_artifact(resolve_data_path(args.prefix, args.large_clusters), LargeClusterPlan),
        output_path=args.out,
        params=ClusterTextParams(
            max_cluster_size=args.max_cluster_size,
            output_shards=args.output_shards,
            groups_per_shard=args.groups_per_shard,
        ),
        shards_per_task=args.shards_per_task,
        max_workers=args.max_workers,
        worker_resources=ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk),
        map_task_resources=ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk=args.task_disk),
        reduce_task_resources=ResourceConfig(cpu=args.task_cpu, ram=args.reduce_task_ram, disk=args.task_disk),
        max_shard_failures=args.max_shard_failures,
    )
    write_artifact(result, args.out)


if __name__ == "__main__":
    main()
