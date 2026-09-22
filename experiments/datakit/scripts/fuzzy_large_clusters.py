# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plan oversized fuzzy clusters before text materialization."""

import argparse
import logging

from fray.types import ResourceConfig
from marin.execution.artifact import write_artifact
from marin.processing.classification.deduplication.large_clusters import LargeClusterParams, plan_large_clusters


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--stride", type=int, default=LargeClusterParams().stride)
    parser.add_argument("--minimum-size", type=int, default=LargeClusterParams().minimum_size)
    parser.add_argument("--shards-per-task", type=int, default=64)
    parser.add_argument("--max-workers", type=int, default=48)
    parser.add_argument("--worker-cpu", type=float, default=16)
    parser.add_argument("--worker-ram", default="128g")
    parser.add_argument("--worker-disk", default="64g")
    parser.add_argument("--task-cpu", type=float, default=1)
    parser.add_argument("--task-ram", default="7g")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO)
    result = plan_large_clusters(
        prefix=args.prefix,
        candidates=args.candidates,
        output_path=args.out,
        params=LargeClusterParams(stride=args.stride, minimum_size=args.minimum_size),
        shards_per_task=args.shards_per_task,
        max_workers=args.max_workers,
        worker_resources=ResourceConfig(cpu=args.worker_cpu, ram=args.worker_ram, disk=args.worker_disk),
        task_resources=ResourceConfig(cpu=args.task_cpu, ram=args.task_ram, disk="32g"),
    )
    write_artifact(result, args.out)


if __name__ == "__main__":
    main()
