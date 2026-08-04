# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Generate and benchmark local synthetic telemetry through Finelog.

The harness never contacts Iris, GCS, or another remote store. Process-cold
means a new Python process, embedded server, and DataFusion metadata cache; it
does not drop the host page cache.

Run from ``lib/finelog``:

    uv run --group dev python -m finelog.benchmarks.telemetry_layout_bench \
        baseline --rows 100000 --work-dir /tmp/finelog-telemetry-baseline \
        --output /tmp/finelog-telemetry-baseline.json
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from finelog.benchmarks.layout_candidates import (
    run_baseline,
    run_catalog_benchmark,
    run_layout_candidates,
    run_manifest_candidate,
    run_nonempty_binding_benchmark,
    run_partition_projection,
    run_rollup_candidates,
)
from finelog.benchmarks.query_measurement import WorkerMode, worker_measure
from finelog.benchmarks.telemetry_workload_corpus import (
    DEFAULT_BATCH_ROWS,
    DEFAULT_COLD_ITERATIONS,
    DEFAULT_DURATION_DAYS,
    DEFAULT_SEGMENTS,
    DEFAULT_WARM_ITERATIONS,
    DEFAULT_WARMUP_ITERATIONS,
    DatasetSpec,
    WorkloadName,
)


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError("must be positive")
    return value


def _nonnegative_int(raw: str) -> int:
    value = int(raw)
    if value < 0:
        raise argparse.ArgumentTypeError("must be non-negative")
    return value


def _add_measurement_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cold-iterations",
        type=_positive_int,
        default=DEFAULT_COLD_ITERATIONS,
    )
    parser.add_argument(
        "--warm-iterations",
        type=_positive_int,
        default=DEFAULT_WARM_ITERATIONS,
    )
    parser.add_argument(
        "--warmup-iterations",
        type=_positive_int,
        default=DEFAULT_WARMUP_ITERATIONS,
    )
    parser.add_argument(
        "--concurrency",
        type=_positive_int,
        nargs="+",
        default=[1, 4, 8],
    )
    parser.add_argument(
        "--concurrency-queries-per-worker",
        type=_positive_int,
        default=2,
    )


def _add_dataset_arguments(
    parser: argparse.ArgumentParser,
    *,
    segments_required: bool,
) -> None:
    parser.add_argument("--rows", type=_positive_int, required=True)
    parser.add_argument(
        "--duration-days",
        type=_positive_int,
        default=DEFAULT_DURATION_DAYS,
    )
    parser.add_argument(
        "--batch-rows",
        type=_positive_int,
        default=DEFAULT_BATCH_ROWS,
    )
    parser.add_argument(
        "--segments",
        type=_positive_int,
        required=segments_required,
        default=None if segments_required else DEFAULT_SEGMENTS,
    )
    parser.add_argument(
        "--distinct-run-ids",
        type=_positive_int,
        default=100_000,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    baseline = subparsers.add_parser(
        "baseline",
        help="generate and measure the current Telltale layout",
    )
    _add_dataset_arguments(baseline, segments_required=False)
    _add_measurement_arguments(baseline)
    baseline.add_argument("--decoy-namespaces", type=_nonnegative_int, default=0)
    baseline.add_argument("--work-dir", type=Path, required=True)
    baseline.add_argument("--output", type=Path, required=True)

    catalog = subparsers.add_parser(
        "catalog",
        help="measure bounded prefix lookup over a generated run catalog",
    )
    _add_measurement_arguments(catalog)
    catalog.add_argument("--run-ids", type=_positive_int, default=100_000)
    catalog.add_argument("--work-dir", type=Path, required=True)
    catalog.add_argument("--output", type=Path, required=True)

    manifest = subparsers.add_parser(
        "manifest",
        help="measure referenced-only binding and manifest pruning",
    )
    _add_dataset_arguments(manifest, segments_required=True)
    _add_measurement_arguments(manifest)
    manifest.add_argument("--decoy-namespaces", type=_nonnegative_int, default=0)
    manifest.add_argument("--source-log-dir", type=Path, required=True)
    manifest.add_argument("--work-dir", type=Path, required=True)
    manifest.add_argument("--output", type=Path, required=True)

    binding = subparsers.add_parser(
        "binding",
        help="compare referenced-only binding with nonempty unrelated namespaces",
    )
    _add_dataset_arguments(binding, segments_required=True)
    _add_measurement_arguments(binding)
    binding.add_argument("--source-log-dir", type=Path, required=True)
    binding.add_argument(
        "--decoy-counts",
        type=_positive_int,
        nargs="+",
        default=[32, 128],
    )
    binding.add_argument("--work-dir", type=Path, required=True)
    binding.add_argument("--output", type=Path, required=True)

    layouts = subparsers.add_parser(
        "layouts",
        help="measure split, clustered, and hidden-partition prototypes",
    )
    _add_dataset_arguments(layouts, segments_required=True)
    _add_measurement_arguments(layouts)
    layouts.add_argument("--source-log-dir", type=Path, required=True)
    layouts.add_argument("--split-files-per-group", type=_positive_int, required=True)
    layouts.add_argument("--partition-buckets", type=_positive_int, default=4)
    layouts.add_argument("--work-dir", type=Path, required=True)
    layouts.add_argument("--output", type=Path, required=True)

    projection = subparsers.add_parser(
        "partition-projection",
        help="compute exact nonempty day/hash partitions and selection cost",
    )
    _add_dataset_arguments(projection, segments_required=True)
    projection.add_argument("--source-log-dir", type=Path, required=True)
    projection.add_argument(
        "--bucket-counts",
        type=_positive_int,
        nargs="+",
        default=[64, 128],
    )
    projection.add_argument("--output", type=Path, required=True)

    rollups = subparsers.add_parser(
        "rollups",
        help="compare materializer and per-segment local rollup proxies",
    )
    _add_dataset_arguments(rollups, segments_required=True)
    rollups.add_argument("--source-log-dir", type=Path, required=True)
    rollups.add_argument(
        "--resolutions",
        choices=("minute", "hour"),
        nargs="+",
        default=["minute", "hour"],
    )
    rollups.add_argument("--work-dir", type=Path, required=True)
    rollups.add_argument("--output", type=Path, required=True)

    worker = subparsers.add_parser("_worker", help=argparse.SUPPRESS)
    _add_dataset_arguments(worker, segments_required=False)
    worker.add_argument("--decoy-namespaces", type=_nonnegative_int, default=0)
    worker.add_argument("--log-dir", type=Path, required=True)
    worker.add_argument("--workload", type=WorkloadName, required=True)
    worker.add_argument("--mode", type=WorkerMode, required=True)
    worker.add_argument("--sql")
    return parser


def _dataset_spec(args: argparse.Namespace) -> DatasetSpec:
    return DatasetSpec(
        rows=cast(int, args.rows),
        duration_days=cast(int, args.duration_days),
        batch_rows=cast(int, args.batch_rows),
        segments=cast(int, args.segments),
        decoy_namespaces=cast(int, getattr(args, "decoy_namespaces", 0)),
        distinct_run_ids=cast(int, args.distinct_run_ids),
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "catalog":
        payload = run_catalog_benchmark(
            args.work_dir,
            args.output,
            run_ids=args.run_ids,
            cold_iterations=args.cold_iterations,
            warm_iterations=args.warm_iterations,
            warmup_iterations=args.warmup_iterations,
            concurrency_levels=args.concurrency,
            concurrency_queries_per_worker=args.concurrency_queries_per_worker,
        )
        print(json.dumps({"output": str(args.output), "dataset": payload["dataset"]}))
        return 0
    spec = _dataset_spec(args)
    if args.command == "partition-projection":
        payload = run_partition_projection(
            args.source_log_dir,
            args.output,
            spec,
            args.bucket_counts,
        )
        print(json.dumps({"output": str(args.output), "source": payload["source"]}))
        return 0
    if args.command == "_worker":
        print(
            json.dumps(
                worker_measure(
                    args.log_dir,
                    args.workload,
                    spec,
                    args.mode,
                    args.sql,
                )
            )
        )
        return 0
    if args.command == "manifest":
        payload = run_manifest_candidate(
            args.source_log_dir,
            args.work_dir,
            args.output,
            spec,
            cold_iterations=args.cold_iterations,
            warm_iterations=args.warm_iterations,
            warmup_iterations=args.warmup_iterations,
            concurrency_levels=args.concurrency,
            concurrency_queries_per_worker=args.concurrency_queries_per_worker,
        )
        candidates = cast(dict[str, object], payload["candidates"])
        print(json.dumps({"output": str(args.output), "candidates": sorted(candidates)}))
        return 0
    if args.command == "binding":
        payload = run_nonempty_binding_benchmark(
            args.source_log_dir,
            args.work_dir,
            args.output,
            spec,
            decoy_counts=args.decoy_counts,
            cold_iterations=args.cold_iterations,
            warm_iterations=args.warm_iterations,
            warmup_iterations=args.warmup_iterations,
            concurrency_levels=args.concurrency,
            concurrency_queries_per_worker=args.concurrency_queries_per_worker,
        )
        variants = cast(dict[str, object], payload["variants"])
        print(json.dumps({"output": str(args.output), "variants": sorted(variants)}))
        return 0
    if args.command == "layouts":
        payload = run_layout_candidates(
            args.source_log_dir,
            args.work_dir,
            args.output,
            spec,
            split_files_per_group=args.split_files_per_group,
            partition_buckets=args.partition_buckets,
            cold_iterations=args.cold_iterations,
            warm_iterations=args.warm_iterations,
            warmup_iterations=args.warmup_iterations,
            concurrency_levels=args.concurrency,
            concurrency_queries_per_worker=args.concurrency_queries_per_worker,
        )
        candidates = cast(dict[str, object], payload["candidates"])
        print(json.dumps({"output": str(args.output), "candidates": sorted(candidates)}))
        return 0
    if args.command == "rollups":
        payload = run_rollup_candidates(
            args.source_log_dir,
            args.work_dir,
            args.output,
            spec,
            args.resolutions,
        )
        candidate = cast(dict[str, object], payload["candidate"])
        print(json.dumps({"output": str(args.output), "candidate": candidate["name"]}))
        return 0
    payload = run_baseline(
        args.work_dir,
        args.output,
        spec,
        cold_iterations=args.cold_iterations,
        warm_iterations=args.warm_iterations,
        warmup_iterations=args.warmup_iterations,
        concurrency_levels=args.concurrency,
        concurrency_queries_per_worker=args.concurrency_queries_per_worker,
    )
    print(json.dumps({"output": str(args.output), "preparation": payload["preparation"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
