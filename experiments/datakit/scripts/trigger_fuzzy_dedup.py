# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run MinHash and global fuzzy dedup for normalized Datakit sources.

The script first confirms that each selected normalize step has ``SUCCESS``
status. It then runs one MinHash step per source and one global fuzzy-dedup
step. It does not run tokenization, embedding, quality, decontamination,
exact dedup, domain assignment, reports, or store construction.

The global stage uses at most 48 CPU-only workers. Each worker requests 120
CPU, 850 GB RAM, and 1 TB disk. This shape can use idle East08 GB200 nodes,
but it does not request their GPUs or use GPU gang scheduling. Zephyr runs
53 map tasks or 4 reduce tasks on each worker. The global stage uses 2,544
input map shards for one full map wave. It keeps 1,248 group-by shards small
enough for each reducer and runs the reducers in seven waves.

Run the script through federation so its entry point and child jobs execute in
East08::

    uv run iris --cluster=marin job run --no-wait --enable-extra-resources \
        --target-cluster cw-us-east-08a --priority interactive \
        --cpu 1 --memory 3g \
        -- python experiments/datakit/scripts/trigger_fuzzy_dedup.py

Use ``--dry-run`` to confirm the source count and output paths without starting
MinHash or fuzzy-dedup jobs.
"""

import argparse
import logging
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from fray.types import ResourceConfig
from marin.execution.step_runner import StepRunner, step_is_built
from marin.execution.step_spec import StepSpec
from marin.execution.step_status import get_status_path
from marin.processing.classification.deduplication.fuzzy_dups import DEFAULT_CC_MAX_ITERATIONS
from rigging.filesystem import StoragePath
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import (
    DEFAULT_SCALE,
    FuzzyDedupSteps,
    PoolConfig,
    build_fuzzy_dedup_steps,
    select_sources,
)

logger = logging.getLogger(__name__)

STATUS_CHECK_WORKERS = 16
DEFAULT_MINHASH_MAX_WORKERS = 1024
DEFAULT_MAX_CONCURRENT = 16
DEFAULT_COORDINATOR_CPU = 1.0
DEFAULT_COORDINATOR_RAM = "3g"
DEFAULT_DEDUP_MAX_WORKERS = 48
DEFAULT_DEDUP_WORKER_CPU = 120.0
DEFAULT_DEDUP_WORKER_RAM = "850g"
DEFAULT_DEDUP_WORKER_DISK = "1t"
DEFAULT_DEDUP_MAP_TASK_CPU = 2.0
DEFAULT_DEDUP_MAP_TASK_RAM = "16g"
DEFAULT_DEDUP_MAP_TASK_DISK = "16g"
DEFAULT_DEDUP_REDUCE_TASK_CPU = 30.0
DEFAULT_DEDUP_REDUCE_TASK_RAM = "40g"
DEFAULT_DEDUP_REDUCE_TASK_DISK = "96g"
DEFAULT_DEDUP_INPUT_SHARDS = 2544
DEFAULT_DEDUP_REDUCE_SHARDS = 1248


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sources",
        default="all",
        help="Comma-separated source names, or 'all' for every registered source. Default: all.",
    )
    parser.add_argument(
        "--minhash-max-workers",
        type=int,
        default=DEFAULT_MINHASH_MAX_WORKERS,
        help=f"Maximum MinHash workers per source. Default: {DEFAULT_MINHASH_MAX_WORKERS}.",
    )
    parser.add_argument(
        "--minhash-worker-cpu",
        type=float,
        default=DEFAULT_SCALE.pool.worker.cpu,
        help=f"CPUs per MinHash worker. Default: {DEFAULT_SCALE.pool.worker.cpu}.",
    )
    parser.add_argument(
        "--minhash-worker-ram",
        default=DEFAULT_SCALE.pool.worker.ram,
        help=f"RAM per MinHash worker. Default: {DEFAULT_SCALE.pool.worker.ram}.",
    )
    parser.add_argument(
        "--minhash-worker-disk",
        default=DEFAULT_SCALE.pool.worker.disk,
        help=f"Disk per MinHash worker. Default: {DEFAULT_SCALE.pool.worker.disk}.",
    )
    parser.add_argument(
        "--coordinator-cpu",
        type=float,
        default=DEFAULT_COORDINATOR_CPU,
        help=f"CPUs for each Zephyr coordinator. Default: {DEFAULT_COORDINATOR_CPU}.",
    )
    parser.add_argument(
        "--coordinator-ram",
        default=DEFAULT_COORDINATOR_RAM,
        help=f"RAM for each Zephyr coordinator. Default: {DEFAULT_COORDINATOR_RAM}.",
    )
    parser.add_argument(
        "--dedup-input-shards",
        type=int,
        default=DEFAULT_DEDUP_INPUT_SHARDS,
        help=f"Input map shards for global fuzzy dedup. Default: {DEFAULT_DEDUP_INPUT_SHARDS}.",
    )
    parser.add_argument(
        "--dedup-output-path",
        default=None,
        help=(
            "Pin the global fuzzy-dedup output tree instead of deriving it from the step hash. "
            "Use it with a raised --cc-max-iterations to continue an existing run: connected "
            "components resumes from the completed it_N directories under this tree."
        ),
    )
    parser.add_argument(
        "--rerun-completed",
        action="store_true",
        help=(
            "Clear a SUCCESS status at the dedup output before running, so a completed step runs "
            "again in place. Required to continue a run that already reached its iteration cap."
        ),
    )
    parser.add_argument(
        "--dedup-reduce-shards",
        type=int,
        default=DEFAULT_DEDUP_REDUCE_SHARDS,
        help=f"Output shards for each global fuzzy-dedup group-by. Default: {DEFAULT_DEDUP_REDUCE_SHARDS}.",
    )
    parser.add_argument(
        "--dedup-max-workers",
        type=int,
        default=DEFAULT_DEDUP_MAX_WORKERS,
        help=f"Maximum global fuzzy-dedup workers. Default: {DEFAULT_DEDUP_MAX_WORKERS}.",
    )
    parser.add_argument(
        "--dedup-worker-cpu",
        type=float,
        default=DEFAULT_DEDUP_WORKER_CPU,
        help=f"CPUs per global fuzzy-dedup worker. Default: {DEFAULT_DEDUP_WORKER_CPU:g}.",
    )
    parser.add_argument(
        "--dedup-worker-ram",
        default=DEFAULT_DEDUP_WORKER_RAM,
        help=f"RAM per global fuzzy-dedup worker. Default: {DEFAULT_DEDUP_WORKER_RAM}.",
    )
    parser.add_argument(
        "--dedup-worker-disk",
        default=DEFAULT_DEDUP_WORKER_DISK,
        help=f"Disk per global fuzzy-dedup worker. Default: {DEFAULT_DEDUP_WORKER_DISK}.",
    )
    parser.add_argument(
        "--dedup-map-task-cpu",
        type=float,
        default=DEFAULT_DEDUP_MAP_TASK_CPU,
        help=f"CPUs per fuzzy-dedup map task. Default: {DEFAULT_DEDUP_MAP_TASK_CPU:g}.",
    )
    parser.add_argument(
        "--dedup-map-task-ram",
        default=DEFAULT_DEDUP_MAP_TASK_RAM,
        help=f"RAM budget per fuzzy-dedup map task. Default: {DEFAULT_DEDUP_MAP_TASK_RAM}.",
    )
    parser.add_argument(
        "--dedup-map-task-disk",
        default=DEFAULT_DEDUP_MAP_TASK_DISK,
        help=f"Disk budget per fuzzy-dedup map task. Default: {DEFAULT_DEDUP_MAP_TASK_DISK}.",
    )
    parser.add_argument(
        "--dedup-reduce-task-cpu",
        type=float,
        default=DEFAULT_DEDUP_REDUCE_TASK_CPU,
        help=f"CPUs per fuzzy-dedup reduce task. Default: {DEFAULT_DEDUP_REDUCE_TASK_CPU:g}.",
    )
    parser.add_argument(
        "--dedup-reduce-task-ram",
        default=DEFAULT_DEDUP_REDUCE_TASK_RAM,
        help=f"RAM budget per fuzzy-dedup reduce task. Default: {DEFAULT_DEDUP_REDUCE_TASK_RAM}.",
    )
    parser.add_argument(
        "--dedup-reduce-task-disk",
        default=DEFAULT_DEDUP_REDUCE_TASK_DISK,
        help=f"Disk budget per fuzzy-dedup reduce task. Default: {DEFAULT_DEDUP_REDUCE_TASK_DISK}.",
    )
    parser.add_argument(
        "--cc-max-iterations",
        type=int,
        default=DEFAULT_CC_MAX_ITERATIONS,
        help=f"Maximum connected-components iterations. Default: {DEFAULT_CC_MAX_ITERATIONS}.",
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=DEFAULT_MAX_CONCURRENT,
        help=f"Maximum concurrent MinHash steps. Default: {DEFAULT_MAX_CONCURRENT}.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Confirm normalized inputs and print the terminal output without starting pipeline steps.",
    )
    args = parser.parse_args(argv)
    for name in (
        "minhash_max_workers",
        "dedup_input_shards",
        "dedup_reduce_shards",
        "dedup_max_workers",
        "cc_max_iterations",
        "max_concurrent",
    ):
        if getattr(args, name) < 1:
            parser.error(f"--{name.replace('_', '-')} must be at least 1")
    for name in ("minhash_worker_cpu", "dedup_worker_cpu", "dedup_map_task_cpu", "dedup_reduce_task_cpu"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be greater than 0")
    if args.coordinator_cpu <= 0:
        parser.error("--coordinator-cpu must be greater than 0")
    return args


def _source_names(value: str) -> list[str] | None:
    if value == "all":
        return None
    return [name.strip() for name in value.split(",") if name.strip()]


def _missing_normalized_sources(sources: dict[str, StepSpec]) -> list[str]:
    items = list(sources.items())
    with ThreadPoolExecutor(max_workers=STATUS_CHECK_WORKERS) as pool:
        cached = pool.map(lambda item: step_is_built(item[1]), items)
    return [name for (name, _), is_cached in zip(items, cached, strict=True) if not is_cached]


def _build_steps(args: argparse.Namespace) -> FuzzyDedupSteps:
    sources = select_sources(_source_names(args.sources))
    missing = _missing_normalized_sources(sources)
    if missing:
        lines = "\n".join(f"  {name}" for name in missing)
        raise SystemExit(
            f"{len(missing)} normalized source(s) are not ready:\n{lines}\n"
            "Run experiments/datakit/scripts/trigger_sources.py for these sources first."
        )

    worker = replace(
        DEFAULT_SCALE.pool.worker,
        cpu=args.minhash_worker_cpu,
        ram=args.minhash_worker_ram,
        disk=args.minhash_worker_disk,
    )
    scale = replace(
        DEFAULT_SCALE,
        pool=PoolConfig(n_workers=args.minhash_max_workers, worker=worker),
        dedup_max_parallelism=args.dedup_input_shards,
    )
    coordinator_resources = ResourceConfig(
        cpu=args.coordinator_cpu,
        ram=args.coordinator_ram,
        preemptible=False,
    )
    dedup_worker_resources = ResourceConfig(
        cpu=args.dedup_worker_cpu,
        ram=args.dedup_worker_ram,
        disk=args.dedup_worker_disk,
    )
    dedup_map_task_resources = replace(
        dedup_worker_resources,
        cpu=args.dedup_map_task_cpu,
        ram=args.dedup_map_task_ram,
        disk=args.dedup_map_task_disk,
    )
    dedup_reduce_task_resources = replace(
        dedup_worker_resources,
        cpu=args.dedup_reduce_task_cpu,
        ram=args.dedup_reduce_task_ram,
        disk=args.dedup_reduce_task_disk,
    )
    return build_fuzzy_dedup_steps(
        sources,
        scale=scale,
        cc_max_iterations=args.cc_max_iterations,
        coordinator_resources=coordinator_resources,
        minhash_max_workers=args.minhash_max_workers,
        dedup_max_workers=args.dedup_max_workers,
        dedup_num_reduce_shards=args.dedup_reduce_shards,
        dedup_worker_resources=dedup_worker_resources,
        dedup_map_task_resources=dedup_map_task_resources,
        dedup_reduce_task_resources=dedup_reduce_task_resources,
        dedup_output_path=args.dedup_output_path,
    )


def _clear_completed_status(output_path: str) -> None:
    """Drop a SUCCESS status so the runner rebuilds this output in place.

    The dedup step is otherwise served from cache once its status file says
    SUCCESS, which blocks a continuation over the same output tree.
    """
    status_path = StoragePath(get_status_path(output_path))
    if not status_path.exists():
        logger.info("No status file at %s; nothing to clear.", status_path)
        return
    logger.warning("Clearing completed status %s so the dedup step runs again in place.", status_path)
    status_path.rm()


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    configure_logging(logging.INFO)

    steps = _build_steps(args)
    logger.info(
        "Fuzzy dedup graph: %d MinHash steps -> %s (%d workers, %d input shards, %d reduce shards)",
        len(steps.minhash),
        steps.dedup.output_path,
        args.dedup_max_workers,
        args.dedup_input_shards,
        args.dedup_reduce_shards,
    )
    if args.dry_run:
        print(f"{len(steps.minhash)} MinHash step(s) would run.")
        print(f"Global fuzzy-dedup output: {steps.dedup.output_path}")
        return

    if args.rerun_completed:
        _clear_completed_status(steps.dedup.output_path)

    StepRunner().run([steps.dedup], max_concurrent=args.max_concurrent)


if __name__ == "__main__":
    main()
