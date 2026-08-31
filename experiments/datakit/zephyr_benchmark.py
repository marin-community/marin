# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Run the Zephyr stages of the Datakit reference pipeline on GCS or S3.

The benchmark starts from an existing normalized Datakit sample, so its measured
pipeline does not include Hugging Face corpus download time. It runs global exact
deduplication, per-source tokenization and MinHash, then cross-source fuzzy dedup.
Every generated stage output is routed under a seven-day temporary prefix keyed by
the required run tag. ``--sample-prefix`` defaults to the 100B GCS sample in
``europe-west4``; select the us-central1 GCS sample for a us-central1 run or
the equivalent S3 path to run on CoreWeave.

Example (the full preset from the `ab-test-zephyr` skill):

    python -m experiments.datakit.zephyr_benchmark \
        --sources all --run-tag zephyr-100b-v1 \
        --pool-workers 48 \
        --target all --max-concurrent 128 \
        --dedup-max-parallelism 1000 --dedup-cc-max-iterations 3

The "light" preset uses ``--source-fraction`` instead of ``--sources`` to
auto-select a byte-budgeted subset of sources, favoring shard-dense sources so
the smaller workload still has enough parquet shards to fill the pool:

    python -m experiments.datakit.zephyr_benchmark \
        --source-fraction 0.1 --run-tag zephyr-100b-light-v1 \
        --pool-workers 48 \
        --target all --max-concurrent 80 \
        --dedup-max-parallelism 500 --dedup-cc-max-iterations 3

Use ``--target map`` to run tokenization and MinHash only. A shuffle-only run
uses ``--target shuffle``; the benchmark verifies that every sample-owned
MinHash input exists before starting the worker pool, then runs exact and fuzzy
dedup with a fresh run tag.
"""

import argparse
import logging
from collections import defaultdict
from dataclasses import replace
from enum import StrEnum
from typing import NamedTuple

from fray.types import ResourceConfig
from marin.execution.step_runner import StepRunner, step_is_built
from marin.execution.step_spec import StepSpec
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging
from zephyr.context import ZephyrContext

from experiments.datakit.materialize_zephyr_benchmark_sample import (
    BENCHMARK_SAMPLE_INPUTS_DIR,
    GCP_BENCHMARK_SAMPLE_PREFIX,
    benchmark_datakit_steps,
    benchmark_sample_fuzzy_steps,
    benchmark_zephyr_context,
)
from experiments.datakit.reference_pipeline import (
    DEFAULT_SCALE,
    SMOKE_SCALE,
    SOURCE_DISCOVERY_DEPTHS,
    PipelineScale,
    ZephyrDatakitSteps,
    sample_sources,
)

logger = logging.getLogger(__name__)

BENCHMARK_OUTPUT_TTL_DAYS = 7
BENCHMARK_OUTPUT_PREFIX = "zephyr-benchmark"
DECIMAL_GB_BYTES = 1_000_000_000
MISSING_ARTIFACT_PREVIEW_LIMIT = 10


class BenchmarkTarget(StrEnum):
    """Datakit work selected for one benchmark run."""

    ALL = "all"
    MAP = "map"
    SHUFFLE = "shuffle"
    EXACT = "exact"
    TOKENIZE = "tokenize"
    MINHASH = "minhash"
    FUZZY = "fuzzy"


def _target_steps(steps: ZephyrDatakitSteps, target: BenchmarkTarget) -> list[StepSpec]:
    stage_steps: dict[BenchmarkTarget, list[StepSpec]] = {
        BenchmarkTarget.EXACT: [steps.exact_dedup],
        BenchmarkTarget.TOKENIZE: list(steps.tokenize.values()),
        BenchmarkTarget.MINHASH: list(steps.minhash.values()),
        BenchmarkTarget.FUZZY: [steps.fuzzy_dedup],
    }
    if target is BenchmarkTarget.MAP:
        return [*stage_steps[BenchmarkTarget.TOKENIZE], *stage_steps[BenchmarkTarget.MINHASH]]
    if target is BenchmarkTarget.SHUFFLE:
        return [*stage_steps[BenchmarkTarget.EXACT], *stage_steps[BenchmarkTarget.FUZZY]]
    if target is BenchmarkTarget.ALL:
        return [
            *stage_steps[BenchmarkTarget.EXACT],
            *stage_steps[BenchmarkTarget.TOKENIZE],
            *stage_steps[BenchmarkTarget.MINHASH],
            *stage_steps[BenchmarkTarget.FUZZY],
        ]
    return stage_steps[target]


class SourceShardStats(NamedTuple):
    """Aggregate parquet-shard metadata for one source under a benchmark sample."""

    total_bytes: int
    shard_count: int


def _source_shard_stats(sample_prefix: str) -> dict[str, SourceShardStats]:
    """Return shard counts and byte sizes, rejecting unsupported sample layouts."""
    root = StoragePath(sample_prefix)
    if root.scheme not in ("gs", "s3") or not root.bucket:
        raise ValueError(f"sample prefix must be a gs:// or s3:// URL: {sample_prefix}")
    fs, root_key = url_to_fs(str(root))
    max_segments = len(SOURCE_DISCOVERY_DEPTHS)
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path, info in fs.find(root_key, detail=True).items():
        if not path.endswith(".parquet"):
            continue
        relative = StoragePath(f"{root.scheme}://{path}").relative_to(root)
        if relative.startswith(f"{BENCHMARK_SAMPLE_INPUTS_DIR}/"):
            continue
        source_name, marker, _ = relative.partition("/outputs/main/")
        if not marker or not source_name:
            raise ValueError(
                f"{path} is not under '<source>/outputs/main/'; the sample tree holds "
                "parquet shards only inside each source's outputs/main directory"
            )
        if source_name.count("/") + 1 > max_segments:
            raise ValueError(
                f"source {source_name!r} is deeper than the {max_segments} segments "
                "sample_sources() discovers; widen SOURCE_DISCOVERY_DEPTHS in "
                "reference_pipeline.py before adding a source this deep"
            )
        stats[source_name][0] += info["size"]
        stats[source_name][1] += 1
    if not stats:
        raise FileNotFoundError(f"no parquet shards found under {sample_prefix}")
    return {name: SourceShardStats(total_bytes, shard_count) for name, (total_bytes, shard_count) in stats.items()}


def _select_source_fraction(stats: dict[str, SourceShardStats], fraction: float) -> list[str]:
    """Select whole sources meeting a byte fraction while favoring shard density."""
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"--source-fraction must be in (0.0, 1.0]; got {fraction}")
    target_bytes = sum(s.total_bytes for s in stats.values()) * fraction
    ordered = sorted(stats, key=lambda name: stats[name].total_bytes / stats[name].shard_count)
    selected: list[str] = []
    selected_bytes = 0
    for name in ordered:
        if selected_bytes >= target_bytes:
            break
        selected.append(name)
        selected_bytes += stats[name].total_bytes
    return selected


def _resolve_sources(
    sample_prefix: str,
    sources_arg: str | None,
    source_fraction: float | None,
    pool_workers: int,
) -> list[str] | None:
    """Resolve ``--sources``/``--source-fraction`` into a name list (``None`` for "all").

    Enforces that ``pool_workers`` doesn't exceed the selected sources' parquet
    shard count, logging the resolved workload size on success.
    """
    stats = _source_shard_stats(sample_prefix)
    if source_fraction is not None:
        selected_sources = _select_source_fraction(stats, source_fraction)
    else:
        selected_sources = None if sources_arg == "all" else [name.strip() for name in sources_arg.split(",")]

    # Dedup before summing: sample_sources() collapses a repeated name into one
    # source, so counting it twice here would overstate coverage and let an
    # oversized --pool-workers pass the guard below.
    selected_names = sorted(set(selected_sources)) if selected_sources is not None else sorted(stats)
    unknown = sorted(set(selected_names) - set(stats))
    if unknown:
        raise KeyError(f"sources {unknown} not found under {sample_prefix}; known: {sorted(stats)}")
    total_shards = sum(stats[name].shard_count for name in selected_names)
    total_bytes = sum(stats[name].total_bytes for name in selected_names)
    if total_shards < pool_workers:
        raise ValueError(
            f"--pool-workers {pool_workers} exceeds the {total_shards} parquet shards available across "
            f"{len(selected_names)} selected sources ({total_bytes / DECIMAL_GB_BYTES:.1f} GB). "
            "compute_minhash_attrs and "
            "tokenize_attributes_step schedule one task per shard and do not split files, so excess workers "
            "would sit idle. Lower --pool-workers, raise --source-fraction, or add sources."
        )
    logger.info(
        "zephyr_benchmark: %d sources, %.1f GB, %d parquet shards (%.1fx --pool-workers %d)",
        len(selected_names),
        total_bytes / DECIMAL_GB_BYTES,
        total_shards,
        total_shards / pool_workers,
        pool_workers,
    )
    return selected_sources


def _benchmark_output_prefix(sample_prefix: str, run_tag: str) -> str:
    return marin_temp_bucket(
        ttl_days=BENCHMARK_OUTPUT_TTL_DAYS,
        prefix=prefix_join(prefix_join(BENCHMARK_OUTPUT_PREFIX, run_tag), "outputs"),
        source_prefix=sample_prefix,
    )


def _benchmark_steps(
    *,
    sample_prefix: str,
    selected_sources: list[str] | None,
    run_tag: str,
    target: BenchmarkTarget,
    scale: PipelineScale,
    zephyr_context: ZephyrContext,
) -> ZephyrDatakitSteps:
    output_prefix = _benchmark_output_prefix(sample_prefix, run_tag)
    sources = sample_sources(sample_prefix, selected_sources, run_tag)
    steps = benchmark_datakit_steps(sources, scale, zephyr_context, output_prefix)

    if target not in (BenchmarkTarget.SHUFFLE, BenchmarkTarget.FUZZY):
        return steps

    input_sources = sample_sources(sample_prefix, selected_sources)
    input_steps = benchmark_sample_fuzzy_steps(
        sample_prefix,
        input_sources,
        scale,
        zephyr_context,
    )
    missing = [name for name, step in input_steps.minhash.items() if not step_is_built(step)]
    if missing:
        shown = ", ".join(missing[:MISSING_ARTIFACT_PREVIEW_LIMIT])
        remainder = (
            f" and {len(missing) - MISSING_ARTIFACT_PREVIEW_LIMIT} more"
            if len(missing) > MISSING_ARTIFACT_PREVIEW_LIMIT
            else ""
        )
        raise RuntimeError(
            f"benchmark sample {sample_prefix!r} is missing MinHash artifacts for {shown}{remainder}; "
            "run experiments.datakit.materialize_zephyr_benchmark_sample in minhash mode"
        )

    return replace(
        steps,
        fuzzy_dedup=replace(input_steps.fuzzy_dedup, output_path_prefix=output_prefix),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sample-prefix",
        default=GCP_BENCHMARK_SAMPLE_PREFIX,
        help=f"Pre-normalized sample root (default: {GCP_BENCHMARK_SAMPLE_PREFIX}).",
    )
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--sources", help="Comma-separated source names or 'all'.")
    source_group.add_argument(
        "--source-fraction",
        type=float,
        help=(
            "Auto-select sources totaling this fraction of the sample's bytes (0.0, 1.0], "
            "favoring shard-dense sources so --pool-workers stays covered by real parquet "
            "shards. Mutually exclusive with --sources."
        ),
    )
    parser.add_argument("--run-tag", required=True, help="Fresh identity tag that forces uncached benchmark stages.")
    parser.add_argument("--pool-workers", required=True, type=int)
    parser.add_argument("--pool-cpu", type=float, default=DEFAULT_SCALE.pool.worker.cpu)
    parser.add_argument("--pool-ram", default=DEFAULT_SCALE.pool.worker.ram)
    parser.add_argument("--pool-disk", default=DEFAULT_SCALE.pool.worker.disk)
    parser.add_argument("--map-task-cpu", type=float, help="Override map-task CPU; omitted uses the whole worker.")
    parser.add_argument("--map-task-ram", help="Override map-task RAM; omitted uses the whole worker.")
    parser.add_argument("--map-task-disk", help="Override map-task disk; omitted uses the whole worker.")
    parser.add_argument("--reduce-task-cpu", type=float, help="Override reduce-task CPU; omitted uses the whole worker.")
    parser.add_argument("--reduce-task-ram", help="Override reduce-task RAM; omitted uses the whole worker.")
    parser.add_argument("--reduce-task-disk", help="Override reduce-task disk; omitted uses the whole worker.")
    parser.add_argument("--target", required=True, type=BenchmarkTarget, choices=list(BenchmarkTarget))
    parser.add_argument("--max-concurrent", required=True, type=int)
    parser.add_argument("--dedup-max-parallelism", required=True, type=int)
    parser.add_argument(
        "--dedup-cc-max-iterations",
        required=True,
        type=int,
        help=(
            "Max connected-components rounds for fuzzy dedup. Both datakit ferries pin this "
            "to 3; each extra round is a full scatter/reduce pass over the whole bucket graph "
            "and the library default of 10 makes fuzzy dedup dominate the benchmark's wall time "
            "without adding representative signal."
        ),
    )
    return parser.parse_args()


def _task_resources(
    worker: ResourceConfig,
    cpu: float | None,
    ram: str | None,
    disk: str | None,
) -> ResourceConfig | None:
    if cpu is None and ram is None and disk is None:
        return None
    return replace(
        worker,
        cpu=worker.cpu if cpu is None else cpu,
        ram=worker.ram if ram is None else ram,
        disk=worker.disk if disk is None else disk,
    )


def _scale_from_args(args: argparse.Namespace) -> PipelineScale:
    worker = ResourceConfig(cpu=args.pool_cpu, ram=args.pool_ram, disk=args.pool_disk)
    map_task = _task_resources(worker, args.map_task_cpu, args.map_task_ram, args.map_task_disk)
    reduce_task = _task_resources(worker, args.reduce_task_cpu, args.reduce_task_ram, args.reduce_task_disk)
    return replace(
        SMOKE_SCALE,
        pool=replace(
            SMOKE_SCALE.pool,
            n_workers=args.pool_workers,
            worker=worker,
            map_task=map_task,
            reduce_task=reduce_task,
        ),
        dedup_max_parallelism=args.dedup_max_parallelism,
        cc_max_iterations=args.dedup_cc_max_iterations,
    )


def _run_benchmark(args: argparse.Namespace) -> None:
    selected_sources = _resolve_sources(args.sample_prefix, args.sources, args.source_fraction, args.pool_workers)
    scale = _scale_from_args(args)
    zephyr_context = benchmark_zephyr_context(
        "zephyr-benchmark",
        scale,
        args.max_concurrent,
    )
    steps = _benchmark_steps(
        sample_prefix=args.sample_prefix,
        selected_sources=selected_sources,
        run_tag=args.run_tag,
        target=args.target,
        scale=scale,
        zephyr_context=zephyr_context,
    )
    with zephyr_context:
        StepRunner().run(
            _target_steps(steps, args.target),
            max_concurrent=args.max_concurrent,
        )


def main() -> None:
    args = _parse_args()
    configure_logging(logging.INFO)
    _run_benchmark(args)


if __name__ == "__main__":
    main()
