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

Example (the "full" preset from the `ab-test-zephyr` skill, sized to match
`datakit_nemotron_ferry.py`'s data volume and worker pool):

    python -m experiments.datakit.zephyr_benchmark \
        --sources all --run-tag zephyr-100b-v1 \
        --pool-workers 512 --pool-cpu 16 --pool-ram 160g --pool-disk 32g \
        --first-stage exact --last-stage fuzzy \
        --max-concurrent 8 --dedup-max-parallelism 1000

The "light" preset uses ``--source-fraction`` instead of ``--sources`` to
auto-select a byte-budgeted subset of sources, favoring shard-dense sources so
the reduced ``--pool-workers`` still has enough parquet shards to fill:

    python -m experiments.datakit.zephyr_benchmark \
        --source-fraction 0.1 --run-tag zephyr-100b-light-v1 \
        --pool-workers 128 --pool-cpu 16 --pool-ram 160g --pool-disk 32g \
        --first-stage exact --last-stage fuzzy \
        --max-concurrent 4 --dedup-max-parallelism 500
"""

import argparse
import logging
from collections import defaultdict
from dataclasses import replace
from enum import StrEnum

from fray.types import ResourceConfig
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from rigging.filesystem.cluster_config import marin_temp_bucket
from rigging.filesystem.factory import url_to_fs
from rigging.filesystem.storage_path import StoragePath, prefix_join
from rigging.log_setup import configure_logging

from experiments.datakit.reference_pipeline import (
    SMOKE_SCALE,
    PoolConfig,
    ZephyrDatakitSteps,
    sample_sources,
    zephyr_datakit_steps,
)

logger = logging.getLogger(__name__)

BENCHMARK_OUTPUT_TTL_DAYS = 7
BENCHMARK_OUTPUT_PREFIX = "zephyr-benchmark"
GCP_BENCHMARK_SAMPLE_PREFIX = "gs://marin-eu-west4/datakit/sample_100b_8ae7a94f"
COREWEAVE_BENCHMARK_SAMPLE_PREFIX = "s3://marin-us-east-02a/marin/datakit/sample_100b_8ae7a94f"


class BenchmarkStage(StrEnum):
    """Stage boundary for a benchmark run."""

    EXACT = "exact"
    TOKENIZE = "tokenize"
    MINHASH = "minhash"
    FUZZY = "fuzzy"


_STAGE_ORDER = tuple(BenchmarkStage)


def _steps_between(
    steps: ZephyrDatakitSteps,
    first_stage: BenchmarkStage,
    last_stage: BenchmarkStage,
) -> list[StepSpec]:
    first_index = _STAGE_ORDER.index(first_stage)
    last_index = _STAGE_ORDER.index(last_stage)
    if first_index > last_index:
        raise ValueError(f"first stage {first_stage} follows last stage {last_stage}")

    stage_steps = {
        BenchmarkStage.EXACT: [steps.exact_dedup],
        BenchmarkStage.TOKENIZE: list(steps.tokenize.values()),
        BenchmarkStage.MINHASH: list(steps.minhash.values()),
        BenchmarkStage.FUZZY: [steps.fuzzy_dedup],
    }
    return [step for stage in _STAGE_ORDER[first_index : last_index + 1] for step in stage_steps[stage]]


def _source_shard_stats(sample_prefix: str) -> dict[str, tuple[int, int]]:
    """Map each source under ``sample_prefix`` to its ``(total_bytes, shard_count)``.

    Reads object metadata only (no shard contents), grouping every
    ``<source>/outputs/main/*.parquet`` file by its source name. Parquet found
    anywhere else in the sample tree means an unexpected layout, so it's an error
    rather than a bogus source entry.
    """
    root = StoragePath(sample_prefix.rstrip("/"))
    if root.scheme not in ("gs", "s3") or not root.bucket:
        raise ValueError(f"sample prefix must be a gs:// or s3:// URL: {sample_prefix}")
    fs, root_key = url_to_fs(str(root))
    stats: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for path, info in fs.find(root_key, detail=True).items():
        if not path.endswith(".parquet"):
            continue
        relative = StoragePath(f"{root.scheme}://{path}").relative_to(root)
        source_name, marker, _ = relative.partition("/outputs/main/")
        if not marker or not source_name:
            raise ValueError(
                f"{path} is not under '<source>/outputs/main/'; the sample tree holds "
                "parquet shards only inside each source's outputs/main directory"
            )
        stats[source_name][0] += info["size"]
        stats[source_name][1] += 1
    if not stats:
        raise FileNotFoundError(f"no parquet shards found under {sample_prefix}")
    return {name: (total_bytes, shard_count) for name, (total_bytes, shard_count) in stats.items()}


def _select_source_fraction(stats: dict[str, tuple[int, int]], fraction: float) -> list[str]:
    """Select sources totaling roughly ``fraction`` of the sample's bytes.

    Orders sources by average shard size, ascending, so the selection favors
    shard-dense sources: that keeps the resulting shard count as high as
    possible for the chosen byte budget, which matters because
    ``compute_minhash_attrs`` and ``tokenize_attributes_step`` schedule one
    Zephyr task per existing parquet file rather than splitting large ones.
    """
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"--source-fraction must be in (0.0, 1.0]; got {fraction}")
    target_bytes = sum(total_bytes for total_bytes, _ in stats.values()) * fraction
    ordered = sorted(stats, key=lambda name: stats[name][0] / stats[name][1])
    selected: list[str] = []
    selected_bytes = 0
    for name in ordered:
        if selected_bytes >= target_bytes:
            break
        selected.append(name)
        selected_bytes += stats[name][0]
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

    selected_names = selected_sources if selected_sources is not None else sorted(stats)
    unknown = sorted(set(selected_names) - set(stats))
    if unknown:
        raise KeyError(f"sources {unknown} not found under {sample_prefix}; known: {sorted(stats)}")
    total_shards = sum(stats[name][1] for name in selected_names)
    total_bytes = sum(stats[name][0] for name in selected_names)
    if total_shards < pool_workers:
        raise ValueError(
            f"--pool-workers {pool_workers} exceeds the {total_shards} parquet shards available across "
            f"{len(selected_names)} selected sources ({total_bytes / 1e9:.1f} GB). compute_minhash_attrs and "
            "tokenize_attributes_step schedule one task per shard and do not split files, so excess workers "
            "would sit idle. Lower --pool-workers, raise --source-fraction, or add sources."
        )
    logger.info(
        "zephyr_benchmark: %d sources, %.1f GB, %d parquet shards (%.1fx --pool-workers %d)",
        len(selected_names),
        total_bytes / 1e9,
        total_shards,
        total_shards / pool_workers,
        pool_workers,
    )
    return selected_sources


def _route_outputs(steps: ZephyrDatakitSteps, output_prefix: str) -> ZephyrDatakitSteps:
    tokenize = {name: replace(step, output_path_prefix=output_prefix) for name, step in steps.tokenize.items()}
    minhash = {name: replace(step, output_path_prefix=output_prefix) for name, step in steps.minhash.items()}
    return ZephyrDatakitSteps(
        exact_dedup=replace(steps.exact_dedup, output_path_prefix=output_prefix),
        tokenize=tokenize,
        minhash=minhash,
        fuzzy_dedup=replace(steps.fuzzy_dedup, output_path_prefix=output_prefix, deps=list(minhash.values())),
    )


def main() -> None:
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
    parser.add_argument("--pool-cpu", required=True, type=float)
    parser.add_argument("--pool-ram", required=True)
    parser.add_argument("--pool-disk", required=True)
    parser.add_argument("--first-stage", required=True, type=BenchmarkStage, choices=list(BenchmarkStage))
    parser.add_argument("--last-stage", required=True, type=BenchmarkStage, choices=list(BenchmarkStage))
    parser.add_argument("--max-concurrent", required=True, type=int)
    parser.add_argument("--dedup-max-parallelism", required=True, type=int)
    args = parser.parse_args()

    configure_logging(logging.INFO)
    selected_sources = _resolve_sources(args.sample_prefix, args.sources, args.source_fraction, args.pool_workers)
    sources = sample_sources(args.sample_prefix, selected_sources, args.run_tag)
    worker = ResourceConfig(cpu=args.pool_cpu, ram=args.pool_ram, disk=args.pool_disk)
    scale = replace(
        SMOKE_SCALE,
        pool=PoolConfig(n_workers=args.pool_workers, worker=worker),
        dedup_max_parallelism=args.dedup_max_parallelism,
    )
    output_prefix = marin_temp_bucket(
        ttl_days=BENCHMARK_OUTPUT_TTL_DAYS,
        prefix=prefix_join(prefix_join(BENCHMARK_OUTPUT_PREFIX, args.run_tag), "outputs"),
        source_prefix=args.sample_prefix,
    )
    steps = _route_outputs(zephyr_datakit_steps(sources, scale), output_prefix)
    StepRunner().run(
        _steps_between(steps, args.first_stage, args.last_stage),
        max_concurrent=args.max_concurrent,
    )


if __name__ == "__main__":
    main()
