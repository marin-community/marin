# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Score document quality across every source, on one shared CPU worker pool.

The reference pipeline runs quality scoring as one step per source, and each step
builds its own Zephyr pool. That is the right shape inside a pipeline where the
step's neighbours also need pools, and the wrong shape when quality scoring is the
only thing running: the median source holds a single input file, so a per-source
pool spends most of its wall clock starting containers, and the model cached in a
worker process is discarded with the pool that held it. A 292-source run measured
1h55m with the cluster mostly idle.

This entry point opens one shared pool and runs every source through it. The pool
outlives each source, so ``InlineRunner``'s per-process model cache is paid for
once rather than 292 times.

It asks for **CPU and memory only**. Scoring is I/O-bound — a worker sits around
25% CPU streaming parquet — so it is a good fit for the CPU on GPU nodes that
would otherwise idle, and requesting no accelerator leaves the GPUs free for the
work those nodes exist for. Submit it in the batch priority band so it yields to
interactive and production jobs rather than competing with them::

    iris --cluster=marin --target-cluster cw-us-east-08a job run \\
        --priority batch --enable-extra-resources --cpu 4 --memory 8g \\
        -- python -m experiments.datakit.scripts.score_quality_only \\
           --source-prefix  s3://.../normalized \\
           --output-prefix  s3://.../quality \\
           --quality-model  s3://.../models/pooled_glm52_v3 \\
           --quality-model-version glm52-v3

``--quality-model-version`` is the model's identity tag, hashed in place of the
region-specific model directory exactly as the reference pipeline does, so two
scorers never collide in one output tree.
"""

import argparse
import logging

from fray.cluster import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner

from experiments.datakit.cluster.quality.fast_transformer.score import TASK_RESOURCES, score_normalized
from experiments.datakit.reference_pipeline import sample_sources

logger = logging.getLogger(__name__)

# A gb200-4x node is 144 vCPU / 960 GB. A worker takes 80% of one, so a node hosts
# exactly one worker and the remaining fifth covers system pods — without that
# headroom Kueue never admits the gang.
NODE_CPU = 144
NODE_RAM_GB = 960
WORKER_FRACTION = 0.8
# One worker per node across the reserved fleet. Asking for more cannot help: the
# request is node-sized, so the 217th worker has nowhere to land.
DEFAULT_MAX_WORKERS = 216


def worker_resources(fraction: float = WORKER_FRACTION) -> ResourceConfig:
    """A node-sized CPU worker: no accelerator, so the node's GPUs stay free."""
    return ResourceConfig(cpu=int(NODE_CPU * fraction), ram=f"{int(NODE_RAM_GB * fraction)}g")


def tasks_per_worker(worker: ResourceConfig, task: ResourceConfig = TASK_RESOURCES) -> int:
    """How many shards fit on one worker, by whichever of CPU or memory binds first.

    Zephyr defaults a task's cost to the whole worker, so without an explicit task
    cost a node-sized worker runs exactly one shard and leaves the other ~113 cores
    idle. ``score_normalized`` states the cost, and this reports what that buys.
    """
    by_cpu = int(worker.cpu // task.cpu)
    by_ram = int(_gb(worker.ram) // _gb(task.ram))
    return max(1, min(by_cpu, by_ram))


def _gb(value: str | int) -> int:
    """Gigabytes from a ``ResourceConfig`` memory string such as ``768g``."""
    text = str(value).lower().rstrip("b")
    return int(float(text.rstrip("g")) * (1 if text.endswith("g") else 1 / 1024))


def score_all(
    *,
    source_prefix: str,
    output_prefix: str,
    quality_model: str,
    quality_model_version: str,
    names: list[str] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    fraction: float = WORKER_FRACTION,
) -> dict[str, dict[str, int | float]]:
    """Score every source through one shared pool; returns each source's counters."""
    sources = sample_sources(source_prefix, names)
    resources = worker_resources(fraction)
    packed = tasks_per_worker(resources)
    logger.info(
        "score_quality_only: %d sources, %d workers of cpu=%s ram=%s, "
        "%d shards per worker (task cpu=%s ram=%s) -> up to %d concurrent shards, model=%s",
        len(sources),
        max_workers,
        resources.cpu,
        resources.ram,
        packed,
        TASK_RESOURCES.cpu,
        TASK_RESOURCES.ram,
        packed * max_workers,
        quality_model_version,
    )

    written: dict[str, dict[str, int | float]] = {}
    with ZephyrContext(
        name="ft-quality-shared",
        resources=resources,
        max_workers=max_workers,
        stage_runner_factory=InlineRunner,
    ) as ctx:
        for i, (name, normalized_path) in enumerate(sorted(sources.items()), start=1):
            out = f"{output_prefix.rstrip('/')}/{name}"
            # One source failing must not take the other 291 with it: the pool is
            # shared, and a raised exception here would tear it down mid-run.
            try:
                scores = score_normalized(
                    output_path=out,
                    normalized=read_artifact(normalized_path, NormalizedData),
                    source=name,
                    model_dir=quality_model,
                    ctx=ctx,
                )
                written[name] = dict(scores.counters)
            except Exception:
                logger.exception("score_quality_only: %s failed, continuing", name)
                continue
            logger.info("score_quality_only: [%d/%d] %s -> %s", i, len(sources), name, out)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-prefix", required=True, help="normalized tree whose sources are scored")
    parser.add_argument("--output-prefix", required=True, help="root the per-source scores are written under")
    parser.add_argument("--quality-model", required=True, help="scorer + calibration + type classifier dir")
    parser.add_argument("--quality-model-version", required=True, help="identity tag for the scorer")
    parser.add_argument("--sources", help="comma-separated source names (default: every source found)")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--worker-fraction",
        type=float,
        default=WORKER_FRACTION,
        help="share of one node's CPU and RAM each worker requests",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = parse_args()
    configure_logging(logging.INFO)
    configure_coreweave_s3()
    names = [s.strip() for s in args.sources.split(",")] if args.sources else None
    written = score_all(
        source_prefix=args.source_prefix,
        output_prefix=args.output_prefix,
        quality_model=args.quality_model,
        quality_model_version=args.quality_model_version,
        names=names,
        max_workers=args.max_workers,
        fraction=args.worker_fraction,
    )
    rows = sum(int(c.get("rows_written", 0)) for c in written.values())
    logger.info("score_quality_only: scored %d sources, %d rows written", len(written), rows)


if __name__ == "__main__":
    main()
