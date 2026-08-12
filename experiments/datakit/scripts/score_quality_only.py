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

This entry point opens one shared pool and runs every source through it, scoring
many sources at the same time so a one-file source does not hold the pool on its
own. Shards run in subprocesses: under the in-process runner a worker's shards are
threads sharing one GIL, which left 265 of 267 threads idle on a 115-CPU worker and
ran ~18x slower for the same scores.

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
from concurrent.futures import ThreadPoolExecutor

from fray.cluster import ResourceConfig
from marin.datakit.normalize import NormalizedData
from marin.execution.artifact import read_artifact
from marin.execution.step_spec import StepSpec
from rigging.filesystem.s3_compat import configure_coreweave_s3
from rigging.log_setup import configure_logging
from zephyr.execution import ZephyrContext
from zephyr.runners import InlineRunner, SubprocessRunner

from experiments.datakit.cluster.quality.fast_transformer.score import TASK_RESOURCES, score_normalized
from experiments.datakit.reference_pipeline import select_sources

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
# Names the pool's actors: zephyr appends its own segments, giving
# ``zephyr-score-<uuid>-coordinator-<pool_id>``. Kept short and free of the stage
# name so a pool is identifiable in the cluster without reading like a step.
POOL_NAME = "score"
# Iris counts a preemption against the worker gang's cumulative failure budget, and
# this job runs at batch priority precisely so it yields to interactive and
# production work -- so it is evicted by design, repeatedly, over a run measured in
# hours. Zephyr's default of 10 is a budget for a short job: a 4-worker smoke spent
# it in nine minutes and the gang was killed with "Job exceeded max_task_failures".
# Scoring resumes per file, so an evicted shard costs its own work and nothing else.
WORKER_MAX_TASK_RETRIES = 5_000
# Sources scored at the same time on the shared pool. Zephyr's default of 16 is a
# limit for a driver running a few pipelines; this one runs 292, most of them a
# single file, so the pool sits idle unless many are in flight together. The pool
# rejects a pipeline past its limit rather than queueing it, so the driver's
# fan-out reads this same number off the context.
MAX_CONCURRENT_PIPELINES = 30
# The coordinator tracks 32 pipelines at once instead of zephyr's default handful,
# so it gets more than the default 0.1 CPU / 1 GB. It stays non-preemptible: losing
# it loses every in-flight pipeline, while losing a worker costs one shard.
COORDINATOR_RESOURCES = ResourceConfig(cpu=2, ram="3g", preemptible=False)


def coordinator_max_concurrency(max_workers: int) -> int:
    """Concurrent calls the coordinator must serve for this pool to make progress.

    ``run_pipeline`` blocks for the whole life of its pipeline, so every running
    pipeline holds one slot permanently and only the remainder serve workers. At
    zephyr's default of 100 with 30 pipelines, 70 slots were left for 64 workers
    polling twice a second: workers queued behind the pipelines, their completions
    timed out, and shards retried forever with nothing in flight. Two slots per
    worker covers a poll overlapping a completion report.
    """
    return MAX_CONCURRENT_PIPELINES + 2 * max_workers + 32


# ``InlineRunner`` runs a worker's shards as threads of one process, so they share
# that process's GIL; ``SubprocessRunner`` gives each shard its own process and pays
# a JAX import and model load per shard. Measured on agenttrove (43 shards, 781k rows,
# 4 workers): inline managed 8 shards in 20 minutes, subprocess did all 43 in about 6
# -- roughly 18x -- for bit-identical scores. A thread dump explains it: a worker
# holding 115 CPUs had 265 of 267 threads idle, because only the tokenizer and numpy
# calls release the GIL and the Python between them does not. Subprocess is therefore
# the default; inline stays available for a stage whose work is mostly native.
STAGE_RUNNERS = {"inline": InlineRunner, "subprocess": SubprocessRunner}
DEFAULT_STAGE_RUNNER = "subprocess"


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


def quality_step(name: str, normalize_step: StepSpec, model_version: str, prefix: str | None) -> StepSpec:
    """The reference pipeline's quality step for one source, built but not run.

    Constructed rather than hand-rolled so the output path is the production one:
    ``StepSpec`` hashes the step name, its ``hash_attrs`` and its dependency
    hashes into the directory, and the model version is one of those attrs — which
    is exactly what keeps two scorers from colliding in the store. Hand-building a
    path would land beside production output with nothing distinguishing it.

    ``fn`` is omitted because this step is never executed here; only its
    ``output_path`` is read. The work runs in-process on the shared pool instead of
    as one remote step per source.
    """
    return StepSpec(
        name=f"datakit/quality/{name}",
        deps=[normalize_step],
        hash_attrs={"model_version": model_version, "v": 1},
        output_path_prefix=prefix,
    )


def score_all(
    *,
    quality_model: str,
    quality_model_version: str,
    output_prefix: str | None = None,
    names: list[str] | None = None,
    max_workers: int = DEFAULT_MAX_WORKERS,
    fraction: float = WORKER_FRACTION,
    stage_runner: str = DEFAULT_STAGE_RUNNER,
) -> dict[str, dict[str, int | float]]:
    """Score every source through one shared pool; returns each source's counters."""
    sources = select_sources(names)
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
    logger.info("score_quality_only: stage runner = %s", stage_runner)

    written: dict[str, dict[str, int | float]] = {}
    with ZephyrContext(
        name=POOL_NAME,
        resources=resources,
        max_workers=max_workers,
        stage_runner_factory=STAGE_RUNNERS[stage_runner],
        worker_max_task_retries=WORKER_MAX_TASK_RETRIES,
        max_concurrent_pipelines=MAX_CONCURRENT_PIPELINES,
        coordinator_resources=COORDINATOR_RESOURCES,
        coordinator_max_concurrency=coordinator_max_concurrency(max_workers),
    ) as ctx:

        def score_one(item: tuple[str, StepSpec]) -> tuple[str, dict[str, int | float] | None]:
            name, normalize_step = item
            out = quality_step(name, normalize_step, quality_model_version, output_prefix).output_path
            # One source failing must not take the other 291 with it: the pool is
            # shared, and a raised exception here would tear it down mid-run.
            try:
                scores = score_normalized(
                    output_path=out,
                    normalized=read_artifact(normalize_step.output_path, NormalizedData),
                    source=name,
                    model_dir=quality_model,
                    ctx=ctx,
                )
            except Exception:
                logger.exception("score_quality_only: %s failed, continuing", name)
                return name, None
            logger.info("score_quality_only: %s -> %s", name, out)
            return name, dict(scores.counters)

        # A source is one pipeline, and one pipeline is capped by its own largest
        # input file -- so running them one at a time leaves the pool almost idle:
        # the median source holds a single file, which occupies one of ~12,000 task
        # slots while the other sources wait. Fan them out instead. The bound is the
        # pool's own limit because a pipeline past it is rejected, not queued.
        lanes = min(len(sources), ctx.max_concurrent_pipelines)
        logger.info("score_quality_only: %d sources over %d concurrent pipelines", len(sources), lanes)
        with ThreadPoolExecutor(max_workers=lanes, thread_name_prefix="score-source") as pool:
            for i, (name, counters) in enumerate(pool.map(score_one, sorted(sources.items())), start=1):
                if counters is not None:
                    written[name] = counters
                logger.info("score_quality_only: [%d/%d] done %s", i, len(sources), name)
    return written


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-prefix",
        default=None,
        help="override the production prefix (default: MARIN_PREFIX, as the pipeline uses)",
    )
    parser.add_argument("--quality-model", required=True, help="scorer + calibration + type classifier dir")
    parser.add_argument("--quality-model-version", required=True, help="identity tag for the scorer")
    parser.add_argument("--sources", help="comma-separated source names (default: every source found)")
    parser.add_argument("--max-workers", type=int, default=DEFAULT_MAX_WORKERS)
    parser.add_argument(
        "--stage-runner",
        choices=sorted(STAGE_RUNNERS),
        default=DEFAULT_STAGE_RUNNER,
        help="inline: shards are threads of one worker process, sharing a cached model. "
        "subprocess: one process per shard, isolated but reloading the model each time.",
    )
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
        output_prefix=args.output_prefix,
        quality_model=args.quality_model,
        quality_model_version=args.quality_model_version,
        names=names,
        max_workers=args.max_workers,
        fraction=args.worker_fraction,
        stage_runner=args.stage_runner,
    )
    rows = sum(int(c.get("rows_written", 0)) for c in written.values())
    logger.info("score_quality_only: scored %d sources, %d rows written", len(written), rows)


if __name__ == "__main__":
    main()
