# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Zephyr pipeline that traces SWE-rebench-V2 instances under gVisor.

For each row in the HuggingFace dataset ``nebius/SWE-rebench-V2`` this
pipeline runs the row's ``test_cmd`` inside a gVisor sandbox with the
marin Python tracer enabled, and writes one JSONL row per instance with
the captured stdout/stderr/trace.

Workers run in the ``iris-task-swetrace`` image (built from
``experiments/swe_rebench_trace/Dockerfile``), which contains ``runsc``,
``skopeo``, and ``umoci``. The image is selected by passing
``ResourceConfig(image=...)`` to ``ZephyrContext`` — that override flows
through Fray → Iris's per-task ``task_image`` field down to the worker.

Usage::

    # Smoke test on a few rows
    uv run python -m experiments.swe_rebench_trace.pipeline --limit 5

    # Full run
    uv run python -m experiments.swe_rebench_trace.pipeline \\
        --output gs://marin-swe-traces/v1
"""

from __future__ import annotations

import argparse
import logging
import os

from fray import ResourceConfig
from zephyr import Dataset, ZephyrContext, load_parquet

from experiments.swe_rebench_trace.run_one import trace_swe_row

logger = logging.getLogger(__name__)


# Default location of the SWE-rebench-V2 parquet shards on the HF Hub.
# Update if the dataset layout changes.
SWE_REBENCH_PARQUET_GLOB = "hf://datasets/nebius/SWE-rebench-V2@main/data/*.parquet"

DEFAULT_OUTPUT_PREFIX = "gs://marin-swe-traces/v1"

DEFAULT_TASK_IMAGE = "ghcr.io/marin-community/iris-task-swetrace:latest"


def build_pipeline(
    *,
    output_prefix: str,
    limit: int | None,
    max_workers: int,
    task_image: str,
) -> tuple[ZephyrContext, Dataset]:
    """Construct (but do not execute) the swe_rebench_trace pipeline.

    Returns the ``ZephyrContext`` and the ``Dataset`` so callers can
    inspect the plan or invoke ``ctx.execute(pipeline)`` separately.
    """
    ctx = ZephyrContext(
        max_workers=max_workers,
        resources=ResourceConfig(
            cpu=4,
            ram="16g",
            disk="100g",
            image=task_image,
        ),
        name="swe-rebench-trace",
    )

    pipeline: Dataset = Dataset.from_files(SWE_REBENCH_PARQUET_GLOB).flat_map(load_parquet)
    if limit is not None and limit > 0:
        pipeline = pipeline.take_per_shard(limit)
    pipeline = pipeline.map(trace_swe_row).write_jsonl(
        f"{output_prefix}/traces-{{shard:05d}}.jsonl.gz",
    )
    return ctx, pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        default=os.environ.get("MARIN_SWE_TRACE_OUTPUT", DEFAULT_OUTPUT_PREFIX),
        help="Output prefix for traces (e.g. gs://bucket/path).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If set, take only this many rows per shard. Use for smoke tests.",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=int(os.environ.get("MARIN_SWE_TRACE_WORKERS", "32")),
        help="Maximum concurrent Zephyr workers (each runs one sandbox at a time).",
    )
    parser.add_argument(
        "--task-image",
        default=os.environ.get("MARIN_SWE_TRACE_IMAGE", DEFAULT_TASK_IMAGE),
        help="Custom task container image with runsc/skopeo/umoci installed.",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    ctx, pipeline = build_pipeline(
        output_prefix=args.output,
        limit=args.limit,
        max_workers=args.max_workers,
        task_image=args.task_image,
    )

    logger.info(
        "Submitting swe_rebench_trace pipeline (output=%s, limit=%s, max_workers=%d, image=%s)",
        args.output,
        args.limit,
        args.max_workers,
        args.task_image,
    )

    output_files = list(ctx.execute(pipeline))
    logger.info("Pipeline complete: wrote %d shard files", len(output_files))
    for f in output_files:
        print(f)


if __name__ == "__main__":
    main()
