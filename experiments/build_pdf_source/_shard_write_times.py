# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- check whether a run's failed rows were actually written by an earlier run.

DELETE once the answer is recorded in ``.agents/ops/``. Nothing in the pipeline imports this.

The control harness writes with ``skip_existing=True`` into a directory keyed by the step hash, so
a killed run and its relaunch share one output prefix and the relaunch never rewrites a shard the
first attempt already produced. If ``/muchanem/control-layout-variance`` (killed 17:03) left half
the shards behind, ``/muchanem/control-layout-variance-2`` (17:11) inherited them and the merged
directory reads as a 50% failure rate that no pod in the second run ever produced.

This lists every shard under the run's ``outputs/texts`` with its object timestamp and its error
tally, so the failures can be matched against when the shard was written.

Runs on the cluster because that is where the bucket credentials are::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name shard-write-times \\
        -- python -m experiments.build_pdf_source._shard_write_times
"""

import logging
from collections import Counter
from functools import partial

import pyarrow.parquet as pq
from fray.types import ResourceConfig
from marin.execution.remote import remote
from marin.execution.step_runner import StepRunner
from marin.execution.step_spec import StepSpec
from pydantic import BaseModel
from rigging.filesystem import url_to_fs
from rigging.log_setup import configure_logging

logger = logging.getLogger(__name__)

_BUCKET = "s3://marin-us-east-02a/marin/data/datakit/validate"
TEXT_DIRS = {
    "control_fp32_vs_fp32": f"{_BUCKET}/layout_control_variance_e180c40a/outputs/texts",
    "compare_int8_vs_fp32": f"{_BUCKET}/layout_backend_text_agreement_cca9a89b/outputs/texts",
}
_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="8g")


class ShardReport(BaseModel):
    """When each shard was written and how many of its rows carry an error."""

    version: str = "v1"
    shards: dict[str, str]
    errors_by_minute: dict[str, int]
    rows_by_minute: dict[str, int]


def shard_write_times(output_path: str) -> ShardReport:
    """Pair each shard's write time with its error count."""
    shards: dict[str, str] = {}
    errors_by_minute: Counter = Counter()
    rows_by_minute: Counter = Counter()

    for name, texts_dir in TEXT_DIRS.items():
        filesystem, path = url_to_fs(texts_dir)
        for shard in sorted(filesystem.glob(f"{path}/*.parquet")):
            info = filesystem.info(shard)
            written = str(info.get("LastModified") or info.get("mtime") or "unknown")
            with filesystem.open(shard, "rb") as stream:
                table = pq.read_table(stream, columns=["error"])
            rows = table.num_rows
            errors = sum(1 for value in table.column("error").to_pylist() if value)
            minute = f"{name} {written[:16]}"
            shards[f"{name}:{shard.rsplit('/', 1)[-1]}"] = f"{written} rows={rows} errors={errors}"
            errors_by_minute[minute] += errors
            rows_by_minute[minute] += rows

    logger.info("=== SHARD WRITE TIMES ===")
    for key, value in sorted(shards.items()):
        logger.info("  %-60s %s", key, value)
    logger.info("=== ROWS AND ERRORS BY WRITE MINUTE ===")
    for minute in sorted(rows_by_minute):
        logger.info("  %-40s rows=%6d errors=%6d", minute, rows_by_minute[minute], errors_by_minute[minute])

    return ShardReport(
        shards=shards,
        errors_by_minute=dict(errors_by_minute),
        rows_by_minute=dict(rows_by_minute),
    )


def shard_write_times_step() -> StepSpec:
    return StepSpec(
        name="data/datakit/validate/shard_write_times",
        deps=[],
        hash_attrs={"text_dirs": sorted(TEXT_DIRS.values()), "attempt": 1},
        fn=remote(partial(shard_write_times), resources=_RESOURCES, pip_dependency_groups=["datakit"]),
    )


def main() -> None:
    configure_logging(logging.INFO)
    StepRunner().run([shard_write_times_step()])


if __name__ == "__main__":
    main()
