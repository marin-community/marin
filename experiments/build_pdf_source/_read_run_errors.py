# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""TEMPORARY -- tally the per-document ``error`` strings a comparison or control run wrote.

DELETE alongside the other ``_``-prefixed modules here. Nothing in the pipeline imports this.

Both harnesses record a failure on the document's row rather than raising, so the exact messages
are already sitting in Parquet. They are hard to get at any other way: worker warnings do not
survive to the launching CLI's stdout, and the log server caps a ``job logs`` response well below
the volume these runs produce -- which is how a run that failed ~46% of its extractions was briefly
mistaken for a clean one.

Runs on the cluster because that is where the bucket credentials are::

    uv run iris --cluster=marin job run --target-cluster cw-us-east-02a \\
        --job-name read-run-errors \\
        -- python -m experiments.build_pdf_source._read_run_errors
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
# The runs to read. Each is a step output directory holding ``outputs/texts/*.parquet``.
TEXT_DIRS = {
    "control_fp32_vs_fp32": f"{_BUCKET}/layout_control_variance_e180c40a/outputs/texts",
}
# The row label distinguishing the two runs within a harness ("arm" for the control, "backend" for
# the comparison). Read whichever the shard actually carries.
_LABEL_COLUMNS = ("arm", "backend")
_RESOURCES = ResourceConfig(cpu=2, ram="8g", disk="8g")


class ErrorReport(BaseModel):
    version: str = "v1"
    rows: dict[str, int]
    errors_by_label: dict[str, int]
    messages: dict[str, int]
    samples: list[str]


def _label_column(shard_columns: list[str]) -> str:
    for candidate in _LABEL_COLUMNS:
        if candidate in shard_columns:
            return candidate
    raise ValueError(f"No label column among {_LABEL_COLUMNS} in {shard_columns}")


def read_errors(output_path: str) -> ErrorReport:
    """Tally the distinct error messages each run recorded."""
    rows: Counter = Counter()
    errors_by_label: Counter = Counter()
    messages: Counter = Counter()
    samples: list[str] = []

    for name, texts_dir in TEXT_DIRS.items():
        filesystem, path = url_to_fs(texts_dir)
        shards = sorted(filesystem.glob(f"{path}/*.parquet"))
        logger.info("%s: %d shards under %s", name, len(shards), texts_dir)
        for shard in shards:
            with filesystem.open(shard, "rb") as stream:
                schema = pq.read_schema(stream)
                stream.seek(0)
                label = _label_column(schema.names)
                table = pq.read_table(stream, columns=[label, "error"])
            for row in table.to_pylist():
                rows[f"{name}:{row[label]}"] += 1
                if not row["error"]:
                    continue
                errors_by_label[f"{name}:{row[label]}"] += 1
                # Keep the exception type and the head of the message; the tail is often a path.
                messages[row["error"][:200]] += 1
                if len(samples) < 8 and row["error"] not in samples:
                    samples.append(row["error"])

    logger.info("=== ROWS PER RUN/LABEL ===")
    for key, count in sorted(rows.items()):
        logger.info("  %-40s %6d rows, %6d errored", key, count, errors_by_label.get(key, 0))
    logger.info("=== DISTINCT ERROR MESSAGES ===")
    for message, count in messages.most_common(20):
        logger.info("  %6d  %s", count, message)
    logger.info("=== FULL SAMPLES ===")
    for sample in samples:
        logger.info("  --- %s", sample[:1500])

    return ErrorReport(
        rows=dict(rows),
        errors_by_label=dict(errors_by_label),
        messages=dict(messages),
        samples=samples,
    )


def read_errors_step() -> StepSpec:
    return StepSpec(
        name="data/datakit/validate/run_error_tally",
        deps=[],
        hash_attrs={"text_dirs": sorted(TEXT_DIRS.values()), "attempt": 1},
        fn=remote(partial(read_errors), resources=_RESOURCES, pip_dependency_groups=["datakit"]),
    )


def main() -> None:
    configure_logging(logging.INFO)
    StepRunner().run([read_errors_step()])


if __name__ == "__main__":
    main()
