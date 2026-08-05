# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Backfill a run's legacy per-(sub)task sample parquets into its finestore archive.

Legacy runs stored one ``samples_<task>_<ts>.parquet`` per (sub)task (each row an ``EvalSample``) and
referenced Harbor trajectories by a ``gs://`` URI. This tool reads those files, writes the same rows
into the run's finestore ``samples`` table, and for agentic samples pulls each referenced trajectory
into the ``blobs`` table (rewriting the sample's ``trajectory_uri`` to a ``finestore://`` reference)
and flattens its steps into the ``steps`` table.

It is idempotent — samples dedupe on ``(task, doc_id, trial_id)``, so a re-run adds nothing — and it
never deletes the source parquet, jsonl, or trajectory objects, so a migration can be validated
before the legacy files are retired.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import click
import pyarrow.parquet as pq
from finestore.layout import FINESTORE_DIR
from finestore.reader import CompositeReader
from marin.evaluation.samples import (
    ARCHIVE_SAMPLES_TABLE,
    SAMPLES_PREFIX,
    SAMPLES_SUFFIX,
    EvaluationStore,
    SampleKind,
    sample_from_archive_row,
)
from rigging.filesystem import StoragePath, url_to_fs

logger = logging.getLogger(__name__)

_ARCHIVE_URI_PREFIX = "finestore://"


@dataclasses.dataclass(frozen=True)
class MigrationCounts:
    """What one migration wrote: sample rows, flattened step rows, and pulled-in trajectory blobs."""

    samples: int
    steps: int
    trajectories: int


def _trial_id_from_uri(uri: str) -> str:
    """Recover a Harbor trial id from a legacy ``.../<trial>/agent/trajectory.json`` trajectory URI."""
    segments = uri.rstrip("/").split("/")
    if len(segments) >= 3 and segments[-2] == "agent":
        return segments[-3]
    return ""


def migrate_run(results_path: str, *, writer_id: str = "migrate") -> MigrationCounts:
    """Backfill one run's legacy sample parquets into its finestore archive. Safe to re-run."""
    fs, root = url_to_fs(results_path)
    store = EvaluationStore.open(results_path, writer_id=writer_id)
    sample_count = step_count = trajectory_count = 0
    try:
        for path in fs.find(root):
            name = path.rsplit("/", 1)[-1]
            if f"/{FINESTORE_DIR}/" in path:
                continue  # skip the archive's own shards
            if not (name.startswith(SAMPLES_PREFIX) and name.endswith(SAMPLES_SUFFIX)):
                continue
            with fs.open(path, "rb") as handle:
                table = pq.read_table(handle)
            for row in table.to_pylist():
                sample = sample_from_archive_row(row)
                trial_id = ""
                uri = sample.trajectory_uri
                if sample.kind == SampleKind.AGENTIC and uri and not uri.startswith(_ARCHIVE_URI_PREFIX):
                    # A read failure aborts the migration -- it is idempotent and only seals on
                    # success, so a transient fault is retried rather than recorded as complete.
                    trial_id = _trial_id_from_uri(uri)
                    stored = store.add_trajectory(
                        StoragePath(uri).read_bytes(), task=sample.task, doc_id=sample.doc_id, trial_id=trial_id
                    )
                    sample = sample.model_copy(update={"trajectory_uri": stored.uri})
                    if stored.steps:
                        step_count += len(stored.steps)
                        trajectory_count += 1
                store.add_sample(sample, trial_id=trial_id)
                sample_count += 1
        store.seal()
    finally:
        store.close()
    counts = MigrationCounts(samples=sample_count, steps=step_count, trajectories=trajectory_count)
    logger.info("migrated %s: %s", results_path, counts)
    return counts


def archive_sample_count(results_path: str) -> int:
    """The number of samples in a run's finestore archive (for post-migration verification)."""
    table = CompositeReader(results_path).scan(ARCHIVE_SAMPLES_TABLE, columns=["task"])
    return 0 if table is None else table.num_rows


@click.command()
@click.argument("results_path")
@click.option("--writer-id", default="migrate", help="Writer identity stamped on the migrated shards.")
def main(results_path: str, writer_id: str) -> None:
    """Backfill the finestore archive for the run at RESULTS_PATH."""
    logging.basicConfig(level=logging.INFO)
    counts = migrate_run(results_path, writer_id=writer_id)
    click.echo(json.dumps(dataclasses.asdict(counts), indent=2))


if __name__ == "__main__":
    main()
