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

import json
import logging

import click
import pyarrow.parquet as pq
from finestore import CompositeReader
from finestore.layout import FINESTORE_DIR
from rigging.filesystem import StoragePath, url_to_fs

from marin.evaluation.samples import (
    ARCHIVE_SAMPLES_TABLE,
    SAMPLES_PREFIX,
    SAMPLES_SUFFIX,
    EvalSample,
    SampleKind,
    archive_samples_table,
    archive_steps_table,
    open_eval_archive,
    sample_from_archive_row,
    sample_to_archive_row,
    trajectory_step_rows,
)

logger = logging.getLogger(__name__)

_ARCHIVE_URI_PREFIX = "finestore://"


def _trial_id_from_uri(uri: str) -> str:
    """Recover a Harbor trial id from a legacy ``.../<trial>/agent/trajectory.json`` trajectory URI."""
    segments = uri.rstrip("/").split("/")
    if len(segments) >= 3 and segments[-2] == "agent":
        return segments[-3]
    return ""


def _migrate_trajectory(store, sample: EvalSample) -> tuple[str, str | None, list[dict]]:
    """Pull an agentic sample's legacy trajectory into the archive; return (trial_id, new_uri, steps)."""
    uri = sample.trajectory_uri
    if sample.kind != SampleKind.AGENTIC or not uri or uri.startswith(_ARCHIVE_URI_PREFIX):
        return "", uri, []
    trial_id = _trial_id_from_uri(uri)
    try:
        raw = StoragePath(uri).read_bytes()
    except Exception as exc:
        logger.warning("trajectory %s unreadable during migration (%s); keeping the original uri", uri, exc)
        return trial_id, uri, []
    new_uri = store.write(
        f"{trial_id}/trajectory.json",
        {"task": sample.task, "doc_id": sample.doc_id, "trial_id": trial_id},
        raw,
    )
    step_rows: list[dict] = []
    try:
        trajectory = json.loads(raw)
        step_rows = trajectory_step_rows(trajectory, task=sample.task, doc_id=sample.doc_id, trial_id=trial_id)
    except (json.JSONDecodeError, ValueError):
        logger.warning("trajectory %s unparseable during migration; skipping steps", uri)
    return trial_id, new_uri, step_rows


def migrate_run(results_path: str, *, writer_id: str = "migrate") -> dict[str, int]:
    """Backfill one run's legacy sample parquets into its finestore archive.

    Returns counts of ``samples``, ``steps``, and ``trajectories`` written. Safe to re-run.
    """
    fs, root = url_to_fs(results_path)
    store = open_eval_archive(results_path, writer_id=writer_id)
    samples = archive_samples_table(store)
    steps = archive_steps_table(store)
    counts = {"samples": 0, "steps": 0, "trajectories": 0}
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
                trial_id, new_uri, step_rows = _migrate_trajectory(store, sample)
                if step_rows:
                    steps.extend(step_rows)
                    counts["steps"] += len(step_rows)
                    counts["trajectories"] += 1
                if new_uri != sample.trajectory_uri:
                    sample = sample.model_copy(update={"trajectory_uri": new_uri})
                samples.append(sample_to_archive_row(sample, trial_id=trial_id))
                counts["samples"] += 1
        store.seal()
    finally:
        store.close()
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
    click.echo(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
