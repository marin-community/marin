# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the finestore eval archive: contract round-trip, evaldash read, migration."""

from __future__ import annotations

import json

from finestore import CompositeReader
from fsspec.core import url_to_fs
from marin.evaluation.migrate_archive import MigrationCounts, archive_sample_count, migrate_run
from marin.evaluation.samples import (
    Choice,
    EvalSample,
    Grading,
    SampleKind,
    archive_samples_table,
    open_eval_archive,
    sample_from_archive_row,
    sample_to_archive_row,
    write_sample_parquet,
)

from infra.evaldash.src.samples import fetch_artifact, fetch_samples, list_sample_tasks


def _mcq(doc_id: str, *, correct: bool) -> EvalSample:
    return EvalSample(
        task="arc",
        doc_id=doc_id,
        kind=SampleKind.MULTIPLE_CHOICE,
        prompt_text="Q?",
        choices=[Choice(label="A", text="a", loglikelihood=-1.0), Choice(label="B", text="b", loglikelihood=-2.0)],
        model_choice=0,
        target_choice=0 if correct else 1,
        grading=Grading(method="lm-eval:acc", metric="acc", score=1.0 if correct else 0.0, passed=correct),
        metrics={"acc": 1.0 if correct else 0.0},
        correct=correct,
    )


def test_archive_row_round_trips_each_kind():
    mcq = _mcq("1", correct=True)
    generation = EvalSample(
        task="gsm8k", doc_id="2", kind=SampleKind.GENERATION, prompt_text="2+2?", output="4", extracted="4"
    )
    agentic = EvalSample(
        task="aime",
        doc_id="3",
        kind=SampleKind.AGENTIC,
        trajectory_uri="finestore://blobs/t3/trajectory.json",
        grading=Grading(method="harbor:verifier", metric="reward", score=1.0, passed=True),
        metrics={"reward": 1.0},
        correct=True,
    )
    for sample in (mcq, generation, agentic):
        row = sample_to_archive_row(sample, trial_id="t")
        assert row["trial_id"] == "t"
        assert sample_from_archive_row(row) == sample


def test_evaldash_reads_the_archive(tmp_path):
    root = str(tmp_path / "run" / "results")
    store = open_eval_archive(root, writer_id="evalchemy")
    samples = archive_samples_table(store)
    samples.append(sample_to_archive_row(_mcq("1", correct=True)))
    samples.append(sample_to_archive_row(_mcq("2", correct=False)))
    store.seal()
    store.close()

    tasks = list_sample_tasks(root)
    assert tasks.available
    assert [task.task for task in tasks.tasks] == ["arc"]

    page = fetch_samples(root, "arc", offset=0, limit=10, correct="all")
    assert page.available
    assert page.counts == page.counts.model_copy(update={"all": 2, "correct": 1, "incorrect": 1, "ungraded": 0})
    assert {row.doc_id for row in page.rows} == {"1", "2"}
    assert page.primary_metric == "acc"

    incorrect = fetch_samples(root, "arc", offset=0, limit=10, correct="incorrect")
    assert [row.doc_id for row in incorrect.rows] == ["2"]


def test_migrate_legacy_run_into_archive(tmp_path):
    results = str(tmp_path / "run" / "results")
    fs, _ = url_to_fs(results)

    # A legacy evalchemy per-(sub)task parquet.
    mcq_path = f"{results}/arc/model/samples_arc_20260101.parquet"
    fs.makedirs(mcq_path.rsplit("/", 1)[0], exist_ok=True)
    write_sample_parquet(fs, mcq_path, [_mcq("1", correct=True)])

    # A legacy Harbor run: one agentic sample referencing a trajectory by an in-place path.
    trajectory_path = f"{results}/harbor_jobs/job/trial-7/agent/trajectory.json"
    fs.makedirs(trajectory_path.rsplit("/", 1)[0], exist_ok=True)
    with fs.open(trajectory_path, "w") as handle:
        handle.write(json.dumps({"steps": [{"step_id": 1, "source": "agent", "message": "solve"}]}))
    agentic = EvalSample(
        task="aime",
        doc_id="prob-1",
        kind=SampleKind.AGENTIC,
        trajectory_uri=trajectory_path,
        grading=Grading(method="harbor:verifier", metric="reward", score=1.0, passed=True),
        metrics={"reward": 1.0},
        correct=True,
    )
    write_sample_parquet(fs, f"{results}/samples_harbor.parquet", [agentic])

    counts = migrate_run(results)
    assert counts == MigrationCounts(samples=2, steps=1, trajectories=1)
    assert archive_sample_count(results) == 2

    # Re-running is idempotent: samples dedupe on their primary key.
    migrate_run(results)
    assert archive_sample_count(results) == 2

    # The migrated agentic sample points at a finestore:// trajectory the archive resolves.
    reader = CompositeReader(results)
    agentic_row = reader.point("samples", task="aime", doc_id="prob-1", trial_id="trial-7")
    assert agentic_row is not None
    uri = agentic_row["trajectory_uri"]
    assert uri.startswith("finestore://blobs/")

    artifact = fetch_artifact(results, uri)
    assert artifact.available
    assert json.loads(artifact.text)["steps"][0]["step_id"] == 1

    # evaldash surfaces both migrated tasks.
    assert {task.task for task in list_sample_tasks(results).tasks} == {"arc", "aime"}


def test_fetch_artifact_keys_cache_by_run(tmp_path):
    # A finestore:// URI is archive-relative, so two runs can share one. Resolving it for run A then
    # run B must return each run's own bytes, not A's cached response for both.
    uri = "finestore://blobs/trial-1/trajectory.json"
    run_a = str(tmp_path / "a" / "results")
    run_b = str(tmp_path / "b" / "results")
    for root, tag in ((run_a, "a"), (run_b, "b")):
        store = open_eval_archive(root, writer_id="w")
        assert store.write("trial-1/trajectory.json", {}, json.dumps({"run": tag}).encode()) == uri
        store.seal()
        store.close()

    first = fetch_artifact(run_a, uri)
    second = fetch_artifact(run_b, uri)
    assert first.available and second.available
    assert json.loads(first.text)["run"] == "a"
    assert json.loads(second.text)["run"] == "b"
