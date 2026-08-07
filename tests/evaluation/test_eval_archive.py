# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the finestore eval archive: contract round-trip, evaldash read, migration."""

from __future__ import annotations

import json

from click.testing import CliRunner
from finestore.eval import (
    Choice,
    EvalSample,
    EvaluationStore,
    Grading,
    SampleKind,
    export_lm_eval_samples,
    sample_from_archive_row,
    sample_to_archive_row,
    write_sample_parquet,
)
from finestore.reader import CompositeReader
from fsspec.core import url_to_fs

from experiments.evaluation.migrate_archive import (
    MigrationCounts,
    archive_sample_count,
    legacy_archive_prefix,
    migrate_run,
)
from experiments.evaluation.migrate_archive import (
    main as migrate_archive_cli,
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
    store = EvaluationStore.open(root, writer_id="evalchemy")
    store.add_sample(_mcq("1", correct=True))
    store.add_sample(_mcq("2", correct=False))
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


def test_ungraded_sample_reads_back_with_empty_metrics(tmp_path):
    # metrics is a pinned map<string,double>. A batch that writes no metrics leaves that column null on
    # disk (the write path drops an all-empty dict column); the reader must normalize the null back to
    # an empty dict, not fail validation nor invent a zero.
    root = str(tmp_path / "run" / "results")
    store = EvaluationStore.open(root, writer_id="evalchemy")
    store.add_sample(EvalSample(task="gsm8k", doc_id="1", kind=SampleKind.GENERATION, output="4"))
    store.seal()
    store.close()

    page = fetch_samples(root, "gsm8k", offset=0, limit=10, correct="all")
    assert page.available
    assert page.counts.ungraded == 1
    assert page.primary_metric is None
    assert page.rows[0].metrics == {}
    assert page.rows[0].correct is None


def test_export_lm_eval_samples_preserves_unicode_line_separator(tmp_path):
    results = tmp_path / "run" / "results"
    sample_path = results / "gsm8k_5shot" / "model" / "samples_gsm8k_20260807.jsonl"
    sample_path.parent.mkdir(parents=True)
    content = "How many?\u2028Show your work."
    prompt = json.dumps([{"role": "user", "content": content}], ensure_ascii=False)
    raw = {
        "doc_id": 604,
        "doc": {"question": content},
        "target": "4",
        "arguments": [[prompt]],
        "resps": [["4"]],
        "filtered_resps": ["4"],
        "exact_match,flexible-extract": 1.0,
    }
    sample_path.write_text(json.dumps(raw, ensure_ascii=False) + "\n")

    assert export_lm_eval_samples(str(results)) == 1

    table = CompositeReader(str(results)).scan("samples")
    assert table is not None
    [row] = table.to_pylist(maps_as_pydicts="strict")
    sample = sample_from_archive_row(row)
    assert sample.prompt_messages is not None
    assert sample.prompt_messages[0].content == content


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


def test_migration_cli_reads_archived_legacy_shards(tmp_path):
    results = (tmp_path / "run" / "results").as_uri()
    missing_results = (tmp_path / "run-without-backup" / "results").as_uri()
    archive = legacy_archive_prefix(results)
    fs, archive_root = url_to_fs(archive)
    legacy_path = f"{archive_root}/arc/model/samples_arc_20260101.parquet"
    fs.makedirs(legacy_path.rsplit("/", 1)[0], exist_ok=True)
    write_sample_parquet(fs, legacy_path, [_mcq("1", correct=True)])

    result = CliRunner().invoke(migrate_archive_cli, [results, missing_results, "--from-legacy-archive"])

    assert result.exit_code == 0, result.output
    assert archive_sample_count(results) == 1
    assert fs.exists(legacy_path)
    summary = json.loads(result.output.splitlines()[-1])
    assert summary["migrated_runs"] == 1
    assert summary["skipped_runs"] == 1


def test_fetch_artifact_keys_cache_by_run(tmp_path):
    # A finestore:// URI is archive-relative, so two runs can share one. Resolving it for run A then
    # run B must return each run's own bytes, not A's cached response for both.
    uri = "finestore://blobs/trial-1/trajectory.json"
    run_a = str(tmp_path / "a" / "results")
    run_b = str(tmp_path / "b" / "results")
    for root, tag in ((run_a, "a"), (run_b, "b")):
        store = EvaluationStore.open(root, writer_id="w")
        stored = store.add_trajectory(json.dumps({"run": tag}).encode(), task="t", doc_id="d", trial_id="trial-1")
        assert stored.uri == uri
        store.seal()
        store.close()

    first = fetch_artifact(run_a, uri)
    second = fetch_artifact(run_b, uri)
    assert first.available and second.available
    assert json.loads(first.text)["run"] == "a"
    assert json.loads(second.text)["run"] == "b"
