# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the finestore eval archive: contract round-trip, evaldash read, migration."""

from __future__ import annotations

import json

import pytest
from click.testing import CliRunner
from finestore.eval import (
    Choice,
    EvalSample,
    EvaluationStore,
    Grading,
    SampleKind,
    export_lm_eval_samples,
    preserved_sample_sources,
    rebuild_lm_eval_samples,
    sample_from_archive_row,
    sample_to_archive_row,
    write_sample_parquet,
)
from finestore.reader import CompositeReader
from fsspec.core import url_to_fs
from rigging.filesystem import StoragePath

from experiments.evaluation.cli import SweepOutcome, _sweep_archives, selected_archives
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


def _lm_eval_row(doc_id: int, extraction_filter: str, score: float, response: str) -> dict:
    """One lm-eval --log_samples row: a task applying two filters writes one of these per filter."""
    return {
        "doc_id": doc_id,
        "doc": {"question": "2+2?"},
        "target": "4",
        "arguments": [["Question: 2+2?"]],
        "resps": [[response]],
        "filtered_resps": [response],
        "filter": extraction_filter,
        "metrics": ["exact_match"],
        "exact_match": score,
        "schema_version": 1,
        "task_name": "gsm8k",
    }


def _backup(tmp_path):
    """Where a replaced samples table is preserved, keyed by the contract version being replaced."""
    return lambda version: str(tmp_path / f"backup-v{version}")


def _write_jsonl(results, rows: list[dict]):
    path = results / "gsm8k_5shot" / "model" / "samples_gsm8k_20260807.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    return path


def test_each_extraction_filter_keeps_its_own_sample(tmp_path):
    # gsm8k scores one document under strict-match and flexible-extract, and they disagree. Both
    # verdicts must survive: collapsing them onto one key discarded half the graded rows.
    results = tmp_path / "run" / "results"
    _write_jsonl(
        results,
        [
            _lm_eval_row(0, "strict-match", 0.0, "[invalid]"),
            _lm_eval_row(0, "flexible-extract", 1.0, "4"),
        ],
    )

    assert export_lm_eval_samples(str(results)) == 2

    rows = CompositeReader(str(results)).scan("samples").to_pylist(maps_as_pydicts="strict")
    assert len(rows) == 2
    by_filter = {row["filter"]: sample_from_archive_row(row) for row in rows}
    assert set(by_filter) == {"strict-match", "flexible-extract"}
    assert by_filter["strict-match"].correct is False
    assert by_filter["flexible-extract"].correct is True
    # The filter is recorded on the grading the UI renders, not only in the archive key.
    assert by_filter["strict-match"].grading.filter == "strict-match"


def test_sample_metrics_exclude_the_row_format_stamp(tmp_path):
    # lm-eval stamps each row with its own numeric schema_version; it is not a score.
    results = tmp_path / "run" / "results"
    _write_jsonl(results, [_lm_eval_row(0, "none", 1.0, "4")])
    export_lm_eval_samples(str(results))

    [row] = CompositeReader(str(results)).scan("samples").to_pylist(maps_as_pydicts="strict")
    assert sample_from_archive_row(row).metrics == {"exact_match": 1.0}


def test_export_preserves_its_sources_and_rebuilds_from_them(tmp_path):
    # The archive keeps the bytes it normalized, so a later contract change can rebuild the tables
    # even if the surrounding results tree is gone.
    results = tmp_path / "run" / "results"
    source = _write_jsonl(results, [_lm_eval_row(0, "none", 1.0, "4"), _lm_eval_row(1, "none", 0.0, "5")])
    (results / "gsm8k_5shot" / "model" / "results_20260807.json").write_text(json.dumps({"results": {}}))
    assert export_lm_eval_samples(str(results)) == 2

    reader = CompositeReader(str(results))
    blob = reader.read_blob(f"sources/{source.relative_to(results)}")
    assert blob == source.read_bytes()

    source.unlink()
    assert rebuild_lm_eval_samples(str(results), superseded_prefix=_backup(tmp_path)) == 2
    assert CompositeReader(str(results)).scan("samples").num_rows == 2


def test_rebuild_reports_when_no_sources_were_preserved(tmp_path):
    # An archive written before source preservation must be rebuilt from the results tree; saying so
    # is what keeps a caller from reading "0 samples" as a successful rebuild.
    results = str(tmp_path / "run" / "results")
    store = EvaluationStore.open(results, writer_id="evalchemy")
    store.add_sample(EvalSample(task="gsm8k", doc_id="1", kind=SampleKind.GENERATION, output="4"))
    store.seal()
    store.close()

    assert preserved_sample_sources(results) == ()
    with pytest.raises(FileNotFoundError):
        rebuild_lm_eval_samples(results)


def test_re_export_replaces_rows_written_under_an_older_contract(tmp_path):
    # A v3 archive folded both filters onto one key. Re-exporting must replace those rows, not leave
    # the folded one beside the two new ones -- their merge keys no longer line up.
    results = tmp_path / "run" / "results"
    _write_jsonl(
        results,
        [
            _lm_eval_row(0, "strict-match", 0.0, "[invalid]"),
            _lm_eval_row(0, "flexible-extract", 1.0, "4"),
        ],
    )
    export_lm_eval_samples(str(results))
    _stamp_schema_version(results, 3)

    assert export_lm_eval_samples(str(results), superseded_prefix=_backup(tmp_path)) == 2
    assert CompositeReader(str(results)).scan("samples").num_rows == 2


def _stamp_schema_version(results, version: int) -> None:
    schema_path = StoragePath(str(results) + "/samples/_schema.json")
    meta = json.loads(schema_path.read_text())
    meta["schema_version"] = version
    schema_path.write_text(json.dumps(meta))


def _harbor_archive(results, version: int) -> None:
    """Leave a samples table from a mechanism that writes no lm-eval jsonl."""
    store = EvaluationStore.open(str(results), writer_id="harbor")
    store.add_sample(
        EvalSample(
            task="tb2",
            doc_id="1",
            kind=SampleKind.AGENTIC,
            trajectory_uri="finestore://blobs/t1/trajectory.json",
            grading=Grading(method="harbor:verifier", metric="reward", score=1.0, passed=True),
            correct=True,
        )
    )
    store.seal()
    store.close()
    _stamp_schema_version(results, version)


def test_export_leaves_an_archive_it_has_no_source_for(tmp_path):
    # A Harbor run's samples arrive through add_sample, not a jsonl on disk. A sweep that visits it
    # must not read the stale schema version as licence to drop rows it cannot put back.
    results = tmp_path / "run" / "results"
    _harbor_archive(results, version=3)

    assert export_lm_eval_samples(str(results)) == 0
    assert CompositeReader(str(results)).scan("samples").num_rows == 1


def test_export_refuses_to_replace_samples_from_another_mechanism(tmp_path):
    # An archive holding both mechanisms cannot be brought forward by re-reading the jsonl, because
    # that only reproduces the lm-eval half. Erroring keeps the other half.
    results = tmp_path / "run" / "results"
    _harbor_archive(results, version=3)
    _write_jsonl(results, [_lm_eval_row(0, "flexible-extract", 1.0, "4")])

    with pytest.raises(ValueError, match="agentic"):
        export_lm_eval_samples(str(results), superseded_prefix=_backup(tmp_path))
    assert CompositeReader(str(results)).scan("samples").num_rows == 1


def test_replacing_a_table_preserves_it_outside_the_run_first(tmp_path):
    # A contract change is the one moment the archive holds rows nothing else does. They go to the
    # 30-day bucket before the drop, so a bad rebuild is recoverable rather than terminal.
    results = tmp_path / "run" / "results"
    _write_jsonl(results, [_lm_eval_row(0, "strict-match", 0.0, "[invalid]")])
    export_lm_eval_samples(str(results), superseded_prefix=_backup(tmp_path))
    _stamp_schema_version(results, 3)

    export_lm_eval_samples(str(results), superseded_prefix=_backup(tmp_path))

    preserved = CompositeReader(str(tmp_path / "backup-v3"))
    assert preserved.scan("samples") is None  # the snapshot is the table itself, not a nested copy
    assert (tmp_path / "backup-v3" / "_schema.json").exists()
    assert list((tmp_path / "backup-v3").rglob("*.parquet"))


def test_replacing_a_table_refuses_without_somewhere_to_preserve_it(tmp_path):
    results = tmp_path / "run" / "results"
    _write_jsonl(results, [_lm_eval_row(0, "strict-match", 0.0, "[invalid]")])
    export_lm_eval_samples(str(results), superseded_prefix=_backup(tmp_path))
    _stamp_schema_version(results, 3)

    with pytest.raises(ValueError, match="superseded_prefix"):
        export_lm_eval_samples(str(results))
    assert CompositeReader(str(results)).scan("samples").num_rows == 1


def test_an_existing_snapshot_is_never_overwritten(tmp_path):
    # The first snapshot is the pristine one. A resumed or repeated replace must not put a degraded
    # table on top of it.
    results = tmp_path / "run" / "results"
    backup = tmp_path / "backup-v3"
    backup.mkdir()
    (backup / "_schema.json").write_text('{"sentinel": true}')
    _write_jsonl(results, [_lm_eval_row(0, "strict-match", 0.0, "[invalid]")])
    export_lm_eval_samples(str(results), superseded_prefix=_backup(tmp_path))
    _stamp_schema_version(results, 3)

    export_lm_eval_samples(str(results), superseded_prefix=_backup(tmp_path))

    assert json.loads((backup / "_schema.json").read_text()) == {"sentinel": True}


def test_a_retried_evaluation_indexes_only_the_published_tree(tmp_path):
    # evalchemy copies its temp working directory into the results tree, so a retry leaves a second
    # complete evaluation whose loglikelihoods differ. Only the canonical tree produced the metrics
    # on the run's record, and indexing both would put two different rows on one merge key.
    results = tmp_path / "run" / "results"
    _write_jsonl(results, [_lm_eval_row(0, "none", 1.0, "4")])
    scratch = results / "gsm8k_5shot" / "tmpp90h6r1d" / "model" / "samples_gsm8k_20260807.jsonl"
    scratch.parent.mkdir(parents=True)
    scratch.write_text(json.dumps(_lm_eval_row(0, "none", 0.0, "5")) + "\n")

    assert export_lm_eval_samples(str(results)) == 1

    [row] = CompositeReader(str(results)).scan("samples").to_pylist(maps_as_pydicts="strict")
    assert sample_from_archive_row(row).output == "4"
    # The retry is still recoverable: it is preserved even though it produced no row.
    assert any("tmpp90h6r1d" in name for name in preserved_sample_sources(str(results)))


def test_writing_to_a_sealed_archive_clears_its_seal(tmp_path):
    # "Sealed" has to mean "these are the finished contents". A stale marker left by an earlier
    # session would vouch for a table a failed export has since replaced.
    root = str(tmp_path / "run" / "results")
    store = EvaluationStore.open(root, writer_id="evalchemy")
    store.add_sample(_mcq("1", correct=True))
    store.seal()
    store.close()
    assert CompositeReader(root).is_sealed()

    reopened = EvaluationStore.open(root, writer_id="evalchemy")
    try:
        assert not CompositeReader(root).is_sealed()
    finally:
        reopened.close()


def test_sweep_visits_an_archive_shared_by_several_runs_once(tmp_path):
    # Several records can name one results tree. Two workers writing it at once make one compact
    # shards the other is reading, so the sweep must group by path before it fans out.
    shared = str(tmp_path / "shared" / "results")
    own = str(tmp_path / "own" / "results")
    visited: list[str] = []

    def work(path: str) -> SweepOutcome:
        visited.append(path)
        return SweepOutcome("exported", "0 sample(s)")

    _sweep_archives({shared: ["run-a", "run-b"], own: ["run-c"]}, 4, work)

    assert sorted(visited) == sorted([shared, own])


def test_naming_an_archive_directly_does_not_pull_in_the_fleet(tmp_path):
    # Targeting a handful of damaged archives must not re-sweep every recorded run beside them.
    named = str(tmp_path / "one" / "results")
    assert selected_archives((), (named + "/",)) == {named: []}


def test_evaldash_serves_one_extraction_filter_at_a_time(tmp_path):
    # Two rows per document would list every question twice and make the correctness counts
    # disagree with the headline score, so the browser picks one filter and names the alternatives.
    results = tmp_path / "run" / "results"
    _write_jsonl(
        results,
        [
            _lm_eval_row(0, "strict-match", 0.0, "[invalid]"),
            _lm_eval_row(0, "flexible-extract", 1.0, "4"),
        ],
    )
    export_lm_eval_samples(str(results))

    page = fetch_samples(str(results), "gsm8k", offset=0, limit=10, correct="all")
    assert page.extraction_filters == ("flexible-extract", "strict-match")
    # FILTER_PRIORITY ranks flexible-extract first, matching the headline metric's filter.
    assert page.extraction_filter == "flexible-extract"
    assert page.counts.all == 1
    assert page.rows[0].correct is True

    strict = fetch_samples(str(results), "gsm8k", offset=0, limit=10, correct="all", extraction_filter="strict-match")
    assert strict.extraction_filter == "strict-match"
    assert strict.counts.all == 1
    assert strict.rows[0].correct is False


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
    legacy_file = StoragePath(archive) / "arc/model/samples_arc_20260101.parquet"
    legacy_file.parent.mkdirs()
    fs, legacy_path = url_to_fs(str(legacy_file))
    write_sample_parquet(fs, legacy_path, [_mcq("1", correct=True)])

    result = CliRunner().invoke(migrate_archive_cli, [results, missing_results, "--from-legacy-archive"])

    assert result.exit_code == 0, result.output
    assert archive_sample_count(results) == 1
    assert legacy_file.exists()
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
