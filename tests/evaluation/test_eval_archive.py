# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the finestore eval archive: contract round-trip, evaldash read, migration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from finestore.admin import set_table_metadata
from finestore.reader import ReadView
from fsspec.core import url_to_fs
from marin.evaluation.archive import (
    SAMPLES_MERGE_KEY,
    Choice,
    EvalSample,
    EvaluationStore,
    Grading,
    SampleKind,
    sample_from_archive_row,
    sample_to_archive_row,
    write_sample_parquet,
)
from marin.evaluation.lm_eval_samples import (
    export_lm_eval_samples,
    preserved_sample_sources,
    rebuild_lm_eval_samples,
    run_artifacts,
)
from marin.evaluation.records import TaskCoverage
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.migrations.cli import SweepOutcome, _sweep_archives, selected_archives
from experiments.evaluation.migrations.cli import cli as migrations_cli
from experiments.evaluation.migrations.format_smoke import smoke_upgrade, smoke_upgrade_fleet
from experiments.evaluation.migrations.migrate_archive import (
    MigrationCounts,
    archive_sample_count,
    legacy_archive_prefix,
    migrate_run,
)
from experiments.evaluation.migrations.migrate_archive import (
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


def test_archive_row_round_trips_each_sample_kind():
    samples = (
        _mcq("1", correct=True),
        EvalSample(task="gsm8k", doc_id="2", kind=SampleKind.GENERATION, output="4", extracted="4"),
        EvalSample(
            task="aime",
            doc_id="3",
            kind=SampleKind.AGENTIC,
            trajectory_uri="finestore://blobs/t3/trajectory.json",
            grading=Grading(method="harbor:verifier", metric="reward", score=1.0, passed=True),
        ),
    )

    for sample in samples:
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

    assert export_lm_eval_samples(str(results)).samples == 1

    table = ReadView(str(results)).scan("samples")
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

    assert export_lm_eval_samples(str(results)).samples == 2

    rows = ReadView(str(results)).scan("samples").to_pylist(maps_as_pydicts="strict")
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

    [row] = ReadView(str(results)).scan("samples").to_pylist(maps_as_pydicts="strict")
    assert sample_from_archive_row(row).metrics == {"exact_match": 1.0}


def test_export_preserves_its_sources_and_rebuilds_from_them(tmp_path):
    # The archive keeps the bytes it normalized, so a later contract change can rebuild the tables
    # even if the surrounding results tree is gone.
    results = tmp_path / "run" / "results"
    source = _write_jsonl(results, [_lm_eval_row(0, "none", 1.0, "4"), _lm_eval_row(1, "none", 0.0, "5")])
    (results / "gsm8k_5shot" / "model" / "results_20260807.json").write_text(json.dumps({"results": {}}))
    assert export_lm_eval_samples(str(results)).samples == 2

    reader = ReadView(str(results))
    blob = reader.read_blob(f"sources/{source.relative_to(results)}")
    assert blob == source.read_bytes()

    source.unlink()
    assert rebuild_lm_eval_samples(str(results)) == 2
    assert ReadView(str(results)).scan("samples").num_rows == 2


def test_export_preserves_every_artifact_the_harness_left(tmp_path):
    # evalchemy's resume state lives under a dot-directory, which the earlier `**/*` globs never
    # matched, so it was the one thing a rebuilt archive could not account for.
    results = tmp_path / "run" / "results"
    _write_jsonl(results, [_lm_eval_row(0, "none", 1.0, "4")])
    (results / "gsm8k_5shot" / "model" / "results_20260807.json").write_text(json.dumps({"results": {}}))
    resume = results / "gsm8k_5shot" / ".resume" / "model" / "gsm8k" / "resume"
    resume.mkdir(parents=True)
    (resume / "fingerprint.json").write_text(json.dumps({"num_fewshot": 5}))
    (resume / "manifest.jsonl").write_text('{"payload": {"output": " 18"}}\n')

    export_lm_eval_samples(str(results))

    preserved = {name for name in ReadView(str(results)).keys("blobs") for name in [name[0]]}
    for relative in run_artifacts(str(results)):
        assert f"sources/{relative}" in preserved, relative
    assert any(".resume" in name for name in preserved)


def test_preserving_artifacts_never_includes_the_archive_itself(tmp_path):
    # The archive shares the run's root, so a re-export that treated its own shards as artifacts
    # would fold the archive into itself and grow without bound.
    results = tmp_path / "run" / "results"
    _write_jsonl(results, [_lm_eval_row(0, "none", 1.0, "4")])
    export_lm_eval_samples(str(results))
    legacy_shard = results / "samples" / "w=old" / "g=0" / "old.parquet"
    legacy_shard.parent.mkdir(parents=True)
    legacy_shard.write_bytes(b"unreachable format-v1 object")
    (results / "SEALED").write_text("{}")

    before = len(run_artifacts(str(results)))
    export_lm_eval_samples(str(results))

    assert run_artifacts(str(results)) == sorted(run_artifacts(str(results)))
    assert len(run_artifacts(str(results))) == before
    assert not any(name.startswith(("samples/", "steps/", "blobs/")) for name in run_artifacts(str(results)))


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


def test_export_refuses_an_archive_written_under_an_older_contract(tmp_path):
    # A v3 archive folded both filters onto one key, and those rows cannot collapse against v4 rows.
    # An ordinary export stops so only the explicit, preserved migration can replace the table.
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

    with pytest.raises(ValueError, match="schema v3"):
        export_lm_eval_samples(str(results))
    assert ReadView(str(results)).scan("samples").num_rows == 2


def _stamp_schema_version(results, version: int) -> None:
    reader = ReadView(str(results))
    metadata = reader.table_metadata("samples").model_copy(update={"schema_version": version})
    set_table_metadata(str(results), "samples", metadata)


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

    assert export_lm_eval_samples(str(results)).samples == 0
    assert ReadView(str(results)).scan("samples").num_rows == 1


def test_a_retried_evaluation_indexes_only_the_published_tree(tmp_path):
    # evalchemy copies its temp working directory into the results tree, so a retry leaves a second
    # complete evaluation whose loglikelihoods differ. Only the canonical tree produced the metrics
    # on the run's record, and indexing both would put two different rows on one primary key.
    results = tmp_path / "run" / "results"
    _write_jsonl(results, [_lm_eval_row(0, "none", 1.0, "4")])
    scratch = results / "gsm8k_5shot" / "tmpp90h6r1d" / "model" / "samples_gsm8k_20260807.jsonl"
    scratch.parent.mkdir(parents=True)
    scratch.write_text(json.dumps(_lm_eval_row(0, "none", 0.0, "5")) + "\n")

    assert export_lm_eval_samples(str(results)).samples == 1

    [row] = ReadView(str(results)).scan("samples").to_pylist(maps_as_pydicts="strict")
    assert sample_from_archive_row(row).output == "4"
    # The retry is still recoverable: it is preserved even though it produced no row.
    assert any("tmpp90h6r1d" in name for name in preserved_sample_sources(str(results)))


def test_export_establishes_what_a_task_set_out_to_grade(tmp_path):
    # lm-eval publishes no attempted-item count in its aggregate results, so a run's own sample rows
    # are the only evidence that it graded everything it enumerated. Without this the dashboard can
    # only report sampling error and has to treat the benchmark's completeness as unknown.
    results = tmp_path / "run" / "results"
    _write_jsonl(results, [_lm_eval_row(doc_id, "none", float(doc_id % 2), "4") for doc_id in range(6)])

    coverage = export_lm_eval_samples(str(results)).coverage

    assert coverage == {"gsm8k_5shot": TaskCoverage(n_attempted=6, n_scored=6, n_correct=3, n_unanswered=0)}


def test_a_document_that_produced_no_row_is_counted_as_attempted(tmp_path):
    # lm-eval indexes documents 0..N-1, so a gap in the indices is a document that was enumerated and
    # never graded. Scoring the survivors as if they were the whole benchmark is the complete-case
    # bias the interval exists to bound.
    results = tmp_path / "run" / "results"
    _write_jsonl(
        results,
        [_lm_eval_row(0, "none", 1.0, "4"), _lm_eval_row(1, "none", 1.0, "4"), _lm_eval_row(3, "none", 1.0, "4")],
    )

    [entry] = export_lm_eval_samples(str(results)).coverage.values()

    assert entry.n_attempted == 4
    assert entry.n_scored == 3


def test_a_document_scored_under_two_filters_counts_once(tmp_path):
    # gsm8k grades each document under strict-match and flexible-extract, so the archive holds two
    # rows per document. Counting rows would report twice the items the benchmark actually has, which
    # halves the interval it deserves.
    results = tmp_path / "run" / "results"
    _write_jsonl(
        results,
        [
            _lm_eval_row(0, "strict-match", 0.0, "[invalid]"),
            _lm_eval_row(0, "flexible-extract", 1.0, "4"),
            _lm_eval_row(1, "strict-match", 0.0, "[invalid]"),
            _lm_eval_row(1, "flexible-extract", 0.0, "nope"),
        ],
    )

    [entry] = export_lm_eval_samples(str(results)).coverage.values()

    assert entry.n_scored == 2
    # flexible-extract leads for exact_match, and the tally comes from the same filter the run's
    # headline metric is reported under rather than mixing the two verdicts.
    assert entry.n_correct == 1


def test_answers_the_grader_could_not_extract_are_counted_apart_from_wrong_ones(tmp_path):
    # A zero because the model answered wrongly and a zero because nothing parseable came back are
    # the same number and different facts; only the second is grounds for doubting the run.
    results = tmp_path / "run" / "results"
    _write_jsonl(results, [_lm_eval_row(0, "none", 0.0, ""), _lm_eval_row(1, "none", 0.0, "5")])

    [entry] = export_lm_eval_samples(str(results)).coverage.values()

    assert entry.n_scored == 2
    assert entry.n_correct == 0
    assert entry.n_unanswered == 1


def test_infrastructure_error_is_unscored_and_recomputes_the_metric(tmp_path):
    results = tmp_path / "run" / "results"
    marker = "[EVALCHEMY_INFRASTRUCTURE_ERROR] ClientResponseError: 400 Bad Request"
    _write_jsonl(
        results,
        [
            _lm_eval_row(0, "none", 1.0, "4"),
            _lm_eval_row(1, "none", 0.0, marker),
        ],
    )

    export = export_lm_eval_samples(str(results))
    [coverage] = export.coverage.values()

    assert coverage == TaskCoverage(
        n_attempted=2,
        n_scored=1,
        n_correct=1,
        n_unanswered=1,
        errors={"EVALCHEMY_INFRASTRUCTURE_ERROR": 1},
    )
    assert export.recovered_metrics == {"gsm8k_5shot": {"exact_match,none": 1.0, "sample_len": 1.0}}


def test_a_group_task_reports_coverage_per_subtask(tmp_path):
    # One task config evaluating several tasks keys its metrics <task_dir>/<task>; coverage keys the
    # same way or a reader cannot line the two up.
    results = tmp_path / "run" / "results"
    directory = results / "mmlu_5shot" / "model"
    directory.mkdir(parents=True)
    for subject in ("anatomy", "astronomy"):
        rows = [_lm_eval_row(doc_id, "none", 1.0, "4") for doc_id in range(3)]
        (directory / f"samples_mmlu_{subject}_20260807.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n"
        )

    coverage = export_lm_eval_samples(str(results)).coverage

    assert sorted(coverage) == ["mmlu_5shot/mmlu_anatomy", "mmlu_5shot/mmlu_astronomy"]
    assert all(entry.n_attempted == 3 for entry in coverage.values())


def test_writing_to_a_sealed_archive_clears_its_seal(tmp_path):
    # "Sealed" has to mean "these are the finished contents". A stale marker left by an earlier
    # session would vouch for a table a failed export has since replaced.
    root = str(tmp_path / "run" / "results")
    store = EvaluationStore.open(root, writer_id="evalchemy")
    store.add_sample(_mcq("1", correct=True))
    store.seal()
    store.close()
    assert ReadView(root).is_sealed()

    reopened = EvaluationStore.open(root, writer_id="evalchemy")
    try:
        assert not ReadView(root).is_sealed()
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


def test_upgrade_format_prefix_migrates_only_sealed_archives(tmp_path, monkeypatch):
    sealed = tmp_path / "evals" / "sealed" / "results"
    unsealed = tmp_path / "evals" / "unsealed" / "results"
    missing = tmp_path / "evals" / "missing" / "results"
    _write_v1_smoke_archive(sealed)
    _write_v1_smoke_archive(unsealed)
    (unsealed / "SEALED").unlink()
    records = [SimpleNamespace(results_path=str(path)) for path in (sealed, unsealed, missing)]
    monkeypatch.setattr("experiments.evaluation.migrations.cli.list_records", lambda _: records)
    monkeypatch.setattr("experiments.evaluation.migrations.format_smoke.read_records", lambda _: (records, ()))

    result = CliRunner().invoke(
        migrations_cli,
        ["upgrade-format", "--prefix", str(tmp_path / "evals"), "--workers", "1"],
    )

    assert result.exit_code == 0, result.output
    assert json.loads((sealed / "_archive.json").read_text())["format_version"] == 2
    assert json.loads((unsealed / "_archive.json").read_text())["format_version"] == 1
    selection = json.loads(result.output.splitlines()[0])
    assert selection["sealed_v1"] == 1
    assert selection["unsealed_v1"] == [str(unsealed)]
    assert selection["missing_archive"] == 1


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
    reader = ReadView(results)
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


def _write_v1_smoke_archive(source) -> None:
    (source / "samples" / "w=legacy" / "g=0").mkdir(parents=True)
    (source / "samples" / "w=compact" / "g=1").mkdir(parents=True)
    (source / "blobs" / "w=legacy" / "g=0").mkdir(parents=True)
    (source / "_archive.json").write_text('{"format_version": 1}')
    (source / "SEALED").write_text('{"writer": "legacy", "superseded": {"samples": 1}}')
    (source / "samples" / "_schema.json").write_text(
        '{"primary_key": ["doc_id"], "schema_version": 4, "on_conflict": "supersede"}'
    )
    (source / "blobs" / "_schema.json").write_text(
        '{"primary_key": ["name"], "schema_version": 1, "on_conflict": "error"}'
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {"doc_id": "a", "score": 0.5, "_seq": 1, "_writer": "legacy"},
                {"doc_id": "b", "score": 0.8, "_seq": 2, "_writer": "legacy"},
            ]
        ),
        source / "samples" / "w=legacy" / "g=0" / "0000000000000001-old.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"doc_id": "a", "score": 0.7, "_seq": 1, "_writer": "compact"}]),
        source / "samples" / "w=compact" / "g=1" / "0000000000000001-compact.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist([{"name": "trajectory.json", "data": b"payload", "_seq": 0, "_writer": "legacy"}]),
        source / "blobs" / "w=legacy" / "g=0" / "0000000000000000-blob.parquet",
    )


def _write_live_v1_eval_archive(source) -> None:
    samples_root = source / "samples"
    blobs_root = source / "blobs"
    for table_root in (samples_root, blobs_root):
        (table_root / "w=legacy" / "g=0").mkdir(parents=True)
    (samples_root / "w=legacy" / "g=1").mkdir(parents=True)
    (source / "_archive.json").write_text('{"format_version": 1}')
    (samples_root / "_schema.json").write_text(
        json.dumps({"primary_key": SAMPLES_MERGE_KEY, "schema_version": 4, "on_conflict": "supersede"})
    )
    (blobs_root / "_schema.json").write_text('{"primary_key": ["name"], "schema_version": 1, "on_conflict": "error"}')

    first = sample_to_archive_row(_mcq("1", correct=False), trial_id="")
    second = sample_to_archive_row(_mcq("2", correct=False), trial_id="")
    pq.write_table(
        pa.Table.from_pylist(
            [
                {**first, "_seq": 0, "_writer": "legacy"},
                {**second, "_seq": 1, "_writer": "legacy"},
            ]
        ),
        samples_root / "w=legacy" / "g=0" / "0000000000000000-samples.parquet",
    )
    replacement = sample_to_archive_row(_mcq("1", correct=True), trial_id="")
    pq.write_table(
        pa.Table.from_pylist([{**replacement, "_seq": 2, "_writer": "legacy"}]),
        samples_root / "w=legacy" / "g=1" / "0000000000000002-samples.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"name": "trajectory.json", "data": b'{"steps":[{"step_id":1}]}', "_seq": 0, "_writer": "legacy"}]
        ),
        blobs_root / "w=legacy" / "g=0" / "0000000000000000-blob.parquet",
    )


def test_evaldash_reads_an_unsealed_v1_finestore_archive(tmp_path):
    results = tmp_path / "run" / "results"
    _write_live_v1_eval_archive(results)

    tasks = list_sample_tasks(str(results))
    assert tasks.available
    assert [(task.task, task.files) for task in tasks.tasks] == [("arc", 2)]

    page = fetch_samples(str(results), "arc", offset=0, limit=10, correct="all")
    assert page.available
    assert page.counts == page.counts.model_copy(update={"all": 2, "correct": 1, "incorrect": 1, "ungraded": 0})
    assert {sample.doc_id: sample.correct for sample in page.rows} == {"1": True, "2": False}

    artifact = fetch_artifact(str(results), "finestore://blobs/trajectory.json")
    assert artifact.available
    assert json.loads(artifact.text) == {"steps": [{"step_id": 1}]}
    assert json.loads((results / "_archive.json").read_text()) == {"format_version": 1}
    assert not (results / "HEAD").exists()


def test_format_smoke_migrates_a_clone_and_preserves_source_and_rows(tmp_path):
    source = tmp_path / "evals" / "run-1" / "results"
    destination = tmp_path / "tmp" / "migration-smoke"
    _write_v1_smoke_archive(source)

    result = smoke_upgrade(str(source), str(destination))

    assert result.source == str(source)
    assert result.destination == str(destination)
    assert result.rows == 3
    assert {table.name: table.rows for table in result.tables} == {"blobs": 1, "samples": 2}
    assert json.loads((source / "_archive.json").read_text())["format_version"] == 1
    assert not (source / "HEAD").exists()
    migrated = ReadView(str(destination))
    assert migrated.point("samples", doc_id="a")["score"] == 0.7
    assert migrated.read_blob("trajectory.json") == b"payload"


def test_fleet_smoke_cleans_up_only_after_every_archive_validates(tmp_path):
    sources = tuple(tmp_path / "evals" / run / "results" for run in ("run-1", "run-2"))
    for source in sources:
        _write_v1_smoke_archive(source)
    (sources[1] / "SEALED").unlink()
    destination = tmp_path / "tmp" / "ttl=1d" / "finestore-migration-fleet" / ("a" * 32)

    result = smoke_upgrade_fleet(
        tuple(str(source) for source in sources),
        str(destination),
        workers=2,
        cleanup=True,
    )

    assert len(result.results) == 2
    assert result.rows == 6
    assert result.cleaned_up
    assert not destination.exists()
    for source in sources:
        assert json.loads((source / "_archive.json").read_text())["format_version"] == 1
        assert not (source / "HEAD").exists()
    assert not (sources[1] / "SEALED").exists()


def test_fleet_smoke_preserves_partial_results_when_validation_fails(tmp_path):
    valid = tmp_path / "evals" / "run-1" / "results"
    invalid = tmp_path / "evals" / "run-2" / "results"
    _write_v1_smoke_archive(valid)
    invalid.mkdir(parents=True)
    (invalid / "_archive.json").write_text('{"format_version": 2}')
    destination = tmp_path / "tmp" / "ttl=1d" / "finestore-migration-fleet" / ("b" * 32)

    with pytest.raises(ValueError):
        smoke_upgrade_fleet(
            (str(valid), str(invalid)),
            str(destination),
            workers=1,
            cleanup=True,
        )

    assert destination.exists()
    assert (destination / "_fleet.json").exists()


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
