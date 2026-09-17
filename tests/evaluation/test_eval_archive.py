# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""End-to-end tests for the finestore eval archive: contract round-trip and migration."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from click.testing import CliRunner
from finestore.admin import set_table_metadata
from finestore.eval import (
    Choice,
    EvalSample,
    EvaluationStore,
    Grading,
    SampleKind,
    sample_from_archive_row,
    sample_to_archive_row,
    write_sample_parquet,
)
from finestore.reader import ReadView
from fsspec.core import url_to_fs
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.evaluation.lm_eval_samples import (
    export_lm_eval_samples,
    preserved_sample_sources,
    rebuild_lm_eval_samples,
    run_artifacts,
    summarize_native_eval_samples,
)
from marin.evaluation.records import DEFAULT_SCAN_PREFIXES, EvalTaskRef, TaskCoverage
from rigging.filesystem.storage_path import StoragePath

from experiments.evaluation.migrations.cli import (
    ArchiveSweep,
    SweepOutcome,
    _resolve_prefixes,
    _sweep_archives,
    selected_archives,
)
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


def test_native_summary_reads_evalchemy_normalized_rows(tmp_path):
    root = str(tmp_path / "run" / "results")
    store = EvaluationStore.open(root, writer_id="evalchemy")
    try:
        store.add_source_artifact(
            "evalchemy/gsm8k_5shot/native/samples_gsm8k_native.jsonl",
            b'{"doc_id": 999}\n',
            content_type="application/x-ndjson",
        )
        store.add_source_artifact(
            "evalchemy/gsm8k_5shot/native/results_gsm8k.json",
            json.dumps(
                {
                    "results": {"gsm8k": {"exact_match,flexible-extract": 1.0}},
                    **_result_contract(
                        "gsm8k",
                        "exact_match",
                        "accuracy",
                        1.0,
                        n_benchmark=1,
                        n_attempted=1,
                    ),
                }
            ).encode(),
            content_type="application/json",
        )
        store.add_sample(
            EvalSample(
                task="gsm8k_5shot",
                doc_id="0",
                kind=SampleKind.GENERATION,
                output="4",
                extracted="4",
                grading=Grading(method="lm-eval:exact_match", metric="exact_match", score=1.0, passed=True),
                metrics={"exact_match": 1.0},
                correct=True,
            )
        )
        store.seal()
    finally:
        store.close()

    summary = summarize_native_eval_samples(root, tasks=(EvalTaskConfig("gsm8k", 5),))

    assert summary.samples == 1
    assert summary.coverage == {
        "gsm8k_5shot": TaskCoverage(
            n_benchmark=1,
            n_attempted=1,
            n_scored=1,
            n_correct=1,
            n_unanswered=0,
        )
    }
    assert summary.canonical_metrics == {"gsm8k_5shot": {"accuracy": 1.0}}
    assert summary.tasks[0].benchmark is not None
    assert summary.tasks[0].benchmark.primary_metric == "accuracy"


def test_native_summary_partitions_repeated_task_configurations(tmp_path):
    root = str(tmp_path / "run" / "results")
    store = EvaluationStore.open(root, writer_id="evalchemy")
    try:
        for task, score in (("hellaswag_0shot", 0.0), ("hellaswag_10shot", 1.0)):
            store.add_source_artifact(
                f"evalchemy/{task}/native/samples_hellaswag_native.jsonl",
                b'{"doc_id": 0}\n',
                content_type="application/x-ndjson",
            )
            store.add_source_artifact(
                f"evalchemy/{task}/native/results_hellaswag.json",
                json.dumps(
                    {
                        "results": {"hellaswag": {"acc,none": score}},
                        **_result_contract(
                            "hellaswag",
                            "acc",
                            "accuracy",
                            score,
                            n_benchmark=1,
                            n_attempted=1,
                        ),
                    }
                ).encode(),
                content_type="application/json",
            )
            store.add_sample(
                EvalSample(
                    task=task,
                    doc_id="0",
                    kind=SampleKind.MULTIPLE_CHOICE,
                    grading=Grading(method="lm-eval:acc", metric="acc", score=score, passed=bool(score)),
                    metrics={"acc": score},
                    correct=bool(score),
                )
            )
        store.seal()
    finally:
        store.close()

    summary = summarize_native_eval_samples(
        root,
        tasks=(EvalTaskConfig("hellaswag", 0), EvalTaskConfig("hellaswag", 10)),
    )

    assert summary.coverage == {
        "hellaswag_0shot": TaskCoverage(
            n_benchmark=1,
            n_attempted=1,
            n_scored=1,
            n_correct=0,
            n_unanswered=0,
        ),
        "hellaswag_10shot": TaskCoverage(
            n_benchmark=1,
            n_attempted=1,
            n_scored=1,
            n_correct=1,
            n_unanswered=0,
        ),
    }
    assert summary.canonical_metrics == {
        "hellaswag_0shot": {"accuracy": 0.0},
        "hellaswag_10shot": {"accuracy": 1.0},
    }


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


def _result_contract(
    task: str,
    source_metric: str,
    canonical_metric: str,
    value: float,
    *,
    n_benchmark: int,
    n_attempted: int,
    kind: str = "binary",
) -> dict:
    return {
        "benchmark_metadata": {
            task: {
                "schema_version": 1,
                "task": task,
                "primary_metric": canonical_metric,
                "metric_kind": kind,
                "metrics": [
                    {
                        "name": canonical_metric,
                        "source_name": source_metric,
                        "kind": kind,
                        "higher_is_better": True,
                    }
                ],
                "n_benchmark": n_benchmark,
                "n_attempted": n_attempted,
            }
        },
        "canonical_results": {task: {canonical_metric: value}},
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


def test_export_records_full_benchmark_and_intended_cap_for_every_group_leaf(tmp_path):
    results = tmp_path / "run" / "results"
    directory = results / "mmlu_5shot" / "model"
    directory.mkdir(parents=True)
    (directory / "results_20260807.json").write_text(
        json.dumps(
            {
                "results": {"mmlu_anatomy": {"acc,none": 1.0}, "mmlu_astronomy": {"acc,none": 0.0}},
                "benchmark_metadata": {
                    task: _result_contract(task, "acc", "accuracy", value, n_benchmark=size, n_attempted=2)[
                        "benchmark_metadata"
                    ][task]
                    for task, value, size in (("mmlu_anatomy", 1.0, 3), ("mmlu_astronomy", 0.0, 4))
                },
                "canonical_results": {
                    "mmlu_anatomy": {"accuracy": 1.0},
                    "mmlu_astronomy": {"accuracy": 0.0},
                },
            }
        )
    )
    rows = [_lm_eval_row(doc_id, "none", 1.0, "4") for doc_id in range(2)]
    for row in rows:
        row["acc"] = row.pop("exact_match")
    (directory / "samples_mmlu_anatomy_20260807.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    coverage = export_lm_eval_samples(
        str(results),
        tasks=(EvalTaskConfig("mmlu", 5, task_alias="mmlu_5shot"),),
    ).coverage

    assert coverage == {
        "mmlu_5shot/mmlu_anatomy": TaskCoverage(n_benchmark=3, n_attempted=2, n_scored=2, n_correct=2),
        "mmlu_5shot/mmlu_astronomy": TaskCoverage(n_benchmark=4, n_attempted=2, n_scored=0),
    }


def test_export_uses_evaluator_metadata_for_chat_native_task(tmp_path):
    results = tmp_path / "run" / "results"
    directory = results / "math500" / "model"
    directory.mkdir(parents=True)
    (directory / "results_20260807.json").write_text(
        json.dumps(
            {
                "results": {"MATH500": {"accuracy": 1.0}},
                **_result_contract("MATH500", "accuracy", "accuracy", 1.0, n_benchmark=500, n_attempted=10),
            }
        )
    )
    row = _lm_eval_row(0, "none", 1.0, "4")
    row["accuracy"] = row.pop("exact_match")
    (directory / "samples_MATH500_20260807.jsonl").write_text(json.dumps(row) + "\n")

    [coverage] = export_lm_eval_samples(
        str(results),
        tasks=(EvalTaskRef(name="MATH500", num_fewshot=0, task_alias="math500"),),
    ).coverage.values()

    assert coverage.n_benchmark == 500
    assert coverage.n_attempted == 10
    assert coverage.n_scored == 1


def test_export_records_aggregate_only_benchmark_metadata(tmp_path):
    results = tmp_path / "run" / "results"
    directory = results / "aime24" / "model"
    directory.mkdir(parents=True)
    (directory / "results_20260807.json").write_text(
        json.dumps(
            {
                "results": {"AIME24": {"accuracy_avg": 0.4}},
                **_result_contract(
                    "AIME24",
                    "accuracy_avg",
                    "accuracy",
                    0.4,
                    n_benchmark=30,
                    n_attempted=30,
                    kind="continuous",
                ),
            }
        )
    )

    exported = export_lm_eval_samples(
        str(results),
        tasks=(EvalTaskConfig("AIME24", 0, task_alias="aime24"),),
    )

    assert exported.canonical_metrics == {"aime24": {"accuracy": 0.4}}
    assert exported.coverage == {"aime24": TaskCoverage(n_benchmark=30, n_attempted=30, n_scored=0)}
    assert exported.tasks[0].benchmark is not None
    assert exported.tasks[0].benchmark.primary_metric == "accuracy"


def test_declared_aggregate_metric_uses_its_per_sample_base_metric(tmp_path):
    results = tmp_path / "run" / "results"
    directory = results / "aime24" / "model"
    directory.mkdir(parents=True)
    (directory / "results_20260807.json").write_text(
        json.dumps(
            {
                "results": {"aime24": {"accuracy_avg": 1.0}},
                **_result_contract("aime24", "accuracy", "accuracy", 1.0, n_benchmark=1, n_attempted=1),
            }
        )
    )
    row = _lm_eval_row(0, "none", 1.0, "4")
    row.pop("exact_match")
    row["accuracy"] = 1.0
    (directory / "samples_aime24_20260807.jsonl").write_text(json.dumps(row) + "\n")
    task = EvalTaskConfig("aime24", 0, task_alias="aime24")

    export_lm_eval_samples(str(results), tasks=(task,))

    [stored] = ReadView(str(results)).scan("samples").to_pylist(maps_as_pydicts="strict")
    assert sample_from_archive_row(stored).grading.metric == "accuracy"


def test_rebuild_keeps_the_recorded_primary_metric(tmp_path):
    results = tmp_path / "run" / "results"
    row = _lm_eval_row(0, "none", 0.0, "4")
    row["f1"] = 1.0
    source = _write_jsonl(results, [row])
    (source.parent / "results_20260807.json").write_text(
        json.dumps(
            {
                "results": {"gsm8k": {"f1": 1.0}},
                **_result_contract("gsm8k", "f1", "f1", 1.0, n_benchmark=1, n_attempted=1, kind="continuous"),
            }
        )
    )
    task = EvalTaskConfig("gsm8k", 5)
    exported = export_lm_eval_samples(str(results), tasks=(task,))
    source.unlink()

    assert rebuild_lm_eval_samples(str(results), tasks=exported.tasks) == 1
    [stored] = ReadView(str(results)).scan("samples").to_pylist(maps_as_pydicts="strict")
    sample = sample_from_archive_row(stored)
    assert sample.grading.metric == "f1"
    assert sample.correct


def test_rebuild_chat_native_samples_from_recorded_task_declaration(tmp_path):
    results = tmp_path / "run" / "results"
    directory = results / "aime24" / "model"
    directory.mkdir(parents=True)
    result_path = directory / "results_20260807.json"
    result_path.write_text(
        json.dumps(
            {
                "results": {"AIME24": {"accuracy_avg": 1.0}},
                **_result_contract("AIME24", "accuracy", "accuracy", 1.0, n_benchmark=30, n_attempted=30),
            }
        )
    )
    row = _lm_eval_row(0, "none", 1.0, "4")
    row.pop("exact_match")
    row["accuracy"] = 1.0
    source = directory / "samples_AIME24_20260807.jsonl"
    source.write_text(json.dumps(row) + "\n")
    task = EvalTaskRef(
        name="AIME24",
        num_fewshot=0,
        task_alias="aime24",
    )
    exported = export_lm_eval_samples(str(results), tasks=(task,))
    source.unlink()

    assert rebuild_lm_eval_samples(str(results), tasks=exported.tasks) == 1
    [stored] = ReadView(str(results)).scan("samples").to_pylist(maps_as_pydicts="strict")
    assert sample_from_archive_row(stored).grading.metric == "accuracy"


def test_two_sample_files_for_one_leaf_use_one_grouped_coverage_key(tmp_path):
    results = tmp_path / "run" / "results"
    directory = results / "gsm8k_5shot" / "model"
    directory.mkdir(parents=True)
    (directory / "results_20260807.json").write_text(
        json.dumps({"results": {"gsm8k": {"exact_match": 1.0}}, "n-samples": {"gsm8k": {"original": 2}}})
    )
    for timestamp in ("20260807", "20260808"):
        (directory / f"samples_gsm8k_{timestamp}.jsonl").write_text(json.dumps(_lm_eval_row(0, "none", 1.0, "4")) + "\n")

    coverage = export_lm_eval_samples(str(results), tasks=(EvalTaskConfig("gsm8k", 5),)).coverage

    assert set(coverage) == {"gsm8k_5shot/gsm8k"}


def test_export_reads_each_result_payload_once(tmp_path, monkeypatch):
    results = tmp_path / "run" / "results"
    source = _write_jsonl(results, [_lm_eval_row(0, "none", 1.0, "4")])
    result_path = source.parent / "results_20260807.json"
    result_path.write_text(
        json.dumps({"results": {"gsm8k": {"exact_match": 1.0}}, "n-samples": {"gsm8k": {"original": 1}}})
    )
    original = StoragePath.read_bytes
    reads = 0

    def counted_read(path: StoragePath) -> bytes:
        nonlocal reads
        if str(path).endswith("results_20260807.json"):
            reads += 1
        return original(path)

    monkeypatch.setattr(StoragePath, "read_bytes", counted_read)

    export_lm_eval_samples(str(results), tasks=(EvalTaskConfig("gsm8k", 5),))

    assert reads == 1


def test_export_rejects_samples_beyond_the_intended_cap(tmp_path):
    results = tmp_path / "run" / "results"
    directory = results / "gsm8k_5shot" / "model"
    directory.mkdir(parents=True)
    (directory / "results_20260807.json").write_text(
        json.dumps(
            {
                "results": {"gsm8k": {"exact_match,none": 1.0}},
                **_result_contract("gsm8k", "exact_match", "accuracy", 1.0, n_benchmark=10, n_attempted=2),
            }
        )
    )
    rows = [_lm_eval_row(doc_id, "none", 1.0, "4") for doc_id in (0, 2)]
    (directory / "samples_gsm8k_20260807.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    with pytest.raises(ValueError, match="sample document extent 3 exceeds intended count 2"):
        export_lm_eval_samples(
            str(results),
            tasks=(EvalTaskConfig("gsm8k", 5),),
        )


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

    def work(path: str, _records) -> SweepOutcome:
        visited.append(path)
        return SweepOutcome("exported", "0 sample(s)")

    _sweep_archives(
        {shared: ArchiveSweep(run_ids=("run-a", "run-b")), own: ArchiveSweep(run_ids=("run-c",))},
        4,
        work,
    )

    assert sorted(visited) == sorted([shared, own])


def test_naming_an_archive_directly_recovers_only_its_records(tmp_path, monkeypatch):
    # Targeting a handful of damaged archives must not re-sweep every recorded run beside them.
    named = str(tmp_path / "one" / "results")
    matching = SimpleNamespace(results_path=named, run_id="run-a")
    other = SimpleNamespace(results_path=str(tmp_path / "other" / "results"), run_id="run-b")
    monkeypatch.setattr(
        "experiments.evaluation.migrations.cli.list_records",
        lambda prefix: (matching, other) if prefix == DEFAULT_SCAN_PREFIXES[0] else (),
    )

    prefixes = _resolve_prefixes((), (named + "/",))

    assert prefixes == tuple(DEFAULT_SCAN_PREFIXES)
    assert selected_archives(prefixes, (named + "/",)) == {named: ArchiveSweep(records=(matching,), run_ids=("run-a",))}


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
    assert json.loads(reader.read_blob(uri.removeprefix("finestore://blobs/")))["steps"][0]["step_id"] == 1


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
