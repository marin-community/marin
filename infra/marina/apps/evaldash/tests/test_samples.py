# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import json

import pyarrow as pa
import pyarrow.parquet as pq
from evaldash.marin_evaluation.archive import (
    SAMPLES_MERGE_KEY,
    Choice,
    EvalSample,
    EvaluationStore,
    SampleKind,
    sample_to_archive_row,
    write_sample_parquet,
)
from evaldash.marin_evaluation.lm_eval_samples import export_lm_eval_samples, sample_from_lm_eval
from evaldash.samples import fetch_artifact, fetch_samples, list_sample_tasks
from fsspec.core import url_to_fs


def test_sample_reader_returns_typed_filtered_page(tmp_path) -> None:
    fs, root = url_to_fs(str(tmp_path))
    samples = [
        EvalSample(
            task="arc",
            doc_id="correct",
            kind=SampleKind.GENERATION,
            prompt_text="Question one",
            output="A",
            target_text="A",
            metrics={"acc_norm,none": 1.0},
            correct=True,
        ),
        EvalSample(
            task="arc",
            doc_id="incorrect",
            kind=SampleKind.GENERATION,
            prompt_text="Question two",
            output="B",
            target_text="A",
            metrics={"acc_norm,none": 0.0},
            correct=False,
        ),
    ]
    write_sample_parquet(fs, f"{root}/samples_arc_20260719.parquet", samples)

    tasks = list_sample_tasks(str(tmp_path))
    page = fetch_samples(str(tmp_path), "arc", offset=0, limit=1, correct="incorrect")

    assert tasks.model_dump(mode="json") == {
        "available": True,
        "error": None,
        "tasks": [{"task": "arc", "files": 1}],
    }
    assert page.primary_metric == "acc_norm,none"
    assert page.counts.model_dump() == {"all": 2, "correct": 1, "incorrect": 1, "ungraded": 0}
    assert page.total == 1
    assert page.offset == 0
    assert page.limit == 1
    assert [row.doc_id for row in page.rows] == ["incorrect"]


def test_grading_is_derived_and_round_trips(tmp_path) -> None:
    fs, root = url_to_fs(str(tmp_path))
    sample = sample_from_lm_eval(
        "gsm8k",
        {
            "doc_id": 3,
            "arguments": [["2+2?", " 4"]],
            "resps": [[" 4"]],
            "target": "4",
            "exact_match,flexible-extract": 1.0,
        },
    )
    write_sample_parquet(fs, f"{root}/samples_gsm8k_20260723.parquet", [sample])

    page = fetch_samples(str(tmp_path), "gsm8k", offset=0, limit=1, correct="all")

    (row,) = page.rows
    assert row.grading is not None
    assert row.grading.method == "lm-eval:exact_match"
    assert row.grading.metric == "exact_match,flexible-extract"
    assert row.grading.filter == "flexible-extract"
    assert row.grading.passed is True


def test_artifact_fetch_returns_run_local_object(tmp_path) -> None:
    results = tmp_path / "results"
    trajectory = results / "trajectories" / "aime_68.json"
    trajectory.parent.mkdir(parents=True)
    trajectory.write_text('{"steps": []}')

    artifact = fetch_artifact(str(results), str(trajectory))

    assert artifact.available is True
    assert artifact.reason is None
    assert artifact.media_type == "application/json"
    assert artifact.truncated is False
    assert artifact.text == '{"steps": []}'


def test_artifact_fetch_rejects_out_of_tree_uri(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()
    secret = tmp_path / "secret.json"
    secret.write_text('{"secret": true}')

    artifact = fetch_artifact(str(results), str(secret))

    assert artifact.available is False
    assert artifact.text is None
    assert "outside the run results directory" in (artifact.reason or "")


def test_artifact_fetch_rejects_parent_traversal(tmp_path) -> None:
    results = tmp_path / "results"
    (results / "trajectories").mkdir(parents=True)

    artifact = fetch_artifact(str(results), f"{results}/trajectories/../../secret.json")

    assert artifact.available is False
    assert "outside the run results directory" in (artifact.reason or "")


def test_artifact_fetch_missing_object_degrades(tmp_path) -> None:
    results = tmp_path / "results"
    results.mkdir()

    artifact = fetch_artifact(str(results), f"{results}/trajectories/absent.json")

    assert artifact.available is False
    assert artifact.text is None
    assert artifact.reason is not None


def test_artifact_fetch_enforces_size_cap(tmp_path) -> None:
    results = tmp_path / "results"
    big = results / "trajectories" / "big.json"
    big.parent.mkdir(parents=True)
    big.write_text("x" * 4096)

    artifact = fetch_artifact(str(results), str(big), max_bytes=1024)

    assert artifact.available is False
    assert artifact.truncated is True
    assert artifact.text is None


def test_artifact_fetch_without_results_path_degrades() -> None:
    artifact = fetch_artifact(None, "gs://bucket/runs/x/results/trajectories/aime_68.json")

    assert artifact.available is False
    assert artifact.reason == "run has no results_path"


# The tests below were part of the eval-archive suite in the main repository; they assert what this
# reader makes of an archive the eval runners wrote, so they live with the reader.


def _mcq(doc_id: str, *, correct: bool) -> EvalSample:
    return EvalSample(
        task="arc",
        doc_id=doc_id,
        kind=SampleKind.MULTIPLE_CHOICE,
        prompt_text="Q?",
        choices=[Choice(label="A", text="a", loglikelihood=-1.0), Choice(label="B", text="b", loglikelihood=-2.0)],
        model_choice=0,
        target_choice=0 if correct else 1,
        metrics={"acc": 1.0 if correct else 0.0},
        correct=correct,
    )


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
