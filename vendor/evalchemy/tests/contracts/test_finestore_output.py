"""Contract tests for Evalchemy's FineStore output."""

from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import shutil
from types import SimpleNamespace
from threading import Event

import pytest
from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView
from finestore.shard_writer import ShardWriter
from finestore.store import DataStore
from sympy import Integer

from eval.contracts.finestore_output import FineStoreOutput, completed_finestore_output, read_finestore_output
from eval.contracts.finestore_resume import FineStoreResumeManager, ResumeRefused
from eval.contracts.lm_eval_normalization import sample_from_lm_eval
from eval.contracts.task_outcome import TaskOutcome, TaskRoute, TaskStatus
from eval.eval import cli_evaluate, handle_evaluation_output, setup_custom_parser
from lm_eval.loggers.evaluation_tracker import GeneralConfigTracker
from eval.lm_eval_compat import setup_parser
from eval.serve_eval.results import EvalResults
from eval.resume.fingerprint import RunFingerprint
from eval.resume import wiring
from eval.resume.wiring import build_resume_wiring
from eval.execution.run_spec import RunSpec
from eval.runtime.checkpoints import CheckpointRow, ResumeStream, input_digest
from eval.graders.answer_equivalence import JudgeConfig


def _write_result(root, source_prefix, results, samples_by_task):
    if completed_finestore_output(root, source_prefix):
        return
    with FineStoreOutput(root, source_prefix, tuple(samples_by_task)) as output:
        for task_name, records in samples_by_task.items():
            for record in records:
                output.write_sample(task_name, record)
            output.finish_task()
        output.finish(results)


@pytest.mark.parametrize("option", ["--output_path", "--finestore_output_path"])
def test_output_path_alias_uses_finestore_destination(option):
    args = setup_parser().parse_args([option, "logs/my-run"])

    assert args.finestore_output_path == "logs/my-run"


def test_lm_eval_normalization_maps_multiple_choice_scores():
    sample = sample_from_lm_eval(
        "arc_easy",
        {
            "doc_id": 3,
            "doc": {"choices": {"label": ["A", "B"], "text": ["3", "4"]}},
            "target": "B",
            "arguments": [["2 + 2 =", "3"], ["2 + 2 =", "4"]],
            "resps": [[-2.0, False], [-0.1, True]],
            "filtered_resps": [],
            "filter": "none",
            "metrics": ["acc_norm"],
            "acc_norm": 1.0,
        },
    )

    assert sample.prompt_text == "2 + 2 ="
    assert sample.model_choice == 1
    assert sample.target_choice == 1
    assert [(choice.label, choice.text, choice.loglikelihood) for choice in sample.choices or []] == [
        ("A", "3", -2.0),
        ("B", "4", -0.1),
    ]
    assert sample.correct is True


def test_normalized_metrics_exclude_sample_identity_coordinates():
    sample = sample_from_lm_eval(
        "MATH500",
        {
            "schema_version": 1,
            "task_name": "MATH500",
            "doc_id": 0,
            "doc": {"problem": "1 + 1"},
            "target": "2",
            "arguments": [["1 + 1", {}]],
            "resps": [["2"]],
            "filtered_resps": ["2"],
            "filter": "none",
            "doc_hash": "doc",
            "prompt_hash": "prompt",
            "target_hash": "target",
            "sample_id": "abc",
            "source_id": 3,
            "sample_ordinal": 7,
            "sample_namespace": "MATH500",
            "sample_shard": None,
            "sample_repeat": 2,
            "metrics": ["accuracy"],
            "accuracy": 1.0,
        },
    )

    assert sample.metrics == {"accuracy": 1.0}
    assert sample.grading is not None
    assert (sample.grading.metric, sample.grading.score) == ("accuracy", 1.0)


def test_normalization_marks_transport_failures_as_infrastructure_errors():
    sample = sample_from_lm_eval(
        "FinanceBench",
        {
            "doc_id": 0,
            "doc": {"question": "question"},
            "target": "answer",
            "arguments": [["question", {}]],
            "resps": [[""]],
            "filtered_resps": [""],
            "filter": "none",
            "failure_category": "model_transport",
            "metrics": ["accuracy"],
            "accuracy": 0.0,
        },
    )

    assert sample.output == "[EVALCHEMY_INFRASTRUCTURE_ERROR] model_transport"
    assert sample.extracted == "[EVALCHEMY_INFRASTRUCTURE_ERROR] model_transport"
    assert sample.metrics == {"accuracy": 0.0}


def test_normalization_does_not_mark_malformed_model_output_as_infrastructure():
    sample = sample_from_lm_eval(
        "task",
        {
            "doc_id": 0,
            "doc": {},
            "target": "answer",
            "arguments": [["question", {}]],
            "resps": [[""]],
            "filtered_resps": [""],
            "filter": "none",
            "failure_category": "malformed_model_response",
            "metrics": ["accuracy"],
            "accuracy": 0.0,
        },
    )

    assert sample.output == ""
    assert sample.extracted == ""


@pytest.mark.parametrize("metric,score", [(None, None), ("accuracy", 0.0), ("score", 0.5)])
def test_finestore_judge_failure_preserves_error_without_scoring_the_failure_counter(tmp_path, metric, score):
    judge_error = {
        "category": "grader_infrastructure", "message": "Judge endpoint unavailable", "exception_type": "ConnectionError",
    }
    record = {
        "doc_id": 0, "doc": {"question": "question"}, "target": "answer",
        "arguments": [["question", {}]], "resps": [["candidate"]], "filtered_resps": ["candidate"],
        "metrics": ["judge_failed"], "judge_failed": 1.0,
        "failure_category": "grader_infrastructure", "judge_error": judge_error,
        "judge_label": None, "judge_raw": None,
    }
    if metric is not None:
        record.update({metric: score, "metrics": [metric, "judge_failed"]})
    root = str(tmp_path / "archive")

    _write_result(root, "run", {"results": {"task": {"accuracy": score}}}, {"task": [record]})

    table = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    sample = sample_from_archive_row(table.to_pylist(maps_as_pydicts="strict")[0])
    assert sample.output == "candidate"
    assert sample.extracted == "candidate"
    assert sample.metrics == ({"judge_failed": 1.0} if metric is None else {"judge_failed": 1.0, metric: score})
    assert sample.correct is (None if metric is None else False)
    assert sample.grading is not None
    assert sample.grading.metric == metric
    assert sample.grading.score == score
    assert sample.grading.passed is sample.correct
    assert json.loads(sample.grading.detail) == {
        "failure_category": "grader_infrastructure", "judge_error": judge_error,
        "judge_label": None, "judge_raw": None,
    }


def test_finestore_output_preserves_repeated_trials(tmp_path: Path):
    root = str(tmp_path / "archive")
    records = [
        {
            "doc_id": 7,
            "doc": {"question": "Choose A"},
            "target": "A",
            "arguments": [["Choose A", {}]],
            "resps": [[answer]],
            "filtered_resps": [answer],
            "filter": "none",
            "sample_repeat": repeat,
            "metrics": ["accuracy"],
            "accuracy": score,
        }
        for repeat, (answer, score) in enumerate((("A", 1.0), ("B", 0.0), ("A", 1.0)))
    ]

    _write_result(
        root,
        "gpqa_diamond",
        {"results": {"GPQADiamond": {"accuracy_avg": 2 / 3}}},
        {"GPQADiamond": records},
    )

    table = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    rows = sorted(table.to_pylist(maps_as_pydicts="strict"), key=lambda row: row["trial_id"])
    assert [row["trial_id"] for row in rows] == ["0", "1", "2"]
    assert [sample_from_archive_row(row).correct for row in rows] == [True, False, True]


def test_finestore_output_expands_filter_variants(
    tmp_path: Path,
):
    root = str(tmp_path / "archive")
    record = {
        "schema_version": 1,
        "task_name": "gsm8k",
        "doc_id": 7,
        "doc": {"question": "What is 40 + 2?"},
        "target": "42",
        "arguments": [["What is 40 + 2?", {}]],
        "resps": [["The answer is 42"]],
        "filtered_resps": ["42"],
        "filter": "strict-match",
        "metrics": ["exact_match"],
        "filter_variants": [
            {
                "filter": "strict-match",
                "filtered_resps": ["42"],
                "metrics": {"exact_match,strict-match": 0.0},
            },
            {
                "filter": "flexible-extract",
                "filtered_resps": ["42"],
                "metrics": {"exact_match,flexible-extract": 1.0},
            },
        ],
    }

    results = {"results": {"gsm8k": {"exact_match": 1.0}}}
    _write_result(root, "gsm8k_5shot", results, {"gsm8k": [record]})

    view = ReadView(root)
    table = view.scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    rows = table.to_pylist(maps_as_pydicts="strict")
    samples = {row["filter"]: sample_from_archive_row(row) for row in rows}
    assert set(samples) == {"strict-match", "flexible-extract"}
    assert samples["strict-match"].correct is False
    assert samples["flexible-extract"].correct is True

    assert read_finestore_output(root, "gsm8k_5shot") == results
    assert view.read_blob("sources/evalchemy/gsm8k_5shot/native/samples_gsm8k_native.jsonl") is None


def test_finestore_output_keeps_repeated_task_configurations_distinct(tmp_path: Path):
    root = str(tmp_path / "archive")
    record = {
        "doc_id": 7,
        "doc": {"query": "Complete the sentence"},
        "target": "answer",
        "arguments": [["Complete the sentence"]],
        "resps": [["answer"]],
        "filtered_resps": ["answer"],
        "filter": "none",
        "acc,none": 1.0,
    }
    results = {"results": {"hellaswag": {"acc,none": 1.0}}}

    _write_result(root, "hellaswag_0shot", results, {"hellaswag": [record]})
    _write_result(root, "hellaswag_10shot", results, {"hellaswag": [record]})

    table = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    assert set(table.column("task").to_pylist()) == {"hellaswag_0shot", "hellaswag_10shot"}


def test_finestore_resume_restores_committed_units(tmp_path: Path):
    root = str(tmp_path / "archive")
    fingerprint = RunFingerprint(inputs={"task_name": "gsm8k", "model_repo": "test-model"})
    with DataStore.open(root) as store:
        first = FineStoreResumeManager(store, "gsm8k_0shot", "gsm8k", fingerprint)
        assert first.decide() == "fresh"
        identity = {"source_id": "0"}
        first.record(CheckpointRow(0, input_digest(identity), {"phase": "generate"}, {"input_digest": "input", "output": "42"}))
        first.finalize()

        changed = FineStoreResumeManager(store, "gsm8k_0shot", "gsm8k", RunFingerprint(inputs={"task_name": "other"}))
        with pytest.raises(ResumeRefused):
            changed.decide()

        retry = FineStoreResumeManager(store, "gsm8k_0shot", "gsm8k", fingerprint)
        assert retry.decide() == "resume"
        restored = ResumeStream(retry).admit(0, identity)
        assert restored.lookup({"phase": "generate"})["output"] == "42"
        retry.record(CheckpointRow(1, input_digest({"source_id": "1"}), {"phase": "generate"}, {"input_digest": "input", "output": "43"}))
        retry.finalize()

        assert [row.payload["output"] for row in FineStoreResumeManager(store, "gsm8k_0shot", "gsm8k", fingerprint).records()] == ["42", "43"]

def test_finestore_resume_refuses_rank_layout_change_and_force_fresh(tmp_path: Path):
    root = str(tmp_path / "archive")
    fingerprint = RunFingerprint(inputs={"task_name": "gsm8k"})
    with DataStore.open(root) as store:
        first = FineStoreResumeManager(store, "run", "gsm8k", fingerprint)
        assert first.decide() == "fresh"
        first.finalize()

        with pytest.raises(ResumeRefused, match="fingerprint changed"):
            FineStoreResumeManager(store, "run", "gsm8k", fingerprint, world_size=2).decide()
        with pytest.raises(ResumeRefused, match="already exists"):
            FineStoreResumeManager(store, "run", "gsm8k", fingerprint, mode="force-fresh").decide()


def test_finestore_resume_refuses_result_without_fingerprint(tmp_path: Path):
    root = str(tmp_path / "archive")
    _write_result(root, "run", {"results": {"gsm8k": {"exact_match": 0.5}}}, {})

    with DataStore.open(root) as store:
        with pytest.raises(ResumeRefused, match="without a resume fingerprint"):
            FineStoreResumeManager(store, "run", "gsm8k", RunFingerprint(inputs={"task_name": "gsm8k"})).decide()


def test_finestore_output_preserves_completed_task_after_archive_reopens(tmp_path: Path):
    root = str(tmp_path / "archive")
    results = {"results": {"gsm8k": {"exact_match": 0.5}}}
    _write_result(root, "gsm8k_0shot", results, {})
    committed = ReadView(root)
    assert completed_finestore_output(root, "gsm8k_0shot")
    assert not committed.is_sealed()
    _write_result(root, "gsm8k_0shot", {"results": {"gsm8k": {"exact_match": 0.0}}}, {})
    _write_result(
        root,
        "gsm8k_0shot",
        {"results": {}, "task_outcomes": {"gsm8k": {"status": "failed"}}},
        {},
    )
    assert read_finestore_output(root, "gsm8k_0shot") == results
    assert ReadView(root).token == committed.token
    with DataStore.open(root) as store:
        other_task = FineStoreResumeManager(store, "arc_0shot", "arc", RunFingerprint(inputs={"task_name": "arc"}))
        other_task.record(CheckpointRow(0, input_digest({"source_id": "0"}), {"phase": "generate"}, {"input_digest": "input", "output": "A"}))
        other_task.finalize()
        assert not ReadView(root).is_sealed()
        assert completed_finestore_output(root, "gsm8k_0shot")


def _triviaqa_record(doc_id: int) -> dict:
    return {
        "doc_id": doc_id,
        "doc": {"question": str(doc_id)},
        "target": "answer",
        "arguments": [[str(doc_id), {}]],
        "resps": [["answer"]],
        "filtered_resps": ["answer"],
        "filter": "none",
        "metrics": ["exact_match"],
        "exact_match": 1.0,
    }


def test_staged_samples_survive_interruption_and_restart(tmp_path: Path):
    root = str(tmp_path / "archive")
    results = {"results": {"triviaqa": {"exact_match": 1.0}}}

    with pytest.raises(MemoryError):
        with FineStoreOutput(root, "triviaqa", ["triviaqa"]) as output:
            for doc_id in range(4):
                output.write_sample("triviaqa", _triviaqa_record(doc_id))
            output.finish_task()
            raise MemoryError("simulated evaluator interruption")

    assert read_finestore_output(root, "triviaqa") is None
    partial = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert partial is not None
    assert {row["doc_id"] for row in partial.to_pylist(maps_as_pydicts="strict")} == {str(i) for i in range(4)}

    with FineStoreOutput(root, "triviaqa", ["triviaqa"]) as output:
        for doc_id in range(6):
            output.write_sample("triviaqa", _triviaqa_record(doc_id))
        output.finish_task()
        output.finish(results)

    rows = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert rows is not None
    assert sorted(row["doc_id"] for row in rows.to_pylist(maps_as_pydicts="strict")) == [str(i) for i in range(6)]
    assert read_finestore_output(root, "triviaqa") == results


def test_large_task_publishes_sample_progress_before_result(tmp_path: Path):
    root = str(tmp_path / "archive")
    with FineStoreOutput(root, "triviaqa", ["triviaqa"]) as output:
        for doc_id in range(1024):
            output.write_sample("triviaqa", _triviaqa_record(doc_id))
            if (doc_id + 1) % 128 == 0:
                output.finish_task()
            if doc_id == 511:
                halfway = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
                assert halfway is not None
                assert halfway.num_rows == 512
                assert read_finestore_output(root, "triviaqa") is None
        output.finish({"results": {"triviaqa": {"exact_match": 1.0}}})

    complete = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert complete is not None
    assert complete.num_rows == 1024


def test_sample_admission_waits_for_stalled_storage_and_preserves_records(tmp_path: Path, monkeypatch):
    """A fast replay must not queue the remaining cohort behind a slow shard."""
    writer_entered = Event()
    release_writer = Event()
    producer_started = Event()
    producer_finished = Event()
    completed = []
    original_write = ShardWriter.write_table

    def stalled_write(writer, table):
        writer_entered.set()
        if not release_writer.wait(timeout=10):
            raise TimeoutError("test did not release its filesystem writer")
        original_write(writer, table)

    # Scale only the byte budget; the DataStore, Arrow conversion, Parquet
    # writer, manifest commits and subsequent reader remain real.
    monkeypatch.setattr("eval.contracts.finestore_output._SAMPLE_BUFFER_BYTES", 1024)
    monkeypatch.setattr(ShardWriter, "write_table", stalled_write)
    root = str(tmp_path / "archive")
    expected = [_triviaqa_record(index) for index in range(17)]
    for record in expected[1:]:
        record["doc"]["question"] = "large restored source document " * 200

    with FineStoreOutput(root, "triviaqa", ["triviaqa"]) as output, ThreadPoolExecutor(max_workers=2) as pool:
        output.write_sample("triviaqa", expected[0])
        writing = pool.submit(output.store.flush)

        def produce():
            producer_started.set()
            for record in expected[1:]:
                output.write_sample("triviaqa", record)
                completed.append(record["doc_id"])
            producer_finished.set()

        try:
            assert writer_entered.wait(timeout=5)
            producing = pool.submit(produce)
            assert producer_started.wait(timeout=5)
            # Observe producer completion while its real filesystem writer is
            # stalled; bounded admission must wait instead of buffering all rows.
            assert not producer_finished.wait(timeout=0.5)
            assert completed == []
        finally:
            release_writer.set()
        writing.result(timeout=10)
        producing.result(timeout=10)
        output.finish({"results": {"triviaqa": {"exact_match": 1.0}}})

    table = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    persisted = {row["doc_id"]: sample_from_archive_row(row) for row in table.to_pylist(maps_as_pydicts="strict")}
    assert set(persisted) == {str(record["doc_id"]) for record in expected}
    for record in expected:
        sample = persisted[str(record["doc_id"])]
        assert json.loads(sample.doc) == record["doc"]
        assert sample.output == "answer"
        assert sample.metrics == {"exact_match": 1.0}
    assert read_finestore_output(root, "triviaqa")["results"]["triviaqa"] == {"exact_match": 1.0}


def test_grouped_lm_eval_children_keep_distinct_sample_keys(tmp_path: Path):
    root = str(tmp_path / "archive")
    with FineStoreOutput(root, "knowledge_group", ["knowledge_group"]) as output:
        output.write_sample("arc_easy", _triviaqa_record(0))
        output.write_sample("arc_challenge", _triviaqa_record(0))
        output.finish_task()

    table = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE)
    assert table is not None
    assert set(table.column("task").to_pylist()) == {
        "knowledge_group/arc_easy",
        "knowledge_group/arc_challenge",
    }

def test_failed_finestore_result_can_be_replaced_by_success(tmp_path: Path):
    root = str(tmp_path / "archive")
    failure = {"results": {}, "task_outcomes": {"gsm8k": {"status": "failed"}}}
    success = {"results": {"gsm8k": {"exact_match": 0.5}}}

    _write_result(root, "gsm8k_0shot", failure, {})
    assert not completed_finestore_output(root, "gsm8k_0shot")
    _write_result(root, "gsm8k_0shot", success, {})

    assert completed_finestore_output(root, "gsm8k_0shot")
    assert read_finestore_output(root, "gsm8k_0shot") == success


@pytest.mark.parametrize("bits", [14_500, 15_001])
def test_finestore_output_serializes_large_sympy_result(tmp_path: Path, bits: int):
    root = str(tmp_path / "archive")
    results = {
        "results": {"gsm8k": {"exact_match": 1.0}},
        "config": {"large_integer": Integer(2) ** bits},
    }

    _write_result(root, "gsm8k_0shot", results, {})

    restored = read_finestore_output(root, "gsm8k_0shot")
    assert restored is not None
    assert restored["results"] == results["results"]
    assert restored["config"]["large_integer"].startswith("<Integer ~")


def test_sample_normalization_failure_does_not_commit_success(tmp_path: Path, monkeypatch):
    root = str(tmp_path / "archive")
    result = {"results": {"gsm8k": {"exact_match": 1.0}}}

    def fail_normalization(task_name, record):
        raise ValueError("invalid sample")

    monkeypatch.setattr("eval.contracts.finestore_output.samples_from_lm_eval", fail_normalization)
    with pytest.raises(ValueError, match="invalid sample"):
        _write_result(root, "gsm8k_0shot", result, {"gsm8k": [{}]})

    assert read_finestore_output(root, "gsm8k_0shot") is None

@pytest.mark.parametrize(
    ("changed_field", "changed_value", "message"),
    [
        ("model_args", "model=other", "fingerprint changed"),
        ("gen_kwargs", "temperature=0.5", "fingerprint changed"),
        ("gen_kwargs", "top_k=50", "fingerprint changed"),
        ("resume_mode", "force-fresh", "already exists"),
    ],
)
def test_cli_refuses_reuse_of_completed_finestore_result(
    tmp_path: Path, monkeypatch, changed_field: str, changed_value: str, message: str
):
    root = str(tmp_path / "archive")
    args = setup_custom_parser().parse_args(
        ["--model", "local-completions", "--tasks", "arc_challenge", "--model_args", "model=served", "--output_path", root]
    )
    model = SimpleNamespace(world_size=1, rank=0, chat_template=lambda _enabled: None)
    with DataStore.open(root) as store:
        manager = build_resume_wiring(args, model, RunSpec.from_args(args), store=store)("arc_challenge")
        assert manager.decide() == "fresh"
        manager.finalize()
        result = {"results": {"arc_challenge": {"acc": 0.5}}, "config": {"tasks": "arc_challenge"}}
        _write_result(root, args.finestore_output_prefix, result, {})

        setattr(args, changed_field, changed_value)
        with pytest.raises(ResumeRefused, match=message):
            build_resume_wiring(args, model, RunSpec.from_args(args), store=store)("arc_challenge").decide()
        assert read_finestore_output(root, args.finestore_output_prefix) == result


@pytest.mark.parametrize("installed_layout", [False, True])
def test_finestore_cli_wiring_resumes_without_local_output_path(tmp_path: Path, monkeypatch, installed_layout):
    if installed_layout:
        import eval.resume.wiring as wiring

        package = tmp_path / "site-packages" / "eval"
        source = Path(wiring.__file__).resolve().parents[1]
        for directory in ("execution", "runtime", "sources", "graders"):
            shutil.copytree(source / directory, package / directory, ignore=shutil.ignore_patterns("__pycache__"))
        monkeypatch.setattr(wiring, "__file__", str(package / "resume" / "wiring.py"))
    root = str(tmp_path / "archive")
    args = Namespace(
        model="local-completions",
        resume_mode="auto",
        output_path=None,
        finestore_output_path=root,
        finestore_output_prefix="gsm8k_0shot",
        model_args="model=test-model",
        include_path=None,
    )
    model = Namespace(world_size=1, rank=0, chat_template=lambda _enabled: None)
    run = RunSpec()
    with DataStore.open(root) as store:
        factory = build_resume_wiring(args, model, run, store=store)
        first = factory("gsm8k")
        identity = {"source_id": "0"}
        first.record(CheckpointRow(0, input_digest(identity), {"phase": "generate"}, {"input_digest": "input", "output": "42"}))
        first.finalize()

        retry = build_resume_wiring(args, model, run, store=store)("gsm8k")
        assert ResumeStream(retry).admit(0, identity).lookup({"phase": "generate"})["output"] == "42"


def test_resume_provenance_ignores_package_builds_and_tracks_runtime_source(tmp_path: Path, monkeypatch):
    package = tmp_path / "source" / "eval"
    (package / "runtime").mkdir(parents=True)
    runtime_source = package / "runtime" / "source.py"
    runtime_source.write_text("generation_protocol = 1\n")
    monkeypatch.setattr(wiring, "__file__", str(package / "resume" / "wiring.py"))
    args = Namespace(
        model="local-completions", model_args="model=test-model", include_path=None,
        resume_mode="auto", finestore_output_prefix="gsm8k_0shot",
    )
    model = Namespace(chat_template=lambda _enabled: None)
    run = RunSpec()
    root = str(tmp_path / "archive")
    fingerprint_blob = "evalchemy/gsm8k_0shot/gsm8k/resume/fingerprint.json"
    with DataStore.open(root) as store:
        first = build_resume_wiring(args, model, run, store=store)("gsm8k")
        identity = {"source_id": "0"}
        first.record(CheckpointRow(0, input_digest(identity), {"phase": "generate"},
                                   {"input_digest": "input", "output": "42"}))
        first.finalize()
        original = json.loads(ReadView(root).read_blob(fingerprint_blob))["provenance"]["implementation"]

        generated = package / "chat_benchmarks" / "optional" / "build" / "lib" / "generated.py"
        generated.parent.mkdir(parents=True)
        generated.write_text("generated_build_copy = 1\n")
        retry = build_resume_wiring(args, model, run, store=store)("gsm8k")
        assert ResumeStream(retry).admit(0, identity).lookup({"phase": "generate"})["output"] == "42"

    rebuilt_root = str(tmp_path / "rebuilt")
    with DataStore.open(rebuilt_root) as store:
        build_resume_wiring(args, model, run, store=store)("gsm8k").decide()
    rebuilt = json.loads(ReadView(rebuilt_root).read_blob(fingerprint_blob))["provenance"]["implementation"]
    assert rebuilt == original

    runtime_source.write_text("generation_protocol = 2\n")
    changed_root = str(tmp_path / "changed")
    with DataStore.open(changed_root) as store:
        build_resume_wiring(args, model, run, store=store)("gsm8k").decide()
    changed = json.loads(ReadView(changed_root).read_blob(fingerprint_blob))["provenance"]["implementation"]
    assert changed != original


def test_finestore_cli_refuses_resume_off(tmp_path: Path):
    args = Namespace(finestore_output_path=str(tmp_path / "archive"), resume_mode="off")
    with pytest.raises(ValueError, match="FineStore output requires"):
        cli_evaluate(args)


@pytest.mark.parametrize("change", [{"model": "other"}, {"base_url": "https://other.example/v1"}])
def test_judge_identity_change_refuses_checkpoint_reuse(tmp_path, change):
    args = setup_custom_parser().parse_args([
        "--tasks", "MTBench", "--model_args", "model=served", "--output_path", str(tmp_path),
    ])
    model = SimpleNamespace(chat_template=lambda _enabled: None)
    config = {"model": "judge", "base_url": "https://judge.example/v1", "api_key": "key"}
    run = RunSpec.from_args(args)
    with DataStore.open(args.finestore_output_path) as store:
        first = build_resume_wiring(args, model, run, judge_configs={"MTBench": JudgeConfig(**config)}, store=store)("MTBench")
        first.decide()
        first.finalize()
        changed = JudgeConfig(**{**config, **change})
        with pytest.raises(ResumeRefused, match="fingerprint changed"):
            build_resume_wiring(args, model, run, judge_configs={"MTBench": changed}, store=store)("MTBench").decide()


def test_finestore_mode_writes_no_local_result_files(tmp_path: Path):
    archive_root = tmp_path / "archive"
    record = {
        "doc_id": 0,
        "doc": {"question": "2 + 2?"},
        "target": "4",
        "arguments": [["2 + 2?"]],
        "resps": [["4"]],
        "filtered_resps": ["4"],
        "exact_match,none": 1.0,
        "doc_hash": "doc",
        "prompt_hash": "prompt",
        "target_hash": "target",
    }
    outcome = TaskOutcome(
        task_name="gsm8k",
        route=TaskRoute.LM_EVAL,
        status=TaskStatus.SUCCEEDED,
        metrics={"exact_match,none": 1.0},
        expected_count=1,
        generated_count=1,
        scored_count=1,
    )
    results = {
        "results": {"gsm8k": {"exact_match,none": 1.0}},
        "samples": {"gsm8k": [record]},
        "task_outcomes": {"gsm8k": outcome.to_dict()},
        "config": {"batch_sizes": [1]},
    }
    args = Namespace(
        show_config=False,
        debug=False,
        model="local-completions",
        model_args="model=test",
        gen_kwargs=None,
        num_fewshot=0,
        max_length=None,
        max_tokens=None,
        limit=None,
        annotator_model="auto",
        batch_size="1",
        finestore_output_path=str(archive_root),
        finestore_output_prefix="gsm8k_0shot",
    )
    tracker = GeneralConfigTracker()
    tracker.model_name = "test-model"
    tracker.model_source = "local-completions"
    tracker.model_name_sanitized = "test-model"

    samples = results.pop("samples")
    with FineStoreOutput(str(archive_root), args.finestore_output_prefix, ["gsm8k"]) as output:
        for sample in samples["gsm8k"]:
            output.write_sample("gsm8k", sample)
        output.finish_task()
        handle_evaluation_output(results, args, tracker, output_sink=output)

    assert list(tmp_path.rglob("samples_*.jsonl")) == []
    assert list(tmp_path.rglob("results_*.json")) == []
    loaded = EvalResults.load_archive(str(archive_root), "gsm8k_0shot")
    assert loaded.results == {"gsm8k": {"exact_match,none": 1.0}}
    assert loaded.model_name == "test-model"
    assert loaded.model_source == "local-completions"
    assert ReadView(str(archive_root)).read_blob("sources/evalchemy/gsm8k_0shot/native/samples_gsm8k_native.jsonl") is None
