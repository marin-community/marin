"""Task outcomes from the single execution lifecycle."""

import asyncio
import json
from argparse import Namespace
from dataclasses import replace

import pytest
from eval.chat_benchmarks.FinanceBench.eval_instruct import FinanceBenchBenchmark
from eval.chat_benchmarks.JEEBench.eval_instruct import JEEBenchBenchmark
from eval.chat_benchmarks.MixEval.eval_instruct import MixEvalBenchmark
from eval.chat_benchmarks.SimpleQA.eval_instruct import SimpleQABenchmark
from eval.chat_benchmarks.zeroeval.eval_instruct import ZeroEvalBenchmark
from eval.completion_response import (
    CompletionContentPolicy,
    CompletionText,
    FailedGeneration,
    completion_response_from_chat_choice,
)
from eval.contracts.eval_spec import EvalSpec, Grade, GraderSpec, ModelRequest, RequestKind, SourceRef
from eval.contracts.failures import FailureCategory
from eval.contracts.finestore_output import FineStoreOutput
from eval.contracts.task_outcome import EvaluationRunError, TaskRoute, TaskStatus, validate_result_document
from eval.eval import evaluate, handle_evaluation_output
from lm_eval.loggers.evaluation_tracker import GeneralConfigTracker
from eval.execution.driver import SourcePlan, run_evaluations, source_plans
from eval.execution.grading import Grader
from eval.execution.metrics import SourceMetrics
from eval.execution.run_spec import RunSpec
from eval.graders.answer_equivalence import JudgeConfig
from eval.serve_eval.results import EvalResults
from eval.task import BaseBenchmark
from finestore.eval import ARCHIVE_SAMPLES_TABLE
from finestore.reader import ReadView
from tests.contracts.source_harness import FixedEndpoint, MemoryCheckpoints


class Model:
    world_size = 1
    rank = 0

    def __init__(self, outputs):
        self.outputs = iter(outputs)

    def tok_encode(self, text):
        return list(text.encode())

    def apply_chat_template(self, messages, **_kwargs):
        return str(messages)

    def generate_until(self, requests):
        output = next(self.outputs)
        if isinstance(output, Exception):
            raise output
        return [output for _ in requests]


def spec(task="task", ordinal=0):
    return EvalSpec(task, SourceRef("fixture"), str(ordinal), ordinal,
                    (ModelRequest(RequestKind.GENERATE, "question"),), GraderSpec("exact_match"), target="yes")


def plan(task="task", items=None, graders=None):
    return SourcePlan(task, TaskRoute.CUSTOM, (items if items is not None else [spec(task)],),
                      (task,), SourceMetrics(repeated=False), graders or {})


def run(outputs, plans=None, **kwargs):
    return asyncio.run(run_evaluations(Model(outputs), plans or [plan()], workers=1, **kwargs))


def test_evaluate_without_args_returns_scores_and_benchmark_metadata():
    class Source(BaseBenchmark):
        METRICS = ("accuracy",)

        def benchmark_size(self):
            return 1

        def iter_specs(self, options):
            return iter([spec(options.task)])

    source = Source()
    result = evaluate(
        Model(["yes"]), Namespace(get_benchmark=lambda _task: source), None,
        ["task"], {"task": TaskRoute.CUSTOM}, [1],
    )

    assert result["results"]["task"]["accuracy"] == 1.0
    assert result["benchmark_metadata"]["task"]["n_attempted"] == 1
    assert result["task_outcomes"]["task"]["status"] is TaskStatus.SUCCEEDED


def test_zero_score_is_a_successful_typed_outcome():
    result = run(["no"])
    outcome = result["task_outcomes"]["task"]
    assert outcome["status"] is TaskStatus.SUCCEEDED
    assert outcome["metrics"]["accuracy"] == 0.0
    assert (outcome["expected_count"], outcome["generated_count"], outcome["scored_count"]) == (1, 1, 1)
    validate_result_document(result)
    assert EvalResults.model_validate(result).task_outcomes["task"].status is TaskStatus.SUCCEEDED


def test_samples_are_durable_before_aggregate_commit(tmp_path):
    root = str(tmp_path / "archive")
    with FineStoreOutput(root, "run", ["task"]) as output:
        result = run(["yes"], sink=output)
        assert "samples" not in result
        assert ReadView(root).scan(ARCHIVE_SAMPLES_TABLE).num_rows == 1
        output.finish(result)


@pytest.mark.parametrize(("error", "category"), [
    (ConnectionError("offline"), FailureCategory.MODEL_TRANSPORT),
    (TimeoutError(), FailureCategory.AGENT_TIMEOUT),
])
def test_generation_failure_has_no_score_and_does_not_stop_other_tasks(error, category):
    result = run([error, "yes"], [plan("broken"), plan("healthy")])
    assert result["task_outcomes"]["broken"]["failure"]["category"] is category
    assert "broken" not in result["results"]
    assert result["task_outcomes"]["healthy"]["status"] is TaskStatus.SUCCEEDED


def test_eager_source_failure_does_not_stop_other_tasks():
    class Source(BaseBenchmark):
        METRICS = ("accuracy",)

        def iter_specs(self, options):
            if options.task == "broken":
                raise FileNotFoundError("dataset unavailable")
            return iter([spec(options.task)])

    tasks = {"broken": Source(), "healthy": Source()}
    model = Model(["yes"])
    plans = source_plans(
        tasks, dict.fromkeys(tasks, TaskRoute.CUSTOM),
        Namespace(get_benchmark=tasks.__getitem__), None, RunSpec(), model,
    )
    result = asyncio.run(run_evaluations(model, plans, workers=1))

    broken = result["task_outcomes"]["broken"]
    assert broken["status"] is TaskStatus.FAILED
    assert broken["failure"]["category"] is FailureCategory.RESOURCE
    assert "broken" not in result["results"]
    healthy = result["task_outcomes"]["healthy"]
    assert healthy["status"] is TaskStatus.SUCCEEDED
    assert healthy["metrics"]["accuracy"] == 1.0
    assert (healthy["expected_count"], healthy["generated_count"], healthy["scored_count"]) == (1, 1, 1)


def test_exhausted_transport_is_not_an_incorrect_answer():
    result = run([FailedGeneration("model_transport")])
    assert result["results"] == {}
    outcome = result["task_outcomes"]["task"]
    assert outcome["failure"]["category"] is FailureCategory.MODEL_TRANSPORT
    assert outcome["failure_counts"] == {FailureCategory.MODEL_TRANSPORT: 1}
    record = result["samples"]["task"][0]
    assert record["failure_category"] == "model_transport"
    assert record["metrics"] == []


@pytest.mark.parametrize("failure", [FailedGeneration(FailureCategory.AGENT_TIMEOUT), ConnectionError("bad request")])
@pytest.mark.parametrize("repeated", [False, True])
def test_partial_transport_failure_keeps_scored_metrics_and_retries_only_failed_work(failure, repeated):
    checkpoints = {"task": MemoryCheckpoints()}
    items = [spec(ordinal=0), spec(ordinal=1)]
    if repeated:
        items = [replace(spec(), trial=trial) for trial in (0, 1)]
    first = run(["yes", failure], [plan(items=items)], checkpoints=checkpoints)
    outcome = first["task_outcomes"]["task"]
    assert outcome["status"] is TaskStatus.SUCCEEDED
    metric = "accuracy_avg" if repeated else "accuracy"
    assert first["results"]["task"][metric] == 1.0
    expected = 1 if repeated else 2
    assert (outcome["expected_count"], outcome["generated_count"], outcome["scored_count"]) == (expected, 1, 1)
    assert sum(outcome["failure_counts"].values()) == 1
    [failed] = [sample for sample in first["samples"]["task"] if sample.get("failure_category")]
    assert failed["metrics"] == []
    assert failed["failure_category"] in {FailureCategory.AGENT_TIMEOUT, FailureCategory.MODEL_TRANSPORT}
    validate_result_document(first)

    recovered = run(["no"], [plan(items=items)], checkpoints=checkpoints)
    assert recovered["results"]["task"][metric] == 0.5
    outcome = recovered["task_outcomes"]["task"]
    assert outcome["status"] is TaskStatus.SUCCEEDED
    assert (outcome["expected_count"], outcome["generated_count"], outcome["scored_count"]) == (expected,) * 3
    assert outcome["failure_counts"] == {}


def test_grader_failure_is_classified_at_the_grading_boundary():
    def broken(_spec, _outputs):
        raise ConnectionError("judge unavailable")

    task = plan(items=[replace(spec(), grader=GraderSpec("broken"))], graders={"broken": Grader(broken)})
    result = run(["yes"], [task])
    assert result["results"] == {}
    assert result["task_outcomes"]["task"]["failure"]["category"] is FailureCategory.GRADER_INFRASTRUCTURE


def test_missing_sampled_answers_report_partial_sample_coverage_and_resume_missing_trials():
    items = [replace(spec(), trial=trial) for trial in range(10)]
    checkpoints = {"task": MemoryCheckpoints()}

    def task():
        result = plan(items=items)
        result.reducer = SourceMetrics(repeated=True, pass_at_k=(1, 10))
        return result

    first = run(["yes", *[FailedGeneration(FailureCategory.AGENT_TIMEOUT) for _ in range(9)]],
                [task()], checkpoints=checkpoints)
    outcome = first["task_outcomes"]["task"]
    assert (outcome["expected_count"], outcome["generated_count"], outcome["scored_count"]) == (1, 1, 1)
    assert (
        outcome["expected_sample_count"], outcome["generated_sample_count"], outcome["scored_sample_count"]
    ) == (10, 1, 1)
    assert outcome["failure_counts"] == {FailureCategory.AGENT_TIMEOUT: 9}
    assert first["results"]["task"].get("pass@10") is None
    assert EvalResults.model_validate(first).task_outcomes["task"].expected_sample_count == 10

    recovered = run(["no"] * 9, [task()], checkpoints=checkpoints)
    outcome = recovered["task_outcomes"]["task"]
    assert (
        outcome["expected_sample_count"], outcome["generated_sample_count"], outcome["scored_sample_count"]
    ) == (10, 10, 10)
    assert recovered["results"]["task"]["pass@10"] == 1.0


@pytest.mark.parametrize("error", [TimeoutError("archive timeout"), ConnectionError("archive offline")])
def test_checkpoint_write_failure_cannot_be_tolerated_as_a_dropped_request(error):
    class Checkpoints(MemoryCheckpoints):
        def record(self, row):
            if row.position == 0:
                raise error
            super().record(row)

    result = run(
        ["yes", "yes"], [plan(items=[spec(ordinal=0), spec(ordinal=1)])],
        checkpoints={"task": Checkpoints()},
    )
    assert result["results"] == {}
    assert result["task_outcomes"]["task"]["failure"]["category"] is FailureCategory.SERIALIZATION


@pytest.mark.parametrize("stage", ["add", "finish"])
def test_reducer_failure_is_a_task_error_and_other_tasks_finish(stage):
    class BrokenMetrics(SourceMetrics):
        def add(self, result):
            if stage == "add":
                raise ValueError("invalid trial grade")
            super().add(result)

        def finish(self, task):
            raise ValueError("invalid trial aggregate")

    broken = plan("broken")
    broken.reducer = BrokenMetrics(repeated=False)

    result = run(["yes", "yes"], [broken, plan("healthy")])

    assert "broken" not in result["results"]
    failure = result["task_outcomes"]["broken"]["failure"]
    assert failure["category"] is FailureCategory.GRADER_INFRASTRUCTURE
    assert failure["exception_type"] == "ItemFailure"
    assert "invalid trial" in failure["message"]
    assert result["task_outcomes"]["healthy"]["status"] is TaskStatus.SUCCEEDED
    assert result["results"]["healthy"]["accuracy"] == 1.0


@pytest.mark.parametrize("task,expected", [("JEEBench", 0.25), ("zeroeval", 100.0), ("MixEval", 0.625)])
def test_nonbinary_sources_keep_native_headlines_with_multiple_samples(monkeypatch, task, expected):
    if task == "JEEBench":
        source = JEEBenchBenchmark()
        monkeypatch.setattr(source, "load_questions", lambda: [{
            "description": "paper", "subject": "math", "index": 1, "type": "MCQ(multiple)",
            "question": "Choose", "gold": "ABC", "options": ["one", "two", "three", "four"],
        }])
        output = r"\boxed{A}"
    elif task == "zeroeval":
        source = ZeroEvalBenchmark(tasks=["math-l5"])
        monkeypatch.setattr(source, "load_dataset", lambda _task: (
            ["one"], [], [[{"role": "user", "content": "Compute"}]], {"answer": ["42"]},
        ))
        output = json.dumps({"answer": "42"})
    else:
        source = MixEvalBenchmark(debug=True)
        output = "candidate"
        monkeypatch.setattr("eval.execution.driver.JudgeConfig.resolve", lambda **_kwargs: JudgeConfig(
            "fixture", "http://offline.invalid/v1", "placeholder",
        ))
        monkeypatch.setattr("eval.execution.driver.JudgeBackend", lambda *_args: FixedEndpoint(
            lambda request: "[[0.25]]" if "correctness" in request.prompt[-1]["content"].lower() else "[[C]]",
        ))
    model = Model([output] * 4)
    plans = source_plans(
        [task], {task: TaskRoute.CUSTOM}, Namespace(get_benchmark=lambda _name: source), None,
        RunSpec(limit=1, num_samples=2, pass_at_k=(1,)), model,
    )

    result = asyncio.run(run_evaluations(model, plans, workers=1, supports_request_seed=True))

    assert result["task_outcomes"][task]["status"] is TaskStatus.SUCCEEDED
    assert result["results"][task][source.PRIMARY_METRIC] == expected
    assert not any(name.startswith("pass@") for name in result["results"][task])
    source.num_samples = 2
    assert source.benchmark_primary_metric() == source.PRIMARY_METRIC


@pytest.mark.parametrize("task", ["FinanceBench", "SimpleQA"])
@pytest.mark.parametrize("available", [0, 1])
def test_unavailable_judge_trials_preserve_headlines_and_make_pass_at_k_unavailable(monkeypatch, task, available):
    source = FinanceBenchBenchmark(debug=True) if task == "FinanceBench" else SimpleQABenchmark(debug=True)
    replies = iter(
        ["correct" if task == "FinanceBench" else "A"] * available
        + [FailedGeneration("model_transport")] * (2 - available)
    )
    monkeypatch.setattr("eval.execution.driver.JudgeConfig.resolve", lambda **_kwargs: JudgeConfig(
        "fixture", "http://offline.invalid/v1", "placeholder",
    ))
    monkeypatch.setattr(
        "eval.execution.driver.JudgeBackend", lambda *_args: FixedEndpoint(lambda _request: next(replies))
    )
    model = Model(["candidate"] * 2)
    plans = source_plans(
        [task], {task: TaskRoute.CUSTOM}, Namespace(get_benchmark=lambda _name: source), None,
        RunSpec(limit=1, num_samples=2, pass_at_k=(1, 2)), model,
    )

    result = asyncio.run(run_evaluations(model, plans, workers=1, supports_request_seed=True))

    metrics = result["results"][task]
    assert result["task_outcomes"][task]["status"] is TaskStatus.SUCCEEDED
    assert metrics["accuracy"] == (1.0 if available else None)
    assert metrics["pass@1"] is metrics["pass@2"] is None
    assert metrics["num_judged"] == available
    assert metrics["num_judge_failed"] == 2 - available
    assert metrics["judge_coverage"] == available / 2


def test_missing_per_sample_metrics_cannot_become_an_aggregate_success():
    task = plan(items=[replace(spec(), grader=GraderSpec("empty"))], graders={"empty": Grader(lambda *_: Grade({}))})
    result = run(["yes"], [task])
    assert result["results"] == {}
    assert result["task_outcomes"]["task"]["status"] is TaskStatus.FAILED


def test_sample_serialization_failure_is_typed_and_closes_the_sink():
    closed = []

    class Sink:
        def write_sample(self, task, record):
            raise ValueError("invalid sample")

        def finish_task(self):
            closed.append(True)

    result = run(["yes"], sink=Sink())
    assert result["task_outcomes"]["task"]["failure"]["category"] is FailureCategory.SERIALIZATION
    assert closed == [True]


def completion(content=None, reasoning=None):
    choice = {"message": {"content": content, "reasoning_content": reasoning}, "finish_reason": "stop"}
    response = completion_response_from_chat_choice({"choices": [choice]}, choice)
    return CompletionText(response.normalized_content(), response, CompletionContentPolicy.COMBINE)


@pytest.mark.parametrize("outputs, status", [
    ([completion(), completion("yes")], TaskStatus.FAILED),
    ([completion(reasoning="yes"), completion("yes")], TaskStatus.SUCCEEDED),
])
def test_empty_response_quality_guard_preserves_reasoning_scores(outputs, status):
    result = run(outputs, [plan(items=[spec(ordinal=0), spec(ordinal=1)])])
    outcome = result["task_outcomes"]["task"]
    assert outcome["status"] is status
    assert sum(outcome["completion_response_summary"].values()) == 2
    assert len(result["samples"]["task"]) == 2


def test_empty_source_is_an_incomplete_evaluation():
    result = run([], [plan(items=[])])
    assert result["task_outcomes"]["task"]["failure"]["category"] is FailureCategory.INCOMPLETE_EVALUATION


@pytest.mark.parametrize(
    "result",
    [{"results": {}}, {"results": {"task": {"accuracy": 1.0}}, "task_outcomes": {"task": {"status": "succeeded"}}}],
)
def test_aggregate_writer_rejects_missing_or_invalid_task_outcomes(result, tmp_path):
    args = Namespace(show_config=False, finestore_output_path=str(tmp_path / "archive"))
    with pytest.raises(EvaluationRunError):
        handle_evaluation_output(result, args, GeneralConfigTracker())
    assert not (tmp_path / "archive").exists()


def test_result_reader_rejects_results_without_outcomes():
    with pytest.raises(EvaluationRunError, match="no task outcomes"):
        EvalResults.model_validate({"results": {"task": {"accuracy": 1.0}}})


def test_predict_only_exports_through_the_same_pool_without_invoking_a_grader():
    task = plan(items=[replace(spec(), grader=GraderSpec("must_not_run"))])
    result = run(["candidate output"], [task], predict_only=True)
    outcome = result["task_outcomes"]["task"]
    assert outcome["status"] is TaskStatus.EXPORTED
    assert outcome["generated_count"] == 1
    assert outcome["scored_count"] is None
    assert result["results"] == {}
    assert result["samples"]["task"][0]["resps"] == [["candidate output"]]
