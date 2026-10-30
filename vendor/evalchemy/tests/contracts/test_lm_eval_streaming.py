"""Native filter variants reach FineStore before the aggregate commit."""

import asyncio

import pytest
from eval.completion_response import FailedGeneration
from eval.contracts.eval_spec import EvalSpec, GraderSpec, ModelRequest, RequestKind, SourceRef
from eval.contracts.finestore_output import FineStoreOutput
from eval.contracts.task_outcome import TaskRoute
from eval.execution.driver import SourcePlan, run_evaluations
from eval.execution.metrics import SourceMetrics
from eval.sources.lm_eval import LMEvalSource
from finestore.eval import ARCHIVE_SAMPLES_TABLE, sample_from_archive_row
from finestore.reader import ReadView
from tests.contracts.source_harness import MemoryCheckpoints


class Filters:
    def __init__(self, name, answer):
        self.name = name
        self.answer = answer

    def apply(self, instances):
        for instance in instances:
            instance.filtered_resps[self.name] = self.answer


class Task:
    def __init__(self, filters):
        self._filters = [Filters(name, answer) for name, answer in filters]

    def process_results(self, doc, responses):
        return {"exact_match": float(responses[0] == "4")}


class Backend:
    world_size = 1
    rank = 0

    def __init__(self, answer):
        self.answer = answer

    def generate_until(self, requests):
        return [self.answer for _request in requests]


@pytest.mark.parametrize("filters", [(("strict-match", "4"),), (("strict", "wrong"), ("flexible", "4"))])
def test_native_filter_variants_supersede_failed_attempts_before_result_commit(tmp_path, filters):
    root = str(tmp_path / "archive")
    spec = EvalSpec("gsm8k", SourceRef("fixture"), "opaque", 0,
                    (ModelRequest(RequestKind.GENERATE, "2 + 2?"),),
                    GraderSpec("lm_eval:gsm8k", {"repeats": [1], "indices": [0]}),
                    doc={"question": "2 + 2?"}, target="4")
    source = LMEvalSource("gsm8k", Task(filters), None, None, {}, None)
    checkpoint = MemoryCheckpoints()

    def run(answer):
        plan = SourcePlan("gsm8k", TaskRoute.LM_EVAL, ([spec],), ("gsm8k",),
                          SourceMetrics(repeated=False), {source.grader_name: source.grader()})
        with FineStoreOutput(root, "run", ["gsm8k"]) as output:
            result = asyncio.run(run_evaluations(Backend(answer), [plan], workers=1,
                                                sink=output, checkpoints={"gsm8k": checkpoint}))
            rows = ReadView(root).scan(ARCHIVE_SAMPLES_TABLE).to_pylist(maps_as_pydicts="strict")
            return result, {row["filter"]: sample_from_archive_row(row) for row in rows}

    result, failed = run(FailedGeneration("model_transport"))
    assert result["results"] == {}
    assert set(failed) == {name for name, _answer in filters}
    assert all(sample.output.startswith("[EVALCHEMY_INFRASTRUCTURE_ERROR]") for sample in failed.values())

    result, recovered = run("4")
    assert result["results"]["gsm8k"][f"exact_match,{filters[-1][0]}"] == 1.0
    assert {name: sample.correct for name, sample in recovered.items()} == {
        name: answer == "4" for name, answer in filters
    }
    assert all(not sample.output.startswith("[EVALCHEMY_INFRASTRUCTURE_ERROR]") for sample in recovered.values())
