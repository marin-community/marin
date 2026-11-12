"""Semantic judge retries preserve native parsing and durable operations."""

import asyncio
from dataclasses import replace
import random
from urllib.error import HTTPError

import pytest

from eval.chat_benchmarks.MixEval.eval_instruct import MixEvalBenchmark
from eval.completion_response import CompletionContentPolicy, CompletionResponse, CompletionText, FailedGeneration
from eval.contracts.eval_spec import GraderSpec, SourceRef
from eval.execution.engine import CollectSamples, Evaluator
from eval.execution.grading import Grader, Graders
from eval.graders.judged import mixeval
from eval.graders.specs import GRADERS
from eval.sources.common import SourceOptions
from tests.contracts.source_harness import FixedEndpoint, MemoryCheckpoints


@pytest.fixture
def mix_specs():
    source = MixEvalBenchmark(debug=True)
    return list(source.iter_specs(SourceOptions("MixEval", SourceRef("fixture"), limit=1)))


def run_items(specs, candidate, judge, checkpoints, *, definitions=GRADERS):
    sink = CollectSamples()
    grading = Graders(definitions)
    engine = Evaluator(candidate, grading, sink, workers=2,
                       judges={"MixEval": judge}, checkpoints={"MixEval": checkpoints})

    async def run():
        return [item async for item in engine.run(specs)]

    try:
        return asyncio.run(run()), sink.samples["MixEval"]
    finally:
        grading.close()


@pytest.mark.parametrize(
    "kind,raw,score",
    [("freeform", "[0.1] then [[0.8]]", 0.8),
     ("multichoice", "[A] then [[C]]", 1.0),
     ("multichoice", "[[Z]]", 0.0)],
)
def test_mixeval_uses_native_double_bracket_priority(mix_specs, kind, raw, score):
    spec = next(spec for spec in mix_specs if spec.namespace == kind)
    judge = FixedEndpoint(raw)

    completed, samples = run_items([spec], FixedEndpoint("candidate"), judge, MemoryCheckpoints())

    assert completed[0].error is None
    assert samples[0]["accuracy"] == score
    assert len(judge.requests) == 1


def test_mixeval_retries_malformed_judgments_and_reuses_each_saved_attempt(mix_specs):
    spec = next(spec for spec in mix_specs if spec.namespace == "multichoice")
    checkpoint = MemoryCheckpoints()
    candidate = FixedEndpoint("candidate")
    judge = FixedEndpoint(lambda _request: "not parsed" if len(judge.requests) == 1 else "[[C]]")

    completed, samples = run_items([spec], candidate, judge, checkpoint)
    resumed_candidate, resumed_judge = FixedEndpoint("unexpected"), FixedEndpoint("unexpected")
    replay, replay_samples = run_items([spec], resumed_candidate, resumed_judge, checkpoint)

    assert completed[0].error is replay[0].error is None
    assert samples == replay_samples
    assert samples[0]["accuracy"] == 1.0
    assert len(candidate.requests) == 1
    assert len(judge.requests) == 2
    assert resumed_candidate.requests == resumed_judge.requests == []
    assert {tuple(sorted(row.unit.items())) for row in checkpoint.records()} == {
        (("phase", "generate"), ("request_index", 0)),
        (("phase", "judge"), ("request_index", 0)),
        (("parse_attempt", 1), ("phase", "judge"), ("request_index", 0)),
    }


@pytest.mark.parametrize("kind,raw", [("freeform", "[0.5] then [[2]]"), ("multichoice", "no choice")])
def test_mixeval_exhaustion_is_ten_attempts_and_replay_stable(mix_specs, kind, raw):
    spec = next(spec for spec in mix_specs if spec.namespace == kind)
    checkpoint = MemoryCheckpoints()
    judge = FixedEndpoint(raw)
    state = random.getstate()

    completed, samples = run_items([spec], FixedEndpoint("candidate"), judge, checkpoint)
    replay_judge = FixedEndpoint("unexpected")
    replay, replay_samples = run_items([spec], FixedEndpoint("unexpected"), replay_judge, checkpoint)

    assert completed[0].error is replay[0].error is None
    assert len(judge.requests) == 10
    assert replay_judge.requests == []
    assert samples == replay_samples
    assert samples[0]["judge_fallback"] == "malformed_response_exhaustion"
    assert samples[0]["judge_raw"] == raw
    assert samples[0]["accuracy"] in ({0.0, 1.0} if kind == "multichoice" else {value / 10 for value in range(11)})
    assert random.getstate() == state


@pytest.mark.parametrize("failure", [
    ConnectionError("judge disconnected"),
    HTTPError("https://judge.invalid", 402, "payment required", None, None),
    FailedGeneration("model_transport"),
])
def test_mixeval_infrastructure_failure_never_becomes_random_accuracy(mix_specs, failure):
    spec = next(spec for spec in mix_specs if spec.namespace == "multichoice")

    def response(_request):
        if isinstance(failure, Exception):
            raise failure
        return failure

    judge = FixedEndpoint(response)
    completed, samples = run_items([spec], FixedEndpoint("candidate"), judge, MemoryCheckpoints())

    assert completed[0].error is not None
    assert len(judge.requests) == 1
    assert all("accuracy" not in sample for sample in samples)


def test_judge_infrastructure_policy_does_not_consume_semantic_retry_budget(mix_specs):
    spec = next(spec for spec in mix_specs if spec.namespace == "multichoice")
    definitions = {**GRADERS, "mixeval": Grader(
        lambda spec, outputs: replace(mixeval(spec, outputs), on_error=GraderSpec("judge_failed")),
    )}
    judge = FixedEndpoint(FailedGeneration("model_transport"))

    completed, samples = run_items([spec], FixedEndpoint("candidate"), judge,
                                    MemoryCheckpoints(), definitions=definitions)

    assert completed[0].error is None
    assert len(judge.requests) == 1
    assert samples[0]["judge_failed"] == 1.0
    assert "accuracy" not in samples[0]
    assert "judge_fallback" not in samples[0]


def test_mixeval_resume_continues_only_missing_semantic_attempt(mix_specs):
    spec = next(spec for spec in mix_specs if spec.namespace == "multichoice")
    checkpoint = MemoryCheckpoints()
    candidate = FixedEndpoint("candidate")

    def interrupted(_request):
        if len(judge.requests) == 2:
            raise ConnectionError("judge disconnected")
        return "malformed"

    judge = FixedEndpoint(interrupted)
    first, _samples = run_items([spec], candidate, judge, checkpoint)
    resumed_candidate, resumed_judge = FixedEndpoint("unexpected"), FixedEndpoint("[[C]]")
    second, samples = run_items([spec], resumed_candidate, resumed_judge, checkpoint)

    assert first[0].error is not None
    assert second[0].error is None
    assert samples[0]["accuracy"] == 1.0
    assert resumed_candidate.requests == []
    assert len(resumed_judge.requests) == 1


def test_empty_candidate_response_remains_eligible_on_replay(mix_specs):
    spec = next(spec for spec in mix_specs if spec.namespace == "multichoice")
    response = CompletionResponse("", None, "stop", None, {}, {})
    candidate = FixedEndpoint(CompletionText("", response, CompletionContentPolicy.FINAL_ONLY))
    judge = FixedEndpoint("[[C]]")
    checkpoint = MemoryCheckpoints()

    first, first_samples = run_items([spec], candidate, judge, checkpoint)
    replay, replay_samples = run_items([spec], candidate, judge, checkpoint)

    assert first[0].error is replay[0].error is None
    assert first_samples == replay_samples
    assert len(candidate.requests) == 2
    assert len(judge.requests) == 1
