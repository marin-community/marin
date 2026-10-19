# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("taskforge")

import json
from typing import Any

from rolloutengine.models import (
    ConversationRecord,
    GradedRollout,
    ModelRequest,
    ModelTurn,
    Rollout,
    RolloutFailure,
    RolloutStep,
    TokenUsage,
    Transition,
)
from rolloutengine.test_support import arithmetic_task, rollout_limits
from taskcompendium.grading_result import GradeResult, Outcome

from experiments.post_training.capability_driven_envs.analyze_failed_runs import (
    EvaluatedRun,
    analyze_failed_runs,
    policy_failures,
)
from experiments.post_training.capability_driven_envs.generate_tasks import task_workpieces

LOOKUP_CALL = {"id": "call", "type": "function", "function": {"name": "lookup", "arguments": '{"key": "six"}'}}
CAPABILITY = {
    "id": "failed-runs.tool-use.stop-repeating-calls",
    "name": "Stop repeating a tool call that does not help",
    "outcome": "Change the approach when a tool call returns the same result two times.",
    "failure_pattern": "The agent called the same tool with the same arguments until its turns ran out.",
    "evidence": ["loop"],
    "includes": ["Notice a repeated tool result"],
    "sample_tasks": [{"kind": "entry", "instruction": "Find the price of an item whose search tool is down."}],
}


def evaluated(run_id: str, status: Outcome, reward: float | None, *turns: dict, failure=None) -> EvaluatedRun:
    """A run of the arithmetic task whose policy produced ``turns``, graded with ``status`` and ``reward``."""
    steps = tuple(RolloutStep(ModelTurn(turn, (), (), None, "stop"), Transition(done=False), 0, ()) for turn in turns)
    prompt = {"role": "user", "content": "What is six plus six?"}
    conversation = ConversationRecord((prompt, *turns), (), (), (), None, steps)
    rollout = Rollout(run_id, (conversation,), "stop", failure=failure)
    return EvaluatedRun(arithmetic_task(id=f"task-{run_id}"), GradedRollout(rollout, GradeResult(status, reward)))


RUNS = [
    evaluated("pass", Outcome.GRADED, 1.0, {"role": "assistant", "content": "12"}),
    evaluated("wrong", Outcome.GRADED, 0.0, {"role": "assistant", "content": "six, " * 600 + "so the answer is 13"}),
    evaluated(
        "loop",
        Outcome.SUBMISSION_FAILURE,
        None,
        {"role": "assistant", "tool_calls": [LOOKUP_CALL]},
        {"role": "assistant", "tool_calls": [LOOKUP_CALL]},
    ),
    evaluated("infra", Outcome.INFRA_ERROR, None, {"role": "assistant", "content": "12"}),
    evaluated(
        "interrupted",
        Outcome.GRADED,
        0.0,
        {"role": "assistant", "content": "1"},
        failure=RolloutFailure("ConnectionError", {"operation": "model"}),
    ),
]


def analyst(*messages: dict) -> tuple[Any, list[ModelRequest]]:
    """A chat model that answers with ``messages`` in order, and the requests that it receives."""
    requests: list[ModelRequest] = []

    async def model(request: ModelRequest) -> ModelTurn:
        requests.append(request)
        return ModelTurn(messages[len(requests) - 1], (), (), None, "stop", usage=TokenUsage(10, 5))

    return model, requests


def tool_call(name: str, **arguments: Any) -> dict:
    function = {"name": name, "arguments": json.dumps(arguments)}
    return {"role": "assistant", "content": "", "tool_calls": [{"id": name, "type": "function", "function": function}]}


def test_only_the_runs_that_the_policy_failed_are_analyzed():
    assert [run.result.rollout.run_id for run in policy_failures(RUNS, pass_threshold=1.0)] == ["wrong", "loop"]


async def test_the_agent_reads_the_failed_runs_and_returns_capability_ideas_for_generate_tasks():
    model, requests = analyst(
        tool_call("search_runs", text="Answer is 13"),
        tool_call("read_run", run_id="loop"),
        {"role": "assistant", "content": json.dumps([CAPABILITY])},
    )

    ideas = await analyze_failed_runs(
        RUNS, model, pass_threshold=1.0, limits=rollout_limits(max_turns=4), catalog_version="round-1"
    )

    overview = requests[0].messages[1]["content"]
    assert overview.startswith("2 of 5 runs failed.")
    assert '"run_id": "loop"' in overview and '"repeated_calls": 1' in overview
    assert '"run_id": "pass"' not in overview and '"run_id": "infra"' not in overview
    search, read = requests[1].messages[-1]["content"], requests[2].messages[-1]["content"]
    assert search.startswith("wrong [1] ...") and search.endswith("so the answer is 13...")
    assert "[0] user: What is six plus six?" in read
    [idea] = ideas
    assert (idea.capability_id, idea.catalog_version) == (CAPABILITY["id"], "round-1")
    assert idea.capability["kind"] == "capability" and idea.capability["evidence"] == ["loop"]
    assert task_workpieces({idea.capability_id: idea}, 1) == [(f"{CAPABILITY['id']}--00", idea)]


async def test_no_failed_run_needs_no_analysis():
    model, requests = analyst()
    assert (
        await analyze_failed_runs(
            RUNS[:1], model, pass_threshold=1.0, limits=rollout_limits(), catalog_version="round-1"
        )
        == []
    )
    assert requests == []


@pytest.mark.parametrize(
    "answer,message",
    [
        ("The agent must learn to stop.", "not a list of capabilities"),
        (json.dumps([{**CAPABILITY, "evidence": ["pass"]}]), "did not fail"),
    ],
    ids=["not-json", "unknown-evidence"],
)
async def test_an_answer_that_is_not_grounded_capabilities_is_rejected(answer, message):
    model, _ = analyst({"role": "assistant", "content": answer})
    with pytest.raises(ValueError, match=message):
        await analyze_failed_runs(RUNS, model, pass_threshold=1.0, limits=rollout_limits(), catalog_version="round-1")
