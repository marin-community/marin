# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest

pytest.importorskip("taskforge")

import json

from rolloutengine.models import (
    ConversationRecord,
    GradedRollout,
    Model,
    ModelRequest,
    ModelTurn,
    Rollout,
    RolloutFailure,
    RolloutStep,
    TokenUsage,
    Transition,
)
from rolloutengine.test_support import arithmetic_task, rollout_limits, shell_call
from taskcompendium.grading_result import GradeResult, Outcome
from taskcompendium.models import AnswerCall, TaskSpec

from experiments.post_training.capability_driven_envs.analyze_failed_runs import analyze_failed_runs, policy_failures
from experiments.post_training.capability_driven_envs.generate_tasks import task_workpieces

LOOKUP_CALL = {"id": "call", "type": "function", "function": {"name": "lookup", "arguments": '{"key": "six"}'}}
CAPABILITY = {
    "id": "failed-runs.tool-use.stop-repeating-calls",
    "name": "Stop repeating a tool call that does not help",
    "outcome": "Change the approach when a tool call returns the same result two times.",
    "evidence": ["loop"],
    "includes": ["Notice a repeated tool result"],
}


def evaluated(
    run_id: str, status: Outcome, reward: float | None, *turns: dict, failure=None
) -> tuple[TaskSpec, GradedRollout]:
    """A rollout of the arithmetic task whose policy produced ``turns``, graded with ``status`` and ``reward``."""
    steps = tuple(RolloutStep(ModelTurn(turn, (), (), None, "stop"), Transition(done=False), 0, ()) for turn in turns)
    prompt = {"role": "user", "content": "What is six plus six?"}
    conversation = ConversationRecord((prompt, *turns), (), (), (), None, steps)
    rollout = Rollout(run_id, (conversation,), "stop", failure=failure)
    return arithmetic_task(id=f"task-{run_id}"), GradedRollout(rollout, GradeResult(status, reward))


ROLLOUTS = [
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


def analyst(*messages: dict) -> tuple[Model, list[ModelRequest]]:
    """A chat model that answers with ``messages`` in order, and the requests that it receives."""
    requests: list[ModelRequest] = []

    async def model(request: ModelRequest) -> ModelTurn:
        requests.append(request)
        return ModelTurn(messages[len(requests) - 1], (), (), None, "stop", usage=TokenUsage(10, 5))

    return model, requests


def shell_output(request: ModelRequest) -> str:
    """The output of the shell command that the agent sent before ``request``."""
    return json.loads(request.messages[-1]["content"])["stdout"]


def test_only_the_rollouts_that_the_policy_failed_are_analyzed():
    failures = policy_failures(ROLLOUTS, pass_threshold=1.0)
    assert [graded.rollout.run_id for _, graded in failures] == ["wrong", "loop"]


async def test_the_agent_reads_the_failed_rollouts_as_files_and_returns_capability_ideas_for_generate_tasks():
    model, requests = analyst(
        shell_call("jq -c '[.run_id, .repeated_calls]' index.jsonl", call_id="index"),
        shell_call("grep -il 'answer is 13' *.txt", call_id="search"),
        shell_call("sed -n 2p loop.txt", call_id="read"),
        {"role": "assistant", "content": json.dumps([CAPABILITY])},
    )

    ideas = await analyze_failed_runs(
        ROLLOUTS, model, pass_threshold=1.0, limits=rollout_limits(max_turns=5), catalog_version="round-1"
    )

    assert requests[0].messages[1]["content"].startswith("2 of 5 rollouts failed.")
    assert [shell_output(request) for request in requests[1:]] == [
        '["wrong",0]\n["loop",1]\n',
        "wrong.txt\n",
        "[0] user: What is six plus six?\n",
    ]
    [idea] = ideas
    assert (idea.capability_id, idea.catalog_version) == (CAPABILITY["id"], "round-1")
    assert idea.capability["kind"] == "capability" and idea.capability["evidence"] == ["loop"]
    assert task_workpieces({idea.capability_id: idea}, 1) == [(f"{CAPABILITY['id']}--00", idea)]


async def test_a_rollout_file_shows_the_tools_that_the_task_gave_and_a_malformed_tool_call():
    policy = {"role": "assistant", "tool_calls": [{"id": "bad", "type": "function"}]}
    _, graded = evaluated("malformed", Outcome.SUBMISSION_FAILURE, None, policy)
    task = arithmetic_task(id="task-malformed", answer_format=AnswerCall())
    model, requests = analyst(
        shell_call("cat malformed.txt"),
        {"role": "assistant", "content": json.dumps([{**CAPABILITY, "evidence": ["malformed"]}])},
    )

    await analyze_failed_runs(
        [(task, graded)], model, pass_threshold=1.0, limits=rollout_limits(max_turns=4), catalog_version="round-1"
    )

    tools, *messages = shell_output(requests[1]).splitlines()
    assert [tool["function"]["name"] for tool in json.loads(tools)["tools"]] == ["submit_answer"]
    assert messages[-1] == '[1] assistant: <tool call> malformed {"id": "bad", "type": "function"}'


async def test_no_failed_rollout_needs_no_analysis():
    model, requests = analyst()
    assert (
        await analyze_failed_runs(
            ROLLOUTS[:1], model, pass_threshold=1.0, limits=rollout_limits(), catalog_version="round-1"
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
        await analyze_failed_runs(
            ROLLOUTS, model, pass_threshold=1.0, limits=rollout_limits(), catalog_version="round-1"
        )
