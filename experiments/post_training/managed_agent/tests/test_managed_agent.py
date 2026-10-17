# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
from types import SimpleNamespace

import pytest
from openai.types.chat import ChatCompletion
from rolloutengine.agent import agent_harness
from rolloutengine.contracts import ModelRequest, ModelTurn, Prompt, TokenUsage
from rolloutengine.engine import RolloutEngine
from rolloutengine.shell import task_prompt
from rolloutengine.spec import RolloutLimits
from taskcompendium.grading_result import Outcome

from experiments.post_training.managed_agent.chat_model import ChatModel
from experiments.post_training.managed_agent.managed_agent import ManagedAgent, agent_spec, agent_task
from experiments.post_training.managed_agent.review_agent import LAST_TURN_PROMPT, PullRequest, reviewer, run_tool

LIMITS = RolloutLimits(
    max_turns=4,
    max_total_tokens=None,
    model_turn_timeout=5,
    tool_turn_timeout=5,
    total_turn_timeout=10,
    attempt_timeout=10,
    verifier_timeout=5,
    cleanup_timeout=5,
)
DIFF = "diff --git a/lib/a.py b/lib/a.py\n--- a/lib/a.py\n+++ b/lib/a.py\n@@ -1 +1,2 @@\n-old\n+new\n+more\n"
PULL_REQUEST = PullRequest(1, "Change a.py", "Changes a.py.", DIFF)


def turn(message: dict) -> ModelTurn:
    return ModelTurn(message, (), (), None, "stop")


def tool_call(name: str) -> dict:
    call = {"id": f"call-{name}", "type": "function", "function": {"name": name, "arguments": "{}"}}
    return {"role": "assistant", "content": "", "tool_calls": [call]}


async def outcomes(engine: RolloutEngine, model, specs, concurrency: int) -> dict:
    return {outcome.id: outcome async for outcome in ManagedAgent(engine, model, concurrency=concurrency).run(specs)}


@pytest.mark.asyncio
async def test_attempts_run_up_to_the_concurrency_and_a_failure_stays_in_its_outcome():
    running = peak = 0

    async def model(request):
        nonlocal running, peak
        running += 1
        peak = max(peak, running)
        try:
            await asyncio.sleep(0.05)
            if request.messages[0]["content"] == "Fail.":
                raise ConnectionError("endpoint unavailable")
            return turn({"role": "assistant", "content": "Done."})
        finally:
            running -= 1

    async def agent(context, complete):
        await complete(Prompt(task_prompt(context.task, None, ()).messages, {}))

    engine = RolloutEngine({}, {"agent": agent_harness(agent)})
    tasks = {"a": "Work.", "b": "Work.", "failing": "Fail.", "c": "Work."}
    specs = [agent_spec(agent_task(task_id, text), "agent", LIMITS) for task_id, text in tasks.items()]
    results = await outcomes(engine, model, specs, concurrency=2)

    assert sorted(results) == sorted(tasks)
    assert peak == 2
    assert (results["a"].answer, results["a"].failure, results["a"].grade.status) == ("Done.", None, Outcome.UNAVAILABLE)
    assert results["failing"].failure.startswith("model: ConnectionError")


@pytest.mark.asyncio
async def test_the_reviewer_answers_after_its_tool_calls_in_one_conversation():
    requests = []

    async def model(request):
        requests.append(request)
        return turn(tool_call("changed_files") if len(requests) == 1 else {"role": "assistant", "content": "Fine."})

    engine = RolloutEngine({}, {"reviewer": agent_harness(reviewer(PULL_REQUEST, max_turns=LIMITS.max_turns))})
    spec = agent_spec(agent_task("review", "Review it."), "reviewer", LIMITS)
    [outcome] = (await outcomes(engine, model, [spec], concurrency=1)).values()

    assert outcome.answer == "Fine."
    assert (requests[1].messages[-1]["role"], requests[1].messages[-1]["content"]) == ("tool", "lib/a.py +2 -1")
    # The reviewer sends the same options on each turn, so its second request is a reply.
    [conversation] = outcome.rollout.conversations
    assert len(conversation.steps) == 2


@pytest.mark.asyncio
async def test_the_reviewer_asks_for_its_answer_when_one_turn_is_left():
    requests = []

    async def model(request):
        requests.append(request)
        return turn(tool_call("changed_files"))

    engine = RolloutEngine({}, {"reviewer": agent_harness(reviewer(PULL_REQUEST, max_turns=2))})
    limits = LIMITS.model_copy(update={"max_turns": 2})
    [outcome] = (
        await outcomes(engine, model, [agent_spec(agent_task("review", "Review it."), "reviewer", limits)], 1)
    ).values()

    assert len(requests) == 2
    assert requests[1].messages[-1] == {"role": "user", "content": LAST_TURN_PROMPT}
    # The agent stops at its own budget, before the recorder refuses a request.
    assert (outcome.failure, outcome.rollout.stop_reason) == (None, "stop")


def test_the_tools_read_the_repository_and_nothing_outside_it():
    own_file = json.dumps({"path": "experiments/post_training/managed_agent/__init__.py"})
    assert "Copyright The Marin Authors" in run_tool(PULL_REQUEST, "read_file", own_file)
    outside = run_tool(PULL_REQUEST, "read_file", json.dumps({"path": "../../etc/passwd"}))
    assert outside.startswith("Tool error:") and "outside the repository" in outside


@pytest.mark.asyncio
async def test_the_chat_model_sends_the_cache_key_and_keeps_only_the_fields_that_go_back():
    call = {"id": "call-1", "type": "function", "function": {"name": "changed_files", "arguments": "{}"}}
    served = {"role": "assistant", "content": None, "tool_calls": [call], "reasoning_content": "Look first."}
    response = ChatCompletion.model_validate(
        {
            "id": "completion",
            "object": "chat.completion",
            "created": 0,
            "model": "glm-5.3",
            "choices": [{"index": 0, "message": served, "finish_reason": "tool_calls"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
    )
    requests = []

    async def create(**request):
        requests.append(request)
        return response

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    model = ChatModel(client, "glm-5.3", max_tokens=100, reasoning_effort="low")  # pyrefly: ignore[bad-argument-type]
    options = {"tools": ["tool"], "prompt_cache_key": "run-1"}
    result = await model(ModelRequest(({"role": "user", "content": "Review it."},), options))

    [request] = requests
    assert request["tools"] == ["tool"]
    assert request["extra_body"] == {"chat_template_kwargs": {"reasoning_effort": "low"}, "prompt_cache_key": "run-1"}
    assert result.message == {
        "role": "assistant",
        "content": "",
        "tool_calls": [call],
        "reasoning_content": "Look first.",
    }
    assert (result.stop_reason, result.usage) == ("tool_calls", TokenUsage(10, 5))
