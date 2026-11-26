# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior tests for the ``repeat_stall`` stop: fire, exemption, precedence, and grading."""

import asyncio
import json
from collections.abc import Callable
from dataclasses import dataclass, field

import pytest

from rolloutengine.harness.core import HarnessContext, ModelSession
from rolloutengine.harness.external import AgentHarness
from rolloutengine.models import ContextRecovery, LimitReached, ModelTurn, Prompt, RolloutLimits
from rolloutengine.test_support import arithmetic_task, engine, rollout_spec


@dataclass
class ScriptedHarness:
    """A harness that replies a fixed number of times, with per-reply observations."""

    replies: int
    observations: tuple[str, ...] | None = None
    """One observation per reply; a harness that never varies it leaves this ``None``."""
    pause: float = 0.0
    """Seconds the harness spends between a turn and its next reply."""
    context_recovery = ContextRecovery.NEVER
    refused: list[str] = field(default_factory=list)
    """The stop reasons of every ``LimitReached`` the harness saw."""

    def check(self, task, limits) -> None:
        pass

    def observation(self, index: int) -> tuple[dict, ...]:
        if self.observations is None:
            return ({"role": "user", "content": "unchanged output"},)
        return ({"role": "user", "content": self.observations[index]},)

    async def run(self, context: HarnessContext, session: ModelSession) -> None:
        await session.start(Prompt(({"role": "user", "content": "Run the check."},), {}))
        for index in range(self.replies):
            if self.pause:
                await asyncio.sleep(self.pause)
            try:
                await session.reply(self.observation(index))
            except LimitReached as error:
                self.refused.append(error.stop_reason)
                raise


@dataclass
class ConstantModel:
    """A model that answers every request with the same message, optionally slowly."""

    content: str
    delay: float = 0.0

    async def complete(self, request) -> ModelTurn:
        if self.delay:
            await asyncio.sleep(self.delay)
        served = request.continuation.served_token_ids if request.continuation is not None else (10, 11)
        return ModelTurn({"role": "assistant", "content": self.content}, served, (20,), (-0.5,), "stop")


@dataclass
class ModelAdapter:
    """Adapts a bare request-to-turn function to the ``Model`` protocol."""

    complete: Callable


def terminus_message(command: str) -> str:
    return json.dumps({"analysis": "a", "plan": "p", "commands": [{"keystrokes": command, "duration": 1.0}]})


def wait_message() -> str:
    return json.dumps({"analysis": "a", "plan": "p", "commands": [{"keystrokes": "", "duration": 5.0}]})


async def run_stall(harness, model, limit_overrides: dict):
    runner = engine(model, harnesses={"scripted": harness})
    return await runner.run_task(rollout_spec(arithmetic_task(), harness="scripted", max_turns=12, **limit_overrides))


STALL_LIMITS = {"repeat_stall_batches": 3}


async def test_stall_stop_fires_and_still_grades():
    """Three identical pairs end the attempt with ``repeat_stall``, graded on the work done."""
    harness = ScriptedHarness(replies=8)
    result = await run_stall(harness, ConstantModel("12"), STALL_LIMITS)
    assert result.rollout.stop_reason == "repeat_stall"
    assert result.grade.reward == 1.0
    # The stop refuses the reply that would make the fourth turn: three identical pairs completed.
    assert len(result.rollout.conversations[0].steps) == 3
    assert harness.refused == ["repeat_stall"]


async def test_stall_fires_on_terminus_command_batches():
    result = await run_stall(ScriptedHarness(replies=8), ConstantModel(terminus_message("cat notes.txt")), STALL_LIMITS)
    assert result.rollout.stop_reason == "repeat_stall"
    assert len(result.rollout.conversations[0].steps) == 3


async def test_changed_observations_prevent_a_stall():
    harness = ScriptedHarness(replies=8, observations=tuple(f"output {index}" for index in range(8)))
    result = await run_stall(harness, ConstantModel(terminus_message("cat notes.txt")), STALL_LIMITS)
    # The harness finishes under the turn limit: no stall fired.
    assert result.rollout.stop_reason == "stop"


async def test_changed_actions_prevent_a_stall():
    harness = ScriptedHarness(replies=8)
    contents = [terminus_message(f"command {index}") for index in range(12)]
    turn = 0

    async def complete(request) -> ModelTurn:
        nonlocal turn
        served = request.continuation.served_token_ids if request.continuation is not None else (10, 11)
        message = ModelTurn({"role": "assistant", "content": contents[turn]}, served, (20,), (-0.5,), "stop")
        turn += 1
        return message

    result = await run_stall(harness, ModelAdapter(complete), STALL_LIMITS)
    assert result.rollout.stop_reason == "stop"


async def test_wait_only_terminus_turns_never_stall():
    """Empty keystrokes with a duration are a wait: an all-wait run never stalls."""
    result = await run_stall(ScriptedHarness(replies=8), ConstantModel(wait_message()), STALL_LIMITS)
    assert result.rollout.stop_reason == "stop"


async def test_stall_is_off_by_default():
    result = await run_stall(ScriptedHarness(replies=8), ConstantModel(terminus_message("cat notes.txt")), {})
    assert result.rollout.stop_reason == "stop"


async def test_short_attempt_never_stalls():
    """A stall needs ``repeat_stall_batches`` completed pairs, so a two-turn attempt cannot fire."""
    result = await run_stall(ScriptedHarness(replies=1), ConstantModel(terminus_message("cat notes.txt")), STALL_LIMITS)
    assert result.rollout.stop_reason == "stop"


async def test_budget_reaches_before_the_stall():
    """An existing limit wins: a budget that fills during the run reports ``budget_reached``."""
    result = await run_stall(
        ScriptedHarness(replies=8),
        ConstantModel(terminus_message("cat notes.txt")),
        {**STALL_LIMITS, "max_total_tokens": 1},
    )
    assert result.rollout.stop_reason == "budget_reached"


async def test_turn_deadline_wins_over_a_stall():
    """A deadline that expires while the pairs repeat reports the timeout, not the stall.

    Two identical pairs complete; the deadline then expires during the harness's pause
    before the third reply, so the stall check is never reached and the timeout stands.
    """
    harness = ScriptedHarness(replies=8, pause=0.05)
    result = await run_stall(
        harness,
        ConstantModel(terminus_message("cat notes.txt"), delay=0.01),
        {**STALL_LIMITS, "total_turn_timeout": 0.15},
    )
    assert result.rollout.stop_reason == "total_turn_timeout"
    assert harness.refused == []


def test_limits_reject_a_non_positive_stall_batch_count():
    with pytest.raises(ValueError):
        RolloutLimits.model_validate(
            {
                "max_turns": 3,
                "max_total_tokens": None,
                "repeat_stall_batches": 0,
                "model_turn_timeout": None,
                "tool_turn_timeout": None,
                "total_turn_timeout": None,
                "attempt_timeout": None,
                "verifier_timeout": 5,
                "cleanup_timeout": 5,
            }
        )


@pytest.mark.parametrize("changed_observation", [False, True])
async def test_external_agent_preserves_observations_with_empty_chat_annotations(changed_observation):
    content = terminus_message("cat notes.txt")
    calls = []

    async def complete(request):
        calls.append(request)
        served = request.continuation.served_token_ids if request.continuation is not None else (10, 11)
        return ModelTurn(
            {"role": "assistant", "content": content, "annotations": [], "audio": None, "refusal": None},
            served,
            (20,),
            (-0.5,),
            "stop",
        )

    async def agent(_context, completion):
        messages = [{"role": "user", "content": "Read the notes."}]
        for index in range(8):
            turn = await completion(Prompt(tuple(messages), {}))
            messages.extend(
                [
                    {"role": "assistant", "content": turn.message["content"]},
                    {"role": "user", "content": f"output {index}" if changed_observation else "unchanged output"},
                ]
            )

    result = await run_stall(AgentHarness(agent), ModelAdapter(complete), STALL_LIMITS)
    assert result.rollout.stop_reason == ("stop" if changed_observation else "repeat_stall")
    assert len(calls) == (8 if changed_observation else 3)
    assert len(result.rollout.conversations) == 1
    assert result.rollout.conversations[0].steps[0].transition.observations == (
        {"role": "user", "content": "output 0" if changed_observation else "unchanged output"},
    )
