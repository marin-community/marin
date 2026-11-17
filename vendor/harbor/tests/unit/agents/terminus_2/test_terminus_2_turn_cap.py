"""Turn-cap exhaustion must be a classified, visible outcome — not a silent
"complete" run.

When ``_run_agent_loop`` exhausts ``max_turns`` without a confirmed task
completion it raises :class:`TurnCapExhaustedError` (``PASSTHROUGH`` in the
``harbor_config.errors`` taxonomy, so the verifier still scores the rollout)
and records a bracketed ``[Turn cap exhausted]`` system step in the
trajectory — the same marker convention as ``[Early termination]``. A
turn-callback early termination, by contrast, is a deliberate external stop
and must keep its own outcome instead of being reclassified as exhaustion.
"""

from collections.abc import Awaitable, Callable
from types import SimpleNamespace
from typing import Any

import pytest

from harbor.agents.base import TurnCapExhaustedError
from harbor_config import AgentStopReason
from tests.unit.agents.terminus_2.bare_agent import make_bare_terminus2


def _response() -> SimpleNamespace:
    return SimpleNamespace(
        content="thinking",
        reasoning_content=None,
        model_name=None,
        prompt_token_ids=None,
        completion_token_ids=None,
        logprobs=None,
    )


def _interaction(*, is_task_complete: bool) -> SimpleNamespace:
    return SimpleNamespace(
        commands=[],
        is_task_complete=is_task_complete,
        feedback="",
        analysis="a",
        plan="p",
        response=_response(),
    )


def _sequenced_interactions(
    completions: list[bool],
) -> Callable[..., Awaitable[SimpleNamespace]]:
    """Async stand-in for ``_handle_llm_interaction``; each call returns one
    interaction whose ``is_task_complete`` is the next entry."""
    iterator = iter(completions)

    async def _call(*args: Any, **kwargs: Any) -> SimpleNamespace:
        return _interaction(is_task_complete=next(iterator))

    return _call


async def _noop_execute(commands: Any, session: Any) -> tuple[bool, str]:
    return False, "output"


async def _alive() -> bool:
    return True


def _chat() -> SimpleNamespace:
    return SimpleNamespace(
        total_input_tokens=0,
        total_output_tokens=0,
        total_cache_tokens=0,
        total_cost=0.0,
        messages=[],
    )


def _make_agent(tmp_path, max_turns: int, turn_callback: Any = None):
    return make_bare_terminus2(
        tmp_path,
        _max_episodes=max_turns,
        _n_episodes=0,
        _session=SimpleNamespace(is_session_alive=_alive),
        _enable_episode_logging=False,
        _save_raw_content_in_trajectory=False,
        _pending_completion=False,
        _pending_subagent_refs=None,
        _pending_handoff_prompt=None,
        _turn_callback=turn_callback,
        _stop_reason=None,
        _early_termination_reason=None,
    )


def _messages(agent) -> list[str | None]:
    return [step.message for step in agent._trajectory_steps]


@pytest.mark.parametrize(
    ("max_turns", "completions", "expected_pending"),
    [
        # Agent never declares the task complete within the cap.
        (2, [False, False], False),
        # Agent declares completion on its final turn; the cap cuts off the
        # confirmation exchange.
        (1, [True], True),
    ],
)
@pytest.mark.asyncio
async def test_turn_cap_exhaustion_raises_and_records_boundary(
    tmp_path, max_turns: int, completions: list[bool], expected_pending: bool
) -> None:
    agent = _make_agent(tmp_path, max_turns=max_turns)
    agent._handle_llm_interaction = _sequenced_interactions(completions)
    agent._execute_commands = _noop_execute

    with pytest.raises(TurnCapExhaustedError) as exc_info:
        await agent._run_agent_loop(
            initial_prompt="hi", chat=_chat(), original_instruction=""
        )

    assert exc_info.value.pending_completion is expected_pending
    assert agent._stop_reason is AgentStopReason.TURN_CAP_EXHAUSTED
    assert any(
        isinstance(m, str) and m.startswith("[Turn cap exhausted]")
        for m in _messages(agent)
    )


@pytest.mark.asyncio
async def test_early_termination_is_not_reclassified_as_exhaustion(tmp_path) -> None:
    """A turn-callback stop leaves the loop via break before the cap; it must
    return normally with its own stop reason, not raise the exhaustion error."""
    agent = _make_agent(
        tmp_path,
        max_turns=5,
        turn_callback=lambda turn, trajectory_steps, messages: True,
    )
    agent._handle_llm_interaction = _sequenced_interactions([False] * 5)
    agent._execute_commands = _noop_execute

    await agent._run_agent_loop(
        initial_prompt="hi", chat=_chat(), original_instruction=""
    )

    assert agent._stop_reason is AgentStopReason.EARLY_TERMINATION
    assert not any(
        isinstance(m, str) and m.startswith("[Turn cap exhausted]")
        for m in _messages(agent)
    )


async def test_agent_loop_reuses_post_action_dead_session_result(tmp_path) -> None:
    async def unexpected_liveness_probe() -> bool:
        raise AssertionError("post-action liveness result was not reused")

    agent = _make_agent(tmp_path, max_turns=1)
    agent._session = SimpleNamespace(is_session_alive=unexpected_liveness_probe)
    agent._session_alive_hint = False

    await agent._run_agent_loop(
        initial_prompt="hi", chat=_chat(), original_instruction=""
    )

    assert agent._stop_reason is AgentStopReason.SESSION_ENDED


@pytest.mark.asyncio
async def test_confirmed_completion_returns_without_boundary_marker(tmp_path) -> None:
    """No-raise is the contract: an agent that confirms completion inside the
    cap must finish the loop as a normal return — the exhaustion error is only
    for runs that actually exhaust the budget."""
    agent = _make_agent(tmp_path, max_turns=5)
    agent._handle_llm_interaction = _sequenced_interactions(
        [
            True,  # first: asks for confirmation
            True,  # second: confirmed
        ]
    )
    agent._execute_commands = _noop_execute

    await agent._run_agent_loop(
        initial_prompt="hi", chat=_chat(), original_instruction=""
    )

    assert agent._stop_reason is AgentStopReason.TASK_COMPLETE
    assert not any(
        isinstance(m, str) and m.startswith("[Turn cap exhausted]")
        for m in _messages(agent)
    )
