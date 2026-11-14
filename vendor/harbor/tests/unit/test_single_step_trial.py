from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from harbor.agents.base import MalformedModelResponseError, TurnCapExhaustedError
from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.agents.terminus_2.tmux_session import ModelInitiatedTmuxFailure
from harbor.llms.base import ContextBudgetExceededError, ContextLengthExceededError
from harbor.models.trial.paths import EnvironmentPaths
from harbor.models.task.config import TaskConfig
from harbor.environments.base import ExecResult
from harbor.trial.errors import AgentTimeoutError
from harbor.trial.single_step import SingleStepTrial


def _agent_phase_trial(raised: Exception) -> SingleStepTrial:
    """A SingleStepTrial whose agent phase raises *raised*, with the
    exception-recording collaborator mocked."""
    trial = object.__new__(SingleStepTrial)
    trial._result = object()
    trial._agent_timeout_sec = 10.0
    trial.task = SimpleNamespace(
        instruction="do it",
        config=SimpleNamespace(agent=SimpleNamespace(user=None)),
    )
    trial._run_agent_phase = AsyncMock(side_effect=raised)
    trial._record_exception = AsyncMock()
    return trial


@pytest.mark.parametrize(
    "exc",
    [
        ContextLengthExceededError("served window exceeded"),
        NonZeroAgentExitCodeError("exit 1"),
        ModelInitiatedTmuxFailure("model killed the terminal runner"),
        AgentTimeoutError("timed out"),
        TurnCapExhaustedError("hit the max_turns cap"),
        MalformedModelResponseError("unpaired Unicode surrogate"),
    ],
)
@pytest.mark.asyncio
async def test_run_agent_passthrough_records_and_continues(exc: Exception) -> None:
    """A context overflow (and the other non-fatal agent-phase errors) must be
    recorded and swallowed so the verifier still runs — NOT propagated."""
    trial = _agent_phase_trial(exc)

    await trial._run_agent()  # must not raise

    trial._record_exception.assert_awaited_once_with(exc)


@pytest.mark.asyncio
async def test_run_agent_does_not_swallow_unexpected_error() -> None:
    """An unrelated error is NOT swallowed by the agent-phase passthrough."""
    trial = _agent_phase_trial(ValueError("boom"))

    with pytest.raises(ValueError):
        await trial._run_agent()

    trial._record_exception.assert_not_awaited()


@pytest.mark.asyncio
async def test_run_agent_does_not_score_invalid_context_budget() -> None:
    error = ContextBudgetExceededError("max_tokens must be positive")
    trial = _agent_phase_trial(error)

    with pytest.raises(ContextBudgetExceededError):
        await trial._run_agent()

    trial._record_exception.assert_not_awaited()
