from contextlib import nullcontext
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
from harbor.models.trial.config import AgentConfig, TrialConfig
from harbor.environments.base import ExecResult
from harbor.trial.errors import AgentTimeoutError
from harbor.trial.single_step import SingleStepTrial
from harbor.trial.hooks import TrialEvent


def _single_step_trial(tmp_path: Path) -> SingleStepTrial:
    trial = object.__new__(SingleStepTrial)
    trial._are_artifacts_collected = False
    trial._artifact_handler = SimpleNamespace(download_artifacts=AsyncMock())
    trial.agent_environment = object()
    trial.agent_env_paths = EnvironmentPaths()
    trial.paths = SimpleNamespace(artifacts_dir=tmp_path / "artifacts")
    trial.task = SimpleNamespace(config=TaskConfig())
    trial._result = object()
    trial._sync_agent_output = AsyncMock()
    trial._stop_agent_environment = AsyncMock()
    trial.logger = MagicMock()
    return trial


@pytest.mark.asyncio
async def test_environment_ready_hook_prepares_once_before_agent_install(tmp_path):
    trial = _single_step_trial(tmp_path)
    trial.config = TrialConfig(task={"path": tmp_path}, agent=AgentConfig(name="pi"))
    trial.task.name = "fixture"
    trial._hooks = {event: [] for event in TrialEvent}
    trial._result = SimpleNamespace(agent_info=None)
    started = tmp_path / "provider-started"
    staged = tmp_path / "setup-file"
    prepared = tmp_path / "dependency"

    async def start_provider():
        started.touch()

    async def healthcheck():
        assert started.exists()

    async def stage_files():
        staged.write_text("staged")

    async def prepare_environment(_event):
        assert started.exists()
        assert staged.read_text() == "staged"
        prepared.write_text("installed")

    async def install_agent():
        assert prepared.read_text() == "installed"

    trial.agent_environment = SimpleNamespace(run_healthcheck=healthcheck, with_default_user=lambda _user: nullcontext())
    trial.agent = SimpleNamespace(to_agent_info=lambda: None)
    trial._setup_agent_environment = start_provider
    trial._upload_injected_skills = AsyncMock()
    trial._upload_setup_files = stage_files
    trial._setup_agent = install_agent
    trial.add_hook(TrialEvent.ENVIRONMENT_READY, prepare_environment)

    await trial._prepare()
    prepared.write_text("agent edit")
    await trial._emit(TrialEvent.AGENT_START)
    await trial._emit(TrialEvent.AGENT_START)
    assert prepared.read_text() == "agent edit"


@pytest.mark.asyncio
async def test_collect_artifacts_is_idempotent(tmp_path: Path) -> None:
    trial = _single_step_trial(tmp_path)
    trial.task = SimpleNamespace(config=TaskConfig.model_validate({"verifier": {"collect": [{"command": "snapshot"}]}}))
    exec_hook = AsyncMock(return_value=ExecResult(stdout="", stderr="", return_code=0))
    trial.agent_environment = SimpleNamespace(exec=exec_hook)

    await trial._collect_artifacts()
    await trial._collect_artifacts()

    exec_hook.assert_awaited_once_with("snapshot", timeout_sec=60, user=None)
    trial._artifact_handler.download_artifacts.assert_awaited_once_with(
        trial.agent_environment,
        tmp_path / "artifacts",
        source_artifacts_dir=EnvironmentPaths().artifacts_dir,
    )


@pytest.mark.asyncio
async def test_recover_outputs_skips_artifact_collection_when_already_collected(
    tmp_path: Path,
) -> None:
    trial = _single_step_trial(tmp_path)
    await trial._collect_artifacts()

    await trial._recover_outputs()

    trial._artifact_handler.download_artifacts.assert_awaited_once()


@pytest.mark.asyncio
async def test_collect_hook_failure_does_not_suppress_artifact_download(
    tmp_path: Path,
) -> None:
    trial = _single_step_trial(tmp_path)
    trial.task = SimpleNamespace(config=TaskConfig.model_validate({"verifier": {"collect": [{"command": "snapshot"}]}}))
    trial.agent_environment = SimpleNamespace(exec=AsyncMock(side_effect=RuntimeError("provider unavailable")))

    await trial._collect_artifacts()

    trial._artifact_handler.download_artifacts.assert_awaited_once()
    trial.logger.warning.assert_called_once()
    assert trial.logger.warning.call_args.kwargs["exc_info"] is True


@pytest.mark.asyncio
async def test_recover_outputs_collects_artifacts_when_not_collected(
    tmp_path: Path,
) -> None:
    trial = _single_step_trial(tmp_path)

    await trial._recover_outputs()

    trial._artifact_handler.download_artifacts.assert_awaited_once()


@pytest.mark.asyncio
async def test_recovery_runs_collect_hook_before_artifact_download(
    tmp_path: Path,
) -> None:
    trial = _single_step_trial(tmp_path)
    trial.task = SimpleNamespace(
        config=TaskConfig.model_validate(
            {
                "verifier": {
                    "collect": [
                        {
                            "command": "make-model-patch",
                            "timeout_sec": 0.25,
                            "user": 1000,
                        }
                    ]
                }
            }
        )
    )
    events: list[str] = []

    async def exec_hook(command: str, **kwargs) -> ExecResult:
        events.append(f"hook:{command}")
        assert kwargs == {"timeout_sec": 1, "user": 1000}
        return ExecResult(stdout="", stderr="", return_code=0)

    async def download_artifacts(*args, **kwargs) -> None:
        events.append("download")

    trial.agent_environment = SimpleNamespace(exec=exec_hook)
    trial._artifact_handler.download_artifacts = AsyncMock(side_effect=download_artifacts)

    await trial._recover_outputs()

    assert events == ["hook:make-model-patch", "download"]


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
