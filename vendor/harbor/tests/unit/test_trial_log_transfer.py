import logging
from pathlib import Path, PurePosixPath
from types import SimpleNamespace

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.agent.context import AgentContext
from harbor.models.environment_type import EnvironmentType
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
    VerifierConfig,
)
from harbor.trial.trial import Trial


class _QuickAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "quick"

    def version(self) -> str:
        return "1.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        pass


class _DiagnosticLogFailureEnvironment(BaseEnvironment):
    @staticmethod
    def type() -> EnvironmentType:
        return EnvironmentType.DOCKER

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(mounted=False)

    def _validate_definition(self) -> None:
        pass

    async def start(self, force_build: bool) -> None:
        pass

    async def stop(self, delete: bool) -> None:
        pass

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        pass

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        pass

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        pass

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        if source_dir == "/logs/agent":
            raise RuntimeError("Daytona filesystem unavailable")

        target.mkdir(parents=True, exist_ok=True)
        if source_dir == "/logs/verifier":
            (target / "reward.txt").write_text("1.0")

    async def is_dir(self, path: str, user: str | int | None = None) -> bool:
        return True

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        return ExecResult(stdout="", stderr="", return_code=0)


class _FlakyLogTransferEnvironment:
    capabilities = SimpleNamespace(mounted=False)

    def __init__(self) -> None:
        self.downloaded = False

    async def download_dir(self, source_dir: str, target_dir: Path) -> None:
        if not self.downloaded:
            self.downloaded = True
            raise RuntimeError("Daytona filesystem unavailable")
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "trajectory.json").write_text("{}")

    async def upload_dir(self, source_dir: Path, target_dir: str) -> None:
        raise RuntimeError("Daytona filesystem unavailable")


def _trial_for_log_transfer(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        _are_agent_logs_downloaded=False,
        agent_environment=_FlakyLogTransferEnvironment(),
        agent_env_paths=SimpleNamespace(agent_dir=PurePosixPath("/logs/agent")),
        paths=SimpleNamespace(agent_dir=tmp_path / "agent"),
        config=SimpleNamespace(
            agent=SimpleNamespace(upload_agent_logs=True),
        ),
        logger=logging.getLogger("test.trial-log-transfer"),
    )


def _scored_trial_config(tmp_path: Path) -> TrialConfig:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text(
        "[agent]\ntimeout_sec = 10.0\n[verifier]\ntimeout_sec = 10.0\n[environment]\n"
    )
    (task_dir / "instruction.md").write_text("Do nothing.\n")
    (task_dir / "environment").mkdir()
    (task_dir / "environment" / "Dockerfile").write_text("FROM ubuntu:24.04\n")
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")

    trials_dir = tmp_path / "trials"
    trials_dir.mkdir()
    return TrialConfig(
        task=TaskConfig(path=task_dir),
        trials_dir=trials_dir,
        agent=AgentConfig(import_path="tests.unit.test_trial_log_transfer:_QuickAgent"),
        environment=EnvironmentConfig(
            import_path=(
                "tests.unit.test_trial_log_transfer:_DiagnosticLogFailureEnvironment"
            ),
            delete=True,
        ),
        verifier=VerifierConfig(),
    )


async def test_log_download_failure_does_not_block_scoring(
    tmp_path: Path, caplog
) -> None:
    trial = await Trial.create(_scored_trial_config(tmp_path))

    with caplog.at_level(logging.WARNING, logger=trial.logger.name):
        result = await trial.run()

    assert result.verifier_result is not None
    assert result.verifier_result.rewards == {"reward": 1.0}
    assert result.exception_info is None
    failure = next(
        record
        for record in caplog.records
        if record.levelno == logging.WARNING
        and record.getMessage().startswith("Failed to download logs")
    )
    assert failure.exc_info is not None
    assert isinstance(failure.exc_info[1], RuntimeError)
    assert str(failure.exc_info[1]) == "Daytona filesystem unavailable"


async def test_failed_log_download_keeps_exception_context_and_can_retry(
    tmp_path: Path, caplog
) -> None:
    trial = _trial_for_log_transfer(tmp_path)

    with caplog.at_level(logging.WARNING, logger=trial.logger.name):
        await Trial._download_agent_logs(trial)
        await Trial._download_agent_logs(trial)

    assert (trial.paths.agent_dir / "trajectory.json").read_text() == "{}"
    failure = next(
        record for record in caplog.records if record.levelno == logging.WARNING
    )
    assert failure.exc_info is not None
    assert isinstance(failure.exc_info[1], RuntimeError)


async def test_failed_log_upload_keeps_exception_context(
    tmp_path: Path, caplog
) -> None:
    trial = _trial_for_log_transfer(tmp_path)

    with caplog.at_level(logging.WARNING, logger=trial.logger.name):
        await Trial._upload_agent_logs(trial)

    failure = next(
        record for record in caplog.records if record.levelno == logging.WARNING
    )
    assert failure.exc_info is not None
    assert isinstance(failure.exc_info[1], RuntimeError)
