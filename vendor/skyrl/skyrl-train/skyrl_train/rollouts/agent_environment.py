"""Harbor agent access to a machine owned by the rollout engine."""

from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from shellbox.machine import HARBOR_EXEC_OUTPUT_LIMIT_BYTES, Command, ExitReason, Machine


class BorrowedMachineEnvironment(BaseEnvironment):
    """Expose commands and transfers without owning container creation or deletion."""

    def __init__(self, machine: Machine, trial_dir: Path, session_id: str, task_env_config: EnvironmentConfig):
        self.machine = machine
        super().__init__(
            environment_dir=trial_dir / "environment",
            environment_name=session_id,
            session_id=session_id,
            trial_paths=TrialPaths(trial_dir),
            task_env_config=task_env_config,
        )

    @staticmethod
    def type() -> str:
        return "engine-owned-machine"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        # Network policy has already been applied when the engine created the machine.
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self) -> None:
        # The engine validates and creates the machine before constructing this view.
        pass

    async def start(self, force_build: bool) -> None:
        raise RuntimeError("The rollout engine owns machine startup")

    async def stop(self, delete: bool) -> None:
        raise RuntimeError("The rollout engine owns machine cleanup")

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        selected_user = self._resolve_user(user)
        result = await self.machine.run(
            Command(
                ("bash", "-c", command),
                cwd=cwd,
                env=self._merge_env(env) or {},
                timeout=timeout_sec,
                output_limit_bytes=HARBOR_EXEC_OUTPUT_LIMIT_BYTES,
                user=None if selected_user is None else str(selected_user),
            )
        )
        if result.reason is ExitReason.TIMED_OUT:
            raise TimeoutError("Installed-agent command timed out")
        assert result.exit_code is not None
        return ExecResult(
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
            return_code=result.exit_code,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self.machine.upload(Path(source_path), target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self.machine.upload(Path(source_dir), target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self.machine.download(source_path, Path(target_path))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        await self.machine.download(source_dir, target)
