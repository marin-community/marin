# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Harbor adapter for ShellSim's built-in Unix-like environment."""

from enum import StrEnum
from pathlib import Path

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities

from harbor_qemu.machine import (
    HARBOR_EXEC_OUTPUT_LIMIT_BYTES,
    Command,
    ExitReason,
    MachineSpec,
    NetworkPolicy,
    ShellSession,
    ShellSimBuiltins,
)
from harbor_qemu.shellsim_machine import (
    DEFAULT_CPU_LIMIT,
    DEFAULT_DISK_LIMIT,
    DEFAULT_MEMORY_MB,
    DEFAULT_OUTPUT_LIMIT,
    ShellSimMachine,
    ShellSimMachineFactory,
)


class TaskNetworkPolicy(StrEnum):
    REQUIRE_OFFLINE_TASK = "require-offline-task"
    DENY = "deny"


class ShellSimEnvironment(BaseEnvironment):
    """Run a Harbor trial in ShellSim, ignoring its Docker image or Dockerfile."""

    def __init__(
        self,
        *args,
        network_policy: TaskNetworkPolicy = TaskNetworkPolicy.REQUIRE_OFFLINE_TASK,
        cpu_limit: int = DEFAULT_CPU_LIMIT,
        memory_mb: int = DEFAULT_MEMORY_MB,
        disk_limit_bytes: int = DEFAULT_DISK_LIMIT,
        output_limit_bytes: int = DEFAULT_OUTPUT_LIMIT,
        **kwargs,
    ):
        self.network_policy = TaskNetworkPolicy(network_policy)
        self.cpu_limit = cpu_limit
        self.memory_mb = memory_mb
        self.disk_limit_bytes = disk_limit_bytes
        self.output_limit_bytes = output_limit_bytes
        self.machine: ShellSimMachine | None = None
        super().__init__(*args, **kwargs)

    @staticmethod
    def type() -> str:
        return "shellsim"

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True)

    def _validate_definition(self) -> None:
        if self.task_env_config.allow_internet and self.network_policy is TaskNetworkPolicy.REQUIRE_OFFLINE_TASK:
            raise ValueError("ShellSim has no guest network; set network_policy='deny' to run this task offline")
        if any((self.environment_dir / name).exists() for name in ("docker-compose.yaml", "docker-compose.yml")):
            raise ValueError("ShellSim does not support Compose tasks")

    async def start(self, force_build: bool) -> None:
        workdir = self.task_env_config.workdir or "/workspace"
        spec = MachineSpec(ShellSimBuiltins(), workdir=workdir, network=NetworkPolicy.DENY, memory_mb=self.memory_mb)
        self.machine = await ShellSimMachineFactory(
            cpu=self.cpu_limit,
            disk=self.disk_limit_bytes,
            output=self.output_limit_bytes,
        ).create(spec)
        try:
            if self.environment_dir.is_dir():
                await self.machine.upload(self.environment_dir, workdir)
        except BaseException:
            await self.stop(True)
            raise

    async def stop(self, delete: bool) -> None:
        if self.machine is not None:
            await self.machine.close()
            self.machine = None

    async def open_bash_session(self) -> ShellSession:
        """Open the agent's persistent simulated shell."""
        if self.machine is None:
            raise RuntimeError("ShellSim environment is not running")
        return await self.machine.open_shell()

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if user not in (None, "root", 0):
            raise ValueError("ShellSim supports only the root user")
        if self.machine is None:
            raise RuntimeError("ShellSim environment is not running")
        result = await self.machine.run(
            Command(
                argv=("sh", "-c", command),
                cwd=cwd,
                env=self._merge_env(env) or {},
                timeout=timeout_sec,
                output_limit_bytes=HARBOR_EXEC_OUTPUT_LIMIT_BYTES,
            )
        )
        if result.reason is ExitReason.TIMED_OUT:
            raise TimeoutError("ShellSim command timed out")
        return ExecResult(
            stdout=result.stdout.decode(errors="replace"),
            stderr=result.stderr.decode(errors="replace"),
            return_code=result.exit_code,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        if self.machine is None:
            raise RuntimeError("ShellSim environment is not running")
        await self.machine.upload(Path(source_path), target_path)

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        if self.machine is None:
            raise RuntimeError("ShellSim environment is not running")
        await self.machine.upload(Path(source_dir), target_dir)

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        if self.machine is None:
            raise RuntimeError("ShellSim environment is not running")
        await self.machine.download(source_path, Path(target_path))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        if self.machine is None:
            raise RuntimeError("ShellSim environment is not running")
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        await self.machine.download(source_dir, target)
