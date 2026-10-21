"""Harbor agent access to a machine owned by the rollout engine."""

import json
from collections.abc import Awaitable
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

from harbor.agents.installed.base import NonZeroAgentExitCodeError
from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities
from harbor.models.task.config import EnvironmentConfig
from harbor.models.trial.paths import TrialPaths
from shellbox.machine import HARBOR_EXEC_OUTPUT_LIMIT_BYTES, Command, ExitReason, Machine

_Result = TypeVar("_Result")

# Only constant Harbor error messages may become diagnostic labels. Unrecognized
# ValueError text may contain task output or credentials and is never persisted.
_PI_FAILURE_CODES = {
    "Pi evidence stdout contained non-frame output": "non_frame_stdout",
    "Pi returned malformed in-memory evidence chunks: missing separator": "missing_frame_separator",
    "Pi returned malformed in-memory evidence chunks: invalid header": "invalid_frame_header",
    "Pi returned malformed in-memory evidence chunks: empty fragment": "empty_frame_fragment",
    "Pi returned malformed in-memory evidence chunks: oversized fragment": "oversized_frame_fragment",
    "Pi returned conflicting in-memory evidence chunks": "conflicting_frame_chunks",
    "Pi did not return exactly one in-memory evidence frame (missing chunks)": "missing_frame_chunks",
    "Pi returned a corrupted in-memory evidence frame": "corrupted_frame",
    "Pi in-memory evidence must be an object": "non_object_frame",
    "Pi evidence frame must contain one collated trajectory": "missing_collated_trajectory",
    "Pi evidence frame must contain model calls and one collated trajectory": "missing_calls_or_trajectory",
}


def _pi_failure_code(error: BaseException) -> str | None:
    if type(error) is not ValueError:
        return None
    return _PI_FAILURE_CODES.get(str(error), "other_value_error")


def _nonzero_agent_failure_code(result: ExecResult) -> str:
    """Classify install/agent exits without persisting command output."""
    output = f"{result.stderr or ''}\n{result.stdout or ''}".lower()
    if "harbor_pi_missing_assistant_result" in output:
        return "missing_assistant_result"
    if "harbor_pi_assistant_error" in output:
        return "assistant_error"
    if "eai_again" in output or "temporary failure resolving" in output:
        return "dns_temporary_failure"
    if "etimedout" in output or "connection timed out" in output:
        return "network_timeout"
    if "econnreset" in output or "connection reset" in output:
        return "connection_reset"
    if "enospc" in output or "no space left on device" in output:
        return "sandbox_disk_full"
    if "npm error code e404" in output or "npm err! code e404" in output:
        return "npm_package_not_found"
    if "failed to fetch" in output:
        return "apt_fetch_failed"
    if "curl: (" in output:
        return "curl_failed"
    if "npm error" in output or "npm err!" in output:
        return "npm_failed"
    return "unclassified_nonzero_exit"


class BorrowedMachineEnvironment(BaseEnvironment):
    """Expose commands and transfers without owning container creation or deletion."""

    def __init__(self, machine: Machine, trial_dir: Path, session_id: str, task_env_config: EnvironmentConfig):
        self.machine = machine
        self._diagnostics_path = trial_dir / "engine" / "diagnostics.jsonl"
        self._operation_index = 0
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

    def record_diagnostic(self, phase: str, *, error: BaseException | None = None) -> None:
        """Record fixed phase names and error classes, never commands, URLs or output."""
        self._diagnostics_path.parent.mkdir(parents=True, exist_ok=True)
        record = {"timestamp_utc": datetime.now(UTC).isoformat(), "phase": phase}
        if error is not None:
            record["error_type"] = type(error).__name__
            if code := _pi_failure_code(error):
                record["error_code"] = code
            elif isinstance(error, NonZeroAgentExitCodeError) and error.result is not None:
                record["error_code"] = _nonzero_agent_failure_code(error.result)
                if 0 <= error.result.return_code <= 255:
                    record["exit_code"] = error.result.return_code
        with self._diagnostics_path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    async def _diagnose(self, operation: str, work: Awaitable[_Result]) -> _Result:
        self._operation_index += 1
        phase = f"{operation}:{self._operation_index}"
        self.record_diagnostic(f"{phase}:start")
        try:
            result = await work
        except BaseException as error:
            self.record_diagnostic(f"{phase}:failed", error=error)
            raise
        self.record_diagnostic(f"{phase}:completed")
        return result

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        selected_user = self._resolve_user(user)
        result = await self._diagnose(
            "machine_run",
            self.machine.run(
                Command(
                    ("bash", "-c", command),
                    cwd=cwd,
                    env=self._merge_env(env) or {},
                    timeout=timeout_sec,
                    output_limit_bytes=HARBOR_EXEC_OUTPUT_LIMIT_BYTES,
                    user=None if selected_user is None else str(selected_user),
                )
            ),
        )
        if result.reason is ExitReason.TIMED_OUT:
            self.record_diagnostic("machine_run:timed_out")
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
        await self._diagnose("machine_upload", self.machine.upload(Path(source_path), target_path))

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        await self._diagnose("machine_upload", self.machine.upload(Path(source_dir), target_dir))

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        await self._diagnose("machine_download", self.machine.download(source_path, Path(target_path)))

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        target = Path(target_dir)
        target.mkdir(parents=True, exist_ok=True)
        await self._diagnose("machine_download", self.machine.download(source_dir, target))
