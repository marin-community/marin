import asyncio
import traceback
from datetime import datetime, timezone
from pathlib import PurePosixPath
from typing import Protocol

from upath import UPath

from harbor.models.trial.config import TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.trial.errors import TrialTimeoutError
from harbor.utils.artifact_writer import ArtifactPersistenceError, get_artifact_writer


class _ResultSource(Protocol):
    result: TrialResult
    first_phase_timeout: asyncio.TimeoutError | None
    has_result: bool


def trial_paths(trial_config: TrialConfig, attempt: int | None = None) -> TrialPaths:
    """Return logical or attempt-scoped paths for a trial configuration."""
    trial_dir = trial_config.trials_dir / trial_config.trial_name
    if attempt is not None:
        trial_dir = trial_dir / "attempts" / f"{attempt:03d}"
    return TrialPaths(trial_dir)


def trial_relative_path(trial_config: TrialConfig, attempt: int | None = None) -> str:
    """Return the portable path from trials_dir to this trial attempt."""
    path = PurePosixPath(trial_config.trial_name)
    if attempt is not None:
        path = path / "attempts" / f"{attempt:03d}"
    return path.as_posix()


async def _write_text(path: UPath, text: str, description: str) -> None:
    writer = get_artifact_writer()
    future = await writer.submit_required(path, lambda: text, description=description)
    await writer.await_write(future)


def _artifact_persistence_error(
    error: BaseExceptionGroup,
) -> ArtifactPersistenceError | None:
    """Return a representative error when every grouped failure is persistence-related."""
    artifact_errors: list[ArtifactPersistenceError] = []
    pending: list[BaseException] = [error]
    while pending:
        current = pending.pop()
        if isinstance(current, BaseExceptionGroup):
            pending.extend(reversed(current.exceptions))
        elif isinstance(current, ArtifactPersistenceError):
            artifact_errors.append(current)
        else:
            return None
    return artifact_errors[0] if artifact_errors else None


async def persist_selected_result(
    trial_config: TrialConfig,
    attempt: int,
    result: TrialResult,
) -> None:
    """Persist an immutable attempt result and select it at the trial root."""
    attempt_paths = trial_paths(trial_config, attempt)
    logical_paths = trial_paths(trial_config)
    if attempt_paths._is_local():
        attempt_paths.trial_dir.mkdir(parents=True, exist_ok=True)
        logical_paths.trial_dir.mkdir(parents=True, exist_ok=True)
    result.trial_relpath = trial_relative_path(trial_config, attempt)
    result_json = result.model_dump_json(indent=4)
    get_artifact_writer().submit_optional(
        logical_paths.config_path,
        lambda: result.config.model_dump_json(indent=4),
        description="selected trial config.json",
    )
    try:
        async with asyncio.TaskGroup() as task_group:
            task_group.create_task(
                _write_text(
                    logical_paths.result_path,
                    result_json,
                    "selected trial result.json",
                )
            )
            task_group.create_task(
                _write_text(
                    attempt_paths.result_path,
                    result_json,
                    "selected trial attempt result.json",
                )
            )
    except ExceptionGroup as error:
        persistence_error = _artifact_persistence_error(error)
        if persistence_error is None:
            raise
        raise persistence_error from error


def _exception_info(error: BaseException) -> ExceptionInfo:
    return ExceptionInfo(
        exception_type=type(error).__name__,
        exception_message=str(error),
        exception_traceback="".join(traceback.format_exception(error)),
        occurred_at=datetime.now(timezone.utc),
    )


def _fallback_result(
    trial_config: TrialConfig,
    attempt: int,
    started_at: datetime,
) -> TrialResult:
    task_id = trial_config.task.get_task_id()
    agent_name = trial_config.agent.name or trial_config.agent.import_path or "unknown"
    version = str(trial_config.agent.kwargs.get("version", "unknown"))
    paths = trial_paths(trial_config, attempt)
    return TrialResult(
        task_name=task_id.get_name(),
        trial_name=trial_config.trial_name,
        trial_uri=paths.trial_dir.expanduser().resolve().as_uri(),
        trial_relpath=trial_relative_path(trial_config, attempt),
        task_id=task_id,
        task_checksum="",
        config=trial_config,
        agent_info=AgentInfo(name=agent_name, version=version),
        started_at=started_at,
    )


def _terminal_result(
    trial_config: TrialConfig,
    attempt: int,
    source: _ResultSource | None,
    started_at: datetime,
) -> TrialResult:
    if source is not None and source.has_result:
        result = source.result.model_copy(deep=True)
    else:
        result = _fallback_result(trial_config, attempt, started_at)
    result.finished_at = datetime.now(timezone.utc)
    return result


def timeout_result(
    trial_config: TrialConfig,
    attempt: int,
    source: _ResultSource | None,
    started_at: datetime,
) -> TrialResult:
    """Build the terminal result selected when an admitted attempt expires."""
    result = _terminal_result(trial_config, attempt, source, started_at)

    timeout_error = (
        source.first_phase_timeout
        if source is not None and source.first_phase_timeout is not None
        else TrialTimeoutError(
            "Trial attempt timed out after "
            f"{trial_config.trial_attempt_timeout_sec} seconds"
        )
    )
    secondary = list(result.secondary_exception_info)
    timeout_info = _exception_info(timeout_error)
    if result.exception_info is not None and (
        result.exception_info.exception_type != timeout_info.exception_type
        or result.exception_info.exception_message != timeout_info.exception_message
    ):
        secondary.append(result.exception_info)
    result.exception_info = timeout_info
    result.secondary_exception_info = secondary
    return result


def exception_result(
    trial_config: TrialConfig,
    attempt: int,
    source: _ResultSource | None,
    started_at: datetime,
    error: BaseException,
) -> TrialResult:
    """Build the terminal result selected when an admitted attempt raises."""
    result = _terminal_result(trial_config, attempt, source, started_at)

    error_info = _exception_info(error)
    if isinstance(error, ArtifactPersistenceError):
        if result.exception_info is not None:
            result.secondary_exception_info = [
                *result.secondary_exception_info,
                result.exception_info,
            ]
        result.exception_info = error_info
    elif result.exception_info is None:
        result.exception_info = error_info
    else:
        result.secondary_exception_info = [
            *result.secondary_exception_info,
            error_info,
        ]
    return result
