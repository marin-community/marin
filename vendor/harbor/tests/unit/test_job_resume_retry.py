"""Tests for the RESUME-level "re-run vs keep" gate in ``harbor jobs resume``.

On a fresh resume, a trial whose on-disk result is *resume-retryable*
(``CancelledError``, ``SandboxBuildFailedError``, or
``ContextManagementInfrastructureError`` — an interrupted / cap-full /
context-recovery-failed attempt) must be RE-RUN, while valid results and
genuinely-terminal errors must be KEPT. See ``Job._RESUME_RETRYABLE_ERROR_TYPES``
/ ``Job._is_resume_retryable_result`` / ``Job._load_existing_trial``.

Legacy results persisted before ``ContextManagementInfrastructureError``
existed carry the provider's ``BadRequestError`` type; a resume still re-runs
the ones whose payload is the exact context-budget-exhausted signal
(``max_tokens must be at least 1, got 0``), and ONLY those.

This is DISTINCT from the in-trial (single-attempt) non-retryable
classification in ``harbor/environments/daytona/utils.py``, which is unchanged.
"""

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harbor.job import Job
from harbor.llms.base import ContextManagementInfrastructureError
from harbor.metrics.mean import Mean
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult

# vLLM's structured context-budget-exhausted payload (the legacy signal).
_MAX_TOKENS_ZERO_MSG = (
    "Error code: 400 - 'max_tokens must be at least 1, got 0. "
    "(parameter=max_tokens, value=0)'"
)


def _make_job(tmp_path: Path, task_configs: list[TaskConfig]) -> Job:
    config = JobConfig(job_name="job-resume-retry-test", jobs_dir=tmp_path)
    metrics = defaultdict(lambda: [Mean()])
    return Job(config, _task_configs=task_configs, _metrics=metrics)


def _trial_result(
    trial_config: TrialConfig,
    *,
    exception_type: str | None = None,
    exception_message: str = "failed",
) -> TrialResult:
    exception_info = None
    if exception_type is not None:
        exception_info = ExceptionInfo(
            exception_type=exception_type,
            exception_message=exception_message,
            exception_traceback=f"Traceback:\n{exception_message}",
            occurred_at=datetime(2026, 4, 28, tzinfo=timezone.utc),
        )
    return TrialResult(
        task_name=trial_config.task.get_task_id().get_name(),
        trial_name=trial_config.trial_name,
        trial_uri=f"file:///tmp/{trial_config.trial_name}",
        task_id=trial_config.task.get_task_id(),
        source=trial_config.task.source,
        task_checksum="abc123",
        config=trial_config,
        agent_info=AgentInfo(name="test-agent", version="1.0"),
        verifier_result=(
            None
            if exception_type is not None
            else VerifierResult(rewards={"reward": 1})
        ),
        exception_info=exception_info,
    )


# task.path -> (exception_type, exception_message) stamped on that trial's
# on-disk result. None type => a valid (scored, no-exception) result.
_ROLES: dict[str, tuple[str | None, str]] = {
    "/tmp/task-sandbox": ("SandboxBuildFailedError", "failed"),
    "/tmp/task-cancelled": ("CancelledError", "failed"),
    "/tmp/task-ctx-mgmt": ("ContextManagementInfrastructureError", "failed"),
    # Legacy pre-fix result: BadRequestError + the structured max_tokens=0 signal.
    "/tmp/task-legacy-400": ("BadRequestError", _MAX_TOKENS_ZERO_MSG),
    "/tmp/task-valid": (None, ""),
    "/tmp/task-terminal": ("RuntimeError", "failed"),
    # Unrelated BadRequestError (a genuine misconfiguration) stays terminal.
    "/tmp/task-bad-request-unrelated": (
        "BadRequestError",
        "Invalid value for 'temperature': must be between 0 and 2.",
    ),
}

_RERUN_PATHS = {
    "/tmp/task-sandbox",
    "/tmp/task-cancelled",
    "/tmp/task-ctx-mgmt",
    "/tmp/task-legacy-400",
}
_KEPT_PATHS = {
    "/tmp/task-valid",
    "/tmp/task-terminal",
    "/tmp/task-bad-request-unrelated",
}


def _seed_completed_job(tmp_path: Path) -> Job:
    """Write an on-disk completed job (config + result + one trial dir per role)."""
    task_configs = [TaskConfig(path=Path(p)) for p in _ROLES]
    job = _make_job(tmp_path, task_configs)

    results: list[TrialResult] = []
    for trial_config in job._trial_configs:
        exc_type, exc_msg = _ROLES[str(trial_config.task.path)]
        result = _trial_result(
            trial_config, exception_type=exc_type, exception_message=exc_msg
        )
        results.append(result)
        trial_dir = job.job_dir / trial_config.trial_name
        trial_dir.mkdir(parents=True, exist_ok=True)
        (trial_dir / "config.json").write_text(trial_config.model_dump_json(indent=4))
        (trial_dir / "result.json").write_text(result.model_dump_json(indent=4))

    job._job_config_path.write_text(job.config.model_dump_json(indent=4))
    job._job_result_path.write_text(
        JobResult(
            id=job.id,
            started_at=datetime(2026, 4, 28, 8, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 4, 28, 8, 0, tzinfo=timezone.utc),
            n_total_trials=len(results),
            stats=JobStats.from_trial_results(results, n_total_trials=len(results)),
        ).model_dump_json(indent=4)
    )
    job._close_logger_handlers()
    return job


@pytest.mark.unit
def test_resume_reruns_retryable_and_keeps_the_rest(tmp_path: Path) -> None:
    _seed_completed_job(tmp_path)

    task_configs = [TaskConfig(path=Path(p)) for p in _ROLES]
    resumed = _make_job(tmp_path, task_configs)
    try:
        rerun_paths = {str(cfg.task.path) for cfg in resumed._remaining_trial_configs}
        kept_paths = {
            str(res.config.task.path) for res in resumed._existing_trial_results
        }

        assert rerun_paths == _RERUN_PATHS
        assert kept_paths == _KEPT_PATHS

        # Re-run and kept partitions are disjoint and cover every trial.
        assert rerun_paths.isdisjoint(kept_paths)
        assert rerun_paths | kept_paths == set(_ROLES)

        # The retryable trials' on-disk dirs were removed (so they re-run clean),
        # while the kept trials' dirs survive.
        surviving = {p.name for p in resumed.job_dir.iterdir() if p.is_dir()}
        kept_dir_names = {res.trial_name for res in resumed._existing_trial_results}
        assert surviving == kept_dir_names
    finally:
        resumed._close_logger_handlers()


@pytest.mark.unit
def test_resume_uses_result_config_when_standalone_config_was_dropped(
    tmp_path: Path,
) -> None:
    seeded = _seed_completed_job(tmp_path)
    valid_result = next(
        result
        for trial_dir in seeded.job_dir.iterdir()
        if trial_dir.is_dir()
        for result in [
            TrialResult.model_validate_json((trial_dir / "result.json").read_text())
        ]
        if str(result.config.task.path) == "/tmp/task-valid"
    )
    standalone_config = seeded.job_dir / valid_result.trial_name / "config.json"
    standalone_config.unlink()

    task_configs = [TaskConfig(path=Path(p)) for p in _ROLES]
    resumed = _make_job(tmp_path, task_configs)
    try:
        loaded = next(
            result
            for result in resumed._existing_trial_results
            if result.trial_name == valid_result.trial_name
        )
        assert loaded.config == valid_result.config
        assert not standalone_config.exists()
    finally:
        resumed._close_logger_handlers()


@pytest.mark.unit
def test_is_resume_retryable_result_predicate() -> None:
    task_config = TaskConfig(path=Path("/tmp/task-pred"))
    job = _make_job(tmp_path=Path("/tmp"), task_configs=[task_config])
    trial_config = job._trial_configs[0]
    job._close_logger_handlers()

    sandbox = _trial_result(trial_config, exception_type="SandboxBuildFailedError")
    cancelled = _trial_result(trial_config, exception_type="CancelledError")
    ctx_mgmt = _trial_result(
        trial_config, exception_type="ContextManagementInfrastructureError"
    )
    valid = _trial_result(trial_config, exception_type=None)
    terminal = _trial_result(trial_config, exception_type="RuntimeError")

    assert Job._is_resume_retryable_result(sandbox) is True
    assert Job._is_resume_retryable_result(cancelled) is True
    assert Job._is_resume_retryable_result(ctx_mgmt) is True
    assert Job._is_resume_retryable_result(valid) is False
    assert Job._is_resume_retryable_result(terminal) is False

    # _is_cancelled_result stays cancelled-specific (bookkeeping), NOT broadened.
    assert Job._is_cancelled_result(sandbox) is False
    assert Job._is_cancelled_result(cancelled) is True
    assert Job._is_cancelled_result(ctx_mgmt) is False


@pytest.mark.unit
def test_legacy_bad_request_predicate_is_narrow() -> None:
    """Resume re-runs legacy BadRequestError results ONLY when their payload is
    the exact context-budget-exhausted signal — a generic BadRequestError stays
    terminal."""
    task_config = TaskConfig(path=Path("/tmp/task-legacy-pred"))
    job = _make_job(tmp_path=Path("/tmp"), task_configs=[task_config])
    trial_config = job._trial_configs[0]
    job._close_logger_handlers()

    legacy_zero = _trial_result(
        trial_config,
        exception_type="BadRequestError",
        exception_message=_MAX_TOKENS_ZERO_MSG,
    )
    unrelated_400 = _trial_result(
        trial_config,
        exception_type="BadRequestError",
        exception_message="Invalid temperature",
    )

    assert Job._is_resume_retryable_result(legacy_zero) is True
    assert Job._is_resume_retryable_result(unrelated_400) is False


@pytest.mark.unit
def test_exception_info_records_infra_exception_type() -> None:
    """The typed infrastructure exception persists under its own type name, so a
    never-scored trial is visible and resume-retryable in exception_stats."""
    info = ExceptionInfo.from_exception(
        ContextManagementInfrastructureError("recovery exhausted")
    )
    assert info.exception_type == "ContextManagementInfrastructureError"
