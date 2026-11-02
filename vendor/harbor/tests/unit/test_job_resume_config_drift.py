"""Tests for the JOB-level config-drift guard in
``Job._maybe_init_existing_job``.

Regression target: the end-of-job ``harbor jobs start ... --export-push``
re-entry (and every preempt-resume) rebuilds the ``JobConfig`` from CLI args,
which no longer byte-matches the ``config.json`` written by the original
launch. A crude exact-equality guard in ``_maybe_init_existing_job`` raised
``FileExistsError`` on ANY such drift — BEFORE the authoritative per-trial
identity reconciliation in ``_init_remaining_trial_configs`` ran — so the whole
attempt FAILED and traces were never exported. Benign job-level drift must now
resume/export; a genuine SEMANTIC mismatch (different task set) must still
raise via the per-trial guard.
"""

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harbor.job import Job
from harbor.metrics.mean import Mean
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.result import AgentInfo, TrialResult
from harbor.models.verifier.result import VerifierResult

_JOB_NAME = "job-config-drift-test"


def _make_job(
    tmp_path: Path,
    task_configs: list[TaskConfig],
    *,
    n_concurrent_trials: int = 4,
) -> Job:
    config = JobConfig(
        job_name=_JOB_NAME,
        jobs_dir=tmp_path,
        n_concurrent_trials=n_concurrent_trials,
    )
    metrics = defaultdict(lambda: [Mean()])
    return Job(config, _task_configs=task_configs, _metrics=metrics)


def _trial_result(trial_config: TrialConfig) -> TrialResult:
    return TrialResult(
        task_name=trial_config.task.get_task_id().get_name(),
        trial_name=trial_config.trial_name,
        trial_uri=f"file:///tmp/{trial_config.trial_name}",
        task_id=trial_config.task.get_task_id(),
        source=trial_config.task.source,
        task_checksum="abc123",
        config=trial_config,
        agent_info=AgentInfo(name="test-agent", version="1.0"),
        verifier_result=VerifierResult(rewards={"reward": 1}),
        exception_info=None,
    )


def _seed_completed_job(tmp_path: Path, task_paths: list[str]) -> None:
    """Write an on-disk completed job (config + result + one trial dir each)."""
    task_configs = [TaskConfig(path=Path(p)) for p in task_paths]
    job = _make_job(tmp_path, task_configs)

    results: list[TrialResult] = []
    for trial_config in job._trial_configs:
        result = _trial_result(trial_config)
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


@pytest.mark.unit
def test_job_config_drift_resumes_and_loads_existing_trials(tmp_path: Path) -> None:
    """A drifted JOB config (here: n_concurrent_trials) that leaves trial
    identity unchanged must NOT raise — the existing trials load and there is
    nothing left to run. This is the export-push / preempt-resume path."""
    task_paths = ["/tmp/task-a", "/tmp/task-b"]
    _seed_completed_job(tmp_path, task_paths)

    task_configs = [TaskConfig(path=Path(p)) for p in task_paths]
    # Drift the job-level config so `existing_config != self.config`.
    resumed = _make_job(tmp_path, task_configs, n_concurrent_trials=99)
    try:
        assert len(resumed._existing_trial_results) == len(task_paths)
        # All on-disk trials exact-match the plan -> nothing left to run.
        assert resumed._remaining_trial_configs == []
    finally:
        resumed._close_logger_handlers()


@pytest.mark.unit
def test_semantic_mismatch_still_raises(tmp_path: Path) -> None:
    """The per-trial guard is still the real safety net: a resume whose planned
    tasks don't cover the on-disk trials (a different evaluation) must abort."""
    _seed_completed_job(tmp_path, ["/tmp/task-a", "/tmp/task-b"])

    # Plan a DIFFERENT task set with no counterpart for the on-disk trials.
    task_configs = [TaskConfig(path=Path("/tmp/task-c"))]
    with pytest.raises(ValueError, match="SEMANTIC mismatch"):
        _make_job(tmp_path, task_configs)
