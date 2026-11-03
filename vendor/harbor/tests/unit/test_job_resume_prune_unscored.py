"""Regression test for the warm-resume UNSCORED-trial prune (``jobs._prune_unscored_trials``).

Bug: a below-gate eval leg re-fired warm never converged because ~31 trials hit an
``AgentTimeout`` whose post-timeout verifier then hit a Daytona 401 -> the trial ended
UNSCORED (``verifier_result=None``) but was mislabeled with the benign ``AgentTimeoutError``
leaf (first-wins). The OLD prune keyed on ``exception_type in filter_error_types``, so the
mislabeled-benign type (not in the list) was neither counted toward completeness nor
re-run -> the leg ceilinged ~72-81% forever. Worse, the type-keyed prune could DELETE a
SCORED trial that also recorded a filtered infra leaf (the clobber vector).

Fix: prune iff ``verifier_result is None`` and NEVER delete a scored trial. This test
locks that behavior.
"""

from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

from harbor.cli.jobs import _prune_unscored_trials
from harbor.job import Job
from harbor.metrics.mean import Mean
from harbor.models.job.config import JobConfig
from harbor.models.trial.config import TaskConfig
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult


def _exc(exc_type: str) -> ExceptionInfo:
    return ExceptionInfo(
        exception_type=exc_type,
        exception_message="msg",
        exception_traceback="tb",
        occurred_at=datetime(2026, 7, 14, tzinfo=timezone.utc),
    )


def _make_job(tmp_path: Path, task_paths: list[str]) -> Job:
    config = JobConfig(job_name="prune-test", jobs_dir=tmp_path, n_concurrent_trials=4)
    metrics = defaultdict(lambda: [Mean()])
    return Job(
        config,
        _task_configs=[TaskConfig(path=Path(p)) for p in task_paths],
        _metrics=metrics,
    )


def _seed_trial(
    job: Job,
    trial_config,
    *,
    verifier_result: VerifierResult | None,
    exception_info: ExceptionInfo | None,
) -> Path:
    result = TrialResult(
        task_name=trial_config.task.get_task_id().get_name(),
        trial_name=trial_config.trial_name,
        trial_uri=f"file:///tmp/{trial_config.trial_name}",
        task_id=trial_config.task.get_task_id(),
        source=trial_config.task.source,
        task_checksum="abc123",
        config=trial_config,
        agent_info=AgentInfo(name="test-agent", version="1.0"),
        verifier_result=verifier_result,
        exception_info=exception_info,
    )
    trial_dir = job.job_dir / trial_config.trial_name
    trial_dir.mkdir(parents=True, exist_ok=True)
    (trial_dir / "result.json").write_text(result.model_dump_json(indent=4))
    return trial_dir


@pytest.mark.unit
def test_prune_keeps_scored_removes_unscored(tmp_path: Path) -> None:
    job = _make_job(
        tmp_path,
        ["/tmp/task-a", "/tmp/task-b", "/tmp/task-c", "/tmp/task-d", "/tmp/task-e"],
    )
    trial_configs = list(job._trial_configs)
    assert len(trial_configs) >= 5

    # (verifier_result, exception_info, expect_kept)
    scenarios = [
        # scored, no exception -> KEEP
        (VerifierResult(rewards={"reward": 1}), None, True),
        # scored (legit reward 0) even WITH an AgentTimeout leaf -> KEEP (counts valid)
        (VerifierResult(rewards={"reward": 0}), _exc("AgentTimeoutError"), True),
        # UNSCORED, mislabeled benign AgentTimeoutError (the culprit) -> REMOVE + re-run
        (None, _exc("AgentTimeoutError"), False),
        # UNSCORED, honest infra leaf -> REMOVE + re-run
        (None, _exc("AddTestsDirError"), False),
        # scored but ALSO carries an infra leaf -> KEEP (closes the clobber vector)
        (VerifierResult(rewards={"reward": 1}), _exc("DaytonaError"), True),
    ]

    expect_kept: dict[str, bool] = {}
    for tc, (vr, exc, kept) in zip(trial_configs, scenarios):
        _seed_trial(job, tc, verifier_result=vr, exception_info=exc)
        expect_kept[tc.trial_name] = kept

    _prune_unscored_trials(job.job_dir)

    for trial_name, kept in expect_kept.items():
        exists = (job.job_dir / trial_name).exists()
        assert exists == kept, (
            f"{trial_name}: expected kept={kept}, got exists={exists}"
        )


@pytest.mark.unit
def test_prune_removes_corrupt_keeps_missing_result(tmp_path: Path) -> None:
    job = _make_job(tmp_path, ["/tmp/task-a", "/tmp/task-b"])
    trial_configs = list(job._trial_configs)
    assert len(trial_configs) >= 2

    # corrupt/truncated result.json -> treated as incomplete -> REMOVE + re-run
    corrupt_dir = job.job_dir / trial_configs[0].trial_name
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    (corrupt_dir / "result.json").write_text("{ not valid json")

    # NO result.json -> left as-is (Job.create re-runs genuinely-missing trials)
    missing_dir = job.job_dir / trial_configs[1].trial_name
    missing_dir.mkdir(parents=True, exist_ok=True)

    _prune_unscored_trials(job.job_dir)

    assert not corrupt_dir.exists(), "corrupt result.json trial should be removed"
    assert missing_dir.exists(), "trial with no result.json should be left as-is"
