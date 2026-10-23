import asyncio
import threading
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pytest

import harbor.utils.artifact_writer as artifact_writer
from harbor.job import Job
from harbor.environments.base import ExecResult
from harbor.metrics.mean import Mean
from harbor.models.job.config import JobConfig
from harbor.models.job.result import JobResult, JobStats
from harbor.models.metric.config import MetricConfig
from harbor.models.metric.type import MetricType
from harbor.models.trial.config import (
    AgentConfig,
    EnvironmentConfig,
    TaskConfig,
    TrialConfig,
)
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.models.verifier.result import VerifierResult
from harbor.trial.hooks import TrialEvent, TrialHookEvent
from harbor.utils.artifact_writer import ArtifactWriter, get_artifact_writer
from tests.unit.test_trial_cleanup import MountedEnvironment


class ScoredEnvironment(MountedEnvironment):
    async def exec(self, command, cwd=None, env=None, timeout_sec=None, user=None):
        self.trial_paths.reward_text_path.write_text("1")
        return ExecResult(return_code=0, stdout="", stderr="")


class _RecoveringCloudResult:
    protocol = "s3"

    def __init__(self, upload_started: Path, release_upload: Path, real_path: Path):
        self._upload_started = upload_started
        self._release_upload = release_upload
        self._real_path = real_path

    def __str__(self) -> str:
        return "s3://bucket/job/result.json"

    def open(self, mode: str):
        self._upload_started.touch()
        _wait_for_path(self._release_upload, timeout=30)
        return self._real_path.open(mode)

    def write_spooled(self, spool_path: Path) -> None:
        self._real_path.write_bytes(spool_path.read_bytes())

    def read_result(self) -> JobResult:
        return JobResult.model_validate_json(self._real_path.read_text())


def _wait_for_path(path: Path, timeout: float) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if path.exists():
            return True
        threading.Event().wait(timeout=0.01)
    return path.exists()


def _make_job(
    tmp_path: Path,
    task_configs: list[TaskConfig] | None = None,
    *,
    release_trial_payloads_in_memory: bool = False,
) -> Job:
    config = JobConfig(
        job_name="job-progress-test",
        jobs_dir=tmp_path,
        release_trial_payloads_in_memory=release_trial_payloads_in_memory,
    )
    metrics = defaultdict(lambda: [Mean()])
    return Job(
        config,
        _task_configs=task_configs or [TaskConfig(path=Path("/tmp/task"))],
        _metrics=metrics,
    )


def _trial_result(
    trial_config: TrialConfig,
    *,
    exception_type: str | None = None,
) -> TrialResult:
    exception_info = None
    if exception_type is not None:
        exception_info = ExceptionInfo(
            exception_type=exception_type,
            exception_message="failed",
            exception_traceback="traceback",
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
        verifier_result=VerifierResult(rewards={"reward": 1}),
        exception_info=exception_info,
    )


def _hook_event(
    event: TrialEvent,
    trial_config: TrialConfig,
    *,
    result: TrialResult | None = None,
) -> TrialHookEvent:
    return TrialHookEvent(
        event=event,
        trial_id=trial_config.trial_name,
        task_name=trial_config.task.get_task_id().get_name(),
        config=trial_config,
        timestamp=datetime(2026, 4, 28, 9, 0, tzinfo=timezone.utc),
        result=result,
    )


def _persist_completed_trial(job: Job, trial_result: TrialResult) -> Path:
    job._job_config_path.write_text(job.config.model_dump_json(indent=4))
    job._job_result_path.write_text(
        JobResult(
            id=job.id,
            started_at=datetime(2026, 4, 28, 8, 0, tzinfo=timezone.utc),
            updated_at=datetime(2026, 4, 28, 8, 0, tzinfo=timezone.utc),
            n_total_trials=1,
            stats=JobStats.from_trial_results([trial_result], n_total_trials=1),
        ).model_dump_json(indent=4)
    )
    trial_dir = job.job_dir / trial_result.trial_name
    trial_dir.mkdir()
    (trial_dir / "config.json").write_text(
        trial_result.config.model_dump_json(indent=4)
    )
    result_path = trial_dir / "result.json"
    result_path.write_text(trial_result.model_dump_json(indent=4))
    job._close_logger_handlers()
    return result_path


async def _flush_job_snapshots(job: Job) -> None:
    failed = await asyncio.to_thread(
        get_artifact_writer().flush,
        prefix=str(job._job_result_path),
    )
    assert failed == 0


@pytest.mark.unit
@pytest.mark.parametrize("metric_type, expected", [(None, 1.0), (MetricType.SUM, 2.0)])
async def test_explicit_task_source_preserves_completed_metrics(
    tmp_path: Path, metric_type: MetricType | None, expected: float
) -> None:
    task_dir = tmp_path / "task"
    task_dir.mkdir()
    (task_dir / "task.toml").write_text('[task]\nname = "test/test-task"\n')
    (task_dir / "instruction.md").write_text("Do nothing.")
    (task_dir / "environment").mkdir()
    (task_dir / "tests").mkdir()
    (task_dir / "tests" / "test.sh").write_text("#!/bin/sh\nexit 0\n")
    job = await Job.create(
        JobConfig(
            job_name="named-source",
            jobs_dir=tmp_path / "jobs",
            tasks=[TaskConfig(path=task_dir, source="named-dataset")],
            n_attempts=2,
            metrics=[] if metric_type is None else [MetricConfig(type=metric_type)],
            agents=[AgentConfig(name="nop")],
            environment=EnvironmentConfig(
                import_path="tests.unit.test_job_status:ScoredEnvironment"
            ),
        )
    )
    try:
        result = await job.run()
        assert len(result.trial_results) == 2
        assert all(trial.exception_info is None for trial in result.trial_results)
        assert all(
            trial.verifier_result.rewards == {"reward": 1}
            for trial in result.trial_results
        )
        persisted = JobResult.model_validate_json(
            (job.job_dir / "result.json").read_text()
        )
        assert persisted.stats.n_completed_trials == 2
        assert persisted.stats.n_errored_trials == 0
        assert len(persisted.stats.evals) == 1
        evaluation = next(iter(persisted.stats.evals.values()))
        assert evaluation.metrics == [{metric_type or "mean": expected}]
    finally:
        job._close_logger_handlers()


@pytest.mark.unit
async def test_job_writes_initial_result_before_trials_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    job = _make_job(tmp_path)
    trial_config = job._trial_configs[0]
    trial_result = _trial_result(trial_config)

    async def fake_run_trials(*args) -> list[TrialResult]:
        result_path = job.job_dir / "result.json"
        assert result_path.exists()
        partial_result = JobResult.model_validate_json(result_path.read_text())
        assert partial_result.n_total_trials == 1
        assert partial_result.stats.n_completed_trials == 0
        assert partial_result.stats.n_pending_trials == 1
        return [trial_result]

    monkeypatch.setattr(job, "_run_trials_with_queue", fake_run_trials)

    try:
        result = await job.run()
    finally:
        job._close_logger_handlers()

    persisted = JobResult.model_validate_json((job.job_dir / "result.json").read_text())
    assert result.stats.n_completed_trials == 1
    assert persisted.stats.n_completed_trials == 1
    assert persisted.stats.n_pending_trials == 0
    assert persisted.updated_at == persisted.finished_at


@pytest.mark.unit
async def test_job_progress_hooks_update_running_and_completed_counts(
    tmp_path: Path,
) -> None:
    job = _make_job(tmp_path)
    trial_config = job._trial_configs[0]
    started_at = datetime(2026, 4, 28, 8, 0, tzinfo=timezone.utc)
    job._job_result = JobResult(
        id=job.id,
        started_at=started_at,
        updated_at=started_at,
        n_total_trials=1,
        stats=JobStats.from_counts(n_total_trials=1),
    )

    try:
        await job._on_trial_started(_hook_event(TrialEvent.START, trial_config))
        await _flush_job_snapshots(job)
        running_result = JobResult.model_validate_json(
            (job.job_dir / "result.json").read_text()
        )
        assert running_result.stats.n_running_trials == 1
        assert running_result.stats.n_pending_trials == 0

        trial_result = _trial_result(trial_config, exception_type="RuntimeError")
        await job._on_trial_completed(
            _hook_event(TrialEvent.END, trial_config, result=trial_result)
        )
        await _flush_job_snapshots(job)
    finally:
        job._close_logger_handlers()

    completed_result = JobResult.model_validate_json(
        (job.job_dir / "result.json").read_text()
    )
    assert completed_result.stats.n_running_trials == 0
    assert completed_result.stats.n_completed_trials == 1
    assert completed_result.stats.n_errored_trials == 1
    assert completed_result.stats.n_pending_trials == 0


@pytest.mark.unit
async def test_stalled_remote_progress_snapshot_allows_required_final_write(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    task_configs = [
        TaskConfig(path=Path("/tmp/task-a")),
        TaskConfig(path=Path("/tmp/task-b")),
    ]
    job = _make_job(tmp_path, task_configs)
    started_at = datetime(2026, 4, 28, 8, 0, tzinfo=timezone.utc)
    job._job_result = JobResult(
        id=job.id,
        started_at=started_at,
        updated_at=started_at,
        n_total_trials=2,
        stats=JobStats.from_counts(n_total_trials=2),
    )

    upload_started = tmp_path / "upload-started"
    release_upload = tmp_path / "release-upload"
    destination = _RecoveringCloudResult(
        upload_started,
        release_upload,
        tmp_path / "uploaded-result.json",
    )
    writer = ArtifactWriter(
        num_workers=1,
        spool_dir=tmp_path / "spool",
        upload_timeout=2,
    )

    upload_count = 0

    def first_upload_stalls_then_recovers(destination, spool_path, timeout):
        nonlocal upload_count
        upload_count += 1
        if upload_count == 1:
            upload_started.touch()
            _wait_for_path(release_upload, timeout)
            return artifact_writer._CloudUploadOutcome.DEADLINE_EXCEEDED
        destination.write_spooled(spool_path)
        return artifact_writer._CloudUploadOutcome.COMPLETE

    monkeypatch.setattr(
        writer._uploaders[0], "upload", first_upload_stalls_then_recovers
    )
    monkeypatch.setattr(Job, "_job_result_path", property(lambda _: destination))
    monkeypatch.setattr("harbor.job.get_artifact_writer", lambda: writer, raising=False)

    first_config, second_config = job._trial_configs
    first_completion = asyncio.create_task(
        job._on_trial_completed(
            _hook_event(
                TrialEvent.END,
                first_config,
                result=_trial_result(first_config),
            )
        )
    )
    assert await asyncio.to_thread(_wait_for_path, upload_started, 5)

    second_completion = asyncio.create_task(
        job._on_trial_completed(
            _hook_event(
                TrialEvent.END,
                second_config,
                result=_trial_result(second_config),
            )
        )
    )
    try:
        await asyncio.wait_for(asyncio.shield(second_completion), timeout=0.2)
        assert job._job_result.stats.n_completed_trials == 2
        await job._flush_job_result_snapshots()
        job._job_result.finished_at = datetime.now(timezone.utc)
        await job._write_required_job_result()

        persisted = destination.read_result()
        assert persisted.stats.n_completed_trials == 2
        assert persisted.finished_at == job._job_result.finished_at
        assert len(list((tmp_path / "spool").iterdir())) == 1
    finally:
        release_upload.touch()
        await asyncio.gather(
            first_completion,
            second_completion,
            return_exceptions=True,
        )
        await asyncio.to_thread(writer.flush, timeout=5)
        job._close_logger_handlers()


@pytest.mark.unit
async def test_job_progress_retry_replaces_previous_attempt_counts(
    tmp_path: Path,
) -> None:
    job = _make_job(tmp_path)
    trial_config = job._trial_configs[0]
    started_at = datetime(2026, 4, 28, 8, 0, tzinfo=timezone.utc)
    job._job_result = JobResult(
        id=job.id,
        started_at=started_at,
        updated_at=started_at,
        n_total_trials=1,
        stats=JobStats.from_counts(n_total_trials=1),
    )

    try:
        await job._on_trial_started(_hook_event(TrialEvent.START, trial_config))
        failed_attempt = _trial_result(trial_config, exception_type="RuntimeError")
        await job._on_trial_completed(
            _hook_event(TrialEvent.END, trial_config, result=failed_attempt)
        )

        await job._on_trial_started(_hook_event(TrialEvent.START, trial_config))
        await _flush_job_snapshots(job)
        retrying_result = JobResult.model_validate_json(
            (job.job_dir / "result.json").read_text()
        )
        assert retrying_result.stats.n_completed_trials == 0
        assert retrying_result.stats.n_running_trials == 1
        assert retrying_result.stats.n_errored_trials == 0
        assert retrying_result.stats.n_retries == 1

        second_failed_attempt = _trial_result(
            trial_config, exception_type="RuntimeError"
        )
        await job._on_trial_completed(
            _hook_event(TrialEvent.END, trial_config, result=second_failed_attempt)
        )

        await job._on_trial_started(_hook_event(TrialEvent.START, trial_config))
        await _flush_job_snapshots(job)
        second_retrying_result = JobResult.model_validate_json(
            (job.job_dir / "result.json").read_text()
        )
        assert second_retrying_result.stats.n_completed_trials == 0
        assert second_retrying_result.stats.n_running_trials == 1
        assert second_retrying_result.stats.n_errored_trials == 0
        assert second_retrying_result.stats.n_retries == 2

        successful_attempt = _trial_result(trial_config)
        await job._on_trial_completed(
            _hook_event(TrialEvent.END, trial_config, result=successful_attempt)
        )
        await _flush_job_snapshots(job)
    finally:
        job._close_logger_handlers()

    completed_result = JobResult.model_validate_json(
        (job.job_dir / "result.json").read_text()
    )
    assert completed_result.stats.n_completed_trials == 1
    assert completed_result.stats.n_running_trials == 0
    assert completed_result.stats.n_errored_trials == 0
    assert completed_result.stats.n_retries == 2


@pytest.mark.unit
async def test_job_progress_hooks_count_cancelled_trials(tmp_path: Path) -> None:
    job = _make_job(tmp_path)
    trial_config = job._trial_configs[0]
    started_at = datetime(2026, 4, 28, 8, 0, tzinfo=timezone.utc)
    job._job_result = JobResult(
        id=job.id,
        started_at=started_at,
        updated_at=started_at,
        n_total_trials=1,
        stats=JobStats.from_counts(n_total_trials=1),
    )

    try:
        await job._on_trial_started(_hook_event(TrialEvent.START, trial_config))
        await job._on_trial_cancelled(_hook_event(TrialEvent.CANCEL, trial_config))
        await _flush_job_snapshots(job)

        cancelled_result = JobResult.model_validate_json(
            (job.job_dir / "result.json").read_text()
        )
        assert cancelled_result.stats.n_running_trials == 1
        assert cancelled_result.stats.n_cancelled_trials == 1

        trial_result = _trial_result(trial_config, exception_type="CancelledError")
        await job._on_trial_completed(
            _hook_event(TrialEvent.END, trial_config, result=trial_result)
        )
        await _flush_job_snapshots(job)
    finally:
        job._close_logger_handlers()

    completed_result = JobResult.model_validate_json(
        (job.job_dir / "result.json").read_text()
    )
    assert completed_result.stats.n_running_trials == 0
    assert completed_result.stats.n_completed_trials == 1
    assert completed_result.stats.n_errored_trials == 1
    assert completed_result.stats.n_cancelled_trials == 1


@pytest.mark.unit
def test_job_resume_progress_starts_from_existing_trial_results(tmp_path: Path) -> None:
    task_config = TaskConfig(path=Path("/tmp/task"))
    first_job = _make_job(tmp_path, [task_config])
    trial_config = first_job._trial_configs[0]
    # A genuinely-terminal error (NOT resume-retryable): it is KEPT on resume and
    # its stats seed the resumed job's progress. (CancelledError / SandboxBuild-
    # FailedError would instead be re-run — see test_job_resume_retry.py.)
    trial_result = _trial_result(trial_config, exception_type="RuntimeError")
    _persist_completed_trial(first_job, trial_result)

    resumed_job = _make_job(tmp_path, [task_config])
    try:
        assert resumed_job._existing_job_result is not None
        resumed_job._job_result = JobResult(
            id=resumed_job.id,
            started_at=resumed_job._existing_job_result.started_at,
            updated_at=resumed_job._existing_job_result.updated_at,
            n_total_trials=len(resumed_job._trial_configs),
            stats=JobStats.from_trial_results(
                resumed_job._existing_trial_results,
                n_total_trials=len(resumed_job._trial_configs),
            ),
        )
        resumed_job._refresh_job_progress()
        assert resumed_job._job_result.stats.n_completed_trials == 1
        assert resumed_job._job_result.stats.n_pending_trials == 0
        assert resumed_job._job_result.stats.n_errored_trials == 1
        assert resumed_job._job_result.stats.n_cancelled_trials == 0
    finally:
        resumed_job._close_logger_handlers()


@pytest.mark.unit
async def test_job_resume_releases_persisted_trial_payloads_from_memory(
    tmp_path: Path,
) -> None:
    task_config = TaskConfig(path=Path("/tmp/task"))
    first_job = _make_job(
        tmp_path,
        [task_config],
        release_trial_payloads_in_memory=True,
    )
    trial_config = first_job._trial_configs[0]
    trial_result = _trial_result(trial_config)
    trial_result.verifier_result.stdout = "verifier stdout\n" * 4096
    trial_result.verifier_result.stderr = "verifier stderr\n" * 4096
    child_result_path = _persist_completed_trial(first_job, trial_result)

    resumed_job = _make_job(
        tmp_path,
        [task_config],
        release_trial_payloads_in_memory=True,
    )
    result = await resumed_job.run()

    assert len(result.trial_results) == 1
    retained = result.trial_results[0]
    assert retained.verifier_result is not None
    assert retained.verifier_result.rewards == {"reward": 1}
    assert retained.verifier_result.stdout is None
    assert retained.verifier_result.stderr is None
    assert JobStats.from_trial_results(
        result.trial_results, n_total_trials=1
    ) == JobStats.from_trial_results(
        [trial_result],
        n_total_trials=1,
    )

    persisted_child = TrialResult.model_validate_json(child_result_path.read_text())
    assert persisted_child.verifier_result is not None
    assert persisted_child.verifier_result.stdout == trial_result.verifier_result.stdout
    assert persisted_child.verifier_result.stderr == trial_result.verifier_result.stderr
