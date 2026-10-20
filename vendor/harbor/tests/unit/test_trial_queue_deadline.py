import asyncio
import concurrent.futures
import contextlib
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

from harbor.environments.base import SandboxDiskFullError
from harbor_config import ErrorCategory, error_category
from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import TaskConfig, TrialConfig
from harbor.models.trial.paths import TrialPaths
from harbor.models.trial.result import AgentInfo, ExceptionInfo, TrialResult
from harbor.trial.errors import (
    AgentDiskFullError,
    AgentTimeoutError,
    PhaseCancellationError,
)
from harbor.trial.hooks import TrialEvent, TrialHookEvent, emit_trial_hooks
from harbor.trial.queue import TrialQueue
from harbor.trial.trial import Trial
from harbor.utils.artifact_writer import ArtifactUploadTimeoutError


Behavior = Callable[["FakeTrial"], Awaitable[None]]


class PhaseTrial(Trial):
    async def _run(self) -> None:
        return None

    async def _recover_outputs(self) -> None:
        return None


class PhaseCancellationTrial(Trial):
    def __init__(self) -> None:
        self.events: list[str] = []
        self._result = object()
        self.config = SimpleNamespace(trial_name="cancellation-resistant")
        self.logger = logging.getLogger(__name__)

    async def _init_result(self) -> None:
        return None

    async def _emit(self, event: TrialEvent) -> None:
        return None

    async def _prepare(self) -> None:
        return None

    async def _run(self) -> None:
        raise PhaseCancellationError("phase did not stop")

    async def _record_exception(self, exc: BaseException) -> None:
        self.events.append(type(exc).__name__)

    async def _recover_outputs(self) -> None:
        self.events.append("unsafe output recovery")

    async def _finalize(self) -> None:
        self.events.append("finalized")

    def _close_logger_handler(self) -> None:
        return None


def _config(
    tmp_path: Path,
    name: str,
    *,
    attempt_timeout: float | None,
    cleanup_grace: float = 0.01,
) -> TrialConfig:
    return TrialConfig(
        task=TaskConfig(path=tmp_path / "task"),
        trial_name=name,
        trials_dir=tmp_path / "trials",
        trial_attempt_timeout_sec=attempt_timeout,
        trial_cleanup_grace_sec=cleanup_grace,
    )


def _result(config: TrialConfig, attempt: int) -> TrialResult:
    paths = TrialPaths(
        config.trials_dir / config.trial_name / "attempts" / f"{attempt:03d}"
    )
    return TrialResult(
        task_name="task",
        trial_name=config.trial_name,
        trial_uri=paths.trial_dir.expanduser().resolve().as_uri(),
        task_id=config.task.get_task_id(),
        task_checksum="checksum",
        config=config,
        agent_info=AgentInfo(name="fake", version="1"),
    )


class FakeTrial:
    def __init__(
        self,
        config: TrialConfig,
        attempt: int,
        behavior: Behavior,
        *,
        phase_timeout: asyncio.TimeoutError | None = None,
    ) -> None:
        self.config = config
        self.paths = TrialPaths(
            config.trials_dir / config.trial_name / "attempts" / f"{attempt:03d}"
        )
        self._result = _result(config, attempt)
        self._behavior = behavior
        self.first_phase_timeout = phase_timeout
        self._hooks = {event: [] for event in TrialEvent}
        self.completed = asyncio.Event()

    @property
    def result(self) -> TrialResult:
        return self._result

    @property
    def has_result(self) -> bool:
        return True

    def add_hook(self, event: TrialEvent, hook) -> None:
        self._hooks[event].append(hook)

    def pending_cleanup_tasks(self) -> tuple[asyncio.Task[None], ...]:
        return ()

    async def _emit(self, event: TrialEvent) -> None:
        await emit_trial_hooks(
            self._hooks[event],
            event=event,
            trial_id=self.config.trial_name,
            task_name=self.result.task_name,
            config=self.config,
            result=self.result,
        )

    async def run(self) -> TrialResult:
        await self._emit(TrialEvent.START)
        try:
            try:
                await self._behavior(self)
            finally:
                await self._emit(TrialEvent.RESULT_PERSISTENCE_START)
                await self._emit(TrialEvent.END)
        finally:
            self.completed.set()
        return self.result


async def _resist_cancellation(
    cancellation_seen: asyncio.Event, provider_terminated: asyncio.Event
) -> None:
    try:
        await asyncio.Event().wait()
    except asyncio.CancelledError:
        cancellation_seen.set()
        await provider_terminated.wait()
        raise


def _phase_trial(cleanup_grace: float) -> PhaseTrial:
    trial = object.__new__(PhaseTrial)
    trial._first_phase_timeout = None
    trial._pending_cleanup_tasks = set()
    trial.config = SimpleNamespace(trial_cleanup_grace_sec=cleanup_grace)
    trial.logger = logging.getLogger(__name__)
    return trial


@pytest.mark.unit
async def test_agent_disk_exhaustion_is_a_model_failure(tmp_path: Path) -> None:
    original = SandboxDiskFullError("no space left on device")

    async def run_agent(**kwargs):
        raise original

    trial = _phase_trial(cleanup_grace=1)
    trial.config = _config(tmp_path, "disk-full", attempt_timeout=60)
    trial._result = _result(trial.config, 0)
    trial._hooks = {event: [] for event in TrialEvent}
    trial.task = SimpleNamespace(name="disk-full")
    trial.agent = SimpleNamespace(run=run_agent)
    trial.user_agent = None
    trial.agent_environment = SimpleNamespace(
        with_default_user=lambda user: contextlib.nullcontext(),
        with_agent_timeout=lambda timeout: contextlib.nullcontext(),
    )

    with pytest.raises(AgentDiskFullError) as raised:
        await trial._run_agent_phase(
            target=trial.result,
            instruction="write a compressor",
            timeout_sec=60,
            user=None,
        )

    assert raised.value.__cause__ is original
    assert error_category(type(raised.value).__name__) is ErrorCategory.AGENT
    assert trial.result.agent_execution.finished_at is not None


@pytest.mark.unit
async def test_cancelled_phase_tracks_cancellation_resistant_provider_work() -> None:
    cancellation_seen = asyncio.Event()
    provider_terminated = asyncio.Event()
    trial = _phase_trial(cleanup_grace=1)

    phase_task = asyncio.create_task(
        trial._await_phase(
            _resist_cancellation(cancellation_seen, provider_terminated),
            timeout_sec=0.01,
            timeout_error=AgentTimeoutError("phase expired"),
        )
    )

    async with asyncio.timeout(1):
        await cancellation_seen.wait()
        phase_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await phase_task
        cleanup_tasks = trial.pending_cleanup_tasks()
        assert len(cleanup_tasks) == 1
        assert not cleanup_tasks[0].done()
        provider_terminated.set()
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)

    assert trial.pending_cleanup_tasks() == ()


@pytest.mark.unit
async def test_phase_timeout_raises_when_cancellation_resists() -> None:
    cancellation_seen = asyncio.Event()
    provider_terminated = asyncio.Event()
    trial = _phase_trial(cleanup_grace=0.01)

    async with asyncio.timeout(1):
        with pytest.raises(PhaseCancellationError):
            await trial._await_phase(
                _resist_cancellation(cancellation_seen, provider_terminated),
                timeout_sec=0.01,
                timeout_error=AgentTimeoutError("phase expired"),
            )
        await cancellation_seen.wait()
        cleanup_tasks = trial.pending_cleanup_tasks()
        assert len(cleanup_tasks) == 1
        assert not cleanup_tasks[0].done()
        provider_terminated.set()
        await asyncio.gather(*cleanup_tasks, return_exceptions=True)


@pytest.mark.unit
async def test_phase_cancellation_timeout_skips_concurrent_output_recovery() -> None:
    trial = PhaseCancellationTrial()

    result = await trial.run()

    assert result is trial.result
    assert trial.events == ["PhaseCancellationError", "finalized"]


@pytest.mark.unit
async def test_trial_reports_result_persistence_after_lifecycle_write(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path, "persisting", attempt_timeout=None)
    trial = object.__new__(PhaseTrial)
    trial.config = config
    trial.task = SimpleNamespace(name="task")
    trial.paths = TrialPaths(config.trials_dir / config.trial_name / "attempts" / "000")
    trial.paths.mkdir()
    trial._attempt_index = 0
    trial._result = _result(config, 0)
    trial._hooks = {event: [] for event in TrialEvent}
    lifecycle_path = trial.paths.lifecycle_result_path
    observed: list[tuple[TrialEvent, bool]] = []

    async def no_op() -> None:
        return None

    async def on_persistence_start(event: TrialHookEvent) -> None:
        observed.append((event.event, lifecycle_path.exists()))

    trial._stop_agent_environment = no_op
    trial._enforce_verifier_result_invariant = no_op
    trial.add_hook(TrialEvent.RESULT_PERSISTENCE_START, on_persistence_start)

    await trial._finalize()

    assert observed == [(TrialEvent.RESULT_PERSISTENCE_START, True)]
    assert lifecycle_path.exists()


def _install_factory(monkeypatch, behaviors: dict[tuple[str, int], FakeTrial]) -> None:
    async def create(cls, config: TrialConfig, *, attempt_index: int | None = None):
        if attempt_index is None:
            raise ValueError("queue attempts require an attempt index")
        return behaviors[(config.trial_name, attempt_index)]

    monkeypatch.setattr(Trial, "create", classmethod(create))


@pytest.mark.unit
async def test_queue_wait_does_not_consume_attempt_deadline(
    tmp_path: Path, monkeypatch
) -> None:
    release_first = asyncio.Event()
    first_started = asyncio.Event()
    second_started = asyncio.Event()

    async def hold_first(trial: FakeTrial) -> None:
        first_started.set()
        await release_first.wait()

    async def finish_second(trial: FakeTrial) -> None:
        second_started.set()

    first = _config(tmp_path, "first", attempt_timeout=None)
    second = _config(tmp_path, "second", attempt_timeout=0.01)
    _install_factory(
        monkeypatch,
        {
            ("first", 0): FakeTrial(first, 0, hold_first),
            ("second", 0): FakeTrial(second, 0, finish_second),
        },
    )
    queue = TrialQueue(n_concurrent=1)

    async with asyncio.timeout(1):
        first_task = asyncio.create_task(queue.submit(first))
        await first_started.wait()
        second_task = asyncio.create_task(queue.submit(second))
        asyncio.get_running_loop().call_later(0.03, release_first.set)
        first_result, second_result = await asyncio.gather(first_task, second_task)

    assert first_result.exception_info is None
    assert second_started.is_set()
    assert second_result.exception_info is None


@pytest.mark.unit
async def test_attempt_failure_returns_terminal_result_and_does_not_cancel_batch(
    tmp_path: Path, monkeypatch
) -> None:
    async def fail_after_admission(trial: FakeTrial) -> None:
        raise OSError("output collection failed")

    async def finish(trial: FakeTrial) -> None:
        return None

    failed = _config(tmp_path, "failed", attempt_timeout=None)
    completed = _config(tmp_path, "completed", attempt_timeout=None)
    _install_factory(
        monkeypatch,
        {
            ("failed", 0): FakeTrial(failed, 0, fail_after_admission),
            ("completed", 0): FakeTrial(completed, 0, finish),
        },
    )

    coroutines = TrialQueue(n_concurrent=2).submit_batch([failed, completed])
    async with asyncio.TaskGroup() as task_group:
        tasks = [task_group.create_task(coroutine) for coroutine in coroutines]
    failed_result, completed_result = [task.result() for task in tasks]

    assert failed_result.finished_at is not None
    assert failed_result.exception_info is not None
    assert failed_result.exception_info.exception_type == "OSError"
    assert completed_result.exception_info is None
    persisted = TrialPaths(failed.trials_dir / failed.trial_name).result_path
    assert persisted.exists()


@pytest.mark.unit
async def test_trial_creation_failure_returns_terminal_result(
    tmp_path: Path, monkeypatch
) -> None:
    config = _config(tmp_path, "creation-failed", attempt_timeout=None)

    async def fail_create(
        cls, trial_config: TrialConfig, *, attempt_index: int | None = None
    ) -> Trial:
        raise RuntimeError("trial creation failed")

    monkeypatch.setattr(Trial, "create", classmethod(fail_create))

    result = await TrialQueue(n_concurrent=1).submit(config)

    assert result.finished_at is not None
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "RuntimeError"
    persisted = TrialPaths(config.trials_dir / config.trial_name).result_path
    assert persisted.exists()


@pytest.mark.unit
async def test_cancelling_queued_trial_does_not_start_or_retry_it(
    tmp_path: Path, monkeypatch
) -> None:
    release_first = asyncio.Event()
    first_started = asyncio.Event()
    created: list[str] = []

    async def hold_first(trial: FakeTrial) -> None:
        first_started.set()
        await release_first.wait()

    first = _config(tmp_path, "first", attempt_timeout=None)
    queued = _config(tmp_path, "queued", attempt_timeout=0.01)
    first_trial = FakeTrial(first, 0, hold_first)

    async def create(cls, config: TrialConfig, *, attempt_index: int | None = None):
        created.append(config.trial_name)
        if config.trial_name == "first" and attempt_index == 0:
            return first_trial
        raise AssertionError("cancelled queued trial was admitted")

    monkeypatch.setattr(Trial, "create", classmethod(create))
    queue = TrialQueue(
        n_concurrent=1,
        retry_config=RetryConfig(max_retries=2, min_wait_sec=0, max_wait_sec=0),
    )

    async with asyncio.timeout(1):
        first_task = asyncio.create_task(queue.submit(first))
        await first_started.wait()
        queued_task = asyncio.create_task(queue.submit(queued))
        await asyncio.sleep(0)
        queued_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await queued_task
        release_first.set()
        await first_task

    assert created == ["first"]


@pytest.mark.unit
async def test_attempt_deadline_returns_trial_timeout_and_persists_selection(
    tmp_path: Path, monkeypatch
) -> None:
    never = asyncio.Event()

    async def hang(trial: FakeTrial) -> None:
        await never.wait()

    config = _config(tmp_path, "hung", attempt_timeout=0.01)
    _install_factory(monkeypatch, {("hung", 0): FakeTrial(config, 0, hang)})

    async with asyncio.timeout(1):
        result = await TrialQueue(n_concurrent=1).submit(config)

    assert result.exception_info is not None
    assert result.exception_info.exception_type == "TrialTimeoutError"
    selected = TrialResult.model_validate_json(
        (tmp_path / "trials" / "hung" / "result.json").read_text()
    )
    assert selected.exception_info == result.exception_info


@pytest.mark.unit
async def test_attempt_deadline_covers_trial_creation(
    tmp_path: Path, monkeypatch
) -> None:
    creation_started = asyncio.Event()

    async def create(cls, config: TrialConfig, *, attempt_index: int | None = None):
        creation_started.set()
        await asyncio.Event().wait()

    monkeypatch.setattr(Trial, "create", classmethod(create))
    config = _config(tmp_path, "create-hung", attempt_timeout=0.01)

    async with asyncio.timeout(1):
        result_task = asyncio.create_task(TrialQueue(n_concurrent=1).submit(config))
        await creation_started.wait()
        result = await result_task

    assert result.exception_info is not None
    assert result.exception_info.exception_type == "TrialTimeoutError"
    assert result.task_checksum == ""


@pytest.mark.unit
async def test_result_persistence_failure_is_primary_and_retryable(
    tmp_path: Path, monkeypatch
) -> None:
    attempts: list[int] = []

    class FailingWriter:
        def submit_optional(self, *args, **kwargs) -> None:
            return None

        async def submit_required(self, *args, **kwargs):
            future: concurrent.futures.Future[None] = concurrent.futures.Future()
            future.set_exception(
                ArtifactUploadTimeoutError(
                    destination="s3://bucket/result.json",
                    byte_count=1,
                    endpoint="https://objects.example",
                    elapsed=1.0,
                    timeout=1.0,
                    spool_path=tmp_path / "spool.json",
                )
            )
            return future

        async def await_write(self, future, **kwargs) -> None:
            future.result()

    async def finish_with_prior_model_error(trial: FakeTrial) -> None:
        attempts.append(len(attempts))
        trial.result.exception_info = ExceptionInfo.from_exception(
            RuntimeError("model failed before persistence")
        )

    config = _config(tmp_path, "persistence-retry", attempt_timeout=None)
    _install_factory(
        monkeypatch,
        {
            ("persistence-retry", 0): FakeTrial(
                config, 0, finish_with_prior_model_error
            ),
            ("persistence-retry", 1): FakeTrial(
                config, 1, finish_with_prior_model_error
            ),
        },
    )
    monkeypatch.setattr("harbor.trial.attempt.get_artifact_writer", FailingWriter)
    queue = TrialQueue(
        n_concurrent=1,
        retry_config=RetryConfig(
            max_retries=1,
            include_exceptions={"ArtifactUploadTimeoutError"},
            min_wait_sec=0,
            max_wait_sec=0,
        ),
    )

    result = await queue.submit(config)

    assert attempts == [0, 1]
    assert result.exception_info is not None
    assert result.exception_info.exception_type == "ArtifactUploadTimeoutError"
    assert any(
        error.exception_type == "RuntimeError"
        for error in result.secondary_exception_info
    )


@pytest.mark.unit
async def test_phase_deadline_observed_first_preserves_phase_classification(
    tmp_path: Path, monkeypatch
) -> None:
    never = asyncio.Event()

    async def hang(trial: FakeTrial) -> None:
        await never.wait()

    config = _config(tmp_path, "agent-hung", attempt_timeout=0.01)
    trial = FakeTrial(
        config,
        0,
        hang,
        phase_timeout=AgentTimeoutError("agent phase expired first"),
    )
    trial.result.exception_info = ExceptionInfo.from_exception(
        RuntimeError("output recovery failed")
    )
    _install_factory(monkeypatch, {("agent-hung", 0): trial})

    async with asyncio.timeout(1):
        result = await TrialQueue(n_concurrent=1).submit(config)

    assert result.exception_info is not None
    assert result.exception_info.exception_type == "AgentTimeoutError"
    assert [error.exception_type for error in result.secondary_exception_info] == [
        "RuntimeError"
    ]


@pytest.mark.unit
async def test_cancellation_resistant_attempt_quarantines_capacity_until_termination(
    tmp_path: Path, monkeypatch
) -> None:
    cancellation_seen = asyncio.Event()
    provider_terminated = asyncio.Event()
    retry_started = asyncio.Event()
    terminal_events: list[str | None] = []

    async def resist_cancellation(trial: FakeTrial) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await provider_terminated.wait()
            raise

    async def finish_retry(trial: FakeTrial) -> None:
        retry_started.set()

    config = _config(tmp_path, "retry", attempt_timeout=0.01)
    _install_factory(
        monkeypatch,
        {
            ("retry", 0): FakeTrial(config, 0, resist_cancellation),
            ("retry", 1): FakeTrial(config, 1, finish_retry),
        },
    )
    queue = TrialQueue(
        n_concurrent=1,
        retry_config=RetryConfig(
            max_retries=1,
            min_wait_sec=0,
            max_wait_sec=0,
        ),
    )

    async def on_end(event: TrialHookEvent) -> None:
        terminal_events.append(
            event.result.exception_info.exception_type
            if event.result is not None and event.result.exception_info is not None
            else None
        )

    queue.on_trial_ended(on_end)

    async with asyncio.timeout(1):
        result_task = asyncio.create_task(queue.submit(config))
        await cancellation_seen.wait()
        assert not retry_started.is_set()
        provider_terminated.set()
        result = await result_task

    assert retry_started.is_set()
    assert result.exception_info is None
    assert terminal_events == ["TrialTimeoutError", None]
    first = TrialResult.model_validate_json(
        (tmp_path / "trials" / "retry" / "attempts" / "000" / "result.json").read_text()
    )
    second = TrialResult.model_validate_json(
        (tmp_path / "trials" / "retry" / "attempts" / "001" / "result.json").read_text()
    )
    assert first.exception_info is not None
    assert first.exception_info.exception_type == "TrialTimeoutError"
    assert second.exception_info is None


@pytest.mark.unit
async def test_late_attempt_cannot_overwrite_retry_selection_or_terminal_hook(
    tmp_path: Path, monkeypatch
) -> None:
    cancellation_seen = asyncio.Event()
    release_timed_out_attempt = asyncio.Event()
    retry_finished = asyncio.Event()
    terminal_events: list[str | None] = []

    async def finish_after_retry(trial: FakeTrial) -> None:
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancellation_seen.set()
            await release_timed_out_attempt.wait()

    async def finish_retry(trial: FakeTrial) -> None:
        retry_finished.set()

    config = _config(tmp_path, "late", attempt_timeout=0.01)
    first_attempt = FakeTrial(config, 0, finish_after_retry)
    retry_attempt = FakeTrial(config, 1, finish_retry)
    _install_factory(
        monkeypatch,
        {
            ("late", 0): first_attempt,
            ("late", 1): retry_attempt,
        },
    )
    queue = TrialQueue(
        n_concurrent=2,
        retry_config=RetryConfig(
            max_retries=1,
            min_wait_sec=0,
            max_wait_sec=0,
        ),
    )

    async def on_end(event: TrialHookEvent) -> None:
        terminal_events.append(
            event.result.exception_info.exception_type
            if event.result is not None and event.result.exception_info is not None
            else None
        )

    queue.on_trial_ended(on_end)

    async with asyncio.timeout(1):
        result = await queue.submit(config)
        await cancellation_seen.wait()
        assert retry_finished.is_set()
        release_timed_out_attempt.set()
        await first_attempt.completed.wait()

    selected = TrialResult.model_validate_json(
        (tmp_path / "trials" / "late" / "result.json").read_text()
    )
    assert result.exception_info is None
    assert selected.exception_info is None
    assert selected.trial_uri.endswith("/attempts/001")
    assert selected.trial_relpath == "late/attempts/001"
    assert terminal_events == ["TrialTimeoutError", None]


@pytest.mark.unit
async def test_terminal_result_is_selected_before_end_hook_stalls(
    tmp_path: Path, monkeypatch
) -> None:
    hook_started = asyncio.Event()
    release_hook = asyncio.Event()
    result_seen_by_hook: list[TrialResult | None] = []

    async def finish(trial: FakeTrial) -> None:
        return None

    config = _config(tmp_path, "completed", attempt_timeout=None)
    _install_factory(
        monkeypatch,
        {("completed", 0): FakeTrial(config, 0, finish)},
    )
    queue = TrialQueue(n_concurrent=1)
    selected_path = TrialPaths(config.trials_dir / config.trial_name).result_path

    async def block_end_hook(event: TrialHookEvent) -> None:
        result_seen_by_hook.append(
            TrialResult.model_validate_json(selected_path.read_text())
            if selected_path.exists()
            else None
        )
        hook_started.set()
        await release_hook.wait()

    queue.on_trial_ended(block_end_hook)
    result_task = asyncio.create_task(queue.submit(config))
    try:
        async with asyncio.timeout(1):
            await hook_started.wait()
        assert result_seen_by_hook[0] is not None
        assert result_seen_by_hook[0].trial_name == config.trial_name
    finally:
        release_hook.set()
        await result_task


@pytest.mark.unit
async def test_hung_terminal_hook_is_bounded_and_quarantines_capacity(
    tmp_path: Path, monkeypatch
) -> None:
    hook_started = asyncio.Event()
    release_hook = asyncio.Event()
    next_started = asyncio.Event()

    async def finish(trial: FakeTrial) -> None:
        return None

    async def start_next(trial: FakeTrial) -> None:
        next_started.set()

    timed = _config(tmp_path, "timed", attempt_timeout=0.01)
    following = _config(tmp_path, "following", attempt_timeout=None)
    _install_factory(
        monkeypatch,
        {
            ("timed", 0): FakeTrial(timed, 0, finish),
            ("following", 0): FakeTrial(following, 0, start_next),
        },
    )
    queue = TrialQueue(n_concurrent=1)

    async def hang_on_end(event: TrialHookEvent) -> None:
        if event.trial_id != "timed":
            return
        hook_started.set()
        try:
            await release_hook.wait()
        except asyncio.CancelledError:
            await release_hook.wait()

    queue.on_trial_ended(hang_on_end)

    async with asyncio.timeout(1):
        timed_task = asyncio.create_task(queue.submit(timed))
        await hook_started.wait()
        result = await timed_task
        following_task = asyncio.create_task(queue.submit(following))
        assert not next_started.is_set()
        release_hook.set()
        following_result = await following_task

    assert result.exception_info is not None
    assert result.exception_info.exception_type == "TrialTimeoutError"
    assert next_started.is_set()
    assert following_result.exception_info is None
