import asyncio
import time
from collections.abc import Coroutine
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from harbor.models.job.config import RetryConfig
from harbor.models.trial.config import TrialConfig
from harbor.models.trial.result import ExceptionInfo, TrialResult
from harbor.telemetry.observer import (
    AttemptIdentity,
    FAILURE_KIND_QUEUE_WAIT,
    OUTCOME_CANCELLED,
    OUTCOME_SUCCEEDED,
    bind_attempt,
    record_failure,
    record_queue_wait,
)
from harbor.trial.attempt import (
    exception_result,
    persist_selected_result,
    timeout_result,
)
from harbor.trial.hooks import (
    HookCallback,
    TrialEvent,
    TrialHookEvent,
    emit_trial_hooks,
)
from harbor.trial.trial import Trial
from harbor.utils.logger import logger


@dataclass
class _HookGate:
    active: bool = True


class _TrialAttempt(Protocol):
    result: TrialResult
    first_phase_timeout: asyncio.TimeoutError | None
    has_result: bool

    def add_hook(self, event: TrialEvent, hook: HookCallback) -> None: ...

    def pending_cleanup_tasks(self) -> tuple[asyncio.Task[Any], ...]: ...

    async def run(self) -> TrialResult: ...


@dataclass
class _AttemptState:
    gate: _HookGate
    trial: _TrialAttempt | None = None
    ended: bool = False


class TrialQueue:
    """Run trials with admission-scoped concurrency, deadlines, and retries."""

    def __init__(
        self,
        n_concurrent: int,
        retry_config: RetryConfig | None = None,
        hooks: dict[TrialEvent, list[HookCallback]] | None = None,
        release_trial_payloads_in_memory: bool = False,
        planned_attempts: dict[str, int] | None = None,
    ):
        if hooks is None:
            hooks = {event: [] for event in TrialEvent}
        else:
            for event in TrialEvent:
                hooks.setdefault(event, [])

        self._n_concurrent = n_concurrent
        self._retry_config = retry_config if retry_config is not None else RetryConfig()
        self._hooks = hooks
        self._release_trial_payloads_in_memory = release_trial_payloads_in_memory
        self._planned_attempts = planned_attempts or {}
        self._logger = logger.getChild(__name__)
        self._semaphore = asyncio.Semaphore(n_concurrent)
        # These strong references intentionally outlive the submitted coroutine.
        # Draining them at job exit could block forever on provider work that
        # ignored cancellation, defeating the bounded caller return.
        self._quarantine_tasks: set[asyncio.Task[None]] = set()
        self._observer_tasks: set[asyncio.Task[None]] = set()

    def add_hook(self, event: TrialEvent, callback: HookCallback) -> "TrialQueue":
        """Register a callback for a trial lifecycle event and return the queue."""
        self._hooks[event].append(callback)
        return self

    def on_trial_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial starts."""
        return self.add_hook(TrialEvent.START, callback)

    def on_environment_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a trial environment starts."""
        return self.add_hook(TrialEvent.ENVIRONMENT_START, callback)

    def on_agent_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when the trial agent starts."""
        return self.add_hook(TrialEvent.AGENT_START, callback)

    def on_verification_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when trial verification starts."""
        return self.add_hook(TrialEvent.VERIFICATION_START, callback)

    def on_verification_ended(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when trial verification ends."""
        return self.add_hook(TrialEvent.VERIFICATION_END, callback)

    def on_result_persistence_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when trial result persistence starts."""
        return self.add_hook(TrialEvent.RESULT_PERSISTENCE_START, callback)

    def on_trial_ended(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial ends."""
        return self.add_hook(TrialEvent.END, callback)

    def on_trial_cancelled(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial is cancelled."""
        return self.add_hook(TrialEvent.CANCEL, callback)

    def _should_retry_exception(self, exception_type: str) -> bool:
        if (
            self._retry_config.exclude_exceptions
            and exception_type in self._retry_config.exclude_exceptions
        ):
            self._logger.debug(
                "Exception %s is in exclude_exceptions, not retrying",
                exception_type,
            )
            return False

        if (
            self._retry_config.include_exceptions
            and exception_type not in self._retry_config.include_exceptions
        ):
            self._logger.debug(
                "Exception %s is not in include_exceptions, not retrying",
                exception_type,
            )
            return False

        return True

    def _calculate_backoff_delay_sec(self, attempt: int) -> float:
        delay_sec = self._retry_config.min_wait_sec * (
            self._retry_config.wait_multiplier**attempt
        )
        return min(delay_sec, self._retry_config.max_wait_sec)

    def _setup_hooks(
        self,
        trial: _TrialAttempt,
        state: _AttemptState,
        trial_config: TrialConfig,
        attempt: int,
    ) -> None:
        async def persist_terminal_result(event: TrialHookEvent) -> None:
            if state.gate.active:
                if event.result is None:
                    raise ValueError(
                        f"Trial {event.trial_id} ended without a result. "
                        "This should never happen."
                    )
                await persist_selected_result(trial_config, attempt, event.result)

        async def mark_ended(_event: TrialHookEvent) -> None:
            state.ended = True

        trial.add_hook(TrialEvent.RESULT_PERSISTENCE_START, persist_terminal_result)
        trial.add_hook(TrialEvent.END, mark_ended)
        for event, hooks in self._hooks.items():
            for hook in hooks:

                async def gated_hook(
                    hook_event: TrialHookEvent,
                    *,
                    callback: HookCallback = hook,
                ) -> None:
                    if state.gate.active:
                        await callback(hook_event)

                trial.add_hook(event, gated_hook)

    async def _emit_hooks(
        self,
        event: TrialEvent,
        trial_config: TrialConfig,
        result: TrialResult,
    ) -> None:
        await emit_trial_hooks(
            self._hooks[event],
            event=event,
            trial_id=trial_config.trial_name,
            task_name=result.task_name,
            config=trial_config,
            result=result,
        )

    def _observe_task(self, task: asyncio.Task[Any]) -> None:
        async def observe() -> None:
            try:
                await task
            except asyncio.CancelledError:
                return
            except Exception:
                self._logger.exception("Detached trial publication failed")

        observer = asyncio.create_task(observe())
        self._observer_tasks.add(observer)
        observer.add_done_callback(self._observer_tasks.discard)

    async def _run_attempt_lifecycle(
        self,
        trial_config: TrialConfig,
        attempt: int,
        state: _AttemptState,
    ) -> TrialResult:
        trial = await Trial.create(trial_config, attempt_index=attempt)
        state.trial = trial
        self._setup_hooks(trial, state, trial_config, attempt)
        return await trial.run()

    async def _publish_timeout_result(
        self,
        trial_config: TrialConfig,
        attempt: int,
        result: TrialResult,
    ) -> None:
        await persist_selected_result(trial_config, attempt, result)
        await self._emit_hooks(
            TrialEvent.RESULT_PERSISTENCE_START,
            trial_config,
            result,
        )
        await self._emit_hooks(TrialEvent.CANCEL, trial_config, result)
        await self._emit_hooks(TrialEvent.END, trial_config, result)

    async def _publish_exception_result(
        self,
        trial_config: TrialConfig,
        attempt: int,
        state: _AttemptState,
        result: TrialResult,
    ) -> None:
        try:
            await persist_selected_result(trial_config, attempt, result)
        except Exception as exc:
            self._logger.exception(
                "Failed to persist terminal result for trial %s",
                trial_config.trial_name,
            )
            result.secondary_exception_info.append(ExceptionInfo.from_exception(exc))
        else:
            await self._emit_hooks(
                TrialEvent.RESULT_PERSISTENCE_START,
                trial_config,
                result,
            )

        if state.ended:
            return
        try:
            await self._emit_hooks(TrialEvent.END, trial_config, result)
        except Exception:
            self._logger.exception(
                "Failed to emit terminal event for trial %s", trial_config.trial_name
            )

    async def _wait_for_termination(
        self, attempt_task: asyncio.Task[TrialResult], state: _AttemptState
    ) -> None:
        try:
            await attempt_task
        except asyncio.CancelledError:
            pass
        except Exception:
            self._logger.exception("Quarantined trial attempt failed during cleanup")

        trial = state.trial
        if trial is not None:
            cleanup_tasks = trial.pending_cleanup_tasks()
            if cleanup_tasks:
                cleanup_results = await asyncio.gather(
                    *cleanup_tasks, return_exceptions=True
                )
                for error in cleanup_results:
                    if isinstance(error, Exception):
                        self._logger.error(
                            "Quarantined provider cleanup failed",
                            exc_info=(type(error), error, error.__traceback__),
                        )

    def _release_or_quarantine(
        self, attempt_task: asyncio.Task[TrialResult], state: _AttemptState
    ) -> None:
        trial = state.trial
        cleanup_pending = bool(trial is not None and trial.pending_cleanup_tasks())
        if attempt_task.done() and not cleanup_pending:
            self._semaphore.release()
            return

        async def release_after_termination() -> None:
            await self._wait_for_termination(attempt_task, state)
            self._semaphore.release()

        quarantine_task = asyncio.create_task(release_after_termination())
        self._quarantine_tasks.add(quarantine_task)
        quarantine_task.add_done_callback(self._quarantine_tasks.discard)

    async def _supervise_attempt(
        self,
        trial_config: TrialConfig,
        attempt: int,
        state: _AttemptState,
        attempt_task: asyncio.Task[TrialResult],
        started_at: datetime,
    ) -> TrialResult:
        timeout_sec = trial_config.trial_attempt_timeout_sec
        if timeout_sec is None:
            return await attempt_task

        done, _ = await asyncio.wait({attempt_task}, timeout=timeout_sec)
        if attempt_task in done:
            return await attempt_task

        state.gate.active = False
        result = timeout_result(trial_config, attempt, state.trial, started_at)
        publish_task = asyncio.create_task(
            self._publish_timeout_result(trial_config, attempt, result)
        )
        attempt_task.cancel()
        done, _ = await asyncio.wait(
            {attempt_task, publish_task},
            timeout=trial_config.trial_cleanup_grace_sec,
        )
        if publish_task in done:
            await publish_task
        else:
            publish_task.cancel()
            self._observe_task(publish_task)
        return result

    async def _cancel_admitted_attempt(
        self,
        trial_config: TrialConfig,
        state: _AttemptState,
        attempt_task: asyncio.Task[TrialResult],
    ) -> None:
        attempt_task.cancel()
        await asyncio.wait({attempt_task}, timeout=trial_config.trial_cleanup_grace_sec)
        if not attempt_task.done():
            state.gate.active = False

    async def _run_admitted_attempt(
        self, trial_config: TrialConfig, attempt: int
    ) -> TrialResult:
        queue_started = time.monotonic()
        try:
            await self._semaphore.acquire()
        except asyncio.CancelledError as error:
            record_queue_wait(
                time.monotonic() - queue_started, outcome=OUTCOME_CANCELLED
            )
            record_failure(FAILURE_KIND_QUEUE_WAIT, error)
            raise
        record_queue_wait(time.monotonic() - queue_started, outcome=OUTCOME_SUCCEEDED)

        state = _AttemptState(gate=_HookGate())
        started_at = datetime.now(timezone.utc)
        attempt_task = asyncio.create_task(
            self._run_attempt_lifecycle(trial_config, attempt, state)
        )

        try:
            result = await self._supervise_attempt(
                trial_config,
                attempt,
                state,
                attempt_task,
                started_at,
            )
        except asyncio.CancelledError:
            await self._cancel_admitted_attempt(trial_config, state, attempt_task)
            self._release_or_quarantine(attempt_task, state)
            raise
        except Exception as exc:
            self._logger.exception(
                "Admitted trial %s failed outside its lifecycle boundary",
                trial_config.trial_name,
            )
            result = exception_result(
                trial_config,
                attempt,
                state.trial,
                started_at,
                exc,
            )
            await self._publish_exception_result(
                trial_config,
                attempt,
                state,
                result,
            )
            self._release_or_quarantine(attempt_task, state)
            return result
        self._release_or_quarantine(attempt_task, state)
        return result

    async def _execute_trial_with_retries(
        self, trial_config: TrialConfig
    ) -> TrialResult:
        result: TrialResult | None = None
        for attempt in range(self._retry_config.max_retries + 1):
            attempt_context = AttemptIdentity(
                trial_name=trial_config.trial_name,
                planned_attempt=self._planned_attempts.get(trial_config.trial_name, 1),
                infrastructure_retry=attempt,
            )
            with bind_attempt(attempt_context):
                result = await self._run_admitted_attempt(trial_config, attempt)

            if result.exception_info is None:
                break
            if not self._should_retry_exception(result.exception_info.exception_type):
                break
            if attempt == self._retry_config.max_retries:
                break

            delay_sec = self._calculate_backoff_delay_sec(attempt)
            self._logger.debug(
                "Trial %s failed with exception %s. Retrying in %.2f seconds...",
                trial_config.trial_name,
                result.exception_info.exception_type,
                delay_sec,
            )
            await asyncio.sleep(delay_sec)

        if result is None:
            raise RuntimeError(f"Trial {trial_config.trial_name} produced no result.")
        return result

    async def _run_trial(self, trial_config: TrialConfig) -> TrialResult:
        result = await self._execute_trial_with_retries(trial_config)
        if self._release_trial_payloads_in_memory:
            return result.slimmed()
        return result

    def submit(self, trial_config: TrialConfig) -> Coroutine[Any, Any, TrialResult]:
        """Return a coroutine that executes one trial."""
        return self._run_trial(trial_config)

    def submit_batch(
        self, configs: list[TrialConfig]
    ) -> list[Coroutine[Any, Any, TrialResult]]:
        """Return trial coroutines in the same order as ``configs``."""
        return [self.submit(config) for config in configs]
