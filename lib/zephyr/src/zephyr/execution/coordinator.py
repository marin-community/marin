# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Central coordinator actor that schedules tasks across the worker pool.

The coordinator owns the task queue, in-flight tracking, retry bookkeeping,
and the heartbeat/status loop. Workers pull tasks from it and report results
back; see ``zephyr.execution.worker.ZephyrWorker``.
"""

from __future__ import annotations

import logging
import sys
import threading
import time
from collections import Counter, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

from fray import ActorHandle, current_actor
from rigging.timing import ExponentialBackoff, RateLimiter

from zephyr.execution.internals import (
    MAX_SHARD_FAILURES,
    MAX_SHARD_INFRA_FAILURES,
    MAX_STATUS_TEXT_LENGTH,
    CounterSnapshot,
    ListShard,
    MemChunk,
    PullStatus,
    ShardFailureKind,
    ShardTask,
    TaskResult,
    WorkerState,
    ZephyrWorkerError,
    _format_bytes,
    _format_count,
    _push_iris_task_status,
    _stage_throughput,
)
from zephyr.plan import (
    Join,
    PhysicalOp,
    PhysicalPlan,
    PhysicalStage,
    Scatter,
    Shard,
    SourceItem,
    StageType,
)

logger = logging.getLogger(__name__)


@dataclass
class JobStatus:
    stage: str
    completed: int
    total: int
    retries: int
    in_flight: int
    queue_depth: int
    done: bool
    fatal_error: str | None
    workers: dict[str, dict[str, Any]]


class ZephyrCoordinator:
    """Central coordinator actor that owns and manages the worker pool.

    The coordinator creates workers via current_client(), runs a background
    loop for discovery and heartbeat checking, and manages all pipeline
    execution internally. Workers poll the coordinator for tasks until
    receiving a SHUTDOWN signal.
    """

    def __init__(self):
        # Task management state
        self._task_queue: deque[ShardTask] = deque()
        self._results: dict[int, TaskResult] = {}
        self._worker_states: dict[str, WorkerState] = {}
        self._last_seen: dict[str, float] = {}
        self._stage_name: str = ""
        # The index of the currently active stage. For joins and reshards, the index of the parent.
        self._current_stage_index: int = 0
        self._plan_stages: list = []  # PhysicalStage list, set in run_pipeline
        self._total_shards: int = 0
        self._completed_shards: int = 0
        self._retries: int = 0
        self._stage_epoch: int = 0
        self._in_flight: dict[str, tuple[ShardTask, int]] = {}
        # _task_attempts: monotonic generation for stale-result rejection (bumps on every
        # requeue). _task_error_attempts: TASK-only counter, bounded by MAX_SHARD_FAILURES.
        self._task_attempts: dict[int, int] = {}
        self._task_error_attempts: dict[int, int] = {}
        self._fatal_error: str | None = None
        self._shard_errors: dict[int, list[str]] = {}
        self._chunk_prefix: str = ""
        self._execution_id: str = ""
        self._no_workers_timeout: float = 60.0
        self._heartbeat_timeout: float = 120.0
        self._max_shard_failures: int = MAX_SHARD_FAILURES
        self._max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES
        # Per-worker in-flight counter snapshots and completed snapshots.
        # Each snapshot carries a monotonic generation so the coordinator
        # can discard stale or out-of-order heartbeats.
        self._worker_counters: dict[str, CounterSnapshot] = {}
        self._completed_counters: list[CounterSnapshot] = []

        # Worker management state (workers self-register via register_worker)
        self._worker_handles: dict[str, ActorHandle] = {}
        self._worker_group: Any = None  # ActorGroup, set via set_worker_group()
        self._coordinator_thread: threading.Thread | None = None
        self._shutdown_event = threading.Event()
        # Set when a stage may have completed (result, failure, or abort) so
        # ``_wait_for_stage`` wakes immediately instead of sleeping out its backoff.
        self._stage_done = threading.Event()
        # When True, pull_task returns SHUTDOWN to idle workers (stage complete).
        self._stage_complete: bool = False
        # When True, idle workers on the last stage receive SHUTDOWN once all
        # tasks are in-flight, so they exit eagerly instead of polling until
        # coordinator.shutdown().
        self._is_last_stage: bool = False
        # Stage type for the currently running stage; workers poll this to know
        # how many subprocess slots to create.
        self._current_stage: PhysicalStage | None = None
        self._initialized: bool = False
        self._pipeline_running: bool = False

        # Set at each _start_stage so _log_status can show average throughput since stage start.
        self._stage_monotonic_start: float | None = None

        # Lock for accessing coordinator state from background thread
        self._lock = threading.Lock()

        # Throttle Iris task-status pushes; the coordinator loop ticks more
        # frequently than the UI needs to refresh.
        self._task_stats_limiter = RateLimiter(interval_seconds=10.0)

        actor_ctx = current_actor()
        self._name = f"{actor_ctx.group_name}"

    def initialize(
        self,
        chunk_prefix: str,
        coordinator_handle: ActorHandle,
        no_workers_timeout: float = 60.0,
        heartbeat_timeout: float = 120.0,
        max_shard_failures: int = MAX_SHARD_FAILURES,
        max_shard_infra_failures: int = MAX_SHARD_INFRA_FAILURES,
    ) -> None:
        """Initialize coordinator for push-based worker registration.

        Workers register themselves by calling register_worker() when they start.
        This eliminates polling overhead and log spam from discover_new() calls.

        Args:
            chunk_prefix: Storage prefix for intermediate chunks
            coordinator_handle: Handle to this coordinator actor (passed from context)
            no_workers_timeout: Seconds to wait for at least one worker before failing.
            heartbeat_timeout: Seconds without a worker heartbeat before requeue.
            max_shard_failures: Per-shard cap on explicit task errors before abort.
            max_shard_infra_failures: Per-shard cap on infra failures (preemption /
                heartbeat) observed while the same shard is in flight before abort.
        """
        self._chunk_prefix = chunk_prefix
        self._self_handle = coordinator_handle
        self._no_workers_timeout = no_workers_timeout
        self._heartbeat_timeout = heartbeat_timeout
        self._max_shard_failures = max_shard_failures
        self._max_shard_infra_failures = max_shard_infra_failures

        logger.info("Coordinator initialized")

        # Start coordinator background loop (heartbeat checking only)
        self._coordinator_thread = threading.Thread(
            target=self._coordinator_loop, daemon=True, name="zephyr-coordinator-loop"
        )
        self._coordinator_thread.start()
        self._initialized = True

    def set_worker_group(self, worker_group: Any) -> None:
        """Set the worker ActorGroup so the coordinator can detect permanent worker death."""
        self._worker_group = worker_group

    def register_worker(self, worker_id: str, worker_handle: ActorHandle) -> int:
        """Called by workers when they come online to register with coordinator.

        Handles re-registration from reconstructed workers (e.g. after node
        preemption) by updating the stale handle and resetting worker state.

        Returns the current stage epoch so the caller can bind its poll loop
        to this stage and detect when it has been superseded.
        """
        with self._lock:
            if worker_id in self._worker_handles:
                logger.info("Worker %s re-registering (likely reconstructed), updating handle", worker_id)
                self._worker_handles[worker_id] = worker_handle
                self._worker_states[worker_id] = WorkerState.READY
                self._last_seen[worker_id] = time.monotonic()
                # NOTE: if there was a task assigned to the worker, there's a race condition between marking
                # the worker as unhealthy via heartbeat and re-registration. If we do not requeue we may silently
                # lose tasks.
                self._maybe_requeue_worker_task(worker_id)
                return self._stage_epoch

            self._worker_handles[worker_id] = worker_handle
            self._worker_states[worker_id] = WorkerState.READY
            self._last_seen[worker_id] = time.monotonic()
            logger.info("Worker %s registered, total: %d", worker_id, len(self._worker_handles))
            return self._stage_epoch

    def deregister_worker(self, worker_id: str) -> None:
        """Remove a sub-worker that has finished its stage pool."""
        with self._lock:
            self._worker_handles.pop(worker_id, None)
            self._worker_states.pop(worker_id, None)
            self._last_seen.pop(worker_id, None)

    def get_current_stage_type(self) -> StageType | None:
        """Return the stage type currently running, or ``None`` if between stages.

        Workers poll this between stages to discover the next stage type and
        spin up the right number of subprocess slots.
        """
        with self._lock:
            return self._current_stage.stage_type if self._current_stage is not None else None

    def _mark_stage_complete(self) -> None:
        with self._lock:
            self._stage_complete = True
            self._current_stage = None

    def _coordinator_loop(self) -> None:
        """Background loop for heartbeat checking and worker job monitoring."""
        last_log_time = 0.0

        while not self._shutdown_event.is_set():
            if sys.is_finalizing():
                return
            try:
                self.check_heartbeats(self._heartbeat_timeout)
                self._check_worker_group()

                now = time.monotonic()
                if self._has_active_execution() and now - last_log_time > 5.0:
                    self._log_status()
                    self._report_task_stats()
                    last_log_time = now
            except Exception:
                if sys.is_finalizing():
                    return
                logger.exception("Coordinator loop crashed, aborting pipeline")
                self.abort("Coordinator loop crashed unexpectedly")
                return

            self._shutdown_event.wait(timeout=0.5)

    def _check_worker_group(self) -> None:
        """Abort the pipeline if the worker job has permanently terminated."""
        if self._worker_group is None or self._fatal_error is not None:
            return
        # After the last stage completes, workers exit cleanly via SHUTDOWN.
        # The worker job finishing at that point is expected, not a crash.
        with self._lock:
            if self._total_shards > 0 and self._completed_shards >= self._total_shards:
                return
        try:
            if self._worker_group.is_done():
                self.abort(
                    "Worker job terminated permanently (all retries exhausted). "
                    "Workers likely crashed (OOM or other fatal error)."
                )
        except Exception:
            logger.debug("Failed to check worker group status", exc_info=True)

    def _has_active_execution(self) -> bool:
        return self._execution_id != "" and self._total_shards > 0 and self._completed_shards < self._total_shards

    def _report_task_stats(self) -> None:
        """Push task status text to the Iris coordinator if available."""

        def build_md() -> tuple[str, str]:
            with self._lock:
                current_stage_index = self._current_stage_index
                stage_name = self._stage_name
                plan_stages = self._plan_stages
                completed = self._completed_shards
                total_shards = self._total_shards
                in_flight = len(self._in_flight)
                queued = len(self._task_queue)
                stage_start = self._stage_monotonic_start

            totals = self.get_counters()
            elapsed = time.monotonic() - (stage_start or time.monotonic())
            throughput = _stage_throughput(totals, stage_name, elapsed)

            lines = ["**Stages**\n"]
            for idx, stage in enumerate(plan_stages):
                stage_desc = _get_stage_description(stage)
                bullet = f"- **{stage_desc}**" if idx == current_stage_index else f"- {stage_desc}"
                lines.append(f"{bullet}")

            pct = int(100 * completed / total_shards) if total_shards > 0 else 0
            lines.append(
                f"\n**Shards** — {completed}/{total_shards} complete ({pct}%), {in_flight} in-flight, {queued} queued"
            )
            if throughput is not None:
                items, bytes_processed, item_rate, byte_rate = throughput
                lines.append(
                    f"\n**Throughput** — {_format_count(items)} items ({_format_count(item_rate)}/s), "
                    f"{_format_bytes(bytes_processed)} ({_format_bytes(byte_rate)}/s)"
                )

            detail_md = "\n".join(lines)[:MAX_STATUS_TEXT_LENGTH]

            current_stage_desc = _get_stage_description(plan_stages[current_stage_index]) if plan_stages else ""
            summary_lines = [f"**{current_stage_desc}** ({current_stage_index + 1}/{len(plan_stages)})"]
            summary_lines.append(f"{completed}/{total_shards} shards ({pct}%)")
            if throughput is not None:
                _, _, item_rate, byte_rate = throughput
                summary_lines.append(f"{_format_count(item_rate)} items/s")
                summary_lines.append(f"{_format_bytes(byte_rate)}/s")
            summary_md = "  \n".join(summary_lines)
            return detail_md, summary_md

        _push_iris_task_status(self._task_stats_limiter, build_md)

    def _log_status(self) -> None:
        with self._lock:
            states = list(self._worker_states.values())
            retried = {idx: att for idx, att in self._task_attempts.items() if att > 0}
        alive = sum(1 for s in states if s in {WorkerState.READY, WorkerState.BUSY})
        dead = sum(1 for s in states if s in {WorkerState.FAILED, WorkerState.DEAD})

        totals = self.get_counters()
        base_msg = "[%s] [%s] %d/%d complete, %d in-flight, %d queued, %d/%d workers alive, %d dead"
        base_args = (
            self._execution_id,
            self._stage_name,
            self._completed_shards,
            self._total_shards,
            len(self._in_flight),
            len(self._task_queue),
            alive,
            len(self._worker_handles),
            dead,
        )

        # Map-only stages don't yield through ``_wrap_stage_stats`` and never
        # populate these counters. Drop the items/bytes_processed segment for
        # those stages.
        elapsed = time.monotonic() - (self._stage_monotonic_start or time.monotonic())
        throughput = _stage_throughput(totals, self._stage_name, elapsed)
        if throughput is not None:
            items, bytes_processed, item_rate, byte_rate = throughput
            logger.info(
                base_msg + "; items=%s (%s/s), bytes_processed=%s (%s/s)",
                *base_args,
                _format_count(items),
                _format_count(item_rate),
                _format_bytes(bytes_processed),
                _format_bytes(byte_rate),
            )
        else:
            logger.info(base_msg, *base_args)
        if retried:
            attempts_histogram = dict(sorted(Counter(retried.values()).items()))
            logger.warning("[%s] Shards retried (attempts: shard count): %s", self._execution_id, attempts_histogram)

    def _record_shard_failure(
        self,
        worker_id: str,
        kind: ShardFailureKind,
        error_info: str | None = None,
    ) -> bool:
        """Requeue the worker's in-flight shard; abort if a per-shard cap is hit.

        TASK errors are bounded by ``MAX_SHARD_FAILURES``. INFRA failures
        observed while the *same* shard was in flight are bounded by
        ``MAX_SHARD_INFRA_FAILURES`` so a payload that deterministically
        crashes its worker (native SIGSEGV, OOM) doesn't loop forever now
        that shard execution is in-process.

        Must be called with lock held. Returns True if the pipeline was aborted.
        """
        task_and_attempt = self._in_flight.pop(worker_id, None)

        # Zero counters but keep the generation watermark so late heartbeats
        # from the old task are rejected.
        existing = self._worker_counters.get(worker_id)
        if existing is not None:
            self._worker_counters[worker_id] = CounterSnapshot.empty(existing.generation)

        if task_and_attempt is None:
            return False

        task, _ = task_and_attempt
        shard_idx = task.shard_idx

        if error_info is not None:
            self._shard_errors.setdefault(shard_idx, []).append(error_info)

        # Bump generation regardless of kind so report_result rejects stale attempts.
        self._task_attempts[shard_idx] += 1
        # Wake _wait_for_stage on every accounted failure (requeue or abort);
        # the waiter re-checks _fatal_error / completed counts after waking.
        self._stage_done.set()

        if kind is ShardFailureKind.TASK:
            self._task_error_attempts[shard_idx] += 1
            error_attempts = self._task_error_attempts[shard_idx]
            if error_attempts >= self._max_shard_failures:
                errors = self._shard_errors.get(shard_idx, [])
                error_detail = f"\nLast error:\n{errors[-1]}" if errors else ""
                logger.error(
                    "Shard %d has failed %d times (max %d), last failure on worker %s, aborting pipeline.",
                    shard_idx,
                    error_attempts,
                    self._max_shard_failures,
                    worker_id,
                )
                self._fatal_error = (
                    f"Shard {shard_idx} failed {error_attempts} times "
                    f"(max {self._max_shard_failures}), last failure on worker {worker_id}.{error_detail}"
                )
                return True

            logger.warning(
                "Shard %d failed on worker %s (task error %d/%d), re-queuing.",
                shard_idx,
                worker_id,
                error_attempts,
                self._max_shard_failures,
            )
        else:
            self._task_infra_attempts[shard_idx] += 1
            infra_attempts = self._task_infra_attempts[shard_idx]
            if infra_attempts >= self._max_shard_infra_failures:
                logger.error(
                    "Shard %d has been in flight during %d infra failures (max %d); "
                    "treating as a deterministic crasher (likely native SIGSEGV / OOM in shard "
                    "code) and aborting pipeline. Last failure on worker %s.",
                    shard_idx,
                    infra_attempts,
                    self._max_shard_infra_failures,
                    worker_id,
                )
                self._fatal_error = (
                    f"Shard {shard_idx} crashed its worker {infra_attempts} times "
                    f"(max {self._max_shard_infra_failures} infra failures while in flight); "
                    f"last failure on worker {worker_id}."
                )
                return True

            logger.warning(
                "Shard %d requeued from worker %s due to infra failure (preemption/heartbeat). "
                "Total generation: %d, task errors so far: %d/%d, infra-while-in-flight: %d/%d.",
                shard_idx,
                worker_id,
                self._task_attempts[shard_idx],
                self._task_error_attempts[shard_idx],
                self._max_shard_failures,
                infra_attempts,
                self._max_shard_infra_failures,
            )

        self._task_queue.append(task)
        self._retries += 1
        return False

    def _maybe_requeue_worker_task(self, worker_id: str) -> None:
        """Requeue the worker's in-flight task as an INFRA failure (preemption/heartbeat)."""
        self._record_shard_failure(worker_id, ShardFailureKind.INFRA)

    def pull_task(self, worker_id: str, epoch: int | None = None) -> tuple[ShardTask, int, dict] | PullStatus:
        """Called by workers to get next task.

        Args:
            worker_id: Unique ID for this worker slot.
            epoch: Stage epoch the slot was registered under. If provided and it
                no longer matches the coordinator's current epoch, the slot has
                survived into a later stage and is told STAGE_COMPLETED so the
                worker re-registers under the new epoch.

        Returns:
            - ``(task, attempt, config)`` if a task is available.
            - ``PullStatus.SHUTDOWN`` if the coordinator is shutting down or the
              last stage has no more work to hand out — the worker should exit.
            - ``PullStatus.STAGE_COMPLETED`` at a non-last stage boundary or when
              the slot's epoch is stale — slot exits, worker re-pools for the
              next stage.
            - ``PullStatus.NO_WORK_BACKOFF`` if the queue is empty mid-stage —
              slot should sleep and retry.
        """
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()
            self._worker_states[worker_id] = WorkerState.READY

            if self._shutdown_event.is_set():
                self._worker_states[worker_id] = WorkerState.DEAD
                return PullStatus.SHUTDOWN

            if epoch is not None and epoch != self._stage_epoch:
                # Slot was created for an earlier stage and missed its STAGE_COMPLETED
                # (e.g. it was sleeping through the _mark_stage_complete window).
                # Tell it the stage is over so the worker re-registers under the new epoch.
                self._worker_states[worker_id] = WorkerState.DEAD
                return PullStatus.STAGE_COMPLETED

            if self._fatal_error:
                return PullStatus.NO_WORK_BACKOFF

            if not self._task_queue:
                if self._is_last_stage:
                    # Last stage has no more tasks to hand out — pipeline is winding
                    # down for this worker. If an in-flight peer crashes and its
                    # shard is requeued, Iris restarts the worker which re-registers
                    # and picks it up. _check_worker_group() detects permanent
                    # worker-job death as a failsafe so we never deadlock.
                    self._worker_states[worker_id] = WorkerState.DEAD
                    return PullStatus.SHUTDOWN
                if self._stage_complete:
                    # Non-last stage boundary — slot exits so the worker can re-pool
                    # at the size required by the next stage (map vs reduce).
                    self._worker_states[worker_id] = WorkerState.DEAD
                    return PullStatus.STAGE_COMPLETED
                return PullStatus.NO_WORK_BACKOFF

            task = self._task_queue.popleft()
            attempt = self._task_attempts[task.shard_idx]
            self._in_flight[worker_id] = (task, attempt)
            self._worker_states[worker_id] = WorkerState.BUSY

            config = {
                "chunk_prefix": self._chunk_prefix,
                "execution_id": self._execution_id,
            }
            return (task, attempt, config)

    def _assert_in_flight_consistent(self, worker_id: str, shard_idx: int) -> None:
        """Assert _in_flight[worker_id], if present, matches the reported shard.

        Workers block on report_result/report_error before calling pull_task,
        so _in_flight can never point to a different shard. It may be absent
        if a heartbeat timeout already re-queued the task.
        """
        in_flight = self._in_flight.get(worker_id)
        if in_flight is not None:
            assert in_flight[0].shard_idx == shard_idx, (
                f"_in_flight mismatch for {worker_id}: reporting shard {shard_idx}, "
                f"but _in_flight tracks shard {in_flight[0].shard_idx}. "
                f"This indicates report_result/pull_task reordering — workers must block on report_result."
            )

    def report_result(
        self,
        worker_id: str,
        shard_idx: int,
        attempt: int,
        result: TaskResult,
        counter_snapshot: CounterSnapshot,
    ) -> None:
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()
            self._assert_in_flight_consistent(worker_id, shard_idx)

            current_attempt = self._task_attempts.get(shard_idx, 0)
            if attempt != current_attempt:
                logger.warning(
                    f"Ignoring stale result from worker {worker_id} for shard {shard_idx} "
                    f"(attempt {attempt}, current {current_attempt})"
                )
                return

            self._results[shard_idx] = result
            self._completed_shards += 1
            self._in_flight.pop(worker_id, None)
            self._worker_states[worker_id] = WorkerState.READY
            self._completed_counters.append(counter_snapshot)
            # Zero the in-flight counters but keep the generation watermark
            # so late heartbeats from this task are rejected.
            self._worker_counters[worker_id] = CounterSnapshot.empty(counter_snapshot.generation)
            self._stage_done.set()

    def report_error(self, worker_id: str, shard_idx: int, error_info: str) -> None:
        """Worker reports a task failure. Re-queues up to MAX_SHARD_FAILURES."""
        with self._lock:
            self._last_seen[worker_id] = time.monotonic()
            self._assert_in_flight_consistent(worker_id, shard_idx)
            aborted = self._record_shard_failure(worker_id, ShardFailureKind.TASK, error_info)
            self._worker_states[worker_id] = WorkerState.DEAD if aborted else WorkerState.READY

    def heartbeat(self, worker_id: str, counter_snapshot: CounterSnapshot | None = None) -> None:
        self._last_seen[worker_id] = time.monotonic()
        if counter_snapshot is not None:
            with self._lock:
                existing = self._worker_counters.get(worker_id)
                if existing is None or counter_snapshot.generation > existing.generation:
                    self._worker_counters[worker_id] = counter_snapshot

    def get_status(self) -> JobStatus:
        with self._lock:
            return JobStatus(
                stage=self._stage_name,
                completed=self._completed_shards,
                total=self._total_shards,
                retries=self._retries,
                in_flight=len(self._in_flight),
                queue_depth=len(self._task_queue),
                done=self._shutdown_event.is_set(),
                fatal_error=self._fatal_error,
                workers={
                    wid: {
                        "state": state.value,
                        "last_seen_ago": time.monotonic() - self._last_seen.get(wid, 0),
                    }
                    for wid, state in self._worker_states.items()
                },
            )

    def get_counters(self, worker_id: str | None = None) -> dict[str, int]:
        """Return counter values, optionally filtered to a single worker.

        Args:
            worker_id: If provided, return the latest snapshot for this worker
                only. If None, return totals derived from completed and
                in-flight snapshots.
        """
        with self._lock:
            if worker_id is not None:
                snap = self._worker_counters.get(worker_id)
                return dict(snap.counters) if snap is not None else {}

            totals: Counter[str] = Counter()
            for snap in self._completed_counters:
                totals.update(snap.counters)
            for snap in self._worker_counters.values():
                totals.update(snap.counters)
            return dict(totals)

    def get_fatal_error(self) -> str | None:
        with self._lock:
            return self._fatal_error

    def abort(self, reason: str) -> None:
        """Set a fatal error that causes the current stage to fail immediately.

        Called by the external worker watchdog when the worker job terminates
        permanently (e.g. all retries exhausted after OOM).
        """
        with self._lock:
            if self._fatal_error is None:
                logger.error("Coordinator aborted: %s", reason)
                self._fatal_error = reason
                self._stage_done.set()

    def _start_stage(
        self,
        stage_name: str,
        current_stage_index: int,
        tasks: list[ShardTask],
        is_last_stage: bool = False,
    ) -> None:
        """Load a new stage's tasks into the queue."""
        with self._lock:
            self._task_queue = deque(tasks)
            self._results = {}
            self._stage_name = stage_name
            self._current_stage_index = current_stage_index
            self._total_shards = len(tasks)
            self._completed_shards = 0
            self._retries = 0
            self._in_flight = {}
            self._task_attempts = {task.shard_idx: 0 for task in tasks}
            self._task_error_attempts = {task.shard_idx: 0 for task in tasks}
            # Counts INFRA failures observed while this specific shard was in
            # flight on the dying worker — bounded by MAX_SHARD_INFRA_FAILURES
            # so a shard that deterministically crashes its worker (native
            # SIGSEGV, OOM) eventually aborts instead of retrying forever.
            self._task_infra_attempts = {task.shard_idx: 0 for task in tasks}
            self._shard_errors = {}
            self._fatal_error = None
            self._is_last_stage = is_last_stage
            self._stage_complete = False
            self._stage_epoch += 1
            # Only reset in-flight worker snapshots; completed snapshots
            # accumulate across stages for full pipeline visibility.
            self._worker_counters = {}
            self._stage_monotonic_start = time.monotonic()
            self._stage_done.clear()

    def _wait_for_stage(self) -> None:
        """Block until current stage completes or error occurs."""
        backoff = ExponentialBackoff(initial=0.1, maximum=1.0)
        last_log_completed = -1
        start_time = time.monotonic()
        all_dead_since: float | None = None
        no_workers_timeout = self._no_workers_timeout

        while True:
            with self._lock:
                if self._fatal_error:
                    raise ZephyrWorkerError(self._fatal_error)

                completed = self._completed_shards
                total = self._total_shards

                if completed >= total:
                    return

                # Count alive workers (READY or BUSY), not just total registered.
                # Dead/failed workers stay in _worker_handles but can't make progress.
                alive_workers = sum(
                    1 for s in self._worker_states.values() if s in {WorkerState.READY, WorkerState.BUSY}
                )

                if alive_workers == 0:
                    now = time.monotonic()
                    elapsed = now - start_time

                    if all_dead_since is None:
                        all_dead_since = now
                        logger.warning("All workers are dead/failed. Waiting for workers to recover...")

                    dead_duration = now - all_dead_since
                    if dead_duration > no_workers_timeout:
                        raise ZephyrWorkerError(
                            f"No alive workers for {dead_duration:.1f}s "
                            f"(total elapsed {elapsed:.1f}s). "
                            f"All {len(self._worker_handles)} registered workers are dead/failed. "
                            "Check cluster resources and worker group configuration."
                        )
                else:
                    # Workers are alive — reset the dead timer
                    all_dead_since = None

            if completed != last_log_completed:
                logger.info("[%s] %d/%d tasks completed", self._stage_name, completed, total)
                last_log_completed = completed
                backoff.reset()

            # Wake promptly on completions / errors / aborts; the timeout still
            # bounds the sleep so the no-alive-workers timer fires regardless.
            if self._stage_done.wait(timeout=backoff.next_interval()):
                self._stage_done.clear()

    def _collect_results(self) -> dict[int, TaskResult]:
        """Return results for the completed stage."""
        with self._lock:
            return dict(self._results)

    def run_pipeline(
        self,
        plan: PhysicalPlan,
        execution_id: str,
    ) -> list:
        """Run complete pipeline, blocking until done. Returns flattened results."""
        with self._lock:
            if self._pipeline_running:
                self._fatal_error = "run_pipeline called while another pipeline is already running"
                raise RuntimeError(self._fatal_error)
            self._pipeline_running = True
            self._execution_id = execution_id

        try:
            shards = _build_source_shards(plan.source_items)
            if not shards:
                return []

            last_worker_stage_idx = max(
                (i for i, s in enumerate(plan.stages) if s.stage_type != StageType.RESHARD),
                default=-1,
            )

            with self._lock:
                self._current_stage_index = 0
                self._plan_stages = list(plan.stages)

            for stage_idx, stage in enumerate(plan.stages):
                if stage.stage_type == StageType.RESHARD:
                    shards = _reshard_refs(shards, stage.output_shards or len(shards))
                    continue

                aux_per_shard = self._compute_join_aux(stage.operations, shards, stage_idx)
                shards = self._run_worker_stage(
                    stage,
                    shards,
                    stage_label=f"stage{stage_idx}-{stage.stage_name(max_length=40)}",
                    stage_index_for_state=stage_idx,
                    aux_per_shard=aux_per_shard,
                    is_last_stage=(stage_idx == last_worker_stage_idx),
                )

            # Flatten final results — each shard may involve I/O (unpickling from
            # remote storage), so parallelize across shards with a thread pool.
            def _materialize_shard(shard):
                return list(shard)

            with ThreadPoolExecutor(max_workers=min(32, len(shards))) as flatten_pool:
                materialized = flatten_pool.map(_materialize_shard, shards)

            flat_result = []
            for items in materialized:
                flat_result.extend(items)

            return flat_result
        finally:
            with self._lock:
                self._pipeline_running = False

    def _run_worker_stage(
        self,
        stage: PhysicalStage,
        shards: list[Shard],
        *,
        stage_label: str,
        stage_index_for_state: int,
        aux_per_shard: list[dict[int, Shard]] | None = None,
        is_last_stage: bool = False,
    ) -> list[Shard]:
        """Submit a worker stage, wait for completion, return regrouped output shards.

        ``stage_index_for_state`` is the index reported in coordinator state for
        UI/logging — for join right-sub-stages this is the *parent* stage index
        so progress reports stay attached to the user-visible stage.
        """
        with self._lock:
            self._current_stage = stage

        tasks = _compute_tasks_from_shards(shards, stage, stage_name=stage_label, aux_per_shard=aux_per_shard)
        logger.info(
            "[%s] Starting stage %s (%s) with %d tasks", self._execution_id, stage_label, stage.stage_type, len(tasks)
        )
        self._start_stage(stage_label, stage_index_for_state, tasks, is_last_stage=is_last_stage)
        self._wait_for_stage()

        self._mark_stage_complete()

        result_refs = self._collect_results()

        stage_is_scatter = any(isinstance(op, Scatter) for op in stage.operations)
        return _regroup_result_refs(
            result_refs,
            len(shards),
            output_shard_count=stage.output_shards,
            is_scatter=stage_is_scatter,
        )

    def _compute_join_aux(
        self,
        operations: list[PhysicalOp],
        shard_refs: list[Shard],
        parent_stage_idx: int,
    ) -> list[dict[int, Shard]] | None:
        """Execute right sub-plans for join operations, returning aux refs per shard."""
        all_right_shard_refs: dict[int, list[Shard]] = {}

        for i, op in enumerate(operations):
            if not isinstance(op, Join) or op.right_plan is None:
                continue

            right_refs = _build_source_shards(op.right_plan.source_items)

            for stage_idx, right_stage in enumerate(op.right_plan.stages):
                if right_stage.stage_type == StageType.RESHARD:
                    right_refs = _reshard_refs(right_refs, right_stage.output_shards or len(right_refs))
                    continue

                right_refs = self._run_worker_stage(
                    right_stage,
                    right_refs,
                    stage_label=f"join-right-{parent_stage_idx}-{i}-stage{stage_idx}",
                    stage_index_for_state=parent_stage_idx,
                )

            if len(shard_refs) != len(right_refs):
                raise ValueError(
                    f"Sorted merge join requires equal shard counts. "
                    f"Left has {len(shard_refs)} shards, right has {len(right_refs)} shards."
                )
            all_right_shard_refs[i] = right_refs

        if not all_right_shard_refs:
            return None

        return [
            {op_idx: right_refs[shard_idx] for op_idx, right_refs in all_right_shard_refs.items()}
            for shard_idx in range(len(shard_refs))
        ]

    def __repr__(self) -> str:
        return f"ZephyrCoordinator(name={self._name})"

    def shutdown(self) -> None:
        """Signal workers to exit. Worker group is managed by ZephyrContext."""
        logger.info("[coordinator.shutdown] Starting shutdown")

        counters = self.get_counters()
        if counters:
            logger.info("[coordinator.shutdown] Final counters: %s", counters)

        self._shutdown_event.set()

        # Wait for coordinator thread to exit
        if self._coordinator_thread is not None:
            self._coordinator_thread.join(timeout=5.0)

        logger.info("Coordinator shutdown complete")

    def set_chunk_config(self, prefix: str, execution_id: str) -> None:
        """Configure chunk storage for this execution."""
        with self._lock:
            self._chunk_prefix = prefix
            self._execution_id = execution_id

    def check_heartbeats(self, timeout: float = 120.0) -> None:
        """Marks stale workers as FAILED, re-queues their in-flight tasks."""
        with self._lock:
            now = time.monotonic()
            for worker_id, last in list(self._last_seen.items()):
                if now - last > timeout and self._worker_states.get(worker_id) not in {
                    WorkerState.FAILED,
                    WorkerState.DEAD,
                }:
                    logger.warning(f"Zephyr worker {worker_id} failed to heartbeat within timeout ({now - last:.1f}s)")
                    self._worker_states[worker_id] = WorkerState.FAILED
                    self._maybe_requeue_worker_task(worker_id)


# ---------------------------------------------------------------------------
# Shard plumbing — shared by ZephyrCoordinator and ZephyrContext
# ---------------------------------------------------------------------------


def _regroup_result_refs(
    result_refs: dict[int, TaskResult],
    input_shard_count: int,
    output_shard_count: int | None = None,
    is_scatter: bool = False,
) -> list[Shard]:
    """Regroup worker output refs by output shard index without loading data.

    Non-scatter: each worker's ListShard maps to its own index (identity).
    Scatter: passes the list of scatter data-file paths to every reducer.
    Each reducer reads the per-mapper ``.scatter_meta`` sidecars in parallel
    to build its own ``ScatterReader`` without coordinator-side consolidation.
    """
    if is_scatter:
        # Scatter routes records into exactly ``output_shard_count`` buckets via
        # ``hash(key) % output_shard_count``; spawning more reduce tasks than that
        # produces empty output files for shard indices that no record hashes to.
        # When output_shard_count is None (group_by auto-detect), inherit the
        # input shard count.
        num_output = output_shard_count if output_shard_count is not None else input_shard_count

        # Collect all scatter file paths from all workers. The coordinator
        # does NOT read the sidecars or write a consolidated manifest —
        # reducers do their own parallel sidecar reads.
        all_paths: list[str] = []
        for result in result_refs.values():
            all_paths.extend(result.shard)

        shared_refs = MemChunk(items=all_paths)
        return [ListShard(refs=[shared_refs]) for _ in range(num_output)]

    # Non-scatter: 1:1 mapping from input shard index to output. Resharding
    # to a different shard count belongs to ReshardOp, not here.
    num_output = max(max(result_refs.keys(), default=0) + 1, input_shard_count)
    return [result_refs[idx].shard if idx in result_refs else ListShard(refs=[]) for idx in range(num_output)]


def _build_source_shards(source_items: list[SourceItem]) -> list[Shard]:
    """Build shard data from source items.

    Each source item becomes a single-element chunk in its assigned shard.
    """
    items_by_shard: dict[int, list] = defaultdict(list)
    for item in source_items:
        items_by_shard[item.shard_idx].append(item.data)

    num_shards = max(items_by_shard.keys()) + 1 if items_by_shard else 0
    shards: list[Shard] = []
    for i in range(num_shards):
        shards.append(ListShard(refs=[MemChunk(items=items_by_shard.get(i, []))]))

    return shards


def _reshard_refs(shards: list[Shard], num_shards: int) -> list[Shard]:
    """Reshard shard refs by output shard index without loading data.

    Only supported on ListShards (non-scatter data).
    """
    output_by_shard: dict[int, list] = defaultdict(list)
    output_idx = 0
    for shard in shards:
        if not isinstance(shard, ListShard):
            raise ValueError("Reshard is only supported on ListShard (non-scatter data)")
        for chunk in shard.refs:
            output_by_shard[output_idx].append(chunk)
            output_idx = (output_idx + 1) % num_shards
    return [ListShard(refs=output_by_shard.get(idx, [])) for idx in range(num_shards)]


def _compute_tasks_from_shards(
    shard_refs: list[Shard],
    stage: PhysicalStage,
    stage_name: str,
    aux_per_shard: list[dict[int, Shard]] | None = None,
) -> list[ShardTask]:
    """Convert shard references into ShardTasks for the coordinator."""
    total = len(shard_refs)
    tasks = []

    for i, shard in enumerate(shard_refs):
        aux_shards = None
        if aux_per_shard and aux_per_shard[i]:
            aux_shards = aux_per_shard[i]

        tasks.append(
            ShardTask(
                shard_idx=i,
                total_shards=total,
                shard=shard,
                operations=stage.operations,
                stage_name=stage_name,
                aux_shards=aux_shards,
            )
        )

    return tasks


def _get_stage_description(stage: PhysicalStage) -> str:
    """Get a description of a stage, including optional hints."""
    name = stage.stage_name()
    hint_parts = []
    if stage.stage_type == StageType.RESHARD:
        hint_parts.append(f"reshard→{stage.output_shards}")
    for op in stage.operations:
        if isinstance(op, Join) and op.right_plan is not None:
            hint_parts.append(f"join({len(op.right_plan.source_items)} items)")
    hint_str = f" [{', '.join(hint_parts)}]" if hint_parts else ""
    return f"{name}{hint_str}"
