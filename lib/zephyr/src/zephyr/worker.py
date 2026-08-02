# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker actor for Zephyr pipelines."""

import logging
import threading
import time
import traceback
from collections import defaultdict
from collections.abc import Callable
from contextlib import suppress

from fray.actor import ActorFuture, ActorHandle, current_actor
from iris.cluster.client.job_info import get_job_info
from rigging.timing import ExponentialBackoff, RateLimiter

from zephyr.coordinator import CoordinatorUnreachable, PullStatus, PullTask
from zephyr.stage_io import ShardTask, StageRunner, TaskResult, ZephyrTaskResources, _stage_throughput
from zephyr.stats import _push_iris_task_status
from zephyr.worker_context import CounterEntry, CounterSnapshot, merge_counter_entries

logger = logging.getLogger(__name__)


def _format_worker_status_md(active_tasks: int, stage: str) -> tuple[str, str]:
    """Return worker status text, or two idle values when no task is active."""
    if active_tasks == 0 or not stage:
        return "idle", "idle"
    summary = f"**{stage}** — {active_tasks} task(s)"
    detail = "  \n".join([f"**Stage**: {stage}", f"**Active tasks**: {active_tasks}"])
    return detail, summary


class ZephyrWorker:
    """Long-lived worker actor with a single poll loop and per-task threads.

    The worker registers once with the coordinator, then loops: polls with its
    current available ``ZephyrTaskResources``, spins off a thread per dispatched task,
    and immediately polls again. The coordinator dispatches when the worker's
    available resources can fit the next task's cost; otherwise it returns
    NO_WORK_BACKOFF.
    """

    def __init__(
        self,
        coordinator_handle: ActorHandle,
        stage_runner_factory: Callable[[], StageRunner],
        total_resources: ZephyrTaskResources,
    ):
        self._coordinator = coordinator_handle
        self._stage_runner_factory = stage_runner_factory
        self._shutdown_event = threading.Event()
        self._counter_generations: dict[str, int] = defaultdict(int)
        self._last_reported_counters: dict[str, dict[str, CounterEntry]] = {}
        # Runners and sub-IDs for currently active slots — written by _stage_manager,
        # read (snapshotted) by heartbeat thread.
        self._active_runners: list[tuple[str, StageRunner]] = []
        self._active_task_count: int = 0
        self._current_stage_name: str = ""

        # Resource pool: each accepted task deducts its cost and restores it on
        # completion. The coordinator gates dispatch on the available amount,
        # so the pool implicitly limits concurrency.
        self._available: ZephyrTaskResources = total_resources
        self._resources_lock = threading.Lock()
        # Set by task threads on completion so the poll loop wakes early.
        self._task_completed_event = threading.Event()

        # Throttle Iris status pushes; the heartbeat loop ticks faster than
        # the UI needs to refresh.
        self._iris_status_limiter = RateLimiter(interval_seconds=10.0)

        # Capture actor context while ContextVar is still set (child threads
        # in Python <3.12 don't inherit it).
        self._actor_ctx = current_actor()
        self._host_shutdown_event = self._actor_ctx.shutdown_event
        self._worker_id = f"{self._actor_ctx.group_name}-{self._actor_ctx.index}"
        self._actor_handle = self._actor_ctx.handle
        job_info = get_job_info()
        self._task_id = job_info.task_id.to_wire() if job_info is not None else ""

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            args=(coordinator_handle,),
            daemon=True,
            name=f"zephyr-hb-{self._worker_id}",
        )
        self._heartbeat_thread.start()

        self._poll_thread = threading.Thread(
            target=self._poll_loop,
            daemon=True,
            name=f"zephyr-poll-{self._worker_id}",
        )
        self._poll_thread.start()

    def _poll_loop(self) -> None:
        """Single poll loop: requests tasks from the coordinator using current available resources.

        Registers once with the coordinator, then loops: polls with available CPU and
        memory, spins off a thread per dispatched task, and immediately polls again.
        At stage boundaries, the loop sleeps briefly and then polls again.
        """
        logger.info("[%s] Poll loop starting", self._worker_id)
        try:
            self._coordinator.register_worker.remote(
                self._worker_id,
                self._actor_handle,
                self._task_id,
            ).result(timeout=30.0)
        except Exception:
            logger.error("[%s] Failed to register with coordinator", self._worker_id, exc_info=True)
            self._shutdown_event.set()
            if self._host_shutdown_event is not None:
                self._host_shutdown_event.set()
            return

        backoff = ExponentialBackoff(initial=0.1, maximum=5.0)
        in_flight_threads: list[threading.Thread] = []
        future: ActorFuture | None = None
        future_start = 0.0
        warned = False

        while not self._shutdown_event.is_set():
            # Prune finished threads to avoid unbounded growth.
            in_flight_threads = [t for t in in_flight_threads if t.is_alive()]

            with self._resources_lock:
                avail = self._available

            # If no resources are available, skip the coordinator round-trip and
            # wait for a task to finish freeing capacity.
            if avail.cpu == 0 and avail.memory == 0:
                self._task_completed_event.wait(timeout=backoff.next_interval())
                self._task_completed_event.clear()
                continue

            # Short timeout keeps the thread responsive to shutdown without
            # killing it on slow coordinator deserialization.
            try:
                if future is None:
                    future = self._coordinator.pull_task.remote(self._worker_id, avail)
                    future_start = time.monotonic()
                    warned = False
                response = future.result(timeout=0.5)
            except TimeoutError:
                elapsed = time.monotonic() - future_start
                if elapsed > 30 and not warned:
                    logger.warning("[%s] Waiting for pull_task response (%.0fs)", self._worker_id, elapsed)
                    warned = True
                continue
            except Exception as e:
                logger.info("[%s] pull_task failed (coordinator may be dead): %s", self._worker_id, e)
                break

            future = None
            status, work = response

            if status == PullStatus.SHUTDOWN:
                logger.info("[%s] Received SHUTDOWN from coordinator", self._worker_id)
                break

            if status != PullStatus.RUN_TASK:
                # Sleep until resources become free or new tasks arrive.
                wait = backoff.next_interval()
                self._task_completed_event.wait(timeout=wait)
                self._task_completed_event.clear()
                continue

            backoff.reset()
            assert work is not None

            with self._resources_lock:
                self._available = self._available - work.task.cost

            runner = self._stage_runner_factory()
            with self._resources_lock:
                self._active_runners.append((work.execution_id, runner))
            self._active_task_count += 1
            self._current_stage_name = work.task.stage_name

            t = threading.Thread(
                target=self._task_thread,
                args=(work, runner),
                daemon=True,
                name=f"zephyr-task-{self._worker_id}-s{work.task.shard_idx}",
            )
            in_flight_threads.append(t)
            t.start()

        # Drain in-flight tasks before deregistering.
        for t in in_flight_threads:
            t.join()

        logger.info("[%s] Poll loop exiting", self._worker_id)
        with suppress(Exception):
            self._coordinator.deregister_worker.remote(self._worker_id).result(timeout=10.0)

        self._shutdown_event.set()
        if self._host_shutdown_event is not None:
            self._host_shutdown_event.set()

    def _task_thread(
        self,
        work: PullTask,
        runner: StageRunner,
    ) -> None:
        """Execute one shard task, report the result, and restore task.cost to the pool."""
        task_start = time.monotonic()
        task = work.task
        try:
            result, task_counters = self._execute_shard(task, work.chunk_prefix, work.execution_id, runner)
            logger.info("[%s] Shard %d done in %.2fs", self._worker_id, task.shard_idx, time.monotonic() - task_start)
            # Block until coordinator records result — prevents _in_flight races.
            counter_generation = self._next_counter_generation(work.execution_id)
            self._coordinator.report_result.remote(
                self._worker_id,
                work.execution_id,
                task.shard_idx,
                work.attempt,
                result,
                CounterSnapshot(counters=dict(task_counters), generation=counter_generation),
                work.stage_generation,
            ).result()
        except Exception:
            logger.error("Worker %s error on shard %d", self._worker_id, task.shard_idx, exc_info=True)
            self._coordinator.report_error.remote(
                self._worker_id,
                work.execution_id,
                task.shard_idx,
                work.attempt,
                "".join(traceback.format_exc()),
                work.stage_generation,
            ).result()
        finally:
            with self._resources_lock:
                self._available = self._available + task.cost
                active_runner = (work.execution_id, runner)
                if active_runner in self._active_runners:
                    self._active_runners.remove(active_runner)
            self._active_task_count = max(0, self._active_task_count - 1)
            self._task_completed_event.set()

    def _report_worker_iris_status(self) -> None:
        """Push worker status text to Iris for UI display. Called on each heartbeat."""

        def build_md() -> tuple[str, str]:
            stage = self._current_stage_name
            stage_values = {
                name: entry.value
                for counters in self._last_reported_counters.values()
                for name, entry in counters.items()
                if entry.stage == stage
            }
            throughput = _stage_throughput(stage_values, 1.0) if stage else None
            if throughput is not None:
                logger.info("[%s] [%s] throughput: %s", self._worker_id, stage, throughput)
            return _format_worker_status_md(self._active_task_count, stage)

        _push_iris_task_status(self._iris_status_limiter, build_md)

    def _next_counter_generation(self, execution_id: str) -> int:
        with self._resources_lock:
            self._counter_generations[execution_id] += 1
            return self._counter_generations[execution_id]

    def _heartbeat_counter_snapshots(self) -> dict[str, CounterSnapshot] | None:
        """Return changed live counters, grouped by pipeline execution."""
        with self._resources_lock:
            active_runners = list(self._active_runners)

        runners_by_execution: dict[str, list[StageRunner]] = defaultdict(list)
        for execution_id, runner in active_runners:
            runners_by_execution[execution_id].append(runner)

        snapshots: dict[str, CounterSnapshot] = {}
        execution_ids = runners_by_execution.keys() | self._last_reported_counters.keys()
        for execution_id in execution_ids:
            current, _ = merge_counter_entries(
                (name, entry)
                for runner in runners_by_execution.get(execution_id, [])
                for name, entry in runner.live_counters().items()
            )
            if current == self._last_reported_counters.get(execution_id, {}):
                continue
            self._last_reported_counters[execution_id] = current
            snapshots[execution_id] = CounterSnapshot(
                counters=current,
                generation=self._next_counter_generation(execution_id),
            )
        return snapshots or None

    def _heartbeat_loop(
        self, coordinator: ActorHandle, interval: float = 5.0, max_consecutive_failures: int = 5
    ) -> None:
        logger.debug("[%s] Heartbeat loop starting", self._worker_id)
        heartbeat_count = 0
        consecutive_failures = 0
        while not self._shutdown_event.is_set():
            try:
                snapshots = self._heartbeat_counter_snapshots()
                coordinator.heartbeat.remote(self._worker_id, snapshots).result()
                heartbeat_count += 1
                consecutive_failures = 0
                if heartbeat_count % 10 == 1:
                    logger.debug("[%s] Sent heartbeat #%d", self._worker_id, heartbeat_count)
                self._report_worker_iris_status()
            except Exception as e:
                consecutive_failures += 1
                logger.warning(
                    "[%s] Heartbeat failed (%d/%d): %s",
                    self._worker_id,
                    consecutive_failures,
                    max_consecutive_failures,
                    e,
                )
                if consecutive_failures >= max_consecutive_failures:
                    logger.error(
                        "[%s] %d consecutive heartbeat failures — coordinator unreachable, shutting down",
                        self._worker_id,
                        consecutive_failures,
                    )
                    self._actor_ctx.fail(
                        CoordinatorUnreachable(f"{consecutive_failures} consecutive heartbeat failures")
                    )
                    self._shutdown_event.set()
                    break
            self._shutdown_event.wait(timeout=interval)
        logger.debug("[%s] Heartbeat loop exiting after %d beats", self._worker_id, heartbeat_count)

    def _execute_shard(
        self,
        task: ShardTask,
        chunk_prefix: str,
        execution_id: str,
        stage_runner: StageRunner,
    ) -> tuple[TaskResult, dict[str, CounterEntry]]:
        logger.info(
            "[%s] [shard %d/%d] stage=%s, %d ops",
            execution_id,
            task.shard_idx,
            task.total_shards,
            task.stage_name,
            len(task.operations),
        )
        result, counters = stage_runner.execute(task, chunk_prefix, execution_id)
        logger.info("[shard %d] Complete: %d refs produced", task.shard_idx, len(result.shard.refs))
        return result, counters

    def __repr__(self) -> str:
        return f"ZephyrWorker(id={self._worker_id})"

    def shutdown(self) -> None:
        """Signal the worker to stop accepting new tasks."""
        self._shutdown_event.set()
        self._task_completed_event.set()
        for thread in (self._poll_thread, self._heartbeat_thread):
            if thread is not threading.current_thread():
                thread.join(timeout=10.0)
        if self._host_shutdown_event is not None:
            self._host_shutdown_event.set()
