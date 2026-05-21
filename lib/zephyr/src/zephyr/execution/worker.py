# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Long-lived worker actor that drives per-stage subprocess thread pools."""

from __future__ import annotations

import logging
import threading
import time
import traceback
from collections import Counter
from collections.abc import Callable
from contextlib import suppress

from fray import ActorFuture, ActorHandle, current_actor
from rigging.timing import ExponentialBackoff, RateLimiter

from zephyr.execution.internals import (
    CoordinatorUnreachable,
    CounterSnapshot,
    PullStatus,
    ShardTask,
    StageRunner,
    TaskResult,
    _format_bytes,
    _format_count,
    _push_iris_task_status,
    _stage_throughput,
)
from zephyr.plan import StageType
from zephyr.runners import InlineRunner

logger = logging.getLogger(__name__)


class ZephyrWorker:
    """Long-lived worker actor that drives per-stage subprocess thread pools.

    For each stage the coordinator makes available, the worker polls
    ``get_current_stage_type()``, creates ``map_workers_per_actor`` or
    ``reduce_workers_per_actor`` subprocess slots, runs them until they drain
    the stage and receive SHUTDOWN, then deregisters and loops.

    Each slot registers with the coordinator under a unique sub-ID
    ``"{base_id}:{i}"`` so the coordinator's task-dispatch and in-flight
    tracking remain clean and predictable.
    """

    def __init__(
        self,
        coordinator_handle: ActorHandle,
        stage_runner_factory: Callable[[int], StageRunner] | None = None,
        map_workers_per_actor: int = 1,
        reduce_workers_per_actor: int = 1,
    ):
        # ZephyrContext normally pre-resolves this via _CoordinatorJobConfig;
        # the fallback covers callers that construct a worker directly (tests).
        if stage_runner_factory is None:
            stage_runner_factory = lambda n: InlineRunner(num_workers=n)  # noqa: E731

        self._coordinator = coordinator_handle
        self._stage_runner_factory = stage_runner_factory
        self._num_map_workers = map_workers_per_actor
        self._num_reduce_workers = reduce_workers_per_actor
        self._shutdown_event = threading.Event()
        self._counter_generation: int = 0
        self._last_reported_counters: dict[str, int] = {}
        # Runners and sub-IDs for currently active slots — written by _stage_manager,
        # read (snapshotted) by heartbeat thread.
        self._active_runners: list[StageRunner] = []
        self._active_sub_ids: list[str] = []
        self._active_task_count: int = 0
        self._current_stage_name: str = ""

        # Throttle Iris status pushes; the heartbeat loop ticks faster than
        # the UI needs to refresh.
        self._iris_status_limiter = RateLimiter(interval_seconds=10.0)

        # Capture actor context while ContextVar is still set (child threads
        # in Python <3.12 don't inherit it).
        self._actor_ctx = current_actor()
        self._host_shutdown_event = self._actor_ctx.shutdown_event
        self._worker_id = f"{self._actor_ctx.group_name}-{self._actor_ctx.index}"
        self._actor_handle = self._actor_ctx.handle

        threading.Thread(
            target=self._heartbeat_loop,
            args=(coordinator_handle,),
            daemon=True,
            name=f"zephyr-hb-{self._worker_id}",
        ).start()

        threading.Thread(
            target=self._stage_manager,
            daemon=True,
            name=f"zephyr-stage-{self._worker_id}",
        ).start()

    def _stage_manager(self) -> None:
        """Drive per-stage thread-pool lifecycle.

        Polls the coordinator for the current stage type, spins up N sub-worker
        threads (N = map or reduce worker count), waits for them to drain and
        exit on SHUTDOWN, then deregisters and loops for the next stage.
        """
        logger.info("[%s] Stage manager starting", self._worker_id)
        backoff = ExponentialBackoff(initial=0.1, maximum=2.0)

        while not self._shutdown_event.is_set():
            try:
                stage_type = self._coordinator.get_current_stage_type.remote().result(timeout=5.0)
            except Exception as e:
                logger.debug("[%s] get_current_stage_type failed: %s", self._worker_id, e)
                self._shutdown_event.wait(timeout=1.0)
                continue

            if stage_type is None:
                self._shutdown_event.wait(timeout=backoff.next_interval())
                continue

            backoff.reset()
            num_workers = self._num_map_workers if stage_type == StageType.MAP_WORKER else self._num_reduce_workers
            logger.info("[%s] Stage '%s' starting, %d slot(s)", self._worker_id, stage_type, num_workers)

            sub_ids = set(f"{self._worker_id}:{i}" for i in range(num_workers))

            stage_epoch: int | None = None
            failed_workers: set[str] = set()
            for sub_id in sub_ids:
                try:
                    epoch = self._coordinator.register_worker.remote(sub_id, self._actor_handle).result(timeout=30.0)
                except Exception:
                    failed_workers.add(sub_id)
                    logger.warning("[%s] Failed to register sub-worker %s", self._worker_id, sub_id, exc_info=True)
                    continue

                if stage_epoch is not None and stage_epoch != epoch:
                    raise RuntimeError(
                        f"Received different stage epochs when registering workers within a stage. "
                        f"Got {epoch} (expected {stage_epoch})"
                    )
                stage_epoch = epoch

            self._active_sub_ids = list(sub_ids - failed_workers)
            num_active_workers = len(self._active_sub_ids)
            self._active_runners = [self._stage_runner_factory(num_active_workers) for _ in range(num_active_workers)]

            # Slots write their final PullStatus here before returning so we can
            # tell a stage-boundary exit from a pipeline-shutdown exit.
            slot_statuses: list[PullStatus | None] = [None] * num_active_workers
            threads = [
                threading.Thread(
                    target=self._poll_loop,
                    args=(self._coordinator, sub_id, runner, stage_epoch, slot_statuses, i),
                    daemon=True,
                    name=f"zephyr-poll-{sub_id}",
                )
                for i, (sub_id, runner) in enumerate(zip(self._active_sub_ids, self._active_runners, strict=True))
            ]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            logger.info("[%s] Stage '%s' slots exited", self._worker_id, stage_type)
            self._active_runners = []
            self._active_sub_ids = []

            for sub_id in sub_ids:
                try:
                    self._coordinator.deregister_worker.remote(sub_id).result(timeout=10.0)
                except Exception:
                    logger.warning("[%s] deregister_worker failed for %s (ignoring)", self._worker_id, sub_id)

            # Any slot seeing SHUTDOWN means the pipeline is done for this worker
            # — no next stage will come, so break instead of polling forever.
            if any(s == PullStatus.SHUTDOWN for s in slot_statuses):
                logger.info("[%s] Pipeline shutdown signaled by slot — exiting stage manager", self._worker_id)
                break

        logger.info("[%s] Stage manager exiting", self._worker_id)
        if self._host_shutdown_event is not None:
            self._host_shutdown_event.set()

    def _report_worker_iris_status(self) -> None:
        """Push worker status text to Iris for UI display. Called on each heartbeat."""

        def build_md() -> tuple[str, str]:
            active = self._active_task_count
            stage = self._current_stage_name
            if active == 0 or not stage:
                summary_md = "idle"
                detail_lines = "idle"
            else:
                summary_md = f"**{stage}** — {active} task(s)"
                detail_lines = [f"**Stage**: {stage}", f"**Active tasks**: {active}"]
                throughput = _stage_throughput(self._last_reported_counters, stage, 1.0)
                if throughput is not None:
                    items, bytes_processed, item_rate, byte_rate = throughput
                    detail_lines += [
                        f"**Items**: {_format_count(items)} ({_format_count(item_rate)}/s)",
                        f"**Throughput**: {_format_bytes(bytes_processed)} ({_format_bytes(byte_rate)}/s)",
                    ]
            return "  \n".join(detail_lines), summary_md

        _push_iris_task_status(self._iris_status_limiter, build_md)

    def _heartbeat_counter_snapshot(self) -> CounterSnapshot | None:
        """Aggregate live counters from all active runners; return None if unchanged."""
        runners = list(self._active_runners)  # GIL-safe snapshot
        current: Counter[str] = Counter()
        for r in runners:
            current.update(r.live_counters())
        if current == self._last_reported_counters:
            return None
        self._last_reported_counters = dict(current)
        self._counter_generation += 1
        return CounterSnapshot(counters=dict(current), generation=self._counter_generation)

    def _heartbeat_loop(
        self, coordinator: ActorHandle, interval: float = 5.0, max_consecutive_failures: int = 5
    ) -> None:
        logger.debug("[%s] Heartbeat loop starting", self._worker_id)
        heartbeat_count = 0
        consecutive_failures = 0
        while not self._shutdown_event.is_set():
            try:
                sub_ids = list(self._active_sub_ids)
                if sub_ids:
                    # Primary slot carries the aggregate counter snapshot; the rest
                    # just refresh last_seen to prevent heartbeat timeout mid-task.
                    snapshot = self._heartbeat_counter_snapshot()
                    coordinator.heartbeat.remote(sub_ids[0], snapshot).result()
                    for sub_id in sub_ids[1:]:
                        with suppress(Exception):
                            coordinator.heartbeat.remote(sub_id).result()
                else:
                    # Between stages: probe for coordinator liveness only.
                    coordinator.get_current_stage_type.remote().result(timeout=10.0)
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

    def _poll_loop(
        self,
        coordinator: ActorHandle,
        worker_id: str,
        stage_runner: StageRunner,
        epoch: int,
        status_holder: list[PullStatus | None],
        slot_idx: int,
    ) -> None:
        """Single-slot polling loop. Exits on STAGE_COMPLETED / SHUTDOWN or coordinator death.

        ``epoch`` is the stage epoch returned by ``register_worker``; it is passed
        on every ``pull_task`` call so the coordinator can detect slots that survived
        past stage completion and send STAGE_COMPLETED before they pick up next-stage tasks.

        On exit, the slot writes its final ``PullStatus`` to
        ``status_holder[slot_idx]`` so the stage manager can distinguish a
        stage-boundary exit (re-pool for next stage) from a pipeline-shutdown
        exit (break the outer loop).
        """
        task_count = 0
        backoff = ExponentialBackoff(initial=0.1, maximum=5.0)
        future: ActorFuture | None = None
        future_start = 0.0
        warned = False

        while not self._shutdown_event.is_set():
            if future is None:
                future = coordinator.pull_task.remote(worker_id, epoch)
                future_start = time.monotonic()
                warned = False

            # Short timeout keeps the thread responsive to shutdown without
            # killing the slot on slow coordinator deserialization.
            try:
                response = future.result(timeout=0.5)
            except TimeoutError:
                elapsed = time.monotonic() - future_start
                if elapsed > 30 and not warned:
                    logger.warning("[%s] Waiting for coordinator pull_task response (%.0fs)", worker_id, elapsed)
                    warned = True
                continue
            except Exception as e:
                logger.info("[%s] pull_task failed (coordinator may be dead): %s", worker_id, e)
                break

            future = None

            if response == PullStatus.NO_WORK_BACKOFF:
                time.sleep(backoff.next_interval())
                continue

            if isinstance(response, PullStatus):
                logger.debug("[%s] Received %s", worker_id, response.name)
                status_holder[slot_idx] = response
                return

            backoff.reset()
            task, attempt, config = response

            logger.info(
                "[%s] Executing stage %s shard %d (attempt %d)",
                worker_id,
                task.stage_name,
                task.shard_idx,
                attempt,
            )
            self._active_task_count += 1
            self._current_stage_name = task.stage_name
            task_start = time.monotonic()
            try:
                result, task_counters = self._execute_shard(task, config, stage_runner)
                logger.info(
                    "[%s] Shard %d done in %.2fs",
                    worker_id,
                    task.shard_idx,
                    time.monotonic() - task_start,
                )
                # Block until coordinator records result — prevents _in_flight races.
                self._counter_generation += 1
                coordinator.report_result.remote(
                    worker_id,
                    task.shard_idx,
                    attempt,
                    result,
                    CounterSnapshot(counters=dict(task_counters), generation=self._counter_generation),
                ).result()
                task_count += 1
            except Exception:
                logger.error("Worker %s error on shard %d", worker_id, task.shard_idx, exc_info=True)
                coordinator.report_error.remote(
                    worker_id,
                    task.shard_idx,
                    "".join(traceback.format_exc()),
                ).result()
            finally:
                self._active_task_count = max(0, self._active_task_count - 1)

    def _execute_shard(
        self, task: ShardTask, config: dict, stage_runner: StageRunner
    ) -> tuple[TaskResult, dict[str, int]]:
        chunk_prefix = config["chunk_prefix"]
        execution_id = config["execution_id"]
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
        """Signal the worker to stop accepting new stages."""
        self._shutdown_event.set()
        if self._host_shutdown_event is not None:
            self._host_shutdown_event.set()
