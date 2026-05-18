# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker-side Reconcile RPC handler and SpecCache.

Phase B implementation: handles `ReconcileRequest` / `ReconcileResponse`
using composite `(task_id, attempt_id)` keys. The `attempt_uid` field is
empty in Phase B; UID-based routing arrives in Phase C.

Design notes:
- `SpecCache` is in-memory only. A cold worker restart loses the cache;
  the controller will receive `TASK_STATE_MISSING` for any attempt whose
  spec was never delivered post-restart, and will fail those attempts
  forward via `worker_lost_spec`.
- `handle_reconcile` is a free function that takes a `ReconcileContext`
  Protocol, keeping the handler logic testable without instantiating a
  full Worker. The Worker satisfies the Protocol and delegates to this
  function from its `handle_reconcile` method.
"""

import logging
import threading
from typing import Protocol

from iris.cluster.worker.env_probe import check_worker_health
from iris.cluster.worker.task_attempt import TaskAttempt
from iris.rpc import job_pb2, worker_pb2
from iris.time_proto import timestamp_to_proto

logger = logging.getLogger(__name__)

# The composite key used in Phase B before attempt_uid routing lands.
_AttemptKey = tuple[str, int]  # (task_id_wire, attempt_id)


class SpecCache:
    """In-memory cache of RunTaskRequest specs for active attempts.

    Keyed by `(task_id_wire, attempt_id)` in Phase B. Phase C migrates to
    `attempt_uid` as the primary key.

    Thread-safe: all mutations are expected to occur under the caller's lock.
    """

    def __init__(self) -> None:
        self._cache: dict[_AttemptKey, job_pb2.RunTaskRequest] = {}

    def add(self, task_id: str, attempt_id: int, request: job_pb2.RunTaskRequest) -> None:
        """Store a spec for the given attempt."""
        self._cache[(task_id, attempt_id)] = request

    def lookup(self, task_id: str, attempt_id: int) -> job_pb2.RunTaskRequest | None:
        """Return the cached spec, or None if not present."""
        return self._cache.get((task_id, attempt_id))

    def evict(self, task_id: str, attempt_id: int) -> None:
        """Remove the spec for a terminal attempt.

        Idempotent: silently ignores missing keys.
        """
        self._cache.pop((task_id, attempt_id), None)

    def __len__(self) -> int:
        return len(self._cache)


class ReconcileContext(Protocol):
    """Subset of Worker that `handle_reconcile` requires.

    Defines only what the handler needs; the full Worker satisfies this
    Protocol without any modifications.
    """

    # Task map and lock — caller must NOT hold the lock before calling
    # handle_reconcile (the function acquires/releases as needed).
    @property
    def _tasks(self) -> dict[_AttemptKey, object]: ...

    @property
    def _lock(self) -> threading.Lock: ...

    @property
    def _TERMINAL_STATES(self) -> frozenset: ...

    def _kill_task_attempt(
        self,
        task_id: str,
        attempt_id: int,
        term_timeout_ms: int = 5000,
        async_kill: bool = False,
    ) -> bool: ...

    def submit_task(self, request: job_pb2.RunTaskRequest) -> str: ...

    def _collect_resource_metrics(self) -> job_pb2.WorkerResourceSnapshot: ...

    @property
    def _cache_dir(self) -> object: ...

    @property
    def _worker_id(self) -> str | None: ...


def _build_observation(
    task_id: str,
    attempt_id: int,
    task: object,
    terminal_states: frozenset,
) -> worker_pb2.Worker.AttemptObservation:
    """Build an AttemptObservation from a local TaskAttempt."""
    assert isinstance(task, TaskAttempt)

    state = task.status
    # Workers never report PENDING to the controller; map it to BUILDING.
    if state == job_pb2.TASK_STATE_PENDING:
        state = job_pb2.TASK_STATE_BUILDING

    obs = worker_pb2.Worker.AttemptObservation(
        task_id=task_id,
        attempt_id=attempt_id,
        state=state,
        exit_code=task.exit_code or 0,
        error=task.error or "",
        container_id=task.platform_container_id or "",
    )
    if task.status in terminal_states and task.finished_at is not None:
        obs.finished_at.CopyFrom(timestamp_to_proto(task.finished_at))
    return obs


def handle_reconcile(
    ctx: ReconcileContext,
    spec_cache: SpecCache,
    request: worker_pb2.Worker.ReconcileRequest,
) -> worker_pb2.Worker.ReconcileResponse:
    """Worker-side Reconcile handler.

    Args:
        ctx: Worker context satisfying ReconcileContext (the Worker instance).
        spec_cache: In-memory spec cache shared across calls. Caller owns it.
        request: The incoming ReconcileRequest from the controller.

    Returns:
        ReconcileResponse with the complete current worker observation and health.

    Raises:
        Exceptions propagate; Connect's error layer converts them to wire errors.
        No defensive try/except here — let the caller's error handler fire.
    """
    # ── Step 1: Process each DesiredAttempt ───────────────────────────────────
    # Desired keys with intent=run. Used to detect MISSING attempts (those
    # the controller wants running but the worker has no record of and no spec).
    desired_run_keys: set[_AttemptKey] = set()
    # All desired keys (run + stop). Used for zombie detection.
    desired_keys: set[_AttemptKey] = set()

    for desired in request.desired:
        task_id = desired.task_id
        attempt_id = desired.attempt_id
        key: _AttemptKey = (task_id, attempt_id)

        # Determine intent. The oneof is either `run` or `stop`.
        # In protobuf Python, HasField works for oneof fields.
        is_run = desired.HasField("run")

        if is_run:
            desired_run_keys.add(key)
            desired_keys.add(key)
            _process_run_intent(ctx, spec_cache, task_id, attempt_id, desired.run)
        else:
            # intent=stop
            desired_keys.add(key)
            _process_stop_intent(ctx, task_id, attempt_id)

    # ── Step 2: Kill any local non-terminal attempts not in desired ────────────
    # (zombies: worker has them, controller didn't include them as desired)
    with ctx._lock:
        local_keys = list(ctx._tasks.keys())

    for key in local_keys:
        task_id, attempt_id = key
        with ctx._lock:
            task = ctx._tasks.get(key)
            if task is None:
                continue
            is_terminal = getattr(task, "status", None) in ctx._TERMINAL_STATES
        if key not in desired_keys and not is_terminal:
            logger.info(
                "Reconcile: killing zombie attempt %s/%d (not in desired set)",
                task_id,
                attempt_id,
            )
            ctx._kill_task_attempt(task_id, attempt_id, async_kill=True)

    # ── Step 3: Build observed set (complete view of all known attempts) ───────
    # Includes: all local attempts + MISSING for desired-run keys not locally known.
    observations: list[worker_pb2.Worker.AttemptObservation] = []
    with ctx._lock:
        snapshot = list(ctx._tasks.items())

    known_keys: set[_AttemptKey] = set()
    for key, task in snapshot:
        task_id, attempt_id = key
        known_keys.add(key)
        obs = _build_observation(task_id, attempt_id, task, ctx._TERMINAL_STATES)
        observations.append(obs)

        # Evict terminal entries from spec_cache so it stays bounded.
        is_terminal = getattr(task, "status", None) in ctx._TERMINAL_STATES
        if is_terminal:
            spec_cache.evict(task_id, attempt_id)

    # Emit MISSING for desired-run keys the worker has no record of.
    for task_id, attempt_id in desired_run_keys:
        if (task_id, attempt_id) not in known_keys:
            observations.append(
                worker_pb2.Worker.AttemptObservation(
                    task_id=task_id,
                    attempt_id=attempt_id,
                    state=job_pb2.TASK_STATE_MISSING,
                )
            )

    # ── Step 4: Build WorkerHealth (mirrors handle_ping health body) ───────────
    resource_snapshot = ctx._collect_resource_metrics()
    health = check_worker_health(disk_path=str(ctx._cache_dir))
    if not health.healthy:
        logger.warning("Reconcile: worker health check failed: %s", health.error)

    worker_health = worker_pb2.Worker.WorkerHealth(
        healthy=health.healthy,
        health_error=health.error,
        resources=resource_snapshot,
    )

    return worker_pb2.Worker.ReconcileResponse(
        worker_id=ctx._worker_id or "",
        observed=observations,
        health=worker_health,
    )


def _process_run_intent(
    ctx: ReconcileContext,
    spec_cache: SpecCache,
    task_id: str,
    attempt_id: int,
    attempt_spec: worker_pb2.Worker.AttemptSpec,
) -> None:
    """Handle a single DesiredAttempt with intent=run.

    Cases:
    - Worker already has the attempt (running or terminal): no-op.
    - Worker doesn't have it, spec is inline: enqueue via submit_task; cache spec.
    - Worker doesn't have it, no spec: do not enqueue; the observation loop
      will report TASK_STATE_MISSING for this key.
    """
    key: _AttemptKey = (task_id, attempt_id)

    with ctx._lock:
        task = ctx._tasks.get(key)

    if task is not None:
        # Already known locally — no-op. Status will be reported in observed.
        return

    # Worker doesn't have this attempt locally.
    spec_has_request = attempt_spec.HasField("request")

    if spec_has_request:
        # Spec inline: enqueue the attempt via submit_task.
        run_request = attempt_spec.request
        spec_cache.add(task_id, attempt_id, run_request)
        logger.info("Reconcile: enqueuing attempt %s/%d (spec inline)", task_id, attempt_id)
        ctx.submit_task(run_request)
    elif spec_cache.lookup(task_id, attempt_id) is not None:
        # No inline spec but we have a cached copy — this shouldn't happen in
        # normal operation (attempt would already be in _tasks), but handle it
        # defensively: enqueue using the cached spec.
        cached_request = spec_cache.lookup(task_id, attempt_id)
        assert cached_request is not None
        logger.warning(
            "Reconcile: attempt %s/%d not in tasks but found in spec_cache; re-enqueuing",
            task_id,
            attempt_id,
        )
        ctx.submit_task(cached_request)
    else:
        # No spec anywhere: will surface as TASK_STATE_MISSING in observed.
        logger.info(
            "Reconcile: attempt %s/%d unknown and no spec; will report MISSING",
            task_id,
            attempt_id,
        )


def _process_stop_intent(
    ctx: ReconcileContext,
    task_id: str,
    attempt_id: int,
) -> None:
    """Handle a single DesiredAttempt with intent=stop.

    Idempotent: silently does nothing if the attempt is already terminal or
    not present locally.
    """
    with ctx._lock:
        task = ctx._tasks.get((task_id, attempt_id))
        if task is None:
            return
        is_terminal = getattr(task, "status", None) in ctx._TERMINAL_STATES

    if is_terminal:
        return

    logger.info("Reconcile: stopping attempt %s/%d (stop intent)", task_id, attempt_id)
    ctx._kill_task_attempt(task_id, attempt_id, async_kill=True)
