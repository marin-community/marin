# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""A deterministic external-boundary backend for Iris journeys."""

import threading
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import ClassVar

from finelog.rpc import logging_pb2
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendCapability,
    BackendRuntime,
    DeviceCapacity,
    ProviderUnsupportedError,
    ReconcileRequest,
    ReconcileResult,
    ScheduleRequest,
    ScheduleResult,
    TaskTarget,
)
from iris.cluster.controller.ops.task import apply_dispatch_updates
from iris.cluster.controller.reconcile.loader import TransitionReader
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.task_state import job_scheduling_deadline
from iris.cluster.controller.worker_health import WorkerHealthTracker
from iris.cluster.types import DEFAULT_BACKEND_ID, JobName, WorkerId
from iris.rpc import controller_pb2, job_pb2, vm_pb2, worker_pb2
from rigging.timing import Timestamp


@dataclass(frozen=True, slots=True)
class ScriptedObservation:
    state: int
    error: str = ""
    exit_code: int | None = None
    attempt_id: int | None = None


@dataclass(frozen=True, slots=True)
class BackendEvent:
    kind: str
    task_id: str
    attempt_id: int
    state: int | None = None
    backend_id: str = DEFAULT_BACKEND_ID


class ScriptedTaskBackend:
    """Placement-owning backend controlled through addressed observations.

    The controller supplies its desired Tasks through ``ReconcileRequest``. New
    Attempts become RUNNING by default; a journey can queue an exact terminal
    observation for the next reconcile. The fake records only backend-visible
    effects and never reads or writes controller tables.
    """

    name = "journey"
    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset({BackendCapability.CLUSTER_VIEW})
    autoscaler = None
    health: WorkerHealthTracker | None = None

    def __init__(self, transition_reader: TransitionReader, *, backend_id: str = DEFAULT_BACKEND_ID) -> None:
        self._transition_reader = transition_reader
        self.backend_id = backend_id
        self._queued: dict[str, deque[ScriptedObservation]] = defaultdict(deque)
        self._desired: set[tuple[str, int]] = set()
        self.events: list[BackendEvent] = []
        self.calls: list[str] = []
        self._reconcile_failures = 0
        self.closed = False
        self.advertised: dict[str, set[str]] = {"region": {"us-central1"}}

    @property
    def has_pending_observations(self) -> bool:
        return any(self._queued.values())

    @property
    def pending_task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(task_id for task_id, queue in self._queued.items() if queue))

    def observe(self, task_id: str, observation: ScriptedObservation) -> None:
        """Queue one observation for the Task's current desired Attempt."""
        self._queued[task_id].append(observation)

    def owns_task(self, task_id: str) -> bool:
        """Whether the controller currently asks this backend to run ``task_id``."""
        return any(desired_task_id == task_id for desired_task_id, _ in self._desired)

    def fail_reconcile(self, *, times: int) -> None:
        self._reconcile_failures += times

    def advertised_attributes(self) -> dict[str, set[str]]:
        return self.advertised

    def configure_routing(self, advertised: dict[str, set[str]]) -> None:
        self.advertised = advertised

    def resource_capacity(self) -> dict[str, DeviceCapacity] | None:
        return None

    def status(self) -> controller_pb2.Controller.BackendStatus:
        return controller_pb2.Controller.BackendStatus()

    def autoscaler_status(self) -> vm_pb2.AutoscalerStatus:
        return vm_pb2.AutoscalerStatus()

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        self.calls.append("schedule")
        return ScheduleResult()

    def reconcile(self, request: ReconcileRequest) -> ReconcileResult:
        self.calls.append("reconcile")
        if self._reconcile_failures:
            self._reconcile_failures -= 1
            raise ConnectionError("scripted backend is unavailable")
        desired = {(run.task_id, run.attempt_id) for run in request.tasks_to_run} | {
            (entry.task_id.to_wire(), entry.attempt_id) for entry in request.running_tasks
        }
        for task_id, attempt_id in sorted(self._desired - desired):
            self.events.append(BackendEvent("stopped", task_id, attempt_id, backend_id=self.backend_id))

        updates: list[TaskUpdate] = []
        newly_launched = {(run.task_id, run.attempt_id) for run in request.tasks_to_run}
        for task_id, attempt_id in sorted(newly_launched - self._desired):
            self.events.append(BackendEvent("launched", task_id, attempt_id, backend_id=self.backend_id))
            queued = self._pop_observation(task_id)
            if queued is None:
                queued = ScriptedObservation(job_pb2.TASK_STATE_RUNNING)
            observed_attempt_id = attempt_id if queued.attempt_id is None else queued.attempt_id
            updates.append(self._task_update(task_id, observed_attempt_id, queued))

        for task_id, attempt_id in sorted(desired - newly_launched):
            queued = self._pop_observation(task_id)
            if queued is not None:
                observed_attempt_id = attempt_id if queued.attempt_id is None else queued.attempt_id
                updates.append(self._task_update(task_id, observed_attempt_id, queued))

        self._desired = desired
        if not updates:
            return ReconcileResult()
        effects = apply_dispatch_updates(self._transition_reader, updates, now=Timestamp.now())
        return ReconcileResult(effects=effects)

    def _pop_observation(self, task_id: str) -> ScriptedObservation | None:
        queue = self._queued.get(task_id)
        if not queue:
            return None
        observation = queue.popleft()
        if not queue:
            self._queued.pop(task_id, None)
        return observation

    def _task_update(self, task_id: str, attempt_id: int, observation: ScriptedObservation) -> TaskUpdate:
        self.events.append(BackendEvent("observed", task_id, attempt_id, observation.state, self.backend_id))
        return TaskUpdate(
            task_id=JobName.from_wire(task_id),
            attempt_id=attempt_id,
            new_state=observation.state,
            error=observation.error or None,
            exit_code=observation.exit_code,
        )

    def run_teardown(self) -> None:
        return None

    def teardown(self, dead_workers: list[WorkerId], *, reason: str) -> None:
        return None

    def prune_dead_workers(self, *, cutoff_ms: int, stop_event: threading.Event | None, pause: float) -> int:
        return 0

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        self.calls.append("autoscale")
        return AutoscaleResult()

    def bind_runtime(self, runtime: BackendRuntime) -> None:
        return None

    def seed_liveness(self) -> None:
        return None

    def get_process_status(
        self,
        target: TaskTarget,
        request: job_pb2.GetProcessStatusRequest,
    ) -> job_pb2.GetProcessStatusResponse:
        raise ProviderUnsupportedError("journey backend has no process runtime")

    def profile_task(
        self,
        target: TaskTarget,
        request: job_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> job_pb2.ProfileTaskResponse:
        raise ProviderUnsupportedError("journey backend has no profiler")

    def exec_in_container(
        self,
        target: TaskTarget,
        request: worker_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int = 60,
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        raise ProviderUnsupportedError("journey backend has no container runtime")

    def fetch_live_logs(
        self,
        task_id: str,
        attempt_id: int,
        cursor: int,
        max_lines: int,
    ) -> tuple[list[logging_pb2.LogEntry], int]:
        return [], cursor

    def close(self) -> None:
        self.closed = True


class UnavailableTaskBackend(ScriptedTaskBackend):
    """Worker-style backend that advertises a route but has no capacity."""

    capabilities: ClassVar[frozenset[BackendCapability]] = frozenset({BackendCapability.WORKER_DAEMON})

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        self.calls.append("schedule")
        expired = []
        for task in request.pending_task_rows:
            deadline = job_scheduling_deadline(task.scheduling_deadline_epoch_ms)
            if deadline is not None and deadline.expired():
                expired.append(task)
        return ScheduleResult(unschedulable=expired)
