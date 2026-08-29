# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Deterministic external-boundary backend for Iris journeys."""

from collections import defaultdict, deque
from dataclasses import dataclass

from finelog.rpc import logging_pb2

from iris.cluster.constraints import Constraint
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendDescriptor,
    BackendKind,
    BackendObservation,
    BackendObservationRequest,
    BackendRecoveryRequest,
    BackendRecoveryResult,
    DirectReconcileRequest,
    ProviderUnsupportedError,
    ReconcileObservation,
    ReconcileRequest,
    RemoveCapacityRequest,
    RemoveCapacityResult,
    ScheduleRequest,
    ScheduleResult,
    TaskTarget,
)
from iris.cluster.controller.reconcile.snapshot import TaskUpdate
from iris.cluster.controller.task_state import job_scheduling_deadline
from iris.cluster.types import DEFAULT_BACKEND_ID, AttemptUid, JobName
from iris.rpc import job_pb2, worker_pb2


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

    def __init__(
        self,
        *,
        backend_id: str = DEFAULT_BACKEND_ID,
        advertised_attributes: dict[str, set[str]] | None = None,
        kind: BackendKind = BackendKind.KUBERNETES,
    ) -> None:
        self.descriptor = BackendDescriptor(
            backend_id=backend_id,
            display_name="journey",
            kind=kind,
            advertised_attributes=advertised_attributes or {"region": {"us-central1"}},
        )
        self._queued: dict[str, deque[ScriptedObservation]] = defaultdict(deque)
        self._desired: dict[tuple[str, int], str] = {}
        self.events: list[BackendEvent] = []
        self.calls: list[str] = []
        self._reconcile_failures = 0
        self.closed = False

    @property
    def has_pending_observations(self) -> bool:
        return any(self._queued.values())

    @property
    def pending_task_ids(self) -> tuple[str, ...]:
        return tuple(sorted(task_id for task_id, queue in self._queued.items() if queue))

    def queue_observation(self, task_id: str, observation: ScriptedObservation) -> None:
        """Queue one observation for the Task's current desired Attempt."""
        self._queued[task_id].append(observation)

    def owns_task(self, task_id: str) -> bool:
        """Whether the controller currently asks this backend to run ``task_id``."""
        return any(desired_task_id == task_id for desired_task_id, _ in self._desired)

    def fail_reconcile(self, *, times: int) -> None:
        self._reconcile_failures += times

    def initialize(self, request: BackendRecoveryRequest) -> BackendRecoveryResult:
        return BackendRecoveryResult()

    def observe(self, request: BackendObservationRequest) -> BackendObservation:
        return BackendObservation()

    def runtime_image(self, requested_image: str) -> str:
        return requested_image

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        self.calls.append("schedule")
        return ScheduleResult()

    def reconcile(self, request: ReconcileRequest) -> ReconcileObservation:
        self.calls.append("reconcile")
        if self._reconcile_failures:
            self._reconcile_failures -= 1
            raise ConnectionError("scripted backend is unavailable")
        if not isinstance(request, DirectReconcileRequest):
            raise ValueError("scripted direct backend requires DirectReconcileRequest")
        desired = {
            **{(run.task_id, run.attempt_id): run.attempt_uid for run in request.tasks_to_run},
            **{(entry.task_id.to_wire(), entry.attempt_id): entry.attempt_uid for entry in request.running_tasks},
        }
        for task_id, attempt_id in sorted(self._desired.keys() - desired.keys()):
            self.events.append(BackendEvent("stopped", task_id, attempt_id, backend_id=self.descriptor.backend_id))

        updates: list[TaskUpdate] = []
        newly_launched = {(run.task_id, run.attempt_id) for run in request.tasks_to_run}
        for task_id, attempt_id in sorted(newly_launched - self._desired.keys()):
            self.events.append(BackendEvent("launched", task_id, attempt_id, backend_id=self.descriptor.backend_id))
            queued = self._pop_observation(task_id)
            if queued is None:
                queued = ScriptedObservation(job_pb2.TASK_STATE_RUNNING)
            observed_attempt_id = attempt_id if queued.attempt_id is None else queued.attempt_id
            updates.append(self._task_update(task_id, observed_attempt_id, desired[(task_id, attempt_id)], queued))

        for task_id, attempt_id in sorted(desired.keys() - newly_launched):
            queued = self._pop_observation(task_id)
            if queued is not None:
                observed_attempt_id = attempt_id if queued.attempt_id is None else queued.attempt_id
                updates.append(self._task_update(task_id, observed_attempt_id, desired[(task_id, attempt_id)], queued))

        self._desired = desired
        return ReconcileObservation(task_updates=updates)

    def _pop_observation(self, task_id: str) -> ScriptedObservation | None:
        queue = self._queued.get(task_id)
        if not queue:
            return None
        observation = queue.popleft()
        if not queue:
            self._queued.pop(task_id, None)
        return observation

    def _task_update(
        self,
        task_id: str,
        attempt_id: int,
        attempt_uid: str,
        observation: ScriptedObservation,
    ) -> TaskUpdate:
        self.events.append(BackendEvent("observed", task_id, attempt_id, observation.state, self.descriptor.backend_id))
        return TaskUpdate(
            attempt_uid=AttemptUid(attempt_uid),
            task_id=JobName.from_wire(task_id),
            attempt_id=attempt_id,
            new_state=observation.state,
            error=observation.error or None,
            exit_code=observation.exit_code,
        )

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        self.calls.append("autoscale")
        return AutoscaleResult()

    def remove_capacity(self, request: RemoveCapacityRequest) -> RemoveCapacityResult:
        return RemoveCapacityResult()

    def job_feasibility(
        self,
        constraints: list[Constraint],
        *,
        replicas: int | None,
        resources: job_pb2.ResourceSpecProto,
    ) -> str | None:
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
    """Backend that advertises a route but has no capacity."""

    def __init__(self, **kwargs) -> None:
        super().__init__(kind=BackendKind.WORKER, **kwargs)

    def reconcile(self, request: ReconcileRequest) -> ReconcileObservation:
        self.calls.append("reconcile")
        return ReconcileObservation()

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        self.calls.append("schedule")
        expired = []
        for task in request.context.pending_task_rows:
            deadline = job_scheduling_deadline(task.scheduling_deadline_epoch_ms)
            if deadline is not None and deadline.expired():
                expired.append(task)
        return ScheduleResult(unschedulable=expired)
