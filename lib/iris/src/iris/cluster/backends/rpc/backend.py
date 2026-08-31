# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RpcTaskBackend: a TaskBackend backed by worker daemons via Connect RPC.

The worker-daemon backend used by the GCP/TPU, CoreWeave-bare-metal, manual, and
local clusters. The Iris scheduler assigns task→worker; this backend fans the
per-worker Reconcile RPC out to worker daemons and translates replies into exact
task and reachability observations. The controller reloads current state,
applies Iris transition and liveness policy, commits effects, and requests teardown.
"""

import asyncio
import logging
import threading
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from typing import Protocol, TypeVar

from rigging.timing import Duration, Timestamp

from iris.chaos import chaos
from iris.cluster.constraints import DeviceType
from iris.cluster.controller.autoscaler import Autoscaler
from iris.cluster.controller.autoscaler.status import overlay_worker_usability
from iris.cluster.controller.backend import (
    AutoscaleRequest,
    AutoscaleResult,
    BackendDescriptor,
    BackendObservation,
    BackendObservationRequest,
    BackendRecoveryRequest,
    BackendRecoveryResult,
    DeviceCapacity,
    JobFeasibilityRequest,
    ProviderError,
    ReconcileObservation,
    ReconcileRequest,
    RemoveCapacityRequest,
    RemoveCapacityResult,
    ScheduleRequest,
    ScheduleResult,
    TaskTarget,
    WorkerFleetReconcileRequest,
    WorkerReconcileTarget,
    run_scheduling_decision,
)
from iris.cluster.controller.reconcile.worker import (
    WorkerReconcilePlan,
    WorkerReconcileResult,
    task_updates_from_result,
)
from iris.cluster.controller.scheduling.scheduler import Scheduler
from iris.cluster.controller.worker_health import (
    WorkerHealthEvent,
    WorkerHealthEventKind,
)
from iris.cluster.types import TERMINAL_TASK_STATES, AttemptUid, WellKnownAttribute, WorkerId
from iris.rpc import controller_pb2, job_pb2, vm_pb2, worker_pb2
from iris.rpc.compression import IRIS_RPC_COMPRESSIONS
from iris.rpc.worker_connect import WorkerServiceClient

logger = logging.getLogger(__name__)

# Per-worker RPC deadline for on-demand worker RPCs (profile_task, exec_in_container,
# get_process_status) and the cached stub's fallback timeout.
DEFAULT_WORKER_RPC_TIMEOUT = Duration.from_seconds(10.0)

# Tighter per-worker deadline for the reconcile fan-out: a hung worker can't gate
# the gather-joined round on the slow straggler, and a missed round never reaps a
# worker (the reconcile-failure threshold is dozens of rounds).
RECONCILE_RPC_TIMEOUT = Duration.from_seconds(3.0)

# Max concurrent in-flight per-worker RPCs in a fan-out (asyncio.Semaphore width).
# Kept >= fleet size so the whole fleet reconciles in one wave and a slow worker
# costs one RPC-timeout window per round, not one per wave.
RECONCILE_FANOUT_PARALLELISM = 512

# Generous deadline for an "unlimited" exec_in_container (negative timeout). Long
# enough for real interactive/debug commands, but not the old ~1-hour stall.
EXEC_IN_CONTAINER_MAX_TIMEOUT = Duration.from_seconds(900.0)

_T = TypeVar("_T")
_R = TypeVar("_R")


def _state_observations(
    result: WorkerReconcileResult,
    release_only_uids: frozenset[AttemptUid],
) -> WorkerReconcileResult:
    """Remove release-only observations before translating task state."""
    return WorkerReconcileResult(
        worker_id=result.worker_id,
        observations=[obs for obs in result.observations if obs.attempt_uid not in release_only_uids],
        error=result.error,
        self_healthy=result.self_healthy,
        responder_worker_id=result.responder_worker_id,
    )


def _confirmed_runtime_releases(
    result: WorkerReconcileResult,
    requested: frozenset[AttemptUid],
) -> set[AttemptUid]:
    """Return requested exact runtimes that a trusted response proves absent."""
    return {
        AttemptUid(obs.attempt_uid)
        for obs in result.observations
        if obs.attempt_uid in requested
        and (obs.state in TERMINAL_TASK_STATES or obs.state == job_pb2.TASK_STATE_MISSING)
        and (obs.runtime_released or obs.state == job_pb2.TASK_STATE_MISSING)
    }


def _targets_with_runtime_releases(
    request: WorkerFleetReconcileRequest,
) -> tuple[list[WorkerReconcileTarget], dict[WorkerId, frozenset[AttemptUid]]]:
    """Merge exact stop intents into immutable per-worker reconcile plans."""
    targets_by_worker: dict[WorkerId, WorkerReconcileTarget] = {}
    for target in request.targets:
        wire_request = worker_pb2.Worker.ReconcileRequest()
        wire_request.CopyFrom(target.plan.request)
        targets_by_worker[target.plan.worker_id] = WorkerReconcileTarget(
            plan=WorkerReconcilePlan(
                worker_id=target.plan.worker_id,
                request=wire_request,
                attempts=target.plan.attempts,
            ),
            address=target.address,
        )

    releases_by_worker: dict[WorkerId, set[AttemptUid]] = {}
    for release in request.release_targets:
        if release.worker_id is None or release.worker_address is None:
            continue
        releases_by_worker.setdefault(release.worker_id, set()).add(release.attempt_uid)
        target = targets_by_worker.get(release.worker_id)
        if target is None:
            target = WorkerReconcileTarget(
                plan=WorkerReconcilePlan(
                    worker_id=release.worker_id,
                    request=worker_pb2.Worker.ReconcileRequest(worker_id=release.worker_id),
                    attempts=(),
                ),
                address=release.worker_address,
            )
            targets_by_worker[release.worker_id] = target
        desired_uids = {desired.attempt_uid for desired in target.plan.request.desired}
        if release.attempt_uid not in desired_uids:
            target.plan.request.desired.append(
                worker_pb2.Worker.DesiredAttempt(
                    attempt_uid=release.attempt_uid,
                    stop=worker_pb2.Worker.STOP_REASON_JOB_TERMINATED,
                )
            )

    ordered = [targets_by_worker[worker_id] for worker_id in sorted(targets_by_worker)]
    return ordered, {worker_id: frozenset(uids) for worker_id, uids in releases_by_worker.items()}


def _fan_out(
    items: Sequence[_T],
    parallelism: int,
    run_one: Callable[[asyncio.Semaphore, _T], Awaitable[_R]],
) -> list[_R]:
    """Run ``run_one`` over every item concurrently, capped at ``parallelism``.

    Each coroutine receives the shared semaphore and is responsible for
    acquiring it; ``gather`` preserves input order in the returned list.
    """
    if not items:
        return []

    async def _run() -> list[_R]:
        sem = asyncio.Semaphore(parallelism)
        return await asyncio.gather(*(run_one(sem, item) for item in items))

    return asyncio.run(_run())


class WorkerStubFactory(Protocol):
    """Factory for getting cached async worker RPC stubs."""

    def get_stub(self, address: str) -> WorkerServiceClient: ...
    def evict(self, address: str) -> None: ...
    def close(self) -> None: ...


class RpcWorkerStubFactory:
    """Caches async WorkerServiceClient stubs by address so each worker gets
    one persistent async HTTP client across RPCs."""

    def __init__(self, timeout: Duration = DEFAULT_WORKER_RPC_TIMEOUT) -> None:
        self._timeout = timeout
        self._stubs: dict[str, WorkerServiceClient] = {}
        self._lock = threading.Lock()

    @property
    def timeout_ms(self) -> int:
        return self._timeout.to_ms()

    def get_stub(self, address: str) -> WorkerServiceClient:
        with self._lock:
            stub = self._stubs.get(address)
            if stub is None:
                stub = WorkerServiceClient(
                    address=f"http://{address}",
                    timeout_ms=self._timeout.to_ms(),
                    accept_compression=IRIS_RPC_COMPRESSIONS,
                    send_compression=None,
                )
                self._stubs[address] = stub
            return stub

    def evict(self, address: str) -> None:
        with self._lock:
            self._stubs.pop(address, None)

    def close(self) -> None:
        with self._lock:
            self._stubs.clear()


@dataclass
class RpcTaskBackend:
    """A worker-daemon :class:`~iris.cluster.controller.backend.TaskBackend`
    backed by async Connect RPCs.

    Each fan-out method spins up an asyncio event loop and dispatches the
    relevant RPC to each worker concurrently via `asyncio.gather`, capped at
    `parallelism` in-flight requests by a local semaphore. Cached stubs in
    the factory keep their pyqwest connection pools across rounds.
    """

    descriptor: BackendDescriptor
    stub_factory: WorkerStubFactory
    parallelism: int = RECONCILE_FANOUT_PARALLELISM
    # The Iris autoscaler that provisions capacity for this backend, passed by the
    # composer at construction after it builds the autoscaler from the provider
    # bundle; None for clusters with no scale groups, where capacity calls are no-ops.
    autoscaler: Autoscaler | None = None
    # One shared scheduler instance reused across cycles; the controller supplies
    # the complete per-tick workspace.
    _scheduler: Scheduler = field(default_factory=Scheduler, init=False, repr=False)

    def initialize(self, request: BackendRecoveryRequest) -> BackendRecoveryResult:
        if self.autoscaler is None or request.autoscaler_checkpoint is None:
            return BackendRecoveryResult()
        self.autoscaler.restore(request.autoscaler_checkpoint)
        return BackendRecoveryResult(autoscaler_state=self.autoscaler.persistable_state())

    def runtime_image(self, requested_image: str) -> str:
        return requested_image

    def observe(self, request: BackendObservationRequest) -> BackendObservation:
        """Build status and capacity from controller-owned worker facts."""
        capacity = self._resource_capacity(request)
        autoscaler_status = self.autoscaler.get_status() if self.autoscaler is not None else vm_pb2.AutoscalerStatus()
        for group in autoscaler_status.groups:
            group.backend_id = self.descriptor.backend_id
        usability_by_id = {str(worker_id): live.usability for worker_id, live in request.liveness.items()}
        overlay_worker_usability(autoscaler_status, usability_by_id, request.running_tasks)
        return BackendObservation(
            status=controller_pb2.Controller.BackendStatus(
                worker=controller_pb2.Controller.WorkerFleetDetail(
                    autoscaler=autoscaler_status,
                    total_worker_count=len(request.liveness),
                    healthy_worker_count=sum(1 for live in request.liveness.values() if live.healthy),
                )
            ),
            resource_capacity=capacity,
            pending_hints=self.autoscaler.get_pending_hints() if self.autoscaler is not None else {},
        )

    def _resource_capacity(self, request: BackendObservationRequest) -> dict[str, DeviceCapacity]:
        """Free and total GPU chips a peer could schedule onto, keyed by lowercased device-variant.

        Counts only capacity the scheduler would actually place onto — chips on
        live, schedulable workers — so the advertised numbers match what a handoff
        can use. v1 is GPU-only; TPU-slice availability is a documented follow-up.
        Always a dict (empty = authoritative "nothing free"), never ``None``: a
        worker-daemon backend always supplies the metric."""
        capacity: dict[str, DeviceCapacity] = {}
        for worker in request.workers:
            device_type = worker.attributes.get(WellKnownAttribute.DEVICE_TYPE)
            variant = worker.attributes.get(WellKnownAttribute.DEVICE_VARIANT)
            if device_type is None or variant is None or str(device_type.value) != DeviceType.GPU.value:
                continue
            token = str(variant.value).strip().lower()
            prior = capacity.get(token, DeviceCapacity(free=0, total=0))
            capacity[token] = DeviceCapacity(
                free=prior.free + max(0, worker.total_gpu_count - worker.committed_gpu_count),
                total=prior.total + worker.total_gpu_count,
            )
        return capacity

    def schedule(self, request: ScheduleRequest) -> ScheduleResult:
        """Run the Iris scheduling pipeline over the controller-built workspace.

        The autoscaler's per-zone accelerator-capability map injects
        ``availability:<variant>`` markers onto workers so a hard availability
        constraint is confined to a capable zone; clusters with no autoscaler pass
        an empty map, so a job carrying an availability constraint there stays
        unschedulable (no zone can satisfy it).
        """
        zone_capabilities = self.autoscaler.zone_capabilities() if self.autoscaler is not None else None
        return run_scheduling_decision(self._scheduler, request, zone_capabilities)

    def _observe_fleet(self, request: WorkerFleetReconcileRequest) -> ReconcileObservation:
        """Source this backend's placement, fan the Reconcile RPC out, classify liveness.

        The reconcile plans and addresses come from the controller. Each per-worker RPC carries the stub
        factory's deadline and the fan-out caps concurrency at
        ``parallelism``, so this returns in bounded time even when the whole fleet
        is hung. Each outcome yields a transport liveness signal:

        * a healthy response is REACHED;
        * an RPC error/timeout is UNREACHABLE, and the (likely broken) stub is
          evicted as I/O hygiene;
        * a response that self-reports unhealthy (e.g. failed disk) is also
          UNREACHABLE so the worker is eventually reaped, but the connection is
          fine so the stub is kept.

        The backend translates worker-protocol replies into exact task state and
        reachability facts. It never decides a worker dead or applies Iris task
        policy; the controller handles both after this I/O returns.
        """
        targets, releases_by_worker = _targets_with_runtime_releases(request)
        plans = [target.plan for target in targets]
        addresses = {target.plan.worker_id: target.address for target in targets}

        async def _one(sem: asyncio.Semaphore, plan: WorkerReconcilePlan) -> WorkerReconcileResult:
            return await self._reconcile_one(sem, plan, addresses[plan.worker_id])

        results = _fan_out(plans, self.parallelism, _one)
        observed_at = Timestamp.now()

        task_updates = []
        released_attempt_uids: set[AttemptUid] = set()
        worker_health_events: list[WorkerHealthEvent] = []
        for plan, result in zip(plans, results, strict=True):
            address = addresses[plan.worker_id]
            releases = releases_by_worker.get(plan.worker_id, frozenset())
            regular_attempt_uids = {row.attempt_uid for row in plan.attempts}
            release_only_uids = releases - regular_attempt_uids
            state_result = _state_observations(result, frozenset(release_only_uids))
            response_is_trusted = result.error is None and (
                result.responder_worker_id is None or result.responder_worker_id == str(plan.worker_id)
            )
            if result.error is not None:
                logger.warning("Reconcile RPC failed for worker %s at %s: %s", plan.worker_id, address, result.error)
                worker_health_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
                self.stub_factory.evict(address)
                task_updates.extend(task_updates_from_result(plan, state_result, observed_at=observed_at))
            elif not response_is_trusted:
                # Misrouted reconcile: a *different* live worker answered at this
                # address. GCP recycles a deleted worker's internal IP onto a new
                # VM, so the controller's stale address for the dead worker now
                # points at someone else. Counting the impostor's healthy reply as
                # REACHED would resurrect the dead worker (reset its failures to 0)
                # and keep it schedulable — a black hole that accepts and kills
                # every task. Treat it as UNREACHABLE so the stale worker accrues
                # failures and is reaped, and drop the impostor's stub.
                logger.warning(
                    "Reconcile for worker %s at %s was answered by %s (recycled address); marking unreachable",
                    plan.worker_id,
                    address,
                    result.responder_worker_id,
                )
                worker_health_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
                self.stub_factory.evict(address)
            elif not result.self_healthy:
                worker_health_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.UNREACHABLE))
                task_updates.extend(task_updates_from_result(plan, state_result, observed_at=observed_at))
            else:
                worker_health_events.append(WorkerHealthEvent(plan.worker_id, WorkerHealthEventKind.REACHED))
                task_updates.extend(task_updates_from_result(plan, state_result, observed_at=observed_at))
            if response_is_trusted:
                released_attempt_uids.update(_confirmed_runtime_releases(result, releases))
        return ReconcileObservation(
            task_updates=task_updates,
            worker_health_events=worker_health_events,
            released_attempt_uids=frozenset(released_attempt_uids),
        )

    def reconcile(self, request: ReconcileRequest) -> ReconcileObservation:
        """Return exact task state and worker reachability observations.

        The controller supplies the complete per-worker plan and address. No Iris
        transition or liveness policy runs in this method.
        """
        if not isinstance(request, WorkerFleetReconcileRequest):
            raise ValueError("worker backend requires WorkerFleetReconcileRequest")
        return self._observe_fleet(request)

    def autoscale(self, request: AutoscaleRequest) -> AutoscaleResult:
        """Run one provisioning cycle from a complete controller snapshot."""
        if self.autoscaler is None:
            return AutoscaleResult()
        self.autoscaler.refresh(request.worker_status)
        self.autoscaler.probe_health()
        self.autoscaler.update(request.residual_demand)
        return AutoscaleResult(autoscaler_state=self.autoscaler.persistable_state())

    def remove_capacity(self, request: RemoveCapacityRequest) -> RemoveCapacityResult:
        if self.autoscaler is None or not request.worker_ids:
            return RemoveCapacityResult()
        siblings = self.autoscaler.terminate_slices_for_workers([str(wid) for wid in request.worker_ids])
        return RemoveCapacityResult(
            sibling_workers=[WorkerId(worker_id) for worker_id in siblings],
            autoscaler_state=self.autoscaler.persistable_state(),
        )

    def job_feasibility(self, request: JobFeasibilityRequest) -> str | None:
        if self.autoscaler is None:
            return None
        return self.autoscaler.job_feasibility(
            request.constraints,
            replicas=request.replicas,
            resources=request.resources,
        )

    def get_process_status(
        self,
        target: TaskTarget,
        request: job_pb2.GetProcessStatusRequest,
    ) -> job_pb2.GetProcessStatusResponse:
        if not target.address:
            raise ProviderError(f"Worker {target.worker_id} has no address")
        stub = self.stub_factory.get_stub(target.address)
        # Forward with target cleared — the worker serves its own process status.
        forwarded = job_pb2.GetProcessStatusRequest(
            max_log_lines=request.max_log_lines,
            log_substring=request.log_substring,
            min_log_level=request.min_log_level,
        )
        return asyncio.run(stub.get_process_status(forwarded, timeout_ms=10000))

    def profile_task(
        self,
        target: TaskTarget,
        request: job_pb2.ProfileTaskRequest,
        timeout_ms: int,
    ) -> job_pb2.ProfileTaskResponse:
        if not target.address:
            raise ProviderError(f"Worker {target.worker_id} has no address")
        stub = self.stub_factory.get_stub(target.address)
        return asyncio.run(stub.profile_task(request, timeout_ms=timeout_ms))

    def exec_in_container(
        self,
        target: TaskTarget,
        request: worker_pb2.Worker.ExecInContainerRequest,
        timeout_seconds: int = 60,
    ) -> worker_pb2.Worker.ExecInContainerResponse:
        if not target.address:
            raise ProviderError(f"Worker {target.worker_id} has no address")
        stub = self.stub_factory.get_stub(target.address)
        # Negative timeout means "no caller limit"; still bound the RPC deadline
        # with a generous cap so a hung exec can't pin the handler indefinitely.
        if timeout_seconds < 0:
            rpc_timeout_ms = EXEC_IN_CONTAINER_MAX_TIMEOUT.to_ms()
        else:
            rpc_timeout_ms = (timeout_seconds + 5) * 1000
        return asyncio.run(stub.exec_in_container(request, timeout_ms=rpc_timeout_ms))

    async def _reconcile_one(
        self,
        sem: asyncio.Semaphore,
        plan: WorkerReconcilePlan,
        address: str,
    ) -> WorkerReconcileResult:
        """Issue a single Reconcile RPC to one worker under the shared semaphore."""
        async with sem:
            try:
                if rule := chaos("controller.reconcile"):
                    await asyncio.sleep(rule.delay_seconds)
                    raise ProviderError("chaos: controller.reconcile")
                stub = self.stub_factory.get_stub(address)
                response = await asyncio.wait_for(
                    stub.reconcile(plan.request), timeout=RECONCILE_RPC_TIMEOUT.to_seconds()
                )
                return WorkerReconcileResult(
                    worker_id=plan.worker_id,
                    observations=list(response.observed),
                    error=None,
                    self_healthy=response.health.healthy,
                    responder_worker_id=response.worker_id or None,
                )
            except Exception as e:
                return WorkerReconcileResult(worker_id=plan.worker_id, observations=[], error=str(e) or type(e).__name__)

    def close(self) -> None:
        if self.autoscaler is not None:
            self.autoscaler.shutdown()
        self.stub_factory.close()
