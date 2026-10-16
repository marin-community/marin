# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Controller read operations for backend, scheduler, and capacity status."""

from dataclasses import dataclass
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from rigging.server_auth import require_identity
from sqlalchemy import func, select

from iris.cluster.controller import reads
from iris.cluster.controller.backend import (
    BackendCapability,
    BackendObservation,
    TaskBackend,
    dashboard_backend_descriptor,
)
from iris.cluster.controller.budget import budget_user_id, compute_effective_band, compute_user_spend
from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import jobs_table, tasks_table, workers_table
from iris.cluster.federation.availability import AVAILABILITY_METRIC_VERSION
from iris.cluster.types import UserBudgetDefaults
from iris.rpc import controller_pb2, job_pb2, vm_pb2
from iris.rpc.proto_display import priority_band_rank


class BackendStatusRuntime(Protocol):
    @property
    def backend(self) -> TaskBackend: ...

    @property
    def backend_observation(self) -> BackendObservation: ...


@dataclass(frozen=True, slots=True)
class BackendStatusDependencies:
    db: ControllerDB
    runtime: BackendStatusRuntime
    user_budget_defaults: UserBudgetDefaults


def get_autoscaler_status(
    dependencies: BackendStatusDependencies,
    request: controller_pb2.Controller.GetAutoscalerStatusRequest,
    context: RequestContext,
) -> controller_pb2.Controller.GetAutoscalerStatusResponse:
    del context
    backend = dependencies.runtime.backend
    if BackendCapability.AUTOSCALER not in backend.descriptor.capabilities:
        return controller_pb2.Controller.GetAutoscalerStatusResponse(status=vm_pb2.AutoscalerStatus())
    if request.backend_id and request.backend_id != backend.descriptor.backend_id:
        return controller_pb2.Controller.GetAutoscalerStatusResponse(status=vm_pb2.AutoscalerStatus())
    return controller_pb2.Controller.GetAutoscalerStatusResponse(
        status=dependencies.runtime.backend_observation.status.worker.autoscaler
    )


def get_kubernetes_cluster_status(
    dependencies: BackendStatusDependencies,
    request: controller_pb2.Controller.GetKubernetesClusterStatusRequest,
    context: RequestContext,
) -> controller_pb2.Controller.GetKubernetesClusterStatusResponse:
    del context
    backend = dependencies.runtime.backend
    if request.backend_id and request.backend_id != backend.descriptor.backend_id:
        raise ConnectError(Code.INVALID_ARGUMENT, f"Backend {request.backend_id!r} does not exist")
    if BackendCapability.DIRECT_DISPATCH in backend.descriptor.capabilities:
        return dependencies.runtime.backend_observation.status.kubernetes
    return controller_pb2.Controller.GetKubernetesClusterStatusResponse()


def get_scheduler_state(
    dependencies: BackendStatusDependencies,
    request: controller_pb2.Controller.GetSchedulerStateRequest,
    context: RequestContext,
) -> controller_pb2.Controller.GetSchedulerStateResponse:
    """Return aggregate pending/running scheduling state."""
    del request, context
    require_identity()
    with dependencies.db.read_snapshot() as snapshot:
        budgets = reads.list_user_budgets(snapshot)
        budget_limits = {budget.user_id: budget.budget_limit for budget in budgets}
        user_spend = compute_user_spend(snapshot)
        pending_rows = reads.pending_tasks_with_jobs(snapshot)
        requested_bands = reads.get_priority_bands(snapshot, {row.job_id for row in pending_rows})
        running_rows = snapshot.execute(
            select(
                tasks_table.c.task_id,
                jobs_table.c.submitting_user,
                tasks_table.c.priority_band,
                tasks_table.c.current_worker_id.label("worker_id"),
                tasks_table.c.backend_id,
            )
            .select_from(tasks_table.join(jobs_table, jobs_table.c.job_id == tasks_table.c.job_id))
            .where(
                tasks_table.c.state == job_pb2.TASK_STATE_RUNNING,
                tasks_table.c.current_worker_id.is_not(None),
            )
        ).all()

    pending_counts: dict[tuple[int, str, str, str], int] = {}
    for row in pending_rows:
        user_id = budget_user_id(row.job_id, row.submitting_user)
        band = compute_effective_band(
            requested_bands.get(row.job_id, row.priority_band),
            user_id,
            user_spend,
            budget_limits,
            dependencies.user_budget_defaults,
        )
        job_id = (row.task_id.parent or row.task_id).to_wire()
        key = (band, user_id, job_id, str(row.backend_id or ""))
        pending_counts[key] = pending_counts.get(key, 0) + 1

    running_counts: dict[tuple[int, str, str, str, str], int] = {}
    for row in running_rows:
        user_id = budget_user_id(row.task_id, str(row.submitting_user))
        job_id = (row.task_id.parent or row.task_id).to_wire()
        key = (row.priority_band, user_id, str(row.worker_id), job_id, str(row.backend_id or ""))
        running_counts[key] = running_counts.get(key, 0) + 1

    defaults = dependencies.user_budget_defaults
    budget_rows = [(budget.user_id, budget.budget_limit, budget.max_band) for budget in budgets]
    configured_users = {budget.user_id for budget in budgets}
    budget_rows.extend(
        (user_id, defaults.budget_limit, defaults.max_band) for user_id in user_spend if user_id not in configured_users
    )
    budget_protos = []
    for user_id, budget_limit, max_band in budget_rows:
        spent = user_spend.get(user_id, 0)
        budget_protos.append(
            controller_pb2.Controller.SchedulerUserBudget(
                user_id=user_id,
                budget_limit=budget_limit,
                budget_spent=spent,
                max_band=max_band,
                effective_band=compute_effective_band(
                    job_pb2.PRIORITY_BAND_INTERACTIVE,
                    user_id,
                    user_spend,
                    budget_limits,
                    defaults,
                ),
                utilization_percent=(spent / budget_limit * 100.0) if budget_limit > 0 else 0.0,
            )
        )

    return controller_pb2.Controller.GetSchedulerStateResponse(
        user_budgets=budget_protos,
        total_pending=sum(pending_counts.values()),
        total_running=sum(running_counts.values()),
        pending_buckets=[
            controller_pb2.Controller.PendingTaskBucket(
                band=band,
                user_id=user_id,
                job_id=job_id,
                backend_id=backend_id,
                count=count,
            )
            for (band, user_id, job_id, backend_id), count in pending_counts.items()
        ],
        running_buckets=[
            controller_pb2.Controller.RunningTaskBucket(
                band=band,
                user_id=user_id,
                worker_id=worker_id,
                job_id=job_id,
                backend_id=backend_id,
                count=count,
            )
            for (band, user_id, worker_id, job_id, backend_id), count in running_counts.items()
        ],
    )


def list_backends(
    dependencies: BackendStatusDependencies,
    request: controller_pb2.Controller.ListBackendsRequest,
    context: RequestContext,
) -> controller_pb2.Controller.ListBackendsResponse:
    """Describe the controller's execution backend and aggregate workload."""
    del request, context
    require_identity()
    backend = dependencies.runtime.backend
    descriptor = backend.descriptor
    dashboard = dashboard_backend_descriptor(backend)
    with dependencies.db.read_snapshot() as snapshot:
        pending_count = int(
            snapshot.execute(
                select(func.count()).select_from(tasks_table).where(tasks_table.c.state == job_pb2.TASK_STATE_PENDING)
            ).scalar_one()
        )
        running_count = int(
            snapshot.execute(
                select(func.count()).select_from(tasks_table).where(tasks_table.c.state == job_pb2.TASK_STATE_RUNNING)
            ).scalar_one()
        )
        worker_count = int(snapshot.execute(select(func.count()).select_from(workers_table)).scalar_one())

    observation = dependencies.runtime.backend_observation
    backend_status = observation.status
    variant = backend_status.WhichOneof("detail")
    capacity_health: dict[str, int] = {}
    if variant == "worker":
        for group in backend_status.worker.autoscaler.groups:
            status = group.availability_status or "unknown"
            capacity_health[status] = capacity_health.get(status, 0) + 1

    summary = controller_pb2.Controller.BackendSummary(
        backend_id=descriptor.backend_id,
        name=dashboard.name,
        kind=descriptor.kind.value,
        capabilities=dashboard.capabilities,
        scale_groups=sorted(descriptor.scale_groups),
        worker_count=worker_count,
        pending_task_count=pending_count,
        running_task_count=running_count,
        has_autoscaler=BackendCapability.AUTOSCALER in descriptor.capabilities,
        capacity_health=capacity_health,
    )
    for key, values in descriptor.advertised_attributes.items():
        summary.advertised_attributes[key].values.extend(sorted(values))

    if observation.resource_capacity is not None:
        summary.availability.version = AVAILABILITY_METRIC_VERSION
        summary.availability.observation_epoch_ms = observation.observed_at.epoch_ms()
        held_by_band: dict[int, dict[str, int]] = {}
        for token, device_capacity in observation.resource_capacity.items():
            summary.availability.amounts[token] = device_capacity.free
            summary.availability.total_amounts[token] = device_capacity.total
            for band, amount in device_capacity.held_by_band.items():
                held_by_band.setdefault(band, {})[token] = amount
        for band, amounts in sorted(held_by_band.items(), key=lambda item: priority_band_rank(item[0])):
            summary.availability.held_by_band.add(band=band, amounts=amounts)

    if variant == "kubernetes":
        summary.detail.kubernetes.CopyFrom(backend_status.kubernetes)
    elif variant == "worker":
        summary.detail.worker.CopyFrom(backend_status.worker)
    return controller_pb2.Controller.ListBackendsResponse(backends=[summary])
