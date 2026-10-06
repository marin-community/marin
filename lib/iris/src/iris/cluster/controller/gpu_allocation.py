# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded, read-only GPU allocation metadata from the owning registry."""

from dataclasses import dataclass
from typing import Protocol

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from rigging.server_auth import require_identity
from rigging.timing import Timestamp
from sqlalchemy import Integer, cast, func, or_, select

from iris.cluster.controller.db import ControllerDB
from iris.cluster.controller.schema import job_config_table, jobs_table, task_attempts_table, tasks_table
from iris.cluster.federation.manager import FederationManager
from iris.cluster.types import LOCAL_CLUSTER, JobName
from iris.rpc import controller_pb2
from iris.rpc.auth import FEDERATION_PEER_ROLE, authorize_method

MAX_WINDOW_MS = 7 * 24 * 60 * 60 * 1000
MAX_ROOTS = 200
MAX_ROWS = 200_000


class GpuAllocationRuntime(Protocol):
    @property
    def federation(self) -> FederationManager: ...


@dataclass(frozen=True)
class GpuAllocationDependencies:
    db: ControllerDB
    runtime: GpuAllocationRuntime


def gpu_allocation_metadata(
    dependencies: GpuAllocationDependencies,
    request: controller_pb2.Controller.GetGpuAllocationMetadataRequest,
) -> controller_pb2.Controller.GetGpuAllocationMetadataResponse:
    """Read attempt lifetimes and child GPU shapes without exposing job payloads."""
    identity = require_identity()
    authorize_method(identity, "GetGpuAllocationMetadata")
    if identity.role not in ("admin", FEDERATION_PEER_ROLE):
        raise ConnectError(Code.PERMISSION_DENIED, "allocation metadata requires admin or federation-peer access")
    if request.from_ms < 0 or not 0 < request.to_ms - request.from_ms <= MAX_WINDOW_MS:
        raise ConnectError(Code.INVALID_ARGUMENT, "allocation metadata requires a window of at most seven days")
    max_rows = request.max_rows or MAX_ROWS
    if not 0 < max_rows <= MAX_ROWS or len(request.root_job_ids) > MAX_ROOTS:
        raise ConnectError(Code.INVALID_ARGUMENT, "allocation metadata exceeds the row or root limit")
    try:
        roots = [JobName.from_wire(root) for root in request.root_job_ids]
        if any(not root.is_root for root in roots):
            raise ValueError("root_job_ids must name root jobs")
    except ValueError as error:
        raise ConnectError(Code.INVALID_ARGUMENT, str(error)) from error

    if request.cluster:
        # A trusted peer receives only a local read, never a transitive proxy.
        if identity.role != "admin":
            raise ConnectError(Code.PERMISSION_DENIED, "only an admin may select a regional metadata peer")
        forwarded = controller_pb2.Controller.GetGpuAllocationMetadataRequest()
        forwarded.CopyFrom(request)
        forwarded.cluster = ""
        if not dependencies.runtime.federation.has_peer(request.cluster):
            raise ConnectError(Code.NOT_FOUND, f"unknown regional metadata peer {request.cluster!r}")
        return dependencies.runtime.federation.proxy_to_peer(
            request.cluster, lambda peer: peer.gpu_allocation_metadata(forwarded)
        )

    jobs, config, tasks, attempts = jobs_table, job_config_table, tasks_table, task_attempts_table
    statement = (
        select(
            jobs.c.root_job_id,
            tasks.c.task_id,
            func.coalesce(cast(func.json_extract(config.c.res_device_json, "$.gpu.count"), Integer), 0).label(
                "gpu_count"
            ),
            func.coalesce(func.json_extract(config.c.res_device_json, "$.gpu.variant"), "").label("gpu_variant"),
            config.c.priority_band.label("requested_priority"),
            tasks.c.priority_band.label("current_applied_priority"),
            tasks.c.current_attempt_id,
            func.coalesce(attempts.c.attempt_id, -1).label("attempt_id"),
            cast(attempts.c.created_at_ms, Integer).label("created_at_ms"),
            cast(attempts.c.started_at_ms, Integer).label("started_at_ms"),
            cast(attempts.c.finished_at_ms, Integer).label("finished_at_ms"),
            func.coalesce(attempts.c.state, tasks.c.state).label("state"),
        )
        .select_from(
            jobs.join(config, config.c.job_id == jobs.c.job_id)
            .join(tasks, tasks.c.job_id == jobs.c.job_id)
            .outerjoin(attempts, attempts.c.task_id == tasks.c.task_id)
        )
        .where(
            jobs.c.cluster == LOCAL_CLUSTER,
            tasks.c.submitted_at_ms < Timestamp.from_ms(request.to_ms),
            or_(attempts.c.attempt_id.is_(None), attempts.c.created_at_ms < Timestamp.from_ms(request.to_ms)),
            or_(attempts.c.finished_at_ms.is_(None), attempts.c.finished_at_ms >= Timestamp.from_ms(request.from_ms)),
        )
        .limit(max_rows + 1)
    )
    if roots:
        statement = statement.where(jobs.c.root_job_id.in_([root.to_wire() for root in roots]))
    with dependencies.db.read_snapshot() as snapshot:
        rows = snapshot.execute(statement).all()
    if len(rows) > max_rows:
        raise ConnectError(Code.RESOURCE_EXHAUSTED, "allocation metadata exceeds the requested row limit")
    response = controller_pb2.Controller.GetGpuAllocationMetadataResponse()
    for row in rows:
        attempt = response.attempts.add(
            root_job_id=row.root_job_id,
            task_id=row.task_id.to_wire(),
            gpu_count=row.gpu_count,
            gpu_variant=row.gpu_variant,
            requested_priority=row.requested_priority,
            current_applied_priority=row.current_applied_priority,
            current_attempt_id=row.current_attempt_id,
            attempt_id=row.attempt_id,
            state=row.state,
        )
        for field in ("created_at_ms", "started_at_ms", "finished_at_ms"):
            value = getattr(row, field)
            if value is not None:
                setattr(attempt, field, value)
    return response
