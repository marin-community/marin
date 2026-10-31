# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Authorize a declared direct endpoint for its live in-cluster driver."""

from dataclasses import dataclass

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from rigging.server_auth import get_verified_identity

from iris.cluster.controller import reads, tasks
from iris.cluster.controller.auth import TASK_ROLE
from iris.cluster.controller.backend import ProviderError, TaskBackend, TaskTarget
from iris.cluster.controller.codec import proto_from_json
from iris.cluster.controller.db import ControllerDB
from iris.cluster.runtime.sandbox import task_isolation
from iris.cluster.task_connection import TaskPod
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2


@dataclass(frozen=True)
class ConnectionDependencies:
    db: ControllerDB
    backend: TaskBackend


def _live_target(dependencies: ConnectionDependencies, name: str, uid: str) -> TaskTarget:
    try:
        task_id = JobName.from_wire(name)
        task_id.require_task()
    except ValueError as error:
        raise ConnectError(Code.INVALID_ARGUMENT, str(error)) from error
    with dependencies.db.read_snapshot() as tx:
        task = tasks.read_task_with_attempts(tx, task_id)
        if reads.federated_handle(tx, task_id.root_job) is not None:
            raise ConnectError(Code.FAILED_PRECONDITION, "Direct task connections require the executing controller")
    if task is None or task.state != job_pb2.TASK_STATE_RUNNING:
        raise ConnectError(Code.FAILED_PRECONDITION, "Connection task is not running")
    current_uid = next((row.attempt_uid for row in task.attempts if row.attempt_id == task.current_attempt_id), "")
    if not uid or uid != current_uid:
        raise ConnectError(Code.FAILED_PRECONDITION, "Connection task attempt is stale")
    return TaskTarget(name, task.current_attempt_id, None, None, uid)


def _driver(
    dependencies: ConnectionDependencies, spec: job_pb2.DriverConnectionSpec, child_job: JobName
) -> tuple[TaskTarget, TaskPod]:
    identity = get_verified_identity()
    try:
        driver_id = JobName.from_wire(spec.driver_task_id)
        driver_job, _ = driver_id.require_task()
    except ValueError as error:
        raise ConnectError(Code.INVALID_ARGUMENT, str(error)) from error
    if identity is None or identity.role != TASK_ROLE or identity.job_id != driver_job.to_wire():
        raise ConnectError(Code.PERMISSION_DENIED, "A direct connection requires the owning in-cluster task identity")
    if identity.user_id != child_job.user or not driver_job.is_ancestor_of(child_job, include_self=False):
        raise ConnectError(Code.PERMISSION_DENIED, "The connection target must be an owned descendant of the driver job")
    with dependencies.db.read_snapshot() as tx:
        job = reads.get_job_detail(tx, driver_job)
    if job is None or not task_isolation(job.container_profile, job.egress_policy).include_task_token:
        raise ConnectError(Code.PERMISSION_DENIED, "An isolated task cannot act as a connection driver")
    target = _live_target(dependencies, spec.driver_task_id, spec.driver_attempt_uid)
    try:
        pod = dependencies.backend.resolve_task_pod(target)
    except ProviderError as error:
        raise ConnectError(Code.FAILED_PRECONDITION, str(error)) from error
    if pod.host_network:
        raise ConnectError(Code.FAILED_PRECONDITION, "A direct connection driver requires POD network attachment")
    return target, pod


def bind_driver_connection(
    dependencies: ConnectionDependencies, request: controller_pb2.Controller.LaunchJobRequest, child_job: JobName
) -> None:
    """Bind a child declaration to the actual driver Pod before persistence."""
    if not request.environment.HasField("driver_connection"):
        return
    spec = request.environment.driver_connection
    if spec.driver_pod_uid or not 1 <= spec.port <= 65535:
        raise ConnectError(Code.INVALID_ARGUMENT, "Declare a valid connection port and leave driver_pod_uid unset")
    if request.environment.network_attachment != job_pb2.NETWORK_ATTACHMENT_POD:
        raise ConnectError(Code.INVALID_ARGUMENT, "Driver connections require POD network attachment")
    driver, pod = _driver(dependencies, spec, child_job)
    try:
        authorized = dependencies.backend.prepare_driver_connection(driver, spec.port)
    except ProviderError as error:
        raise ConnectError(Code.FAILED_PRECONDITION, str(error)) from error
    if authorized.pod_uid != pod.pod_uid:
        raise ConnectError(Code.FAILED_PRECONDITION, "The owning driver Pod was replaced during launch")
    _live_target(dependencies, spec.driver_task_id, spec.driver_attempt_uid)
    spec.driver_pod_uid = pod.pod_uid


def get_task_connection(
    dependencies: ConnectionDependencies,
    request: controller_pb2.Controller.GetTaskConnectionRequest,
    context: RequestContext,
) -> controller_pb2.Controller.GetTaskConnectionResponse:
    del context
    try:
        task_id = JobName.from_wire(request.task_id)
        child_job, _ = task_id.require_task()
    except ValueError as error:
        raise ConnectError(Code.INVALID_ARGUMENT, str(error)) from error
    with dependencies.db.read_snapshot() as tx:
        job = reads.get_job_detail(tx, child_job)
    if job is None:
        raise ConnectError(Code.NOT_FOUND, "Connection task does not exist")
    environment = proto_from_json(job.environment_json, job_pb2.EnvironmentConfig)
    if not environment.HasField("driver_connection"):
        raise ConnectError(Code.FAILED_PRECONDITION, "Task has no declared driver endpoint")
    spec = environment.driver_connection
    if request.driver_task_id != spec.driver_task_id or request.driver_attempt_uid != spec.driver_attempt_uid:
        raise ConnectError(Code.PERMISSION_DENIED, "Connection belongs to another driver attempt")
    driver, driver_pod = _driver(dependencies, spec, child_job)
    if driver_pod.pod_uid != spec.driver_pod_uid:
        raise ConnectError(Code.FAILED_PRECONDITION, "The owning driver Pod was replaced")
    target = _live_target(dependencies, request.task_id, request.attempt_uid)
    try:
        pod = dependencies.backend.resolve_task_connection(driver, target, spec)
    except ProviderError as error:
        raise ConnectError(Code.FAILED_PRECONDITION, str(error)) from error
    # Provider reads may race a retry or cancellation.
    _live_target(dependencies, request.task_id, request.attempt_uid)
    _live_target(dependencies, request.driver_task_id, request.driver_attempt_uid)
    return controller_pb2.Controller.GetTaskConnectionResponse(
        pod_ip=pod.pod_ip, port=spec.port, pod_uid=pod.pod_uid, attempt_uid=pod.attempt_uid
    )
