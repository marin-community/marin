# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Wire helpers shared by controller-owned resource operations."""

import uuid

from google.protobuf import any_pb2
from google.protobuf.message import Message
from rigging.server_auth import ANONYMOUS_ADMIN, get_verified_identity
from rigging.timing import Timestamp

from iris.cluster.authorization import DASHBOARD_ROLE, FEDERATION_PEER_ROLE, authorize_resource_owner
from iris.cluster.controller.controller import Controller
from iris.resources.errors import InvalidResourceKey, InvalidResourceRequest, ResourcePermissionDenied, ResourceReplaced
from iris.resources.identity import ResourceKey
from iris.resources.names import JobName
from iris.rpc import resource_action_pb2, resource_endpoint_pb2, resource_identity_pb2, resource_pb2
from iris.rpc.resource_types import ATTEMPT, ENDPOINT, JOB, NODE, OPERATION, SLICE, TASK
from iris.time_proto import timestamp_to_proto

_DEFAULT_JOB_PAGE_SIZE = 50
_DEFAULT_RESOURCE_PAGE_SIZE = 100
_DEFAULT_ACTIVITY_PAGE_SIZE = 200


def _action_principal(resource_id: str) -> str:
    owner = JobName.from_wire(resource_id.rpartition(":")[0] or resource_id).user
    identity = get_verified_identity()
    if identity is None:
        return ANONYMOUS_ADMIN.user_id
    if identity.role in {"admin", DASHBOARD_ROLE}:
        return identity.user_id
    return authorize_resource_owner(owner).user_id


def _resource_principal(resources: Controller, resource_id: str) -> str:
    identity = get_verified_identity()
    if identity is None or identity.role != FEDERATION_PEER_ROLE:
        return _action_principal(resource_id)
    root_job = JobName.from_wire(resource_id.rpartition(":")[0] or resource_id).root_job
    if resources.received_job_from_peer(root_job, identity.user_id):
        return root_job.user
    raise ResourcePermissionDenied(f"Peer {identity.user_id!r} did not federate job {root_job}")


def _authorized_owner(requested_owner: str | None = None) -> str | None:
    identity = get_verified_identity()
    if identity is None or identity.role in {"admin", DASHBOARD_ROLE}:
        return requested_owner
    if requested_owner is not None and requested_owner != identity.user_id:
        authorize_resource_owner(requested_owner)
    return identity.user_id


def _authorize_key_owner(key: ResourceKey) -> None:
    identity = get_verified_identity()
    if identity is None or identity.role in {"admin", DASHBOARD_ROLE}:
        return
    owner = JobName.from_wire(key.resource_id.rpartition(":")[0] or key.resource_id).user
    authorize_resource_owner(owner)


def _pack(value: Message) -> any_pb2.Any:
    packed = any_pb2.Any()
    packed.Pack(value)
    return packed


def _resource_ref(
    authority_cluster_id: str,
    resource_type: str,
    resource_id: str,
    uid: str | None = None,
) -> resource_pb2.ResourceRef:
    result = resource_pb2.ResourceRef(
        authority_cluster_id=authority_cluster_id,
        type=resource_type,
        id=resource_id,
    )
    if uid is not None:
        result.uid = uid
    return result


def _resource(ref: resource_pb2.ResourceRef, body: Message) -> resource_pb2.Resource:
    return resource_pb2.Resource(ref=ref, body=_pack(body))


def _require_ref_type(ref: resource_pb2.ResourceRef, expected: str) -> None:
    if ref.type != expected:
        raise InvalidResourceRequest(f"expected resource type {expected!r}, got {ref.type!r}")
    if not ref.authority_cluster_id or not ref.id:
        raise InvalidResourceRequest("resource authority and id are required")


def _require_exact_uid(ref: resource_pb2.ResourceRef, actual_uid: str) -> None:
    if ref.HasField("uid") and ref.uid != actual_uid:
        raise ResourceReplaced(f"resource {ref.id!r} was replaced")


def _legacy_key(ref: resource_pb2.ResourceRef, kind: int) -> resource_identity_pb2.ResourceKey:
    return resource_identity_pb2.ResourceKey(
        cluster_id=ref.authority_cluster_id,
        kind=kind,
        resource_id=ref.id,
    )


def _backend_ref_id(backend_id: str, resource_id: str) -> str:
    if ":" in backend_id:
        raise ValueError("backend IDs used in ResourceRef must not contain ':'")
    return f"{backend_id}:{resource_id}"


def _parse_backend_ref_id(value: str) -> tuple[str, str]:
    backend_id, separator, resource_id = value.partition(":")
    if not separator or not backend_id or not resource_id:
        raise InvalidResourceKey("backend resource id must be '<backend>:<id>'")
    return backend_id, resource_id


def _attempt_locator(ref: resource_pb2.ResourceRef) -> resource_identity_pb2.AttemptLocator:
    task_id, separator, attempt = ref.id.rpartition(":")
    if not separator or not task_id:
        raise InvalidResourceKey("Attempt id must be '<task>:<number|current>'")
    locator = resource_identity_pb2.AttemptLocator(
        task=_legacy_key(
            _resource_ref(ref.authority_cluster_id, TASK, task_id),
            resource_identity_pb2.RESOURCE_KIND_TASK,
        )
    )
    if attempt != "current":
        if not attempt.isdecimal() or str(int(attempt)) != attempt:
            raise InvalidResourceKey("Attempt number must be canonical and non-negative")
        locator.attempt_number = int(attempt)
    return locator


def _job_ref(identity: resource_identity_pb2.JobIdentity) -> resource_pb2.ResourceRef:
    return _resource_ref(identity.key.cluster_id, JOB, identity.key.resource_id, identity.job_uid)


def _task_ref(identity: resource_identity_pb2.TaskIdentity) -> resource_pb2.ResourceRef:
    return _resource_ref(identity.key.cluster_id, TASK, identity.key.resource_id, identity.task_uid)


def _attempt_ref(identity: resource_identity_pb2.AttemptIdentity) -> resource_pb2.ResourceRef:
    return _resource_ref(
        identity.task.cluster_id,
        ATTEMPT,
        f"{identity.task.resource_id}:{identity.attempt_number}",
        identity.attempt_uid,
    )


def _node_ref(identity: resource_identity_pb2.NodeIdentity) -> resource_pb2.ResourceRef:
    return _resource_ref(
        identity.key.cluster_id,
        NODE,
        _backend_ref_id(identity.backend_id, identity.key.resource_id),
        identity.node_uid,
    )


def _slice_ref(identity: resource_identity_pb2.SliceIdentity) -> resource_pb2.ResourceRef:
    return _resource_ref(
        identity.key.cluster_id,
        SLICE,
        _backend_ref_id(identity.backend_id, identity.key.resource_id),
        identity.slice_uid,
    )


def _endpoint_ref(summary: resource_endpoint_pb2.EndpointSummary) -> resource_pb2.ResourceRef:
    return _resource_ref(summary.key.cluster_id, ENDPOINT, summary.key.resource_id)


def _operation(
    request_id: str,
    *,
    verb: str,
    requested_ref: resource_pb2.ResourceRef | None,
    resolved_ref: resource_pb2.ResourceRef | None,
    result: Message,
    phase: int = resource_pb2.OPERATION_PHASE_VERIFIED,
) -> resource_pb2.Operation:
    operation_id = request_id or uuid.uuid4().hex
    cluster_id = "system"
    if resolved_ref is not None:
        cluster_id = resolved_ref.authority_cluster_id
    elif requested_ref is not None:
        cluster_id = requested_ref.authority_cluster_id
    now = timestamp_to_proto(Timestamp.now())
    operation = resource_pb2.Operation(
        ref=_resource_ref(cluster_id, OPERATION, operation_id, operation_id),
        phase=phase,
        verb=verb,
        result=_pack(result),
        accepted_at=now,
        applied_at=now,
    )
    if requested_ref is not None:
        operation.requested_ref.CopyFrom(requested_ref)
    if resolved_ref is not None:
        operation.resolved_ref.CopyFrom(resolved_ref)
    if phase in {resource_pb2.OPERATION_PHASE_VERIFIED, resource_pb2.OPERATION_PHASE_FAILED}:
        operation.completed_at.CopyFrom(now)
    return operation


def _action_target_ref(receipt: resource_action_pb2.ActionReceipt) -> resource_pb2.ResourceRef:
    if receipt.target.kind == resource_identity_pb2.RESOURCE_KIND_JOB:
        return _resource_ref(
            receipt.target.cluster_id,
            JOB,
            receipt.target.resource_id,
            receipt.expected_target_uid,
        )
    if receipt.target.kind == resource_identity_pb2.RESOURCE_KIND_TASK:
        return _resource_ref(
            receipt.target.cluster_id,
            TASK,
            receipt.target.resource_id,
            receipt.expected_target_uid,
        )
    return _resource_ref(
        receipt.target.cluster_id,
        ATTEMPT,
        receipt.target.resource_id,
        receipt.expected_attempt_uid,
    )


def _selected_ref_from_action(
    requested_ref: resource_pb2.ResourceRef,
    receipt: resource_action_pb2.ActionReceipt,
) -> resource_pb2.ResourceRef:
    if requested_ref.type == JOB:
        return _resource_ref(
            receipt.target.cluster_id,
            JOB,
            requested_ref.id,
            receipt.expected_target_uid,
        )
    if requested_ref.type == TASK:
        task_id = receipt.target.resource_id
        if receipt.target.kind == resource_identity_pb2.RESOURCE_KIND_ATTEMPT:
            task_id, _, _ = task_id.rpartition(":")
        return _resource_ref(
            receipt.target.cluster_id,
            TASK,
            task_id,
            receipt.expected_target_uid,
        )
    if requested_ref.type == ATTEMPT:
        if not receipt.HasField("expected_attempt_number") or not receipt.expected_attempt_uid:
            raise RuntimeError("Attempt action did not record its exact target")
        task_id = receipt.target.resource_id
        if receipt.target.kind == resource_identity_pb2.RESOURCE_KIND_ATTEMPT:
            task_id, _, _ = task_id.rpartition(":")
        return _resource_ref(
            receipt.target.cluster_id,
            ATTEMPT,
            f"{task_id}:{receipt.expected_attempt_number}",
            receipt.expected_attempt_uid,
        )
    raise RuntimeError(f"action cannot resolve selected resource type {requested_ref.type!r}")


def _operation_from_action(
    requested_ref: resource_pb2.ResourceRef,
    resolved_ref: resource_pb2.ResourceRef,
    receipt: resource_action_pb2.ActionReceipt,
) -> resource_pb2.Operation:
    state_to_phase = {
        resource_action_pb2.ACTION_STATE_ACCEPTED: resource_pb2.OPERATION_PHASE_ACCEPTED,
        resource_action_pb2.ACTION_STATE_VERIFYING: resource_pb2.OPERATION_PHASE_VERIFYING,
        resource_action_pb2.ACTION_STATE_SUCCEEDED: resource_pb2.OPERATION_PHASE_APPLIED,
        resource_action_pb2.ACTION_STATE_FAILED: resource_pb2.OPERATION_PHASE_FAILED,
    }
    operation = _operation(
        receipt.action_id,
        verb="update",
        requested_ref=requested_ref,
        resolved_ref=resolved_ref,
        result=receipt,
        phase=state_to_phase[receipt.state],
    )
    if receipt.HasField("expected_attempt_number"):
        task_id = receipt.target.resource_id
        if receipt.target.kind == resource_identity_pb2.RESOURCE_KIND_ATTEMPT:
            task_id, _, _ = task_id.rpartition(":")
        operation.affected.append(
            _resource_ref(
                receipt.target.cluster_id,
                ATTEMPT,
                f"{task_id}:{receipt.expected_attempt_number}",
                receipt.expected_attempt_uid,
            )
        )
    operation.ref.CopyFrom(_resource_ref(receipt.target.cluster_id, OPERATION, receipt.action_id, receipt.action_id))
    operation.accepted_at.CopyFrom(receipt.created_at)
    operation.applied_at.CopyFrom(receipt.updated_at)
    if operation.phase in {
        resource_pb2.OPERATION_PHASE_VERIFIED,
        resource_pb2.OPERATION_PHASE_FAILED,
    } and receipt.HasField("completed_at"):
        operation.completed_at.CopyFrom(receipt.completed_at)
    return operation
