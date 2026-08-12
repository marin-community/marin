# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registered Task and Attempt operations owned by the controller."""

from connectrpc.request import RequestContext
from google.protobuf.message import Message

from iris.cluster.controller.controller import Controller
from iris.cluster.controller.resource_operations.support import (
    _DEFAULT_RESOURCE_PAGE_SIZE,
    _attempt_locator,
    _attempt_ref,
    _authorize_key_owner,
    _authorized_owner,
    _operation_from_action,
    _require_exact_uid,
    _require_ref_type,
    _resource,
    _resource_principal,
    _selected_ref_from_action,
    _task_ref,
)
from iris.resources.action import ActionKind, ActionReceipt
from iris.resources.attempt import AttemptSummary
from iris.resources.errors import (
    InvalidResourceRequest,
    ResourcePermissionDenied,
    ResourcePreconditionFailed,
    ResourceReplaced,
)
from iris.resources.identity import AttemptIdentity, ResourceKey, ResourceKind, TaskIdentity
from iris.resources.state import TaskState
from iris.resources.task import TaskDetail, TaskQuery, TaskSummary
from iris.rpc import resource_pb2, resource_task_pb2
from iris.rpc.resource_codec import (
    action_receipt_to_proto as _action_receipt_to_proto,
)
from iris.rpc.resource_codec import (
    attempt_identity_to_proto as _attempt_identity_to_proto,
)
from iris.rpc.resource_codec import attempt_locator_from_proto
from iris.rpc.resource_codec import (
    job_identity_to_proto as _job_identity_to_proto,
)
from iris.rpc.resource_codec import (
    node_identity_to_proto as _node_identity_to_proto,
)
from iris.rpc.resource_codec import (
    resource_key_from_proto as _resource_key_from_proto,
)
from iris.rpc.resource_codec import (
    resource_source_status_to_proto as _source_status_to_proto,
)
from iris.rpc.resource_codec import (
    task_identity_to_proto as _task_identity_to_proto,
)
from iris.rpc.resource_registry import ResourceWireContract
from iris.rpc.resource_types import ATTEMPT, TASK
from iris.time_proto import timestamp_to_proto


def _task_summary_to_proto(value: TaskSummary) -> resource_task_pb2.TaskSummary:
    result = resource_task_pb2.TaskSummary(
        identity=_task_identity_to_proto(value.identity),
        job=_job_identity_to_proto(value.job),
        task_index=value.task_index,
        state=value.state,
        execution_cluster_id=value.execution_cluster_id,
        backend_id=value.backend_id,
        failure_count=value.failure_count,
        preemption_count=value.preemption_count,
        submitted_at=timestamp_to_proto(value.submitted_at),
        status_message=value.status_message,
        error_message=value.error_message,
    )
    if value.current_attempt is not None:
        result.current_attempt.CopyFrom(_attempt_identity_to_proto(value.current_attempt))
    if value.current_node is not None:
        result.current_node.CopyFrom(_node_identity_to_proto(value.current_node))
    if value.started_at is not None:
        result.started_at.CopyFrom(timestamp_to_proto(value.started_at))
    if value.finished_at is not None:
        result.finished_at.CopyFrom(timestamp_to_proto(value.finished_at))
    return result


def _attempt_summary_to_proto(value: AttemptSummary) -> resource_task_pb2.AttemptSummary:
    result = resource_task_pb2.AttemptSummary(
        identity=_attempt_identity_to_proto(value.identity),
        state=value.state,
        execution_cluster_id=value.execution_cluster_id,
        backend_id=value.backend_id,
        created_at=timestamp_to_proto(value.created_at),
        error_message=value.error_message,
        terminal_reason=value.terminal_reason,
    )
    if value.node is not None:
        result.node.CopyFrom(_node_identity_to_proto(value.node))
    if value.started_at is not None:
        result.started_at.CopyFrom(timestamp_to_proto(value.started_at))
    if value.finished_at is not None:
        result.finished_at.CopyFrom(timestamp_to_proto(value.finished_at))
    if value.exit_code is not None:
        result.exit_code = value.exit_code
    return result


def _task_detail_to_proto(value: TaskDetail) -> resource_task_pb2.TaskDetail:
    return resource_task_pb2.TaskDetail(
        summary=_task_summary_to_proto(value.summary),
        attempts=[_attempt_summary_to_proto(item) for item in value.attempts],
        source_statuses=[_source_status_to_proto(status) for status in value.source_statuses],
        root_cause_highlights=value.root_cause_highlights,
    )


def _attempt_detail_to_proto(detail) -> resource_task_pb2.AttemptDetail:
    result = resource_task_pb2.AttemptDetail(
        summary=_attempt_summary_to_proto(detail.summary),
        source_statuses=[_source_status_to_proto(status) for status in detail.source_statuses],
    )
    if detail.runtime is not None:
        result.runtime.CopyFrom(
            resource_task_pb2.AttemptRuntimeObject(
                provider_kind=detail.runtime.provider_kind,
                namespace=detail.runtime.namespace,
                name=detail.runtime.name,
                provider_uid=detail.runtime.provider_uid,
                provider_node_id=detail.runtime.provider_node_id,
                provider_node_uid=detail.runtime.provider_node_uid,
                container_id=detail.runtime.container_id,
                observed_at=timestamp_to_proto(detail.runtime.observed_at),
            )
        )
    return result


def _replayed_operation(
    resources: Controller,
    request: resource_pb2.UpdateResourceRequest,
    *,
    kind: ActionKind,
    reason: str,
) -> resource_pb2.Operation | None:
    receipt = resources.replay_action(
        principal_id=_resource_principal(resources, request.ref.id),
        idempotency_key=request.mutation.request_id,
        kind=kind,
        reason=reason,
    )
    if receipt is None:
        return None
    receipt_proto = _action_receipt_to_proto(receipt)
    return _operation_from_action(
        request.ref,
        _selected_ref_from_action(request.ref, receipt_proto),
        receipt_proto,
    )


def _attempt_action_kind_and_reason(
    intent: str | None,
    requested_reason: str,
    resource_name: str,
) -> tuple[ActionKind, str]:
    if intent == "preempt":
        return ActionKind.RETRY_TASK, requested_reason or "Requested through the resource API"
    if intent == "terminate":
        return ActionKind.TERMINATE_ATTEMPT, requested_reason or "Requested through the resource API"
    if intent == "fail":
        return ActionKind.FAIL_ATTEMPT, requested_reason or "Failed through the resource API"
    raise InvalidResourceRequest(f"{resource_name} update requires preempt, terminate, or fail")


def _apply_attempt_action(
    resources: Controller,
    attempt: AttemptIdentity,
    retry_task: TaskIdentity | None,
    *,
    kind: ActionKind,
    request: resource_pb2.UpdateResourceRequest,
    reason: str,
    principal: str,
) -> ActionReceipt:
    if kind is ActionKind.TERMINATE_ATTEMPT:
        return resources.terminate_attempt(
            attempt,
            idempotency_key=request.mutation.request_id,
            reason=reason,
            principal_id=principal,
        )
    if kind is ActionKind.FAIL_ATTEMPT:
        return resources.fail_attempt(
            attempt,
            idempotency_key=request.mutation.request_id,
            reason=reason,
            principal_id=principal,
        )
    if kind is ActionKind.RETRY_TASK:
        task = retry_task if retry_task is not None else resources.describe_task(attempt.task).summary.identity
        return resources.retry_task(
            task,
            expected_attempt_uid=attempt.attempt_uid,
            idempotency_key=request.mutation.request_id,
            reason=reason,
            principal_id=principal,
        )
    raise AssertionError(f"unsupported Attempt action kind: {kind}")


class GetTask:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_BASIC, resource_pb2.RESOURCE_VIEW_FULL),
        body_types=(resource_task_pb2.TaskSummary, resource_task_pb2.TaskDetail),
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(self, request: resource_pb2.GetResourceRequest, _context: RequestContext):
        _require_ref_type(request.ref, TASK)
        key = ResourceKey(request.ref.authority_cluster_id, ResourceKind.TASK, request.ref.id)
        _authorize_key_owner(key)
        detail = self._resources.describe_task(key)
        proto = _task_detail_to_proto(detail)
        _require_exact_uid(request.ref, proto.summary.identity.task_uid)
        body: Message = proto.summary if request.view == resource_pb2.RESOURCE_VIEW_BASIC else proto
        return resource_pb2.GetResourceResponse(
            resource=_resource(_task_ref(proto.summary.identity), body),
            source_statuses=proto.source_statuses,
        )


class ListTasks:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_BASIC, resource_pb2.RESOURCE_VIEW_FULL),
        body_types=(resource_task_pb2.TaskSummary,),
        input_type=resource_task_pb2.TaskQuery,
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(
        self,
        request: resource_pb2.ListResourcesRequest,
        query: resource_task_pb2.TaskQuery,
        _context: RequestContext,
    ):
        job = _resource_key_from_proto(query.job) if query.HasField("job") else None
        if job is not None:
            _authorize_key_owner(job)
        owner_id = _authorized_owner()
        job_id_prefix = query.job_id_prefix or None
        if owner_id is not None:
            owner_prefix = f"/{owner_id}/"
            if job_id_prefix is not None and not job_id_prefix.startswith(owner_prefix):
                raise ResourcePermissionDenied(f"User {owner_id!r} cannot list Tasks outside {owner_prefix!r}")
            job_id_prefix = job_id_prefix or owner_prefix
        page = self._resources.list_tasks(
            TaskQuery(
                job=job,
                job_id_prefix=job_id_prefix,
                states=frozenset(TaskState(state) for state in query.states),
                backend_id=query.backend_id or None,
                authority_cluster_id=query.authority_cluster_id or None,
                execution_cluster_id=query.execution_cluster_id or None,
                page_size=request.page_size or query.page.page_size or _DEFAULT_RESOURCE_PAGE_SIZE,
                page_token=request.page_token or query.page.page_token or None,
            )
        )
        summaries = [_task_summary_to_proto(item) for item in page.items]
        return resource_pb2.ListResourcesResponse(
            resources=[_resource(_task_ref(item.identity), item) for item in summaries],
            next_page_token=page.next_page_token or "",
            source_statuses=[_source_status_to_proto(status) for status in page.source_statuses],
        )


class BatchGetTasks:
    contract = GetTask.contract

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(self, request: resource_pb2.BatchGetResourcesRequest, _context: RequestContext):
        for ref in request.refs:
            _require_ref_type(ref, TASK)
        keys = tuple(ResourceKey(ref.authority_cluster_id, ResourceKind.TASK, ref.id) for ref in request.refs)
        for key in keys:
            _authorize_key_owner(key)
        details = self._resources.describe_tasks(keys)
        results: list[resource_pb2.BatchGetResourceResult] = []
        for ref, detail in zip(request.refs, details, strict=True):
            proto = _task_detail_to_proto(detail)
            _require_exact_uid(ref, proto.summary.identity.task_uid)
            body: Message = proto.summary if request.view == resource_pb2.RESOURCE_VIEW_BASIC else proto
            results.append(
                resource_pb2.BatchGetResourceResult(resource=_resource(_task_ref(proto.summary.identity), body))
            )
        return resource_pb2.BatchGetResourcesResponse(results=results)


class UpdateTask:
    contract = ResourceWireContract(
        input_type=resource_task_pb2.TaskUpdate,
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(
        self,
        request: resource_pb2.UpdateResourceRequest,
        update: resource_task_pb2.TaskUpdate,
        _context: RequestContext,
    ):
        _require_ref_type(request.ref, TASK)
        kind, reason = _attempt_action_kind_and_reason(
            update.WhichOneof("intent"),
            request.mutation.reason,
            "Task",
        )
        replay = _replayed_operation(self._resources, request, kind=kind, reason=reason)
        if replay is not None:
            return replay

        principal = _resource_principal(self._resources, request.ref.id)
        detail = self._resources.describe_task(
            ResourceKey(request.ref.authority_cluster_id, ResourceKind.TASK, request.ref.id)
        )
        summary = detail.summary
        _require_exact_uid(request.ref, summary.identity.task_uid)
        attempt = summary.current_attempt
        if attempt is None:
            raise ResourcePreconditionFailed("Task has no current Attempt")
        if update.expected_attempt_uid and update.expected_attempt_uid != attempt.attempt_uid:
            raise ResourceReplaced("Task's current Attempt was replaced")
        receipt = _apply_attempt_action(
            self._resources,
            attempt,
            summary.identity,
            kind=kind,
            request=request,
            reason=reason,
            principal=principal,
        )
        receipt_proto = _action_receipt_to_proto(receipt)
        return _operation_from_action(
            request.ref,
            _task_ref(_task_identity_to_proto(summary.identity)),
            receipt_proto,
        )


class GetAttempt:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_FULL,),
        body_types=(resource_task_pb2.AttemptDetail,),
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(self, request: resource_pb2.GetResourceRequest, _context: RequestContext):
        _require_ref_type(request.ref, ATTEMPT)
        locator = attempt_locator_from_proto(_attempt_locator(request.ref))
        _authorize_key_owner(locator.task)
        detail = self._resources.describe_attempt(locator)
        proto = _attempt_detail_to_proto(detail)
        _require_exact_uid(request.ref, proto.summary.identity.attempt_uid)
        return resource_pb2.GetResourceResponse(
            resource=_resource(_attempt_ref(proto.summary.identity), proto),
            source_statuses=proto.source_statuses,
        )


class UpdateAttempt:
    contract = ResourceWireContract(
        input_type=resource_task_pb2.AttemptUpdate,
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(
        self,
        request: resource_pb2.UpdateResourceRequest,
        update: resource_task_pb2.AttemptUpdate,
        _context: RequestContext,
    ):
        _require_ref_type(request.ref, ATTEMPT)
        kind, reason = _attempt_action_kind_and_reason(
            update.WhichOneof("intent"),
            request.mutation.reason,
            "Attempt",
        )
        replay = _replayed_operation(self._resources, request, kind=kind, reason=reason)
        if replay is not None:
            return replay

        principal = _resource_principal(self._resources, request.ref.id)
        detail = self._resources.describe_attempt(attempt_locator_from_proto(_attempt_locator(request.ref)))
        identity = detail.summary.identity
        _require_exact_uid(request.ref, identity.attempt_uid)
        receipt = _apply_attempt_action(
            self._resources,
            identity,
            None,
            kind=kind,
            request=request,
            reason=reason,
            principal=principal,
        )
        receipt_proto = _action_receipt_to_proto(receipt)
        return _operation_from_action(
            request.ref,
            _attempt_ref(_attempt_identity_to_proto(identity)),
            receipt_proto,
        )
