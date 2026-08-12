# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Read-only observability operations installed by controller composition."""

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.cluster.controller.controller import Controller
from iris.resources.activity import ActivityEntry, ActivityQuery
from iris.resources.errors import InvalidPageToken, InvalidResourceKey, ResourceNotFound, ResourceReplaced
from iris.resources.identity import AttemptIdentity
from iris.resources.log import LogEntry, LogLevel, LogQuery
from iris.resources.user import UserSummary
from iris.rpc import iris_logging_pb2, resource_observability_pb2, resource_pb2
from iris.rpc.proto_display import job_state_friendly, task_state_friendly
from iris.rpc.resource_codec import (
    action_receipt_to_proto as _action_receipt_to_proto,
)
from iris.rpc.resource_codec import (
    attempt_identity_from_proto as _attempt_identity_from_proto,
)
from iris.rpc.resource_codec import (
    job_identity_from_proto as _job_identity_from_proto,
)
from iris.rpc.resource_codec import (
    resource_key_from_proto as _resource_key_from_proto,
)
from iris.rpc.resource_codec import (
    resource_key_to_proto as _resource_key_to_proto,
)
from iris.rpc.resource_codec import (
    resource_source_status_to_proto as _source_status_to_proto,
)
from iris.rpc.resource_codec import (
    task_identity_from_proto as _task_identity_from_proto,
)
from iris.rpc.resource_endpoint_support import (
    _DEFAULT_ACTIVITY_PAGE_SIZE,
    _action_principal,
    _action_target_ref,
    _authorized_owner,
    _operation_from_action,
    _require_exact_uid,
    _require_ref_type,
    _resource,
    _resource_ref,
    _unpack,
    type_url,
)
from iris.rpc.resource_registry import ResourceWireContract
from iris.rpc.resource_types import ACTIVITY_ENTRY, LOG_ENTRY, OPERATION, USER_SUMMARY
from iris.time_proto import timestamp_from_proto, timestamp_to_proto


def _user_summary_to_proto(value: UserSummary) -> resource_observability_pb2.UserSummary:
    result = resource_observability_pb2.UserSummary(
        user_id=value.user_id,
        task_state_counts={task_state_friendly(state): count for state, count in value.task_state_counts},
        job_state_counts={job_state_friendly(state): count for state, count in value.job_state_counts},
        role=value.role,
        budget_spent=value.budget_spent,
        budget_configured=value.budget_limit is not None,
    )
    if value.budget_limit is not None:
        result.budget_limit = value.budget_limit
    if value.max_band is not None:
        result.max_band = value.max_band
    return result


def _log_entry_to_proto(value: LogEntry) -> iris_logging_pb2.LogEntry:
    result = iris_logging_pb2.LogEntry(
        source=value.source,
        data=value.data,
        attempt_id=value.attempt_id,
        level=int(value.level),
        key=value.key,
        seq=value.sequence,
    )
    if value.timestamp is not None:
        result.timestamp.CopyFrom(timestamp_to_proto(value.timestamp))
    return result


def _activity_entry_to_proto(value: ActivityEntry) -> resource_observability_pb2.ActivityEntry:
    return resource_observability_pb2.ActivityEntry(
        entry_id=value.entry_id,
        occurred_at=timestamp_to_proto(value.occurred_at),
        source=value.source,
        severity=value.severity,
        kind=value.kind,
        message=value.message,
        target=_resource_key_to_proto(value.target),
        attempt_uid=value.attempt_uid or "",
        correlation_id=value.correlation_id or "",
        attributes=value.attributes,
    )


class ListUsers:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_BASIC,),
        body_type_urls=(type_url(resource_observability_pb2.UserSummary),),
        accepted_type_urls=(type_url(resource_observability_pb2.UserQuery),),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(
        self, request: resource_pb2.ListResourcesRequest, _context: RequestContext
    ) -> resource_pb2.ListResourcesResponse:
        _unpack(request.query, resource_observability_pb2.UserQuery)
        owner_id = _authorized_owner()
        users = self._resources.list_users()
        if owner_id is not None:
            users = tuple(user for user in users if user.user_id == owner_id)
        summaries = [_user_summary_to_proto(user) for user in users]
        return resource_pb2.ListResourcesResponse(
            resources=[
                _resource(_resource_ref("system", USER_SUMMARY, summary.user_id), summary) for summary in summaries
            ]
        )


class ListActivity:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_FULL,),
        body_type_urls=(type_url(resource_observability_pb2.ActivityEntry),),
        accepted_type_urls=(type_url(resource_observability_pb2.ActivityQuery),),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(
        self, request: resource_pb2.ListResourcesRequest, _context: RequestContext
    ) -> resource_pb2.ListResourcesResponse:
        query = _unpack(request.query, resource_observability_pb2.ActivityQuery)
        try:
            target = _resource_key_from_proto(query.target)
            _action_principal(target.resource_id)
            page = self._resources.list_activity(
                ActivityQuery(
                    target=target,
                    attempt_uid=query.attempt_uid or None,
                    after=timestamp_from_proto(query.after) if query.HasField("after") else None,
                    page_size=request.page_size or query.page.page_size or _DEFAULT_ACTIVITY_PAGE_SIZE,
                    page_token=request.page_token or query.page.page_token or None,
                )
            )
        except (InvalidPageToken, InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        entries = [_activity_entry_to_proto(item) for item in page.items]
        return resource_pb2.ListResourcesResponse(
            resources=[
                _resource(_resource_ref(entry.target.cluster_id, ACTIVITY_ENTRY, entry.entry_id), entry)
                for entry in entries
            ],
            next_page_token=page.next_page_token or "",
            source_statuses=[_source_status_to_proto(status) for status in page.source_statuses],
        )


class ListLogs:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_FULL,),
        body_type_urls=(type_url(iris_logging_pb2.LogEntry),),
        accepted_type_urls=(type_url(resource_observability_pb2.LogQuery),),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(
        self, request: resource_pb2.ListResourcesRequest, _context: RequestContext
    ) -> resource_pb2.ListResourcesResponse:
        query = _unpack(request.query, resource_observability_pb2.LogQuery)
        if request.page_token:
            try:
                query.filter.cursor = int(request.page_token)
            except ValueError as exc:
                raise ConnectError(Code.INVALID_ARGUMENT, "log page token must be an integer cursor") from exc
        if request.page_size:
            query.filter.max_lines = request.page_size
        try:
            target_kind = query.target.WhichOneof("target")
            if target_kind == "job":
                target = _job_identity_from_proto(query.target.job)
            elif target_kind == "task":
                target = _task_identity_from_proto(query.target.task)
            elif target_kind == "attempt":
                target = _attempt_identity_from_proto(query.target.attempt)
            else:
                raise InvalidResourceKey("log target is required")
            resource_id = target.task.resource_id if isinstance(target, AttemptIdentity) else target.key.resource_id
            _action_principal(resource_id)
            page = self._resources.fetch_logs(
                target,
                LogQuery(
                    after=timestamp_from_proto(query.filter.after) if query.filter.HasField("after") else None,
                    cursor=query.filter.cursor,
                    max_lines=query.filter.max_lines or 1_000,
                    substring=query.filter.substring,
                    minimum_level=LogLevel(query.filter.minimum_level),
                    tail=query.filter.tail,
                ),
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        key = target.task if isinstance(target, AttemptIdentity) else target.key
        entries = [_log_entry_to_proto(entry) for entry in page.entries]
        return resource_pb2.ListResourcesResponse(
            resources=[
                _resource(_resource_ref(key.cluster_id, LOG_ENTRY, f"{entry.key}:{entry.seq}"), entry)
                for entry in entries
            ],
            next_page_token=str(page.next_cursor),
            source_statuses=[_source_status_to_proto(status) for status in page.source_statuses],
        )


class GetOperation:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_FULL,),
        body_type_urls=(type_url(resource_pb2.Operation),),
        features=("durable-update-v1",),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(
        self, request: resource_pb2.GetResourceRequest, _context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, OPERATION)
        try:
            receipt = self._resources.get_action_receipt(request.ref.id)
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        _action_principal(receipt.target.resource_id)
        receipt_proto = _action_receipt_to_proto(receipt)
        _require_exact_uid(request.ref, receipt_proto.action_id)
        operation = _operation_from_action(
            _action_target_ref(receipt_proto),
            _action_target_ref(receipt_proto),
            receipt_proto,
        )
        return resource_pb2.GetResourceResponse(resource=_resource(operation.ref, operation))
