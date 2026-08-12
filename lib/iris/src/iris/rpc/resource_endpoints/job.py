# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Job endpoint operations installed by controller composition."""

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf.message import Message

from iris.cluster.controller.controller import Controller
from iris.cluster.federation.protocol import PeerCallError
from iris.resources.action import ActionKind
from iris.resources.errors import (
    ActionIdempotencyConflict,
    InvalidPageToken,
    InvalidResourceKey,
    InvalidResourceRequest,
    ResourceConflict,
    ResourceExhausted,
    ResourceNotFound,
    ResourcePermissionDenied,
    ResourcePreconditionFailed,
    ResourceReplaced,
)
from iris.resources.identity import ResourceKey, ResourceKind
from iris.resources.job import JobQuery, JobSummary
from iris.resources.state import JobState
from iris.rpc import resource_job_pb2, resource_pb2
from iris.rpc.federation_client import peer_connect_error
from iris.rpc.resource_codec import (
    action_receipt_to_proto as _action_receipt_to_proto,
)
from iris.rpc.resource_codec import (
    job_identity_to_proto as _job_identity_to_proto,
)
from iris.rpc.resource_codec import job_spec_from_proto, redacted_job_spec_to_proto, resource_spec_to_proto
from iris.rpc.resource_codec import (
    resource_key_from_proto as _resource_key_from_proto,
)
from iris.rpc.resource_codec import (
    resource_source_status_to_proto as _source_status_to_proto,
)
from iris.rpc.resource_endpoint_support import (
    _DEFAULT_JOB_PAGE_SIZE,
    _authorize_key_owner,
    _authorized_owner,
    _job_ref,
    _operation,
    _operation_from_action,
    _require_exact_uid,
    _require_ref_type,
    _resource,
    _resource_principal,
    _resource_ref,
    _selected_ref_from_action,
    _unpack,
    type_url,
)
from iris.rpc.resource_registry import ResourceWireContract
from iris.rpc.resource_types import JOB
from iris.time_proto import timestamp_to_proto


def _job_summary_to_proto(value: JobSummary) -> resource_job_pb2.JobSummary:
    result = resource_job_pb2.JobSummary(
        identity=_job_identity_to_proto(value.identity),
        owner_id=value.owner_id,
        state=value.state,
        execution_cluster_id=value.execution_cluster_id,
        backend_id=value.backend_id,
        num_tasks=value.num_tasks,
        submitted_at=timestamp_to_proto(value.submitted_at),
        error_message=value.error_message,
        pending_reason=value.pending_reason,
        resources=resource_spec_to_proto(value.resources),
    )
    if value.exit_code is not None:
        result.exit_code = value.exit_code
    if value.parent is not None:
        result.parent.CopyFrom(_job_identity_to_proto(value.parent))
    if value.started_at is not None:
        result.started_at.CopyFrom(timestamp_to_proto(value.started_at))
    if value.finished_at is not None:
        result.finished_at.CopyFrom(timestamp_to_proto(value.finished_at))
    return result


class CreateJob:
    """Create one Job through the generic resource endpoint."""

    contract = ResourceWireContract(
        body_type_urls=(type_url(resource_job_pb2.JobSummary), type_url(resource_job_pb2.JobDetail)),
        accepted_type_urls=(type_url(resource_job_pb2.CreateJob),),
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(
        self,
        request: resource_pb2.CreateResourceRequest,
        context: RequestContext,
    ) -> resource_pb2.Operation:
        body = _unpack(request.body, resource_job_pb2.CreateJob)
        try:
            identity = self._resources.submit_job(
                job_spec_from_proto(body.spec),
                body.bundle_blob,
                enforce_client_freshness=context is not None,
            )
        except (InvalidResourceKey, InvalidResourceRequest, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourcePermissionDenied as exc:
            raise ConnectError(Code.PERMISSION_DENIED, str(exc)) from exc
        except ResourcePreconditionFailed as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ResourceConflict as exc:
            raise ConnectError(Code.ALREADY_EXISTS, str(exc)) from exc
        except ResourceExhausted as exc:
            raise ConnectError(Code.RESOURCE_EXHAUSTED, str(exc)) from exc

        response = resource_job_pb2.CreatedJob(job=_job_identity_to_proto(identity))
        resolved = _job_ref(response.job)
        requested = _resource_ref(response.job.key.cluster_id, JOB, request.id or response.job.key.resource_id)
        return _operation(
            request.mutation.request_id,
            verb="create",
            requested_ref=requested,
            resolved_ref=resolved,
            result=response,
        )


class GetJob:
    """Read one current or exact Job through the generic resource endpoint."""

    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_BASIC, resource_pb2.RESOURCE_VIEW_FULL),
        body_type_urls=(type_url(resource_job_pb2.JobSummary), type_url(resource_job_pb2.JobDetail)),
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(
        self,
        request: resource_pb2.GetResourceRequest,
        _context: RequestContext,
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, JOB)
        try:
            key = ResourceKey(request.ref.authority_cluster_id, ResourceKind.JOB, request.ref.id)
            _authorize_key_owner(key)
            detail = self._resources.describe_job(key)
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc

        summary = _job_summary_to_proto(detail.summary)
        _require_exact_uid(request.ref, summary.identity.job_uid)
        body: Message
        if request.view == resource_pb2.RESOURCE_VIEW_BASIC:
            body = summary
        else:
            body = resource_job_pb2.JobDetail(summary=summary, spec=redacted_job_spec_to_proto(detail.spec))
        return resource_pb2.GetResourceResponse(resource=_resource(_job_ref(summary.identity), body))


class ListJobs:
    """List authorized Jobs through the generic resource endpoint."""

    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_BASIC, resource_pb2.RESOURCE_VIEW_FULL),
        body_type_urls=(type_url(resource_job_pb2.JobSummary),),
        accepted_type_urls=(type_url(resource_job_pb2.JobQuery),),
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(
        self,
        request: resource_pb2.ListResourcesRequest,
        _context: RequestContext,
    ) -> resource_pb2.ListResourcesResponse:
        query = _unpack(request.query, resource_job_pb2.JobQuery)
        try:
            owner_id = _authorized_owner(query.owner_id or None)
            page = self._resources.list_jobs(
                JobQuery(
                    resource_id=query.resource_id or None,
                    owner_id=owner_id,
                    parent=_resource_key_from_proto(query.parent) if query.HasField("parent") else None,
                    job_id_prefix=query.job_id_prefix or None,
                    states=frozenset(JobState(state) for state in query.states),
                    backend_id=query.backend_id or None,
                    execution_cluster_id=query.execution_cluster_id or None,
                    top_level_only=query.top_level_only,
                    page_size=request.page_size or query.page.page_size or _DEFAULT_JOB_PAGE_SIZE,
                    page_token=request.page_token or query.page.page_token or None,
                )
            )
        except (InvalidPageToken, InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        summaries = [_job_summary_to_proto(item) for item in page.items]
        return resource_pb2.ListResourcesResponse(
            resources=[_resource(_job_ref(item.identity), item) for item in summaries],
            next_page_token=page.next_page_token or "",
            source_statuses=[_source_status_to_proto(status) for status in page.source_statuses],
        )


class UpdateJob:
    """Apply one idempotent Job lifecycle update."""

    contract = ResourceWireContract(
        accepted_type_urls=(type_url(resource_job_pb2.JobUpdate),),
        features=("current-ref-v1", "exact-ref-v1"),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(
        self,
        request: resource_pb2.UpdateResourceRequest,
        _context: RequestContext,
    ) -> resource_pb2.Operation:
        _require_ref_type(request.ref, JOB)
        update = _unpack(request.update, resource_job_pb2.JobUpdate)
        if update.WhichOneof("intent") != "cancel":
            raise ConnectError(Code.INVALID_ARGUMENT, "Job update requires cancel")
        reason = request.mutation.reason or "Cancelled through the resource API"
        principal = _resource_principal(self._resources, request.ref.id)
        try:
            duplicate = self._resources.replay_action(
                principal_id=principal,
                idempotency_key=request.mutation.request_id,
                kind=ActionKind.CANCEL_JOB,
                reason=reason,
            )
            if duplicate is not None:
                receipt = _action_receipt_to_proto(duplicate)
                return _operation_from_action(
                    request.ref,
                    _selected_ref_from_action(request.ref, receipt),
                    receipt,
                )

            detail = self._resources.describe_job(
                ResourceKey(request.ref.authority_cluster_id, ResourceKind.JOB, request.ref.id)
            )
            identity = detail.summary.identity
            _require_exact_uid(request.ref, identity.job_uid)
            receipt = self._resources.cancel_job(
                identity,
                idempotency_key=request.mutation.request_id,
                reason=reason,
                principal_id=principal,
            )
        except ActionIdempotencyConflict as exc:
            raise ConnectError(Code.ALREADY_EXISTS, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except PeerCallError as exc:
            raise peer_connect_error(exc) from exc

        receipt_proto = _action_receipt_to_proto(receipt)
        return _operation_from_action(
            request.ref,
            _job_ref(_job_identity_to_proto(identity)),
            receipt_proto,
        )
