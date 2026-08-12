# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Endpoint resource operations installed by controller composition."""

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext
from google.protobuf.message import Message
from rigging.server_auth import get_verified_identity

from iris.cluster.controller.auth import WORKER_ROLE
from iris.cluster.controller.controller import Controller
from iris.resources.endpoint import EndpointDetail, EndpointQuery, EndpointSummary
from iris.resources.errors import (
    InvalidPageToken,
    InvalidResourceKey,
    ResourceNotFound,
    ResourcePermissionDenied,
)
from iris.resources.identity import ResourceKey, ResourceKind
from iris.rpc import resource_endpoint_pb2, resource_pb2
from iris.rpc.auth import DASHBOARD_ROLE
from iris.rpc.resource_codec import endpoint_access_to_proto
from iris.rpc.resource_codec import (
    resource_key_from_proto as _resource_key_from_proto,
)
from iris.rpc.resource_codec import (
    resource_key_to_proto as _resource_key_to_proto,
)
from iris.rpc.resource_codec import (
    resource_source_status_to_proto as _source_status_to_proto,
)
from iris.rpc.resource_endpoint_support import (
    _DEFAULT_RESOURCE_PAGE_SIZE,
    _authorize_key_owner,
    _authorized_owner,
    _endpoint_ref,
    _operation,
    _require_ref_type,
    _resource,
    _unpack,
    type_url,
)
from iris.rpc.resource_registry import ResourceWireContract
from iris.rpc.resource_types import ENDPOINT
from iris.time_proto import duration_from_proto, timestamp_to_proto


def _endpoint_summary_to_proto(value: EndpointSummary) -> resource_endpoint_pb2.EndpointSummary:
    result = resource_endpoint_pb2.EndpointSummary(
        key=_resource_key_to_proto(value.key),
        endpoint_id=value.endpoint_id,
        name=value.name,
        execution_cluster_id=value.execution_cluster_id,
        access=endpoint_access_to_proto(value.access),
    )
    if value.task is not None:
        result.task.CopyFrom(_resource_key_to_proto(value.task))
    if value.lease_deadline is not None:
        result.lease_deadline.CopyFrom(timestamp_to_proto(value.lease_deadline))
    return result


def _endpoint_detail_to_proto(value: EndpointDetail) -> resource_endpoint_pb2.EndpointDetail:
    return resource_endpoint_pb2.EndpointDetail(
        summary=_endpoint_summary_to_proto(value.summary),
        address=value.address,
        metadata=value.metadata,
    )


def _authorize_endpoint_owner(detail: EndpointDetail) -> None:
    if detail.summary.task is not None:
        _authorize_key_owner(detail.summary.task)
        return
    identity = get_verified_identity()
    if identity is not None and identity.role not in {"admin", DASHBOARD_ROLE}:
        raise ConnectError(Code.PERMISSION_DENIED, "System Endpoints require administrator access")


def _authorize_endpoint_read(detail: EndpointDetail) -> None:
    if detail.summary.task is not None:
        _authorize_key_owner(detail.summary.task)
        return
    identity = get_verified_identity()
    if identity is not None and identity.role == WORKER_ROLE:
        return
    _authorize_endpoint_owner(detail)


class GetEndpoint:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_BASIC, resource_pb2.RESOURCE_VIEW_FULL),
        body_type_urls=(type_url(resource_endpoint_pb2.EndpointSummary), type_url(resource_endpoint_pb2.EndpointDetail)),
        features=("current-ref-v1",),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(
        self, request: resource_pb2.GetResourceRequest, _context: RequestContext
    ) -> resource_pb2.GetResourceResponse:
        _require_ref_type(request.ref, ENDPOINT)
        if request.ref.HasField("uid"):
            raise ConnectError(Code.FAILED_PRECONDITION, "Endpoint exact identity is not available")
        try:
            detail = self._resources.describe_endpoint(
                ResourceKey(request.ref.authority_cluster_id, ResourceKind.ENDPOINT, request.ref.id)
            )
            _authorize_endpoint_read(detail)
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        proto = _endpoint_detail_to_proto(detail)
        body: Message = proto.summary if request.view == resource_pb2.RESOURCE_VIEW_BASIC else proto
        return resource_pb2.GetResourceResponse(resource=_resource(_endpoint_ref(proto.summary), body))


class ListEndpoints:
    contract = ResourceWireContract(
        views=(resource_pb2.RESOURCE_VIEW_BASIC, resource_pb2.RESOURCE_VIEW_FULL),
        body_type_urls=(type_url(resource_endpoint_pb2.EndpointSummary),),
        accepted_type_urls=(type_url(resource_endpoint_pb2.EndpointQuery),),
        features=("current-ref-v1",),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(
        self, request: resource_pb2.ListResourcesRequest, _context: RequestContext
    ) -> resource_pb2.ListResourcesResponse:
        query = _unpack(request.query, resource_endpoint_pb2.EndpointQuery)
        try:
            task = _resource_key_from_proto(query.task) if query.HasField("task") else None
            identity = get_verified_identity()
            worker_system_lookup = identity is not None and identity.role == WORKER_ROLE
            if worker_system_lookup and task is not None:
                raise ConnectError(Code.PERMISSION_DENIED, "Workers may only resolve system Endpoints")
            if task is not None:
                _authorize_key_owner(task)
            page = self._resources.list_endpoints(
                EndpointQuery(
                    name_prefix=query.name_prefix or None,
                    task=task,
                    owner_id=None if worker_system_lookup else _authorized_owner(),
                    page_size=request.page_size or query.page.page_size or _DEFAULT_RESOURCE_PAGE_SIZE,
                    page_token=request.page_token or query.page.page_token or None,
                    system_only=worker_system_lookup,
                )
            )
        except (InvalidPageToken, InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        summaries = [_endpoint_summary_to_proto(item) for item in page.items]
        return resource_pb2.ListResourcesResponse(
            resources=[_resource(_endpoint_ref(item), item) for item in summaries],
            next_page_token=page.next_page_token or "",
            source_statuses=[_source_status_to_proto(status) for status in page.source_statuses],
        )


class BatchGetEndpoints:
    contract = GetEndpoint.contract

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(
        self, request: resource_pb2.BatchGetResourcesRequest, _context: RequestContext
    ) -> resource_pb2.BatchGetResourcesResponse:
        for ref in request.refs:
            _require_ref_type(ref, ENDPOINT)
            if ref.HasField("uid"):
                raise ConnectError(Code.FAILED_PRECONDITION, "Endpoint exact identity is not available")
        try:
            details = self._resources.describe_endpoints(
                tuple(ResourceKey(ref.authority_cluster_id, ResourceKind.ENDPOINT, ref.id) for ref in request.refs)
            )
            for detail in details:
                _authorize_endpoint_read(detail)
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        results: list[resource_pb2.BatchGetResourceResult] = []
        for detail in details:
            proto = _endpoint_detail_to_proto(detail)
            body: Message = proto.summary if request.view == resource_pb2.RESOURCE_VIEW_BASIC else proto
            results.append(resource_pb2.BatchGetResourceResult(resource=_resource(_endpoint_ref(proto.summary), body)))
        return resource_pb2.BatchGetResourcesResponse(results=results)


class CreateEndpointCapability:
    contract = ResourceWireContract(
        body_type_urls=(
            type_url(resource_endpoint_pb2.CreateEndpointCapability),
            type_url(resource_endpoint_pb2.EndpointCapability),
        ),
        accepted_type_urls=(type_url(resource_endpoint_pb2.CreateEndpointCapability),),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(self, request: resource_pb2.CreateResourceRequest, _context: RequestContext) -> resource_pb2.Operation:
        _require_ref_type(request.parent, ENDPOINT)
        body = _unpack(request.body, resource_endpoint_pb2.CreateEndpointCapability)
        try:
            match body.WhichOneof("selector"):
                case "endpoint":
                    key = _resource_key_from_proto(body.endpoint)
                    detail = self._resources.describe_endpoint(key)
                case "endpoint_name":
                    page = self._resources.list_endpoints(
                        EndpointQuery(name_prefix=body.endpoint_name, page_size=_DEFAULT_RESOURCE_PAGE_SIZE)
                    )
                    summary = next((item for item in page.items if item.name == body.endpoint_name), None)
                    if summary is None:
                        raise ResourceNotFound(body.endpoint_name)
                    key = summary.key
                    detail = self._resources.describe_endpoint(key)
                case _:
                    raise ValueError("Endpoint capability selector is required")
            _authorize_endpoint_owner(detail)
            token = self._resources.mint_endpoint_token(
                key,
                duration_from_proto(body.ttl) if body.HasField("ttl") else None,
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourcePermissionDenied as exc:
            raise ConnectError(Code.PERMISSION_DENIED, str(exc)) from exc
        result = resource_endpoint_pb2.EndpointCapability(
            token=token.token,
            expires_at=timestamp_to_proto(token.expires_at),
            capability_url=token.capability_url,
        )
        return _operation(
            request.mutation.request_id,
            verb="create",
            requested_ref=request.parent,
            resolved_ref=_endpoint_ref(_endpoint_summary_to_proto(detail.summary)),
            result=result,
        )
