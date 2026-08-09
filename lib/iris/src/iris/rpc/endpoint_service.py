# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Legacy EndpointService protobuf adapter over the native endpoint registry."""

from typing import Any

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.controller.endpoint_registry import (
    EndpointAttemptStale,
    EndpointRegistry,
    EndpointTaskNotFound,
    EndpointTaskTerminal,
)
from iris.cluster.types import JobName
from iris.rpc import controller_pb2, job_pb2
from iris.time_proto import duration_from_proto, duration_to_proto


class EndpointServiceImpl:
    """Encode legacy EndpointService calls around native registry operations."""

    def __init__(self, registry: EndpointRegistry) -> None:
        self._registry = registry

    @property
    def registry(self) -> EndpointRegistry:
        return self._registry

    def register_endpoint(
        self,
        request: controller_pb2.Controller.RegisterEndpointRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.RegisterEndpointResponse:
        del ctx
        task_id = JobName.from_wire(request.task_id)
        requested_lease = duration_from_proto(request.lease_duration) if request.HasField("lease_duration") else None
        try:
            endpoint_id, granted = self._registry.register_endpoint(
                endpoint_id=request.endpoint_id or None,
                name=request.name,
                address=request.address,
                task_id=task_id,
                attempt_id=request.attempt_id,
                metadata=dict(request.metadata),
                access=request.access,
                requested_lease=requested_lease,
            )
        except EndpointTaskNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, f"Task {request.task_id} not found") from exc
        except EndpointTaskTerminal as exc:
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                f"Task {request.task_id} is already terminal; endpoint not registered",
            ) from exc
        except EndpointAttemptStale as exc:
            raise ConnectError(
                Code.FAILED_PRECONDITION,
                f"Task {request.task_id} attempt {request.attempt_id} is no longer current",
            ) from exc
        return controller_pb2.Controller.RegisterEndpointResponse(
            endpoint_id=endpoint_id,
            lease_duration=duration_to_proto(granted),
        )

    def unregister_endpoint(
        self,
        request: controller_pb2.Controller.UnregisterEndpointRequest,
        ctx: Any,
    ) -> job_pb2.Empty:
        del ctx
        self._registry.unregister_endpoint(request.endpoint_id)
        return job_pb2.Empty()

    def list_endpoints(
        self,
        request: controller_pb2.Controller.ListEndpointsRequest,
        ctx: Any,
    ) -> controller_pb2.Controller.ListEndpointsResponse:
        del ctx
        if request.prefix.startswith("/system/"):
            endpoints = [
                controller_pb2.Controller.Endpoint(endpoint_id=name, name=name, address=address)
                for name, address in self._registry.system_endpoints()
                if (name == request.prefix if request.exact else name.startswith(request.prefix))
            ]
            return controller_pb2.Controller.ListEndpointsResponse(endpoints=endpoints)

        rows = self._registry.list_endpoint_rows(
            prefix=request.prefix,
            exact=request.exact,
            task_ids=tuple(JobName.from_wire(value) for value in request.task_ids),
        )
        return controller_pb2.Controller.ListEndpointsResponse(
            endpoints=[
                controller_pb2.Controller.Endpoint(
                    endpoint_id=row.endpoint_id,
                    name=row.name,
                    address=row.address,
                    task_id=row.task_id.to_wire(),
                    metadata=row.metadata,
                    access=row.access,
                    peer_id=row.peer_id or "",
                )
                for row in rows
            ]
        )
