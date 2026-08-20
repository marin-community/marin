# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registered Attempt command operations owned by the controller."""

from connectrpc.request import RequestContext

from iris.cluster.controller.controller import Controller
from iris.cluster.controller.resource_operations.support import (
    _operation,
    _require_ref_type,
    _resource_principal,
)
from iris.rpc import resource_command_pb2, resource_pb2
from iris.rpc.resource_codec import (
    attempt_identity_from_proto as _attempt_identity_from_proto,
)
from iris.rpc.resource_codec import profile_configuration_from_proto
from iris.rpc.resource_registry import ResourceWireContract
from iris.rpc.resource_types import ATTEMPT
from iris.time_proto import duration_from_proto


class CreateExecSession:
    contract = ResourceWireContract(
        body_types=(resource_command_pb2.ExecSessionResult,),
        input_type=resource_command_pb2.CreateExecSession,
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(
        self,
        request: resource_pb2.CreateResourceRequest,
        body: resource_command_pb2.CreateExecSession,
        _context: RequestContext,
    ) -> resource_pb2.Operation:
        _require_ref_type(request.parent, ATTEMPT)
        identity = _attempt_identity_from_proto(body.attempt)
        _resource_principal(self._resources, identity.task.resource_id)
        result = self._resources.exec_attempt(
            identity,
            tuple(body.command),
            duration_from_proto(body.timeout) if body.HasField("timeout") else None,
        )
        response = resource_command_pb2.ExecSessionResult(
            exit_code=result.exit_code,
            stdout=result.stdout,
            stderr=result.stderr,
            error_message=result.error_message,
        )
        return _operation(
            request.mutation.request_id,
            verb="create",
            requested_ref=request.parent,
            resolved_ref=request.parent,
            result=response,
        )


class CreateProfileCapture:
    contract = ResourceWireContract(
        body_types=(resource_command_pb2.ProfileCaptureResult,),
        input_type=resource_command_pb2.CreateProfileCapture,
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def run(
        self,
        request: resource_pb2.CreateResourceRequest,
        body: resource_command_pb2.CreateProfileCapture,
        _context: RequestContext,
    ) -> resource_pb2.Operation:
        _require_ref_type(request.parent, ATTEMPT)
        identity = _attempt_identity_from_proto(body.attempt)
        _resource_principal(self._resources, identity.task.resource_id)
        result = self._resources.profile_attempt(
            identity,
            profile_configuration_from_proto(body.profile),
            duration_from_proto(body.duration) if body.HasField("duration") else None,
        )
        response = resource_command_pb2.ProfileCaptureResult(
            profile_data=result.profile_data,
            error_message=result.error_message,
        )
        return _operation(
            request.mutation.request_id,
            verb="create",
            requested_ref=request.parent,
            resolved_ref=request.parent,
            result=response,
        )
