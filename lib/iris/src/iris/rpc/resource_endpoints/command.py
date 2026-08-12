# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Attempt command operations installed by controller composition."""

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from connectrpc.request import RequestContext

from iris.cluster.controller.controller import Controller
from iris.cluster.federation.protocol import PeerCallError
from iris.resources.errors import InvalidResourceKey, ResourceNotFound, ResourceReplaced, ResourceSourceUnavailable
from iris.rpc import resource_command_pb2, resource_pb2
from iris.rpc.federation_client import peer_connect_error
from iris.rpc.resource_codec import (
    attempt_identity_from_proto as _attempt_identity_from_proto,
)
from iris.rpc.resource_codec import profile_configuration_from_proto
from iris.rpc.resource_endpoint_support import (
    _operation,
    _require_ref_type,
    _resource_principal,
    _unpack,
    type_url,
)
from iris.rpc.resource_registry import ResourceWireContract
from iris.rpc.resource_types import ATTEMPT
from iris.time_proto import duration_from_proto


class CreateExecSession:
    contract = ResourceWireContract(
        body_type_urls=(
            type_url(resource_command_pb2.CreateExecSession),
            type_url(resource_command_pb2.ExecSessionResult),
        ),
        accepted_type_urls=(type_url(resource_command_pb2.CreateExecSession),),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(self, request: resource_pb2.CreateResourceRequest, _context: RequestContext) -> resource_pb2.Operation:
        _require_ref_type(request.parent, ATTEMPT)
        body = _unpack(request.body, resource_command_pb2.CreateExecSession)
        try:
            identity = _attempt_identity_from_proto(body.attempt)
            _resource_principal(self._resources, identity.task.resource_id)
            result = self._resources.exec_attempt(
                identity,
                tuple(body.command),
                duration_from_proto(body.timeout) if body.HasField("timeout") else None,
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ResourceSourceUnavailable as exc:
            raise ConnectError(Code.UNAVAILABLE, str(exc)) from exc
        except PeerCallError as exc:
            raise peer_connect_error(exc) from exc
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
        body_type_urls=(
            type_url(resource_command_pb2.CreateProfileCapture),
            type_url(resource_command_pb2.ProfileCaptureResult),
        ),
        accepted_type_urls=(type_url(resource_command_pb2.CreateProfileCapture),),
    )

    def __init__(self, resources: Controller) -> None:
        self._resources = resources

    def __call__(self, request: resource_pb2.CreateResourceRequest, _context: RequestContext) -> resource_pb2.Operation:
        _require_ref_type(request.parent, ATTEMPT)
        body = _unpack(request.body, resource_command_pb2.CreateProfileCapture)
        try:
            identity = _attempt_identity_from_proto(body.attempt)
            _resource_principal(self._resources, identity.task.resource_id)
            result = self._resources.profile_attempt(
                identity,
                profile_configuration_from_proto(body.profile),
                duration_from_proto(body.duration) if body.HasField("duration") else None,
            )
        except (InvalidResourceKey, ValueError) as exc:
            raise ConnectError(Code.INVALID_ARGUMENT, str(exc)) from exc
        except ResourceNotFound as exc:
            raise ConnectError(Code.NOT_FOUND, str(exc)) from exc
        except ResourceReplaced as exc:
            raise ConnectError(Code.FAILED_PRECONDITION, str(exc)) from exc
        except ResourceSourceUnavailable as exc:
            raise ConnectError(Code.UNAVAILABLE, str(exc)) from exc
        except PeerCallError as exc:
            raise peer_connect_error(exc) from exc
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
