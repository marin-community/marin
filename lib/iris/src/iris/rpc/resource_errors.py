# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Translate native resource failures at Connect RPC boundaries."""

from collections.abc import Callable
from typing import TypeVar

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.resources.errors import (
    ActionIdempotencyConflict,
    ActionPolicyRejected,
    BackendIdentityUnknown,
    InvalidPageToken,
    InvalidResourceKey,
    InvalidResourceRequest,
    ResourceConflict,
    ResourceError,
    ResourceExhausted,
    ResourceNotFound,
    ResourcePermissionDenied,
    ResourcePreconditionFailed,
    ResourceReplaced,
    ResourceSourceUnavailable,
    UnsupportedResourceVerb,
)

_T = TypeVar("_T")


def resource_call(operation: Callable[[], _T]) -> _T:
    """Run one native operation and expose its failure as Connect status."""
    try:
        return operation()
    except (InvalidPageToken, InvalidResourceKey, InvalidResourceRequest) as error:
        raise ConnectError(Code.INVALID_ARGUMENT, str(error)) from error
    except ResourcePermissionDenied as error:
        raise ConnectError(Code.PERMISSION_DENIED, str(error)) from error
    except ResourceNotFound as error:
        raise ConnectError(Code.NOT_FOUND, str(error)) from error
    except (ResourceReplaced, ResourcePreconditionFailed, ActionPolicyRejected, BackendIdentityUnknown) as error:
        raise ConnectError(Code.FAILED_PRECONDITION, str(error)) from error
    except (ResourceConflict, ActionIdempotencyConflict) as error:
        raise ConnectError(Code.ALREADY_EXISTS, str(error)) from error
    except ResourceExhausted as error:
        raise ConnectError(Code.RESOURCE_EXHAUSTED, str(error)) from error
    except ResourceSourceUnavailable as error:
        raise ConnectError(Code.UNAVAILABLE, str(error)) from error
    except UnsupportedResourceVerb as error:
        raise ConnectError(Code.UNIMPLEMENTED, str(error)) from error
    except ResourceError as error:
        raise ConnectError(Code.INTERNAL, str(error)) from error
