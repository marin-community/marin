# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""RPC error handling utilities with full traceback support."""

import traceback
from contextlib import contextmanager
from collections.abc import Generator

from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.time_utils import Timestamp


@contextmanager
def rpc_error_handler(
    operation: str,
    code: Code = Code.INTERNAL,
) -> Generator[None, None, None]:
    """Context manager for consistent RPC error handling with full tracebacks.

    Usage:
        with rpc_error_handler("launching job"):
            # code that might fail
            docker.run(...)

    Args:
        operation: Description of what operation is being performed (for error message)
        code: ConnectRPC error code to use (default: INTERNAL)
    """
    try:
        yield
    except ConnectError:
        # Don't wrap ConnectErrors - they already have proper codes/messages
        raise
    except Exception as e:
        raise connect_error_with_traceback(code, f"Error {operation}: {e}", exc=e) from e


def connect_error_with_traceback(
    code: Code,
    message: str,
    exc: Exception | None = None,
) -> ConnectError:
    """Create a ConnectError with full traceback details attached.

    Args:
        code: ConnectRPC error code
        message: Human-readable error message
        exc: Exception to extract traceback from (uses current if None)
    """
    # Import here to avoid circular import during module load
    from iris.rpc import errors_pb2

    details = errors_pb2.ErrorDetails(
        message=message,
    )
    details.timestamp.epoch_ms = Timestamp.now().to_proto()

    if exc is not None:
        details.exception_type = f"{type(exc).__module__}.{type(exc).__name__}"
        details.traceback = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    else:
        details.traceback = traceback.format_exc()

    # Pass details to constructor (ConnectError takes Iterable[Message])
    return ConnectError(code, message, details=[details])


def format_exception_with_traceback(exc: Exception) -> str:
    """Format an exception with its full traceback as a string.

    Suitable for embedding in JobStatus.error field.
    """
    tb = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    return f"{type(exc).__name__}: {exc}\n\nTraceback:\n{tb}"


def extract_error_details(error: ConnectError):
    """Extract ErrorDetails from a ConnectError if present.

    Useful for clients that want to programmatically access traceback info.

    Returns:
        ErrorDetails proto if found, None otherwise
    """
    from iris.rpc import errors_pb2

    for detail in error.details:
        # Details are wrapped in google.protobuf.Any
        if hasattr(detail, "type_url") and "ErrorDetails" in detail.type_url:
            error_details = errors_pb2.ErrorDetails()
            detail.Unpack(error_details)
            return error_details
    return None
