# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RPC error handling utilities with full traceback support."""

import logging
import time
import traceback
from contextlib import contextmanager
from typing import TypeVar
from collections.abc import Callable, Generator

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.protobuf.any_pb2 import Any as AnyProto

from iris.rpc import errors_pb2
from iris.time_utils import ExponentialBackoff, Timestamp

logger = logging.getLogger(__name__)

T = TypeVar("T")


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
    details = errors_pb2.ErrorDetails(
        message=message,
    )
    details.timestamp.CopyFrom(Timestamp.now().to_proto())

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
    for detail in error.details:
        if isinstance(detail, AnyProto) and "ErrorDetails" in detail.type_url:
            error_details = errors_pb2.ErrorDetails()
            detail.Unpack(error_details)
            return error_details
    return None


def format_connect_error(error: ConnectError) -> str:
    """Format a ConnectError, including server traceback if available."""
    details = extract_error_details(error)
    if details and details.traceback:
        return f"{error}\n\nServer traceback:\n{details.traceback}"
    return str(error)


def is_retryable_error(exc: Exception) -> bool:
    """Check if an RPC error should trigger retry.

    Retries on:
    - ConnectError with Code.UNAVAILABLE (controller temporarily down)
    - ConnectError with Code.INTERNAL (network errors bubble up as INTERNAL)
    - ConnectError with Code.DEADLINE_EXCEEDED (client-side httpx read timeout)

    Does not retry on:
    - Application errors (NOT_FOUND, INVALID_ARGUMENT, ALREADY_EXISTS, etc.)
    - These indicate issues with the request itself, not transient failures
    """
    if isinstance(exc, ConnectError):
        return exc.code in (Code.UNAVAILABLE, Code.INTERNAL, Code.DEADLINE_EXCEEDED)
    return False


def call_with_retry(
    operation: str,
    call_fn: Callable[[], T],
    *,
    on_retry: Callable[[Exception], None] | None = None,
    max_attempts: int = 20,
    backoff: ExponentialBackoff | None = None,
) -> T:
    """Execute an RPC call with exponential backoff retry.

    Args:
        operation: Description of the operation for logging
        call_fn: Callable that performs the RPC
        on_retry: Optional callback invoked with the exception on every retryable
            failure, including the final attempt. Useful for clearing cached
            connections so subsequent calls can re-resolve endpoints.
        max_attempts: Maximum number of attempts (default: 20)
        backoff: Backoff configuration. A fresh copy is made internally so the
            caller's instance is not mutated. Defaults to
            ExponentialBackoff(initial=0.5, maximum=10.0, factor=2.0).

    Returns:
        Result from call_fn

    Raises:
        Exception from call_fn if all retries exhausted or error is not retryable
    """
    if backoff is None:
        backoff = ExponentialBackoff(initial=0.5, maximum=10.0, factor=2.0)
    else:
        backoff = backoff.copy()
    last_exception = None

    for attempt in range(max_attempts):
        try:
            return call_fn()
        except Exception as e:
            last_exception = e
            if not is_retryable_error(e):
                # Non-retryable error, fail immediately
                raise

            # Always clear stale state on retryable errors, even on the final
            # attempt, so the next call from the caller can re-resolve.
            if on_retry is not None:
                on_retry(e)

            if attempt + 1 >= max_attempts:
                # Final attempt failed, raise
                logger.exception(
                    "Operation %s failed after %d attempts: %s",
                    operation,
                    max_attempts,
                    e,
                )
                raise

            # Log and retry
            delay = backoff.next_interval()
            logger.exception(
                "Operation %s failed (attempt %d/%d), retrying in %.2fs: %s",
                operation,
                attempt + 1,
                max_attempts,
                delay,
                e,
            )
            time.sleep(delay)

    # Should not reach here due to raise in loop, but satisfy type checker
    assert last_exception is not None
    raise last_exception
