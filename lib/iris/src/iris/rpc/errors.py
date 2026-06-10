# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""RPC error handling utilities with full traceback support."""

import logging
import time
import traceback
from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TypeVar

from connectrpc.code import Code
from connectrpc.errors import ConnectError
from google.protobuf.any_pb2 import Any as AnyProto
from rigging.timing import Deadline, ExponentialBackoff, Timestamp, retry_with_backoff

from iris.rpc import errors_pb2
from iris.time_proto import timestamp_to_proto

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


def connect_error_sanitized(
    code: Code,
    message: str,
    exc: Exception | None = None,
) -> ConnectError:
    """Create a ConnectError WITHOUT traceback details. For production use."""
    details = errors_pb2.ErrorDetails(message=message)
    details.timestamp.CopyFrom(timestamp_to_proto(Timestamp.now()))
    if exc is not None:
        details.exception_type = f"{type(exc).__module__}.{type(exc).__name__}"
    return ConnectError(code, message, details=[details])


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
    details.timestamp.CopyFrom(timestamp_to_proto(Timestamp.now()))

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
    - ConnectError with Code.RESOURCE_EXHAUSTED (server-side load shed; safe to
      retry because the handler did not run)

    Does not retry on:
    - Application errors (NOT_FOUND, INVALID_ARGUMENT, ALREADY_EXISTS, etc.)
    - These indicate issues with the request itself, not transient failures
    """
    if isinstance(exc, ConnectError):
        return exc.code in (
            Code.UNAVAILABLE,
            Code.INTERNAL,
            Code.DEADLINE_EXCEEDED,
            Code.RESOURCE_EXHAUSTED,
        )
    return False


# Default retry budget: tolerate up to 30 minutes of transient controller
# unavailability. The controller can stall for several minutes under heavy
# load; a short budget (~3 min) caused clients — notably the
# ExecutorStep → GetJobStatus polling loop — to crash even though the job
# was still running server-side. See issue #4913.
DEFAULT_RETRY_MAX_ELAPSED = 1800.0
DEFAULT_RETRY_MAX_ATTEMPTS = 240


def call_with_retry(
    operation: str,
    call_fn: Callable[[], T],
    *,
    on_retry: Callable[[Exception], None] | None = None,
    max_attempts: int = DEFAULT_RETRY_MAX_ATTEMPTS,
    max_elapsed: float | None = DEFAULT_RETRY_MAX_ELAPSED,
    backoff: ExponentialBackoff | None = None,
) -> T:
    """Execute an RPC call with exponential backoff retry.

    Retries stop when either ``max_attempts`` is exhausted **or**
    ``max_elapsed`` seconds have passed, whichever comes first. Defaults
    tolerate ~30 minutes of transient controller unavailability so callers
    survive heavy-load stalls without losing track of long-running jobs.

    Args:
        operation: Description of the operation for logging
        call_fn: Callable that performs the RPC
        on_retry: Optional callback invoked with the exception on every retryable
            failure, including the final attempt. Useful for clearing cached
            connections so subsequent calls can re-resolve endpoints.
        max_attempts: Maximum number of attempts (default: 240; secondary cap —
            ``max_elapsed`` is the authoritative wall-clock budget).
        max_elapsed: Maximum wall-clock seconds to keep retrying. ``None``
            means no time limit (only ``max_attempts`` is used). Default:
            1800s (30 min) — long enough to ride out controller restarts.
        backoff: Backoff configuration. A fresh copy is made internally so the
            caller's instance is not mutated. Defaults to
            ExponentialBackoff(initial=0.5, maximum=10.0, factor=2.0).

    Returns:
        Result from call_fn

    Raises:
        Exception from call_fn if all retries exhausted or error is not retryable
    """
    wrapped_on_retry: Callable[[Exception, int], None] | None = None
    if on_retry is not None:

        def wrapped_on_retry(exc: Exception, _attempt: int) -> None:
            assert on_retry is not None
            on_retry(exc)

    return retry_with_backoff(
        call_fn,
        retryable=is_retryable_error,
        max_attempts=max_attempts,
        max_elapsed=max_elapsed,
        backoff=backoff,
        on_retry=wrapped_on_retry,
        operation=operation,
    )


def poll_with_retries(
    operation: str,
    poll_fn: Callable[[], T],
    *,
    deadline: Deadline,
    unavailable_tolerance: float = 3600.0,
    backoff: ExponentialBackoff | None = None,
) -> T:
    """Poll an RPC endpoint, tolerating transient unavailability.

    Calls ``poll_fn`` in a loop.  On retryable errors the function backs off
    and keeps trying for up to ``unavailable_tolerance`` seconds **or** until
    ``deadline`` expires — whichever comes first.  When the call succeeds the
    unavailability timer resets.

    This is designed for monitoring loops (e.g. ``wait_for_job``) where the
    server-side work continues regardless of client polling failures.

    Args:
        operation: Human-readable description for log messages.
        poll_fn: Callable that performs the RPC.  Should raise on failure.
        deadline: Caller-supplied deadline — polling stops with ``TimeoutError``
            if the deadline expires, even during unavailability.
        unavailable_tolerance: Maximum seconds to tolerate continuous
            controller unavailability before re-raising the RPC error.
        backoff: Backoff for unavailability retries.  Defaults to 1 s → 60 s.

    Returns:
        The successful result of ``poll_fn``.

    Raises:
        TimeoutError: If *deadline* expires while the controller is unavailable.
        Exception: The last RPC error if unavailability exceeds the tolerance,
            or any non-retryable error from ``poll_fn``.
    """

    if backoff is None:
        backoff = ExponentialBackoff(initial=1.0, maximum=60.0, factor=2.0)
    else:
        backoff = backoff.copy()

    unavailable_since: float | None = None

    while True:
        try:
            result = poll_fn()
        except Exception as e:
            if not is_retryable_error(e):
                raise

            now = time.monotonic()
            if unavailable_since is None:
                unavailable_since = now
            elapsed_unavailable = now - unavailable_since

            if elapsed_unavailable >= unavailable_tolerance:
                logger.error(
                    "Controller unavailable for %.0fs, giving up on %s",
                    elapsed_unavailable,
                    operation,
                )
                raise

            if deadline.expired():
                raise TimeoutError(
                    f"{operation}: deadline expired after {elapsed_unavailable:.0f}s of controller unavailability"
                ) from e

            logger.warning(
                "Controller unavailable for %s (%.0fs), job is still running server-side: %s",
                operation,
                elapsed_unavailable,
                e,
            )
            interval = backoff.next_interval()
            time.sleep(min(interval, deadline.remaining_seconds()))
            continue

        # Success — reset unavailability tracking.
        if unavailable_since is not None:
            elapsed_unavailable = time.monotonic() - unavailable_since
            logger.info(
                "Controller back online for %s after %.0fs of unavailability",
                operation,
                elapsed_unavailable,
            )
            unavailable_since = None
            backoff.reset()

        return result
