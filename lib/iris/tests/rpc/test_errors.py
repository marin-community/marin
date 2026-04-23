# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0


import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.rpc.errors import (
    call_with_retry,
    connect_error_sanitized,
    connect_error_with_traceback,
    extract_error_details,
    poll_with_retries,
)
from rigging.timing import Deadline, ExponentialBackoff


def test_connect_error_with_traceback_populates_timestamp() -> None:
    try:
        raise ValueError("boom")
    except ValueError as exc:
        err = connect_error_with_traceback(Code.INTERNAL, "Error launching job", exc=exc)

    details = extract_error_details(err)
    assert details is not None
    assert details.exception_type.endswith("ValueError")
    assert details.timestamp.epoch_ms > 0


def test_connect_error_sanitized_omits_traceback() -> None:
    try:
        raise ValueError("boom")
    except ValueError as exc:
        err = connect_error_sanitized(Code.INTERNAL, "Error launching job", exc=exc)

    details = extract_error_details(err)
    assert details is not None
    assert details.traceback == ""
    assert details.message == "Error launching job"
    assert details.exception_type.endswith("ValueError")
    assert details.timestamp.epoch_ms > 0


def test_connect_error_sanitized_without_exception() -> None:
    err = connect_error_sanitized(Code.NOT_FOUND, "Job not found")
    details = extract_error_details(err)
    assert details is not None
    assert details.traceback == ""
    assert details.message == "Job not found"
    assert details.exception_type == ""


def test_call_with_retry_succeeds_first_attempt() -> None:
    """call_with_retry should succeed on first attempt if no error."""
    call_count = 0

    def success_fn():
        nonlocal call_count
        call_count += 1
        return "success"

    result = call_with_retry("test_op", success_fn)
    assert result == "success"
    assert call_count == 1


def test_call_with_retry_retries_on_unavailable() -> None:
    """call_with_retry should retry on UNAVAILABLE errors."""
    call_count = 0

    def retry_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectError(Code.UNAVAILABLE, "Service down")
        return "success"

    result = call_with_retry("test_op", retry_then_succeed, backoff=ExponentialBackoff(initial=0.01, maximum=0.05))

    assert result == "success"
    assert call_count == 3


def test_call_with_retry_fails_after_max_attempts() -> None:
    """call_with_retry should give up after max_attempts."""

    def always_fail():
        raise ConnectError(Code.UNAVAILABLE, "Always down")

    with pytest.raises(ConnectError) as exc_info:
        call_with_retry("test_op", always_fail, max_attempts=3, backoff=ExponentialBackoff(initial=0.01, maximum=0.05))

    assert exc_info.value.code == Code.UNAVAILABLE


def test_call_with_retry_retries_on_deadline_exceeded() -> None:
    call_count = 0

    def retry_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectError(Code.DEADLINE_EXCEEDED, "Request timed out")
        return "success"

    result = call_with_retry("test_op", retry_then_succeed, backoff=ExponentialBackoff(initial=0.001, maximum=0.001))
    assert result == "success"
    assert call_count == 3


def test_call_with_retry_retries_on_resource_exhausted() -> None:
    call_count = 0

    def retry_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectError(Code.RESOURCE_EXHAUSTED, "shed by concurrency limiter")
        return "success"

    result = call_with_retry("test_op", retry_then_succeed, backoff=ExponentialBackoff(initial=0.001, maximum=0.001))
    assert result == "success"
    assert call_count == 3


def test_call_with_retry_no_retry_on_not_found() -> None:
    """call_with_retry should not retry on non-retryable errors."""
    call_count = 0

    def not_found():
        nonlocal call_count
        call_count += 1
        raise ConnectError(Code.NOT_FOUND, "Not found")

    with pytest.raises(ConnectError) as exc_info:
        call_with_retry("test_op", not_found)

    assert exc_info.value.code == Code.NOT_FOUND
    # Should only call once (no retries)
    assert call_count == 1


def test_call_with_retry_max_elapsed_stops_retrying() -> None:
    """call_with_retry should stop retrying after max_elapsed seconds."""
    call_count = 0

    def always_fail():
        nonlocal call_count
        call_count += 1
        raise ConnectError(Code.UNAVAILABLE, "Always down")

    with pytest.raises(ConnectError):
        call_with_retry(
            "test_op",
            always_fail,
            max_attempts=1000,
            max_elapsed=0.5,
            backoff=ExponentialBackoff(initial=0.05, maximum=0.1),
        )

    # Should have retried several times within the 0.5s window, but not all 1000.
    assert 2 <= call_count <= 30


def test_call_with_retry_max_elapsed_succeeds_within_window() -> None:
    """call_with_retry should succeed if the call recovers within max_elapsed."""
    call_count = 0

    def fail_then_succeed():
        nonlocal call_count
        call_count += 1
        if call_count <= 3:
            raise ConnectError(Code.UNAVAILABLE, "Temporarily down")
        return "recovered"

    result = call_with_retry(
        "test_op",
        fail_then_succeed,
        max_attempts=1000,
        max_elapsed=5.0,
        backoff=ExponentialBackoff(initial=0.01, maximum=0.05),
    )
    assert result == "recovered"
    assert call_count == 4


# -- poll_with_retries tests --


def test_poll_with_retries_succeeds_immediately() -> None:
    result = poll_with_retries(
        "test",
        lambda: "ok",
        deadline=Deadline.from_seconds(5.0),
    )
    assert result == "ok"


def test_poll_with_retries_retries_then_succeeds() -> None:
    call_count = 0

    def flaky():
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectError(Code.UNAVAILABLE, "down")
        return "recovered"

    result = poll_with_retries(
        "test",
        flaky,
        deadline=Deadline.from_seconds(5.0),
        backoff=ExponentialBackoff(initial=0.01, maximum=0.05),
    )
    assert result == "recovered"
    assert call_count == 3


def test_poll_with_retries_respects_deadline() -> None:
    """Deadline expiry during unavailability raises TimeoutError, not the RPC error."""

    def always_fail():
        raise ConnectError(Code.UNAVAILABLE, "down")

    with pytest.raises(TimeoutError, match="deadline expired"):
        poll_with_retries(
            "test",
            always_fail,
            deadline=Deadline.from_seconds(0.3),
            unavailable_tolerance=3600.0,
            backoff=ExponentialBackoff(initial=0.01, maximum=0.05),
        )


def test_poll_with_retries_respects_unavailable_tolerance() -> None:
    """Unavailability tolerance expiry re-raises the RPC error."""

    def always_fail():
        raise ConnectError(Code.UNAVAILABLE, "down")

    with pytest.raises(ConnectError) as exc_info:
        poll_with_retries(
            "test",
            always_fail,
            deadline=Deadline.from_seconds(10.0),
            unavailable_tolerance=0.3,
            backoff=ExponentialBackoff(initial=0.01, maximum=0.05),
        )
    assert exc_info.value.code == Code.UNAVAILABLE


def test_poll_with_retries_raises_non_retryable_immediately() -> None:
    call_count = 0

    def not_found():
        nonlocal call_count
        call_count += 1
        raise ConnectError(Code.NOT_FOUND, "gone")

    with pytest.raises(ConnectError) as exc_info:
        poll_with_retries(
            "test",
            not_found,
            deadline=Deadline.from_seconds(5.0),
        )
    assert exc_info.value.code == Code.NOT_FOUND
    assert call_count == 1
