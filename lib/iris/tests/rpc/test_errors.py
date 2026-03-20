# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import time

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.rpc.errors import (
    call_with_retry,
    connect_error_sanitized,
    connect_error_with_traceback,
    extract_error_details,
)
from iris.time_utils import ExponentialBackoff


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

    start = time.monotonic()
    result = call_with_retry("test_op", retry_then_succeed)
    elapsed = time.monotonic() - start

    assert result == "success"
    assert call_count == 3
    # Should have some delay due to backoff (initial is 0.1s)
    assert elapsed >= 0.1


def test_call_with_retry_fails_after_max_attempts() -> None:
    """call_with_retry should give up after max_attempts."""

    def always_fail():
        raise ConnectError(Code.UNAVAILABLE, "Always down")

    with pytest.raises(ConnectError) as exc_info:
        call_with_retry("test_op", always_fail, max_attempts=3)

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
