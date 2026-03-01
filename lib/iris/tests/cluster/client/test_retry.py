# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for call_with_retry logic, exercised directly without RPC mocks."""

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.rpc.errors import call_with_retry


def test_retries_on_unavailable():
    attempts = 0

    def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise ConnectError(Code.UNAVAILABLE, "down")
        return "ok"

    result = call_with_retry("test", flaky, initial_backoff=0.01, max_backoff=0.01)
    assert result == "ok"
    assert attempts == 3


def test_retries_on_internal():
    attempts = 0

    def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectError(Code.INTERNAL, "connection reset")
        return 42

    result = call_with_retry("test", flaky, initial_backoff=0.01, max_backoff=0.01)
    assert result == 42
    assert attempts == 2


def test_no_retry_on_not_found():
    attempts = 0

    def always_not_found():
        nonlocal attempts
        attempts += 1
        raise ConnectError(Code.NOT_FOUND, "gone")

    with pytest.raises(ConnectError) as exc_info:
        call_with_retry("test", always_not_found)

    assert exc_info.value.code == Code.NOT_FOUND
    assert attempts == 1


def test_exhausts_max_attempts():
    attempts = 0

    def always_unavailable():
        nonlocal attempts
        attempts += 1
        raise ConnectError(Code.UNAVAILABLE, "down")

    with pytest.raises(ConnectError) as exc_info:
        call_with_retry("test", always_unavailable, max_attempts=3, initial_backoff=0.01, max_backoff=0.01)

    assert exc_info.value.code == Code.UNAVAILABLE
    assert attempts == 3


def test_on_retry_callback_invoked():
    retried_errors: list[Exception] = []
    attempts = 0

    def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectError(Code.UNAVAILABLE, "down")
        return "ok"

    result = call_with_retry(
        "test",
        flaky,
        on_retry=lambda e: retried_errors.append(e),
        initial_backoff=0.01,
        max_backoff=0.01,
    )
    assert result == "ok"
    assert len(retried_errors) == 1
    assert isinstance(retried_errors[0], ConnectError)


def test_retries_on_deadline_exceeded():
    attempts = 0

    def flaky():
        nonlocal attempts
        attempts += 1
        if attempts < 2:
            raise ConnectError(Code.DEADLINE_EXCEEDED, "timeout")
        return "done"

    result = call_with_retry("test", flaky, initial_backoff=0.01, max_backoff=0.01)
    assert result == "done"
    assert attempts == 2
