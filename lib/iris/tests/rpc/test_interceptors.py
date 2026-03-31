# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from connectrpc.errors import ConnectError

from iris.rpc.errors import extract_error_details
from iris.rpc.interceptors import RequestTimingInterceptor


@dataclass(frozen=True)
class FakeMethodInfo:
    name: str


def _make_ctx(method_name: str):
    ctx = Mock()
    ctx.method.return_value = FakeMethodInfo(name=method_name)
    return ctx


def test_interceptor_passes_through_response():
    interceptor = RequestTimingInterceptor()
    ctx = _make_ctx("FetchLogs")
    result = interceptor.intercept_unary_sync(lambda req, ctx: "ok", "request", ctx)
    assert result == "ok"


def test_interceptor_wraps_exceptions_as_connect_error_with_traceback():
    interceptor = RequestTimingInterceptor(include_traceback=True)
    ctx = _make_ctx("LaunchJob")

    def failing_handler(req, ctx):
        raise ValueError("boom")

    with pytest.raises(ConnectError, match="boom") as exc_info:
        interceptor.intercept_unary_sync(failing_handler, "request", ctx)

    error = exc_info.value
    assert error.__cause__ is not None
    assert isinstance(error.__cause__, ValueError)
    # Verify traceback details are attached
    assert len(error.details) > 0
    details = extract_error_details(error)
    assert details is not None
    assert details.traceback != ""


def test_interceptor_sanitized_by_default():
    interceptor = RequestTimingInterceptor()
    ctx = _make_ctx("LaunchJob")

    def failing_handler(req, ctx):
        raise ValueError("boom")

    with pytest.raises(ConnectError, match="boom") as exc_info:
        interceptor.intercept_unary_sync(failing_handler, "request", ctx)

    error = exc_info.value
    assert error.__cause__ is not None
    assert isinstance(error.__cause__, ValueError)
    details = extract_error_details(error)
    assert details is not None
    assert details.traceback == ""
    assert details.exception_type.endswith("ValueError")
    assert details.message != ""


def test_interceptor_passes_through_connect_errors():
    interceptor = RequestTimingInterceptor()
    ctx = _make_ctx("LaunchJob")

    original = ConnectError(code=5, message="not found")

    def failing_handler(req, ctx):
        raise original

    with pytest.raises(ConnectError) as exc_info:
        interceptor.intercept_unary_sync(failing_handler, "request", ctx)

    assert exc_info.value is original


def test_interceptor_logs_traceback_regardless_of_flag(caplog):
    """Server-side logging always includes exc_info regardless of include_traceback."""
    interceptor = RequestTimingInterceptor(include_traceback=False)
    ctx = _make_ctx("Explode")

    def failing_handler(req, ctx):
        raise RuntimeError("kaboom")

    with caplog.at_level(logging.WARNING, logger="iris.rpc.interceptors"):
        with pytest.raises(ConnectError):
            interceptor.intercept_unary_sync(failing_handler, "request", ctx)

    assert any("kaboom" in r.message and r.exc_info is not None for r in caplog.records)
