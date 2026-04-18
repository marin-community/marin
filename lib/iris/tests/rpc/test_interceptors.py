# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from connectrpc.errors import ConnectError

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
    ctx = _make_ctx("GetTaskLogs")
    result = interceptor.intercept_unary_sync(lambda req, ctx: "ok", "request", ctx)
    assert result == "ok"


def test_interceptor_wraps_exceptions_as_connect_error_with_traceback():
    interceptor = RequestTimingInterceptor()
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


def test_interceptor_passes_through_connect_errors():
    interceptor = RequestTimingInterceptor()
    ctx = _make_ctx("LaunchJob")

    original = ConnectError(code=5, message="not found")

    def failing_handler(req, ctx):
        raise original

    with pytest.raises(ConnectError) as exc_info:
        interceptor.intercept_unary_sync(failing_handler, "request", ctx)

    assert exc_info.value is original
