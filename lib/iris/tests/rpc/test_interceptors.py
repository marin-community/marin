# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from unittest.mock import Mock

import pytest

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


def test_interceptor_reraises_exceptions():
    interceptor = RequestTimingInterceptor()
    ctx = _make_ctx("LaunchJob")

    def failing_handler(req, ctx):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        interceptor.intercept_unary_sync(failing_handler, "request", ctx)
