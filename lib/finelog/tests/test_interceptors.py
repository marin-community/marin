# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for the finelog server interceptors."""

from __future__ import annotations

import asyncio
import time

import pytest
from finelog.server.interceptors import SlowRpcInterceptor


class _FakeCtx:
    def __init__(self, method_name: str) -> None:
        self._name = method_name

    def method(self):  # mimic connectrpc RequestContext shape
        class _Method:
            name = self._name  # type: ignore[misc]

        # Capture method name in a local for the inner class.
        name = self._name
        return type("M", (), {"name": name})()

    def timeout_ms(self):
        return None


def test_slow_rpc_interceptor_logs_when_over_threshold(caplog):
    interceptor = SlowRpcInterceptor({"Query": 5})

    def slow_handler(request, ctx):
        time.sleep(0.05)
        return "ok"

    with caplog.at_level("WARNING"):
        result = interceptor.intercept_unary_sync(slow_handler, "req", _FakeCtx("Query"))

    assert result == "ok"
    assert any("Slow RPC Query" in r.message for r in caplog.records), caplog.records


def test_slow_rpc_interceptor_silent_when_under_threshold(caplog):
    # Generous threshold; fast handler should produce no WARNING.
    interceptor = SlowRpcInterceptor({"Query": 5000})

    def fast_handler(request, ctx):
        return "ok"

    with caplog.at_level("WARNING"):
        result = interceptor.intercept_unary_sync(fast_handler, "req", _FakeCtx("Query"))

    assert result == "ok"
    assert not any("Slow RPC" in r.message for r in caplog.records), caplog.records


def test_slow_rpc_interceptor_uses_default_for_unlisted_methods(caplog):
    interceptor = SlowRpcInterceptor({"FetchLogs": 5000}, default_threshold_ms=5)

    def slow_handler(request, ctx):
        time.sleep(0.05)
        return "ok"

    with caplog.at_level("WARNING"):
        interceptor.intercept_unary_sync(slow_handler, "req", _FakeCtx("PushLogs"))

    assert any("Slow RPC PushLogs" in r.message for r in caplog.records)


def test_slow_rpc_interceptor_zero_threshold_disables_method(caplog):
    interceptor = SlowRpcInterceptor({"PushLogs": 0})

    def slow_handler(request, ctx):
        time.sleep(0.05)
        return "ok"

    with caplog.at_level("WARNING"):
        interceptor.intercept_unary_sync(slow_handler, "req", _FakeCtx("PushLogs"))

    assert not any("Slow RPC" in r.message for r in caplog.records)


def test_slow_rpc_interceptor_async_path(caplog):
    interceptor = SlowRpcInterceptor({"Query": 5})

    async def slow_handler(request, ctx):
        await asyncio.sleep(0.05)
        return "ok"

    with caplog.at_level("WARNING"):
        result = asyncio.run(interceptor.intercept_unary(slow_handler, "req", _FakeCtx("Query")))

    assert result == "ok"
    assert any("Slow RPC Query" in r.message for r in caplog.records)


@pytest.mark.parametrize("method", ["Query", "WriteRows", "PushLogs", "FetchLogs"])
def test_slow_rpc_interceptor_handles_finelog_method_names(method, caplog):
    interceptor = SlowRpcInterceptor(default_threshold_ms=5)

    def slow_handler(request, ctx):
        time.sleep(0.02)
        return "ok"

    with caplog.at_level("WARNING"):
        interceptor.intercept_unary_sync(slow_handler, "req", _FakeCtx(method))

    assert any(f"Slow RPC {method}" in r.message for r in caplog.records)
