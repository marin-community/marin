# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import logging
import threading
import time
from dataclasses import dataclass
from unittest.mock import Mock

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from connectrpc.code import Code

from iris.rpc.errors import extract_error_details
from iris.rpc.interceptors import ConcurrencyLimitInterceptor, RequestTimingInterceptor


@dataclass(frozen=True)
class FakeMethodInfo:
    name: str


def _make_ctx(method_name: str, timeout_ms: int | None = None):
    ctx = Mock()
    ctx.method.return_value = FakeMethodInfo(name=method_name)
    ctx.timeout_ms.return_value = timeout_ms
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


def test_concurrency_limit_passes_through_unlimited_methods():
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": 1})
    ctx = _make_ctx("LaunchJob")
    result = interceptor.intercept_unary_sync(lambda req, ctx: "ok", "request", ctx)
    assert result == "ok"


def test_concurrency_limit_caps_in_flight_calls():
    limit = 3
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": limit})
    release = threading.Event()
    in_flight = 0
    peak = 0
    lock = threading.Lock()

    def handler(req, ctx):
        nonlocal in_flight, peak
        with lock:
            in_flight += 1
            peak = max(peak, in_flight)
        try:
            assert release.wait(timeout=5.0), "handler never released"
            return "ok"
        finally:
            with lock:
                in_flight -= 1

    num_callers = limit + 3

    def run():
        interceptor.intercept_unary_sync(handler, "request", _make_ctx("FetchLogs"))

    threads = [threading.Thread(target=run) for _ in range(num_callers)]
    for t in threads:
        t.start()

    # Wait for the first wave to saturate the semaphore.
    deadline = time.monotonic() + 2.0
    while time.monotonic() < deadline:
        with lock:
            if in_flight >= limit:
                break

    with lock:
        assert in_flight == limit

    release.set()
    for t in threads:
        t.join(timeout=5.0)
        assert not t.is_alive()

    assert peak == limit


def test_concurrency_limit_sheds_when_deadline_expired_before_acquire():
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": 1})
    ctx = _make_ctx("FetchLogs", timeout_ms=0)
    called = False

    def handler(req, ctx):
        nonlocal called
        called = True
        return "ok"

    with pytest.raises(ConnectError) as exc_info:
        interceptor.intercept_unary_sync(handler, "request", ctx)
    assert exc_info.value.code == Code.DEADLINE_EXCEEDED
    assert not called


def test_concurrency_limit_sheds_when_deadline_expires_during_wait():
    """Deadline check after semaphore acquire: caller queues, then deadline lapses."""
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": 1})
    release = threading.Event()

    def blocking_handler(req, ctx):
        assert release.wait(timeout=5.0)
        return "ok"

    # Hold the single slot with one thread.
    holder = threading.Thread(
        target=lambda: interceptor.intercept_unary_sync(blocking_handler, "req", _make_ctx("FetchLogs", timeout_ms=5000))
    )
    holder.start()

    # A ctx whose timeout_ms flips to 0 after the semaphore finally releases.
    expired_ctx = Mock()
    expired_ctx.method.return_value = FakeMethodInfo(name="FetchLogs")
    expired_ctx.timeout_ms.side_effect = [5000, 0]

    late_called = False

    def late_handler(req, ctx):
        nonlocal late_called
        late_called = True
        return "ok"

    result: list = []

    def run_late():
        try:
            interceptor.intercept_unary_sync(late_handler, "req", expired_ctx)
        except ConnectError as e:
            result.append(e)

    late = threading.Thread(target=run_late)
    late.start()
    # Let the late caller block on the semaphore, then free the holder.
    time.sleep(0.1)
    release.set()
    holder.join(timeout=5.0)
    late.join(timeout=5.0)

    assert len(result) == 1
    assert result[0].code == Code.DEADLINE_EXCEEDED
    assert not late_called


def test_concurrency_limit_releases_slot_on_exception():
    interceptor = ConcurrencyLimitInterceptor({"FetchLogs": 1})

    def boom(req, ctx):
        raise ValueError("nope")

    ctx = _make_ctx("FetchLogs")
    for _ in range(3):
        with pytest.raises(ValueError):
            interceptor.intercept_unary_sync(boom, "request", ctx)
    # If the slot leaked, the fourth call would deadlock — a successful
    # passthrough proves the semaphore was released on each exception.
    assert interceptor.intercept_unary_sync(lambda req, ctx: "ok", "request", ctx) == "ok"


# --- Stats collector integration -------------------------------------------


class _RecordingCollector:
    """Stub that captures every record() call for assertions."""

    def __init__(self):
        self.calls: list[dict] = []

    def record(self, *, method, duration_ms, request, ctx, error_code="", error_message=""):
        self.calls.append(
            {
                "method": method,
                "duration_ms": duration_ms,
                "error_code": error_code,
                "error_message": error_message,
            }
        )


def test_interceptor_records_successful_call():
    collector = _RecordingCollector()
    interceptor = RequestTimingInterceptor(collector=collector)
    ctx = _make_ctx("ListJobs")

    result = interceptor.intercept_unary_sync(lambda req, ctx: "ok", "request", ctx)

    assert result == "ok"
    assert len(collector.calls) == 1
    call = collector.calls[0]
    assert call["method"] == "ListJobs"
    assert call["error_code"] == ""
    assert call["duration_ms"] >= 0


def test_interceptor_records_connect_error_with_code_name():
    collector = _RecordingCollector()
    interceptor = RequestTimingInterceptor(collector=collector)
    ctx = _make_ctx("ListJobs")

    def boom(req, ctx):
        raise ConnectError(code=Code.NOT_FOUND, message="not found")

    with pytest.raises(ConnectError):
        interceptor.intercept_unary_sync(boom, "request", ctx)

    assert len(collector.calls) == 1
    assert collector.calls[0]["error_code"] == "NOT_FOUND"
    assert "not found" in collector.calls[0]["error_message"]


def test_interceptor_records_unhandled_exception_as_internal():
    collector = _RecordingCollector()
    interceptor = RequestTimingInterceptor(collector=collector)
    ctx = _make_ctx("LaunchJob")

    def boom(req, ctx):
        raise RuntimeError("kaboom")

    with pytest.raises(ConnectError):
        interceptor.intercept_unary_sync(boom, "request", ctx)

    assert len(collector.calls) == 1
    assert collector.calls[0]["error_code"] == "INTERNAL"
    assert collector.calls[0]["error_message"] == "kaboom"
