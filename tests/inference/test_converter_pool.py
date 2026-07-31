# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Behavior of the brokered CPU converter pool.

These run the real broker, the real proxy (via its synchronous ``forward_raw_request`` API, no
uvicorn), and real ``serve_leases`` loops on threads -- only the Iris job plumbing is absent. What
is under test is the pool's contract: single-slot leasing, the readiness probe answering without a
handler, handler failures becoming error envelopes rather than lost leases, and dead converters
being respawned into their slots.
"""

import json
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass

import marin.inference.converter_pool as converter_pool
import pytest
from fray.types import ResourceConfig, create_environment
from marin.inference.broker import InferenceBroker
from marin.inference.config import BrokerConfig
from marin.inference.converter_pool import ConverterPoolConfig, serve_leases
from marin.inference.proxy import InferenceProxy
from marin.inference.types import InferenceRequest, InferenceResponse
from rigging.timing import ExponentialBackoff

_MODEL_ID = "test-converter"
_FAST_POLL = ExponentialBackoff(initial=0.01, maximum=0.05, factor=1.5)


def _payload_response(request: InferenceRequest, payload: bytes) -> InferenceResponse:
    return InferenceResponse(
        request_id=request.request_id,
        status_code=200,
        payload=payload,
        headers=(("content-type", "application/octet-stream"),),
    )


@contextmanager
def _running_pool(handlers):
    """A broker, one serve_leases thread per handler, and a proxy over the same broker."""
    broker = InferenceBroker(request_lease_timeout_seconds=300.0)
    stop = threading.Event()
    threads = [
        threading.Thread(
            target=serve_leases,
            args=(broker, handler, _MODEL_ID),
            kwargs={"stop_event": stop, "backoff": _FAST_POLL},
            name=f"test-converter-{index}",
        )
        for index, handler in enumerate(handlers)
    ]
    with InferenceProxy(
        broker=broker,
        model=_MODEL_ID,
        request_timeout_seconds=10.0,
        readiness_timeout_seconds=10.0,
        max_pending_requests=16,
        response_fetch_batch_size=8,
    ) as proxy:
        for thread in threads:
            thread.start()
        try:
            yield proxy
        finally:
            stop.set()
            for thread in threads:
                thread.join()


def _forward(proxy: InferenceProxy, path: str, body: bytes = b"", method: str = "POST"):
    return proxy.forward_raw_request(path, body, method=method, query_string="", headers={}, timeout_seconds=10.0)


def test_a_converter_answers_a_brokered_convert_request() -> None:
    def handler(request: InferenceRequest) -> InferenceResponse:
        return _payload_response(request, request.payload.upper())

    with _running_pool([handler]) as proxy:
        response = _forward(proxy, "/v1/convert", b"pdf bytes")

    assert response.status_code == 200
    assert response.body == b"PDF BYTES"


def test_the_models_probe_is_answered_without_reaching_the_handler() -> None:
    """Readiness must not depend on the handler understanding /v1/models.

    The proxy's startup probe is a real GET /v1/models through the broker; if it reached a
    conversion handler the whole fleet would fail readiness with a 404 and never start.
    """

    def handler(request: InferenceRequest) -> InferenceResponse:
        raise AssertionError("the models probe must not reach the handler")

    with _running_pool([handler]) as proxy:
        response = _forward(proxy, "/v1/models", method="GET")

    assert response.status_code == 200
    listed = json.loads(response.body)
    assert [entry["id"] for entry in listed["data"]] == [_MODEL_ID]


def test_a_raising_handler_becomes_an_error_envelope_and_the_converter_survives() -> None:
    """An exception escaping the handler is a handler bug, not a reason to lose the lease.

    The lease must get an explicit error response (otherwise the request is redelivered until the
    proxy times out), and the lease loop must keep serving afterwards.
    """
    calls = {"count": 0}

    def handler(request: InferenceRequest) -> InferenceResponse:
        calls["count"] += 1
        if calls["count"] == 1:
            raise ValueError("boom")
        return _payload_response(request, b"recovered")

    with _running_pool([handler]) as proxy:
        first = _forward(proxy, "/v1/convert", b"poison")
        second = _forward(proxy, "/v1/convert", b"fine")

    assert first.status_code == 502
    assert "converter handler raised" in json.loads(first.body)["error"]["message"]
    assert second.status_code == 200
    assert second.body == b"recovered"


def test_two_single_slot_converters_serve_two_documents_concurrently() -> None:
    """A busy converter must not hold queued work hostage.

    Both handlers block on a shared barrier, so the barrier can only be crossed if each converter
    leased exactly one of the two requests -- a converter that leased both could never serve them
    at the same time.
    """
    barrier = threading.Barrier(3)

    def handler(request: InferenceRequest) -> InferenceResponse:
        barrier.wait(timeout=10.0)
        return _payload_response(request, b"served")

    broker = InferenceBroker(request_lease_timeout_seconds=300.0)
    stop = threading.Event()
    threads = [
        threading.Thread(
            target=serve_leases,
            args=(broker, handler, _MODEL_ID),
            kwargs={"stop_event": stop, "backoff": _FAST_POLL},
        )
        for _ in range(2)
    ]
    for thread in threads:
        thread.start()
    try:
        for index in range(2):
            broker.submit_request(
                InferenceRequest(request_id=f"req-{index}", method="POST", path="/v1/convert", payload=b"doc")
            )
        barrier.wait(timeout=10.0)
        deadline = time.monotonic() + 10.0
        responses = []
        while len(responses) < 2 and time.monotonic() < deadline:
            responses.extend(broker.fetch_responses(max_items=8))
            time.sleep(0.01)
    finally:
        stop.set()
        for thread in threads:
            thread.join()

    assert sorted(response.request_id for response in responses) == ["req-0", "req-1"]


def test_dead_converters_are_respawned_in_their_slots(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(converter_pool, "_RESPAWN_DELAY_SECONDS", 0.0)

    @dataclass
    class _FakeProcess:
        """The slice of subprocess.Popen the supervisor reads: poll() and returncode."""

        returncode: int | None

        def poll(self) -> int | None:
            return self.returncode

    spawned: list[int] = []

    def spawn(slot: int) -> _FakeProcess:
        spawned.append(slot)
        return _FakeProcess(returncode=None)

    children = {
        0: _FakeProcess(returncode=None),
        1: _FakeProcess(returncode=-11),
        2: _FakeProcess(returncode=1),
    }
    survivor = children[0]

    respawned = converter_pool._respawn_dead(children, spawn)

    assert respawned == [1, 2]
    assert spawned == [1, 2]
    assert children[0] is survivor
    assert children[1].poll() is None and children[2].poll() is None


def test_an_unpicklable_handler_factory_is_rejected_at_config_time() -> None:
    """The factory crosses a spawn boundary on a pod; a lambda would fail there, hours in."""
    with pytest.raises(ValueError, match="picklable"):
        ConverterPoolConfig(
            handler_factory=lambda: None,  # type: ignore[arg-type, return-value]
            model_id=_MODEL_ID,
            instances=1,
            processes_per_instance=1,
            worker_resources=ResourceConfig(cpu=1, ram="1g", disk="1g"),
            worker_environment=create_environment(),
            broker=BrokerConfig(),
        )
