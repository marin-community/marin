# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Endpoint-wait behavior under queue delay: queue time must not consume the startup budget."""

from unittest.mock import MagicMock

import pytest
from marin.inference import iris as iris_inference


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += seconds


def _fake_job(states):
    job = MagicMock()
    job.job_id = "/test/inference-job"
    iterator = iter(states)
    last = {"state": states[0]}

    def status():
        try:
            last["state"] = next(iterator)
        except StopIteration:
            pass
        result = MagicMock()
        result.value = last["state"]
        return result

    job.status.side_effect = status
    return job


def _patch(monkeypatch, clock, endpoints=()):
    ctx = MagicMock()
    ctx.client.list_endpoint_instances.return_value = list(endpoints)
    monkeypatch.setattr(iris_inference, "iris_ctx", lambda: ctx)
    monkeypatch.setattr(iris_inference.time, "monotonic", clock.monotonic)
    monkeypatch.setattr(iris_inference.time, "sleep", clock.sleep)


def test_queue_time_does_not_consume_startup_budget(monkeypatch):
    clock = FakeClock()
    _patch(monkeypatch, clock)
    # Pending far beyond the ready timeout, then running without ever registering:
    # the ready timeout must start at the running transition, not submission.
    pending_polls = int(3000 / iris_inference._ENDPOINT_READY_POLL_SECONDS)
    job = _fake_job(["pending"] * pending_polls + ["running"] * 10_000)
    with pytest.raises(TimeoutError, match="inference endpoint"):
        iris_inference._wait_for_endpoint(job, "endpoint", timeout_seconds=100.0)
    # Ran past the queue phase plus a full startup budget.
    assert clock.now >= 3000 + 100


def test_placement_timeout_bounds_queue_wait(monkeypatch):
    clock = FakeClock()
    _patch(monkeypatch, clock)
    job = _fake_job(["pending"])
    with pytest.raises(TimeoutError, match="to be placed"):
        iris_inference._wait_for_endpoint(job, "endpoint", timeout_seconds=100.0)
    assert clock.now >= iris_inference._ENDPOINT_PLACEMENT_TIMEOUT_SECONDS


def test_requeue_resets_startup_budget(monkeypatch):
    clock = FakeClock()
    _patch(monkeypatch, clock)
    # Running long enough to eat half the budget, preempted back to pending,
    # then running again: the second run gets a fresh budget.
    half = int(50 / iris_inference._ENDPOINT_READY_POLL_SECONDS)
    job = _fake_job(["running"] * half + ["pending"] * 10 + ["running"] * 10_000)
    with pytest.raises(TimeoutError, match="inference endpoint"):
        iris_inference._wait_for_endpoint(job, "endpoint", timeout_seconds=100.0)
    assert clock.now >= 50 + 10 * iris_inference._ENDPOINT_READY_POLL_SECONDS + 100


def test_returns_endpoint_when_registered(monkeypatch):
    clock = FakeClock()
    endpoint = MagicMock()
    endpoint.address = "http://host:1234"
    endpoint.metadata = {"tensor_parallel_size": "1", "backend": "vllm"}
    _patch(monkeypatch, clock, endpoints=[endpoint])
    job = _fake_job(["running"])
    address, metadata = iris_inference._wait_for_endpoint(job, "endpoint", timeout_seconds=100.0)
    assert address == "http://host:1234"
    assert metadata["backend"] == "vllm"
