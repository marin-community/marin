# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace
from types import SimpleNamespace

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from marin.inference import iris_vllm


def test_healthy_follower_outlives_rendezvous_deadline(monkeypatch):
    coordinator = iris_vllm.VllmCoordinatorActor("10.0.0.1")
    launch = iris_vllm.IrisVllmLaunch(
        task_index=1,
        num_tasks=2,
        extra_cli_args=(),
        host_ip="10.0.0.2",
        gloo_interface="eth0",
        coordinator_name="test",
        tensor_parallel_size=8,
        data_parallel_size=1,
    )
    clock = SimpleNamespace(now=0.0)

    def advance(seconds):
        clock.now += seconds
        if clock.now > 1800:
            coordinator.request_shutdown()

    monkeypatch.setattr(iris_vllm, "_coordinator_client", lambda _: coordinator)
    monkeypatch.setattr("time.monotonic", lambda: clock.now)
    monkeypatch.setattr("time.sleep", advance)
    iris_vllm.wait_for_iris_vllm_shutdown(launch, lambda: None)
    assert clock.now > 1800
    iris_vllm.notify_iris_vllm_stopped(launch)
    assert coordinator.followers_stopped((1,))

    def died():
        raise RuntimeError("vLLM exited")

    coordinator = iris_vllm.VllmCoordinatorActor("10.0.0.1")
    with pytest.raises(RuntimeError, match="vLLM exited"):
        iris_vllm.wait_for_iris_vllm_shutdown(launch, died)
    assert not coordinator.followers_stopped((1,))

    with iris_vllm.iris_vllm_followers(replace(launch, task_index=0)):
        coordinator.follower_stopped(1)
    assert coordinator.shutdown_requested()


def test_coordinator_latches_shutdown_and_tracks_stopped_followers():
    coordinator = iris_vllm.VllmCoordinatorActor("10.0.0.1")

    assert coordinator.vllm_primary_address() == "10.0.0.1"
    assert not coordinator.shutdown_requested()
    assert not coordinator.followers_stopped((1, 2))

    coordinator.request_shutdown()
    coordinator.request_shutdown()
    assert coordinator.shutdown_requested()

    coordinator.follower_stopped(1)
    assert not coordinator.followers_stopped((1, 2))
    coordinator.follower_stopped(2)
    assert coordinator.followers_stopped((1, 2))


def test_leader_failure_leaves_follower_running_for_iris_retry(monkeypatch):
    coordinator = iris_vllm.VllmCoordinatorActor("10.0.0.1")
    coordinator.follower_stopped(1)
    launch = iris_vllm.IrisVllmLaunch(
        task_index=0,
        num_tasks=2,
        extra_cli_args=(),
        host_ip="10.0.0.1",
        gloo_interface="eth0",
        coordinator_name="test-coordinator",
        tensor_parallel_size=1,
        data_parallel_size=8,
    )
    monkeypatch.setattr(iris_vllm, "_coordinator_client", lambda _: coordinator)

    with pytest.raises(RuntimeError, match="leader failed"):
        with iris_vllm.iris_vllm_followers(launch):
            raise RuntimeError("leader failed")

    assert not coordinator.shutdown_requested()


@pytest.mark.parametrize(
    "code",
    [Code.UNAVAILABLE, Code.NOT_FOUND, Code.UNIMPLEMENTED, Code.DEADLINE_EXCEEDED],
)
def test_wait_until_retries_transient_actor_errors(monkeypatch, code):
    attempts = 0

    def transient_outage() -> bool:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ConnectError(code, "coordinator unavailable")
        return True

    monkeypatch.setattr(iris_vllm, "_POLL_SECONDS", 0.001)
    iris_vllm._wait_until(transient_outage, error_message="did not recover")
    assert attempts == 2
