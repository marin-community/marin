# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import argparse
from dataclasses import replace
from types import SimpleNamespace

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.client.job_info import JobInfo
from iris.cluster.types import JobName
from marin.inference import iris_vllm


def test_distributed_dp_launch_covers_sixteen_ranks_with_one_global_load_balancer(monkeypatch):
    monkeypatch.setenv("VLLM_HOST_IP", "127.0.0.1")
    monkeypatch.setenv("GLOO_SOCKET_IFNAME", "lo")
    parser = argparse.ArgumentParser()
    for option in (
        "tensor-parallel-size",
        "pipeline-parallel-size",
        "data-parallel-size",
        "data-parallel-size-local",
        "nnodes",
        "node-rank",
        "master-port",
        "data-parallel-rpc-port",
    ):
        parser.add_argument(f"--{option}", type=int)
    for option in ("master-addr", "device-ids", "data-parallel-address", "data-parallel-backend"):
        parser.add_argument(f"--{option}")
    parser.add_argument("--headless", action="store_true")
    ranks = []
    api_hosts = []
    for index in range(4):
        info = JobInfo(task_id=JobName.from_string(f"/fixture/hero/{index}"), num_tasks=4)
        launch = iris_vllm._build_task_launch(
            info,
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=16,
            data_parallel_size_local=4,
            master_addr="10.0.0.1",
            coordinator_name="fixture",
        )
        native = parser.parse_args(launch.extra_cli_args)
        first = native.node_rank * native.data_parallel_size_local
        ranks.extend(range(first, first + native.data_parallel_size_local))
        assert native.device_ids.split(",") == ["0", "1", "2", "3"]
        assert native.data_parallel_size == 16
        assert native.nnodes == 4
        assert native.data_parallel_address == "10.0.0.1"
        if not native.headless:
            api_hosts.append(index)
        # An explicit start-rank would switch the API process into local hybrid LB.
        assert "--data-parallel-start-rank" not in launch.extra_cli_args
    assert sorted(ranks) == list(range(16))
    assert api_hosts == [0]


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
