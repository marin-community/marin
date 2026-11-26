# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Worker groups can publish and use distinct live rendezvous stores on one host."""

import asyncio
import multiprocessing
import threading
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.distributed as dist

from skyrl_train.distributed import utils as distributed_utils
from skyrl_train.distributed.utils import init_custom_process_group, init_worker_process_group_with_device
from skyrl_train.utils import get_tcp_url
from skyrl_train.workers import worker as worker_module
from skyrl_train.workers.worker import DistributedTorchRayActor, Worker


def _worker(rank, group, channel):
    # Each process holds its own actor state and default group. Ray placement
    # inputs are fixed to a local CPU host for this Gloo test.
    with (
        patch("skyrl_train.workers.worker.ray.get_gpu_ids", return_value=[]),
        patch.object(DistributedTorchRayActor, "_get_current_node_ip", return_value="127.0.0.1"),
        patch("skyrl_train.workers.worker.torch.cuda.device_count", return_value=0),
    ):
        actor = DistributedTorchRayActor(world_size=2, rank=rank, local_rank=0, sequence_parallel_size=1)
    actor.cfg = SimpleNamespace(trainer=SimpleNamespace(distributed=SimpleNamespace(worker_collective_timeout_seconds=10)))
    if rank == 0:
        channel.send(actor.get_master_addr_port())
    master_addr, master_port = channel.recv()
    actor._init_default_process_group(master_addr, master_port, backend="gloo")
    try:
        value = torch.tensor(group * 100 + rank + 1)
        dist.all_reduce(value)
        assert value.item() == group * 200 + 3
    finally:
        dist.destroy_process_group()


def test_concurrent_worker_groups_accept_clients_before_process_group_init():
    context = multiprocessing.get_context("spawn")
    processes = []
    channels = {}

    def start_rank(rank, group):
        parent, child = context.Pipe()
        process = context.Process(target=_worker, args=(rank, group, child))
        process.start()
        child.close()
        processes.append(process)
        channels[rank, group] = parent

    try:
        for group in range(2):
            start_rank(0, group)
            start_rank(1, group)
        endpoints = []
        for group in range(2):
            leader = channels[0, group]
            assert leader.poll(30), "rank zero did not publish a rendezvous"
            endpoints.append(leader.recv())
        assert endpoints[0] != endpoints[1]
        # Neither group has begun process-group init. Both published ports
        # already accept TCPStore clients, closing the old probe-to-bind gap.
        for addr, port in endpoints:
            client = dist.TCPStore(addr, port, is_master=False, timeout=timedelta(seconds=2), wait_for_workers=False)
            client.set("before_init", "reachable")
        for (_, group), channel in channels.items():
            channel.send(endpoints[group])
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
    finally:
        for channel in channels.values():
            channel.close()
        for process in processes:
            if process.is_alive():
                process.terminate()
            process.join()


def test_worker_rendezvous_pins_cuda_before_initializing_nccl(monkeypatch):
    monkeypatch.setenv("LOCAL_RANK", "3")
    monkeypatch.setenv("MASTER_ADDR", "unused")
    monkeypatch.setenv("MASTER_PORT", "0")
    pinned = []
    initialized = []
    store = SimpleNamespace(port=12345)

    def init_process_group(**kwargs):
        assert pinned == [3]
        initialized.append(kwargs)

    monkeypatch.setattr(torch.cuda, "set_device", pinned.append)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(dist, "is_initialized", lambda: False)
    monkeypatch.setattr(dist, "init_process_group", init_process_group)
    init_worker_process_group_with_device("127.0.0.1", store.port, 60, rank=0, world_size=2, master_store=store)
    assert initialized == [
        {
            "backend": "nccl",
            "store": store,
            "rank": 0,
            "world_size": 2,
            "timeout": timedelta(seconds=60),
            "device_id": torch.device("cuda", 3),
        }
    ]


def _weight_sync_peer(channel, timeout):
    channel.send("ready")
    request = channel.recv()
    # The inference worker still joins through the helper's tcp:// rendezvous.
    group = init_custom_process_group(
        backend=request["backend"],
        init_method=get_tcp_url(request["master_addr"], request["master_port"]),
        timeout=timeout,
        world_size=request["world_size"],
        rank=request["rank_offset"],
        group_name=request["group_name"],
    )
    try:
        value = torch.tensor(41)
        dist.all_reduce(value, group=group)
        assert value.item() == 42
    finally:
        dist.destroy_process_group(group)


def test_weight_sync_worker_advertises_live_store_to_tcp_peer(monkeypatch):
    timeout = timedelta(seconds=10)
    monkeypatch.setattr(distributed_utils, "default_pg_timeout", timeout)
    # Hold the trainer group initializer until the inference RPC has confirmed
    # the port is live. A probe that closes before publishing cannot pass.
    endpoint_open = threading.Event()

    def init_trainer_group(**kwargs):
        assert endpoint_open.wait(timeout.total_seconds()), "advertised weight-sync store was unreachable"
        return init_custom_process_group(**kwargs)

    monkeypatch.setattr(worker_module, "init_custom_process_group", init_trainer_group)
    context = multiprocessing.get_context("spawn")
    trainer_channel, peer_channel = context.Pipe()
    peer = context.Process(target=_weight_sync_peer, args=(peer_channel, timeout))
    peer.start()
    peer_channel.close()

    class InferencePeer:
        async def init_weight_update_communicator(self, **request):
            self.endpoint = request["master_addr"], request["master_port"]
            store = dist.TCPStore(*self.endpoint, is_master=False, timeout=timedelta(seconds=3))
            store.set("probe", "reachable")
            endpoint_open.set()
            trainer_channel.send(request)

    worker = object.__new__(Worker)
    worker._master_addr = "127.0.0.1"
    worker._weight_sync_pg_options = None
    worker.cfg = SimpleNamespace(
        generator=SimpleNamespace(
            num_inference_engines=1,
            inference_engine_tensor_parallel_size=1,
            inference_engine_pipeline_parallel_size=1,
            inference_engine_data_parallel_size=1,
            weight_sync_backend="gloo",
            override_existing_update_group="disable",
        )
    )
    try:
        assert trainer_channel.poll(45) and trainer_channel.recv() == "ready"
        dist.init_process_group("gloo", store=dist.HashStore(), rank=0, world_size=1, timeout=timeout)
        try:
            client = InferencePeer()
            asyncio.run(worker.init_weight_sync_state(client))
            value = torch.tensor(1)
            dist.all_reduce(value, group=worker._model_update_group)
            assert value.item() == 42
            peer.join(timeout.total_seconds())
            assert peer.exitcode == 0
            store = dist.TCPStore(*client.endpoint, is_master=False, timeout=timedelta(seconds=3))
            assert store.get("probe") == b"reachable"
        finally:
            dist.destroy_process_group()
    finally:
        trainer_channel.close()
        if peer.is_alive():
            peer.terminate()
        peer.join()
