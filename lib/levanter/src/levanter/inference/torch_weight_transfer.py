# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Torch broadcast receiver for single-device native serving."""

from datetime import timedelta
import threading

import jax
import jax.dlpack
import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _new_process_group_helper, _world

from levanter.inference.weight_reload import WeightTransferConfig


class TorchWeightReceiver:
    def __init__(self, config: WeightTransferConfig):
        self.config = config
        platform = jax.devices()[0].platform
        expected_platform = "gpu" if config.backend == "nccl" else "cpu"
        if platform != expected_platform:
            raise ValueError(f"{config.backend} weight transfer requires a {expected_platform} serving device")
        self.device = torch.device("cuda:0" if config.backend == "nccl" else "cpu")
        self.group = None
        self.lock = threading.RLock()

    def initialize(
        self,
        master_address: str,
        master_port: int,
        rank_offset: int,
        world_size: int,
        group_name: str,
        backend: str,
        override_existing: bool = False,
    ) -> None:
        with self.lock:
            if backend != self.config.backend:
                raise ValueError("Requested transport backend differs from the serving configuration")
            if self.group is not None:
                if not override_existing:
                    raise ValueError("Weight update communicator already exists")
                self.close()
            if dist.is_initialized():
                raise ValueError("Native weight transfer requires a process without a default Torch process group")
            if not group_name or rank_offset < 1 or rank_offset >= world_size:
                raise ValueError("Invalid weight update group or serving rank")
            if self.device.type == "cuda":
                torch.cuda.set_device(self.device)
            timeout = timedelta(seconds=self.config.timeout)
            store = dist.TCPStore(master_address, master_port, world_size, False, timeout=timeout)
            store = dist.PrefixStore(group_name, store)
            # Match SkyRL's standalone trainer/serving group without creating a default Torch world.
            group, _ = _new_process_group_helper(
                world_size,
                rank_offset,
                [],
                backend,
                store,
                group_name=group_name,
                timeout=timeout,
            )
            assert isinstance(group, dist.ProcessGroup)
            self.group = group
            _world.pg_group_ranks[group] = {rank: rank for rank in range(world_size)}

    def receive(self, dtype: str, shape: tuple[int, ...]) -> jax.Array:
        with self.lock:
            if self.group is None:
                raise ValueError("Weight update communicator is not initialized")
            tensor = torch.empty(shape, dtype=getattr(torch, dtype.removeprefix("torch.")), device=self.device)
            dist.broadcast(tensor, src=0, group=self.group)
            # Keep the received allocation on its device; JAX owns the DLPack buffer lifetime.
            return jax.dlpack.from_dlpack(tensor, copy=False)

    def close(self) -> None:
        with self.lock:
            if self.group is not None:
                dist.destroy_process_group(self.group)
                self.group = None
