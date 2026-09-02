# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Minimal standalone torch process-group construction for SkyRL weight sync."""

import inspect
from datetime import timedelta

import torch
from torch.distributed.distributed_c10d import (
    Backend,
    PrefixStore,
    _new_process_group_helper,
    _world,
    default_pg_timeout,
    rendezvous,
)


def init_custom_process_group(
    *,
    backend: str,
    master_addr: str,
    master_port: int,
    world_size: int,
    rank: int,
    group_name: str,
    timeout: timedelta | None = None,
):
    """Join SkyRL's cross-runtime weight-transfer process group."""
    torch.cuda.set_device(0)
    timeout = timeout or default_pg_timeout
    iterator = rendezvous(
        f"tcp://{master_addr}:{master_port}",
        rank,
        world_size,
        timeout=timeout,
    )
    store, rank, world_size = next(iterator)
    store.set_timeout(timeout)
    store = PrefixStore(group_name, store)

    parameters = inspect.signature(_new_process_group_helper).parameters
    options_name = "backend_options" if "backend_options" in parameters else "pg_options"
    process_group, _ = _new_process_group_helper(
        world_size,
        rank,
        [],
        Backend(backend),
        store,
        group_name=group_name,
        **{options_name: None},
        timeout=timeout,
    )
    _world.pg_group_ranks[process_group] = {index: index for index in range(world_size)}
    return process_group
