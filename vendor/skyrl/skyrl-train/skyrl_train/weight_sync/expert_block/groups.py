"""NCCL groups between trainer and receiver processes.

Trainers and receivers do not share a default process group, so each sync group is a
standalone communicator. The driver hosts one ``TCPStore``. Each participant creates the
groups it belongs to, in schedule order, each under its own store prefix. A small
broadcast on every group makes NCCL connect at startup instead of during the first sync.
"""

from dataclasses import dataclass
from datetime import timedelta
import time

import ray
import torch
import torch.distributed as dist

from skyrl_train.distributed.utils import init_custom_process_group
from skyrl_train.weight_sync.expert_block.schedule import Group

WARMUP_VALUE = 1729


@dataclass(frozen=True)
class Rendezvous:
    address: str
    port: int
    namespace: str
    timeout_seconds: int


class RendezvousStore:
    """The driver's TCP store. It stays open until every group is destroyed."""

    def __init__(self, namespace: str, *, timeout_seconds: int):
        self.address = ray.util.get_node_ip_address()
        self.store = dist.TCPStore(
            self.address,
            0,
            world_size=None,
            is_master=True,
            timeout=timedelta(seconds=timeout_seconds),
            wait_for_workers=False,
        )
        self.rendezvous = Rendezvous(self.address, self.store.port, namespace, timeout_seconds)

    def close(self) -> None:
        self.store = None


def create_groups(
    participant: int, groups: tuple[Group, ...], rendezvous: Rendezvous, *, backend: str = "nccl"
) -> dict[str, dist.ProcessGroup]:
    """Create the groups this participant belongs to. If one fails, destroy those already created."""
    client = dist.TCPStore(
        rendezvous.address,
        rendezvous.port,
        world_size=None,
        is_master=False,
        timeout=timedelta(seconds=rendezvous.timeout_seconds),
        wait_for_workers=False,
    )
    created: dict[str, dist.ProcessGroup] = {}
    try:
        for group in groups:
            if participant not in group.members:
                continue
            created[group.name] = init_custom_process_group(
                backend,
                store=dist.PrefixStore(f"{rendezvous.namespace}/{group.name}", client),
                rank=group.members.index(participant),
                world_size=len(group.members),
                group_name=group.name,
                timeout=timedelta(seconds=rendezvous.timeout_seconds),
            )
    except BaseException:
        destroy_groups(created)
        raise
    return created


def warm_groups(
    participant: int, groups: tuple[Group, ...], created: dict[str, dist.ProcessGroup], device
) -> dict[str, float]:
    """Broadcast a known value on each group and check it arrived. Returns the seconds per group."""
    wire = torch.empty(1, dtype=torch.int32, device=device)
    seconds = {}
    for index, group in enumerate(groups):
        if group.name not in created:
            continue
        started = time.monotonic()
        expected = WARMUP_VALUE + index
        wire.fill_(expected if group.members[0] == participant else -1)
        dist.broadcast(wire, src=0, group=created[group.name])
        if wire.item() != expected:
            raise RuntimeError(f"Warm-up broadcast on group {group.name} delivered {wire.item()}, expected {expected}")
        seconds[group.name] = time.monotonic() - started
    return seconds


def destroy_groups(created: dict[str, dist.ProcessGroup]) -> None:
    for name in reversed(list(created)):
        dist.destroy_process_group(created.pop(name))
