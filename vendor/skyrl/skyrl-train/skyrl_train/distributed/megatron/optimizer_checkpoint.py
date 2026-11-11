"""Compatibility for MCore 0.18 CPU-offloaded Adam checkpoint restore."""

from collections import Counter
from collections.abc import Callable
import socket

import torch
import torch.distributed as dist


def rebind_master_parameters(state: dict, masters: dict) -> None:
    """Keep optimizer state attached to the master's current storage."""
    for parameter, values in state.items():
        values["master_param"] = masters[parameter]


def restore_hybrid_adam_bucket_steps(state: dict) -> None:
    """Recover omitted bucket steps from the saved optimizer group metadata."""
    children = [state] if "param_state" in state else state.values()
    for child in children:
        if child["param_state_sharding_type"] != "dp_reshardable":
            continue
        groups = child["optimizer"]["param_groups"]
        steps = {group.get("step") for group in groups}
        if len(steps) != 1 or None in steps:
            raise ValueError("CPU-offloaded Adam restore requires one saved optimizer step")
        step = torch.tensor(steps.pop(), dtype=torch.float32)

        def complete(value):
            if isinstance(value, dict):
                if "param" in value and "exp_avg" in value and "exp_avg_sq" in value:
                    # The bucket format omits scalar steps, but MCore's loader
                    # asks for them after initializing Torch Adam's state.
                    value.setdefault("step", step)
                else:
                    for item in value.values():
                        complete(item)
            elif isinstance(value, list):
                for item in value:
                    complete(item)

        complete(child["param_state"])


def restore_in_host_order(restore: Callable[[], None]) -> None:
    """Bound temporary host copies to one offloaded optimizer restore per host."""
    hosts = [None] * dist.get_world_size()
    host = socket.gethostname()
    dist.all_gather_object(hosts, host)
    local_ranks = [rank for rank, name in enumerate(hosts) if name == host]
    local_turn = local_ranks.index(dist.get_rank())
    for turn in range(max(Counter(hosts).values())):
        if local_turn == turn:
            restore()
        dist.barrier()
