# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""FP32 master records in model space, assembled one parameter at a time.

Weight analysis reads every checkpoint's FP32 optimizer masters by model parameter
name (``optimizer.state.fp32_param.<model key>``). Megatron's distributed optimizer
holds each master as flat slices spread over its data-parallel group. Its
``fully_reshardable`` checkpoint format produces model-space records by gathering a
group's whole optimizer state (masters and both moments) onto one rank, which crashed
Snowball 67B; ``fully_sharded_model_space`` would avoid the gather but writes flattened
ShardedTensors, which Megatron 0.18's dist-checkpointing rejects.

This module builds the master records directly: for each parameter of each gradient
buffer, in the buffer's own order on every rank, the DP group reduces a zero-filled
full-size FP32 tensor, into which each rank has copied its slice, onto the group's first
rank. Only one parameter is materialized on the GPU at a time, and only that writer keeps
the result (on the CPU), so a node's host memory holds each master once. The record
reuses the model's ShardedTensor metadata (key, shape) as the main replica, so the
checkpoint writer can alias the BF16 model record to it. Optimizer moments are left to the
``dp_reshardable`` format, which writes DP-local shards without a gather.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import replace
from typing import Any

import torch
from torch import distributed as dist

MASTER_RECORD_PREFIX = "optimizer.state.fp32_param."


def master_record_key(model_key: str) -> str:
    return f"{MASTER_RECORD_PREFIX}{model_key}"


def full_master(
    numel: int, local_range: range | None, local_shard: torch.Tensor | None, group, device
) -> torch.Tensor | None:
    """One parameter's whole FP32 master on the group's first rank (the writer); ``None`` elsewhere.

    Each rank adds its slice into zeros and the group sums onto its first rank, so only
    the writer ever holds the whole master. Every rank of ``group`` must call this for
    the same parameter in the same order; a rank that holds no slice passes
    ``local_range=None``.
    """
    full = torch.zeros(numel, dtype=torch.float32, device=device)
    if local_range is not None:
        full[local_range.start : local_range.stop] = local_shard.to(device=device, dtype=torch.float32)
    writer = dist.get_global_rank(group, 0) if group is not None else 0
    dist.reduce(full, dst=writer, group=group)
    return full if dist.get_rank() == writer else None


def _child_optimizers(optimizer) -> list:
    return list(getattr(optimizer, "chained_optimizers", None) or [optimizer])


def _local_ranges(child) -> dict[int, range]:
    """This rank's slice of each of the child's parameters, keyed by parameter identity."""
    ranges = {}
    for gbuf_range_maps in child.gbuf_ranges:
        for buckets in gbuf_range_maps.values():
            for bucket in buckets:
                for model_param, range_map in bucket["param_map"].items():
                    span = range_map["param"]
                    ranges[id(model_param)] = range(span.start, span.end)
    return ranges


def _buffer_params(child) -> Iterable[torch.nn.Parameter]:
    """Every parameter of the child's gradient buffers, in the same order on every rank."""
    for buffer in child.buffers:
        yield from buffer.params


def model_space_masters(optimizer, model_sharded_state_dict: Mapping[str, Any]) -> dict[str, Any]:
    """FP32 master records for every optimized parameter, keyed like ``model_sharded_state_dict``.

    ``model_sharded_state_dict`` is the model's flat sharded state dict; each entry whose
    data is an optimized parameter gets a twin with key ``master_record_key(entry.key)``
    and the parameter's FP32 master as data (on the CPU).
    """
    entries = {
        entry.data.data_ptr(): (name, entry)
        for name, entry in model_sharded_state_dict.items()
        if hasattr(entry, "key") and hasattr(entry, "data") and isinstance(entry.data, torch.Tensor)
    }
    device = torch.device("cuda", torch.cuda.current_device()) if torch.cuda.is_available() else torch.device("cpu")
    records: dict[str, Any] = {}
    for child in _child_optimizers(optimizer):
        local = _local_ranges(child)
        group = child.data_parallel_group
        for param in _buffer_params(child):
            local_range = local.get(id(param))
            shard = child._get_main_param_and_optimizer_states(param)["param"] if local_range else None
            if param.data_ptr() not in entries:
                raise ValueError(f"optimized parameter of shape {tuple(param.shape)} has no model checkpoint record")
            master = full_master(param.numel(), local_range, shard, group, device)
            if master is None:
                continue  # another rank of the DP group writes this master
            name, entry = entries[param.data_ptr()]
            # A ShardedTensor declares its dtype; a ShardedTensorFactory carries none.
            dtype = {"dtype": torch.float32} if hasattr(entry, "dtype") else {}
            records[name] = replace(
                entry,
                key=master_record_key(entry.key),
                data=master.view(param.shape).cpu(),
                # The writer holds the DP group's only copy: it is the DP main replica.
                replica_id=_dp_main_replica(entry.replica_id),
                **dtype,
            )
    return records


def _dp_main_replica(replica_id):
    """The replica id with its data-parallel coordinate (the last) set to 0; other axes kept."""
    return (*replica_id[:-1], 0) if isinstance(replica_id, tuple) else 0
