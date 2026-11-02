"""Resolve publication records to live trainer and receiver storage.

The Megatron adapter supplies expert identities and dense runs. These views
validate whole expert matrices and receiver slots without interpreting Bridge
mapping classes or expert checkpoint schemas.
"""

from dataclasses import dataclass
import re
from typing import Literal

import torch

from skyrl_train.weight_sync.expert_block.schedule import (
    BF16,
    FP32,
    WIRE_DTYPE_BYTES,
    DenseSlice,
    ExpertEntry,
    TrainerRank,
)

LAYER_PREFIX = re.compile(r"model\.layers\.(\d+)\.")
ROUTED_EXPERTS = "mlp.experts.routed_experts"
ROUTER_WEIGHT_SUFFIX = ".mlp.router.weight"


def dtype_name(dtype: torch.dtype) -> str:
    """``torch.bfloat16`` -> ``"bfloat16"``, as used in inventories and the schedule."""
    return str(dtype).removeprefix("torch.")


def is_widened_router(hf_name: str, wire_dtype: str, installed_dtype: str) -> bool:
    """Whether this is the router weight, which is sent as BF16 and stored as FP32 in vLLM."""
    return hf_name.endswith(ROUTER_WEIGHT_SUFFIX) and wire_dtype == BF16 and installed_dtype == FP32


@dataclass(frozen=True)
class ExpertSlice:
    """One normalized expert projection and the live trainer parameter that stores it."""

    layer: int
    expert: int
    part: Literal["gate", "up", "down"]
    source_key: str


@dataclass(frozen=True)
class ExpertSource:
    """One expert matrix on this rank and the trainer parameter that stores it."""

    entry: ExpertEntry
    source_key: str
    shape: tuple[int, int]


@dataclass(frozen=True)
class LocalSources:
    """One trainer rank's expert slices, dense slices and the parameters that store them."""

    experts: list[ExpertSlice]
    dense: list[DenseSlice]
    sources: dict[str, torch.Tensor]


def local_expert_sources(
    expert_slices: list[ExpertSlice],
    sources: dict[str, torch.Tensor],
    trainer: TrainerRank,
    *,
    num_experts: int,
    expert_parallel_size: int,
    expert_hidden_size: int,
    intermediate_size: int,
) -> list[ExpertSource]:
    """Group expert slices into whole matrices. Check that each is one contiguous BF16 parameter."""
    per_block = num_experts // expert_parallel_size
    grouped: dict[tuple[int, int, str], dict[str, ExpertSlice]] = {}
    for item in expert_slices:
        layer = item.layer
        if not trainer.ep * per_block <= item.expert < (trainer.ep + 1) * per_block:
            raise ValueError(f"Expert {item.expert} of layer {layer} is not owned by EP rank {trainer.ep}")
        parts = grouped.setdefault((layer, item.expert, "fc2" if item.part == "down" else "fc1"), {})
        if item.part in parts:
            raise ValueError(f"Duplicate {item.part} slice for expert {item.expert} of layer {layer}")
        parts[item.part] = item
    result = []
    for (layer, expert, projection), parts in sorted(grouped.items()):
        needed = {"down"} if projection == "fc2" else {"gate", "up"}
        if set(parts) != needed or len({item.source_key for item in parts.values()}) != 1:
            raise ValueError(
                f"Expert {expert} of layer {layer} ({projection}) is incomplete or split across parameters"
            )
        first = next(iter(parts.values()))
        source = sources[first.source_key]
        shape = (
            (expert_hidden_size, intermediate_size)
            if projection == "fc2"
            else (2 * intermediate_size, expert_hidden_size)
        )
        if tuple(source.shape) != shape or source.dtype != torch.bfloat16:
            raise ValueError(f"Parameter {first.source_key} is not the {shape} BF16 matrix of expert {expert}")
        entry = ExpertEntry(
            f"model.layers.{layer}.mlp.experts.{projection}.expert{expert}",
            layer,
            trainer.pp,
            expert,
            projection,
            source.numel() * WIRE_DTYPE_BYTES[BF16],
        )
        result.append(ExpertSource(entry, first.source_key, shape))
    return result


def expert_source_view(item: ExpertSource, sources: dict[str, torch.Tensor]) -> torch.Tensor:
    source = sources[item.source_key]
    if tuple(source.shape) != item.shape or source.dtype != torch.bfloat16 or not source.is_contiguous():
        raise ValueError(f"Expert parameter {item.source_key} changed shape, dtype or layout")
    return source.detach().view(-1)


def expert_slot_view(entry: ExpertEntry, parameters, expert_maps) -> torch.Tensor:
    """The receiver's slot for a global expert, as a flat view. The layer's expert map gives the local index."""
    prefix = f"model.layers.{entry.layer}.{ROUTED_EXPERTS}"
    local = expert_maps[prefix][entry.expert]
    if local < 0:
        raise ValueError(f"This receiver does not serve expert {entry.expert} of layer {entry.layer}")
    parameter = parameters[f"{prefix}.w13_weight" if entry.projection == "fc1" else f"{prefix}.w2_weight"]
    slot = parameter.detach()[local]
    if (
        slot.dtype != torch.bfloat16
        or not slot.is_contiguous()
        or slot.numel() * WIRE_DTYPE_BYTES[BF16] != entry.nbytes
    ):
        raise ValueError(f"Receiver slot for {entry.name} differs from the scheduled matrix")
    return slot.view(-1)


def dense_source_view(item: DenseSlice, sources: dict[str, torch.Tensor]) -> torch.Tensor:
    source = sources[item.source_key]
    if dtype_name(source.dtype) != item.wire_dtype or not source.is_contiguous():
        raise ValueError(f"Parameter {item.source_key} changed dtype or layout")
    return source.detach().view(-1).narrow(0, item.source_offset, item.numel)


def dense_installed_view(item: DenseSlice, parameters) -> torch.Tensor:
    """The part of the receiver's tensor a slice is written to. The router weight is FP32 here and BF16 on the wire."""
    parameter = parameters[item.hf_name]
    if not parameter.is_contiguous():
        raise ValueError(f"Installed parameter {item.hf_name} is not contiguous")
    installed = dtype_name(parameter.dtype)
    if installed != item.wire_dtype and not is_widened_router(item.hf_name, item.wire_dtype, installed):
        raise ValueError(f"Installed dtype of {item.hf_name} differs from the wire dtype {item.wire_dtype}")
    if item.hf_offset + item.numel > parameter.numel():
        raise ValueError(f"Slice of {item.hf_name} exceeds the installed tensor")
    return parameter.detach().view(-1).narrow(0, item.hf_offset, item.numel)
