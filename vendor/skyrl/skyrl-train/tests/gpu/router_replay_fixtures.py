"""Shared valid route samples for the opt-in Megatron router-replay GPU tests."""

from __future__ import annotations

import torch


def random_unique_routes(shape: tuple[int, ...], num_experts: int, *, generator: torch.Generator) -> torch.Tensor:
    """Return random top-K indices with no duplicate expert in a token row."""
    if not shape or shape[-1] > num_experts:
        raise ValueError(f"invalid route shape {shape} for {num_experts} experts")
    scores = torch.rand((*shape[:-1], num_experts), generator=generator)
    return scores.argsort(dim=-1)[..., : shape[-1]]
