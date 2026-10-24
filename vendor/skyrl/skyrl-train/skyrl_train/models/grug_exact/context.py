"""The per-forward object the exact model owns: what the forward's layers hand each other and keep for the
activation recompute."""

from dataclasses import dataclass, field

import torch

from skyrl_train.inference_engines.base import UNKNOWN_ENGINE_DP_RANK


@dataclass(frozen=True)
class ResidualSum:
    """The three bf16 tensors, detached, that a layer adds in fp32 to form its output
    ``residual + (routed + shared)``."""

    residual: torch.Tensor
    routed: torch.Tensor
    shared: torch.Tensor


@dataclass
class ExactForward:
    """What the layers of one model forward share: each sequence's engine data-parallel rank and the engines'
    expert-parallel size, which set the order of the expert-parallel combine, and, by layer number, the attention
    output each checkpointed layer keeps for its recompute."""

    engine_dp_ranks: torch.Tensor
    expert_parallel_size: int
    attention: dict[int, torch.Tensor] = field(default_factory=dict)

    def row_engine_dp_ranks(self, rows: int) -> torch.Tensor:
        """Each of ``rows`` router rows' engine data-parallel rank: row ``s * B + b`` is position ``s`` of sequence
        ``b``."""
        return self.engine_dp_ranks.repeat(rows // self.engine_dp_ranks.numel())


def check_engine_dp_ranks(ranks: torch.Tensor, scored: torch.Tensor, data_parallel_size: int) -> None:
    """Raise unless every sequence with loss tokens (``scored``, ``[B]`` bool) has an engine data-parallel rank below
    ``data_parallel_size``; a sequence without loss tokens may carry ``UNKNOWN_ENGINE_DP_RANK``."""
    if bool(((ranks < 0) & scored).any()):
        raise ValueError(f"every sequence with loss tokens needs its engine data-parallel rank, got {ranks.tolist()}")
    if bool((ranks < UNKNOWN_ENGINE_DP_RANK).any()) or bool((ranks >= data_parallel_size).any()):
        raise ValueError(
            f"engine data-parallel ranks must be below the engines' data-parallel size {data_parallel_size} or "
            f"{UNKNOWN_ENGINE_DP_RANK}, got {ranks.tolist()}"
        )
