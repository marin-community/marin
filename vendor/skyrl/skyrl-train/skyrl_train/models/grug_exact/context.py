"""The per-forward object the exact model owns: what the forward's layers hand each other and keep for the
activation recompute."""

from dataclasses import dataclass, field

import torch


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
