"""Test-only snapshot type used by the Megatron Grug policy worker."""

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class GrugValidationSnapshot:
    """One policy rank's loaded Grug state as gathered by ``grug_validation_snapshot``."""

    rank: int
    attention_backend: str
    weights: dict[str, torch.Tensor]
