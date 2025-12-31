# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug: a grug-simple explicit-sharding transformer trainer.

This package intentionally exposes only the "grug core" (raw-array model + kernels).
Levanter integration adapters live under `levanter.models`.
"""

from .attention import apply_rotary_embedding, attention
from .config import GrugModelConfig, GrugTrainingConfig, RotaryConfig
from .data import build_token_loader, make_dataloader, make_token_dataset
from .model import (
    GrugAttentionParams,
    GrugBlockParams,
    GrugModelParameters,
    forward,
    init_parameters,
)

__all__ = [
    "GrugTrainingConfig",
    "GrugModelConfig",
    "RotaryConfig",
    "GrugAttentionParams",
    "GrugBlockParams",
    "GrugModelParameters",
    "apply_rotary_embedding",
    "attention",
    "forward",
    "init_parameters",
    "make_token_dataset",
    "make_dataloader",
    "build_token_loader",
]
