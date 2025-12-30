# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Grug: a grug-simple explicit-sharding transformer trainer.

This package intentionally exposes only the "grug core" (raw-array model + kernels).
Levanter integration adapters live under `levanter.models`.
"""

from .attention import apply_rotary_embedding, default_attention_mask, resolve_attention_backend
from .config import AttentionRuntimeConfig, GrugModelConfig, GrugTrainingConfig, RotaryConfig
from .data import build_token_loader, make_dataloader, make_token_dataset
from .model import (
    GrugAttentionParams,
    GrugBlockParams,
    GrugModelParameters,
    forward,
    init_parameters,
)

__all__ = [
    "AttentionRuntimeConfig",
    "GrugTrainingConfig",
    "GrugModelConfig",
    "RotaryConfig",
    "GrugAttentionParams",
    "GrugBlockParams",
    "GrugModelParameters",
    "apply_rotary_embedding",
    "default_attention_mask",
    "forward",
    "init_parameters",
    "resolve_attention_backend",
    "make_token_dataset",
    "make_dataloader",
    "build_token_loader",
]
