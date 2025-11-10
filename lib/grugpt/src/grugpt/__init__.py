"""GruGPT: a grug-simple explicit-sharding transformer trainer."""

from .attention import apply_rotary_embedding, default_attention_mask, resolve_attention_backend
from .config import AttentionRuntimeConfig, GruGPTModelConfig, RotaryConfig, TrainingConfig
from .data import build_token_loader, make_dataloader, make_token_dataset
from .model import (
    GruGPTAttentionParams,
    GruGPTBlockParams,
    GruGPTModelParameters,
    forward,
    init_parameters,
)

__all__ = [
    "AttentionRuntimeConfig",
    "TrainingConfig",
    "GruGPTModelConfig",
    "RotaryConfig",
    "GruGPTAttentionParams",
    "GruGPTBlockParams",
    "GruGPTModelParameters",
    "apply_rotary_embedding",
    "default_attention_mask",
    "forward",
    "init_parameters",
    "resolve_attention_backend",
    "make_token_dataset",
    "make_dataloader",
    "build_token_loader",
]
