"""GruGPT: a grug-simple explicit-sharding transformer trainer."""

from .attention import default_attention_mask, resolve_attention_backend
from .config import AttentionRuntimeConfig, GruGPTModelConfig, RotaryConfig, TrainingConfig
from .model import (
    GruGPTAttentionParams,
    GruGPTBlockParams,
    GruGPTModelParameters,
    apply_rotary_embedding,
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
]
