# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass, field


@dataclass(frozen=True)
class RotaryConfig:
    """Lightweight rotary embedding configuration."""

    theta: float = 10000.0
    scaling_factor: float | None = None


@dataclass(frozen=True)
class GrugModelConfig:
    """Hyperparameters for the Grug Llama-style transformer."""

    vocab_size: int
    hidden_dim: int = 2048
    intermediate_dim: int = 5632
    num_layers: int = 24
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int | None = None
    max_seq_len: int = 4096
    dropout_rate: float = 0.0
    layer_norm_eps: float = 1e-5
    initializer_std: float = 0.02
    rope: RotaryConfig = field(default_factory=RotaryConfig)
    tie_embeddings: bool = False

    # Controls how we compute logsumexp over the vocab in `levanter.grug.loss_fn`.
    #
    # - `None` means "single full-vocab block" (often faster for small-ish models/vocabs).
    # - Smaller values reduce peak memory, but can be significantly slower in practice.
    #
    # TODO(grug): Replace with a faster large-vocab CE kernel so we don't have to pick between
    # speed and memory.
    cross_entropy_block_size: int | None = 32768

    @property
    def inferred_head_dim(self) -> int:
        if self.head_dim is not None:
            return self.head_dim
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} is not divisible by num_heads={self.num_heads}; set head_dim explicitly"
            )
        return self.hidden_dim // self.num_heads


@dataclass(frozen=True)
class GrugTrainingConfig:
    """Full training recipe, nested around a model + attention backend."""

    model: GrugModelConfig = field(default_factory=GrugModelConfig)
    learning_rate: float = 1e-3
    weight_decay: float = 0.01
    seed: int = 0
    steps: int = 10
    global_batch_size: int = 8


def validate_config(cfg: GrugModelConfig) -> None:
    _ = cfg.inferred_head_dim
    if cfg.num_heads % cfg.num_kv_heads != 0:
        raise ValueError("num_heads must be divisible by num_kv_heads for grouped-query attention")
    if cfg.vocab_size <= 0:
        raise ValueError("vocab_size must be positive")
    if cfg.max_seq_len <= 0:
        raise ValueError("max_seq_len must be positive")


__all__ = [
    "RotaryConfig",
    "GrugModelConfig",
    "GrugTrainingConfig",
    "validate_config",
]
