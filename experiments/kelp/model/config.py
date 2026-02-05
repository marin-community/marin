# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Configuration for Kelp tree diffusion models."""

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Type

if TYPE_CHECKING:
    from experiments.kelp.model.model import TreeDiffusionModel


@dataclass(frozen=True)
class TreeDiffusionConfig:
    """Configuration for tree diffusion models.

    This follows the grugformer pattern of using simple dataclasses rather than
    the Levanter LmConfig registry system, for simplicity and fast iteration.
    """

    vocab_size: int
    """Size of the token vocabulary."""

    hidden_dim: int = 512
    """Hidden dimension size."""

    intermediate_dim: int = 2048
    """FFN intermediate dimension (typically 4x hidden_dim)."""

    num_layers: int = 6
    """Number of transformer layers."""

    num_heads: int = 8
    """Number of attention heads."""

    num_kv_heads: int = 8
    """Number of key-value heads (for GQA). Set equal to num_heads for MHA."""

    head_dim: int | None = None
    """Per-head dimension. If None, computed as hidden_dim // num_heads."""

    max_seq_len: int = 2048
    """Maximum sequence length."""

    num_diffusion_steps: int = 100
    """Number of diffusion steps for generation."""

    noise_schedule: str = "cosine"
    """Noise schedule type: 'cosine' or 'linear'."""

    prefix_max_len: int = 256
    """Maximum length of the conditioning prefix (docstring/signature)."""

    layer_norm_eps: float = 1e-5
    """Epsilon for layer normalization."""

    initializer_std: float = 0.02
    """Standard deviation for weight initialization."""

    dropout_rate: float = 0.0
    """Dropout rate (0.0 for no dropout)."""

    use_rope: bool = True
    """Whether to use rotary positional embeddings."""

    rope_base: float = 10000.0
    """Base for rotary embeddings."""

    mask_token_id: int | None = None
    """Token ID for [MASK] token. If None, uses vocab_size - 1."""

    def __post_init__(self) -> None:
        """Validate configuration."""
        if self.hidden_dim % self.num_heads != 0 and self.head_dim is None:
            raise ValueError(
                f"hidden_dim={self.hidden_dim} must be divisible by "
                f"num_heads={self.num_heads}, or set head_dim explicitly"
            )
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError(f"num_heads={self.num_heads} must be divisible by num_kv_heads={self.num_kv_heads}")

    @property
    def inferred_head_dim(self) -> int:
        """Get per-head dimension."""
        if self.head_dim is not None:
            return self.head_dim
        return self.hidden_dim // self.num_heads

    @property
    def effective_mask_token_id(self) -> int:
        """Get the mask token ID."""
        if self.mask_token_id is not None:
            return self.mask_token_id
        return self.vocab_size - 1


# Size presets following grugformer convention
def tiny_config(vocab_size: int) -> TreeDiffusionConfig:
    """Tiny model for testing (~10M params)."""
    return TreeDiffusionConfig(
        vocab_size=vocab_size,
        hidden_dim=256,
        intermediate_dim=1024,
        num_layers=4,
        num_heads=4,
        num_kv_heads=4,
        max_seq_len=512,
    )


def small_config(vocab_size: int) -> TreeDiffusionConfig:
    """Small model for laptop development (~125M params)."""
    return TreeDiffusionConfig(
        vocab_size=vocab_size,
        hidden_dim=512,
        intermediate_dim=2048,
        num_layers=6,
        num_heads=8,
        num_kv_heads=8,
        max_seq_len=2048,
    )


def medium_config(vocab_size: int) -> TreeDiffusionConfig:
    """Medium model (~300M params, 1x A100)."""
    return TreeDiffusionConfig(
        vocab_size=vocab_size,
        hidden_dim=768,
        intermediate_dim=3072,
        num_layers=12,
        num_heads=12,
        num_kv_heads=12,
        max_seq_len=2048,
    )


def large_config(vocab_size: int) -> TreeDiffusionConfig:
    """Large model (~8B params, v5p-8 TPU)."""
    return TreeDiffusionConfig(
        vocab_size=vocab_size,
        hidden_dim=4096,
        intermediate_dim=14336,
        num_layers=32,
        num_heads=32,
        num_kv_heads=8,
        max_seq_len=4096,
    )


SIZE_PRESETS = {
    "tiny": tiny_config,
    "small": small_config,
    "medium": medium_config,
    "large": large_config,
}
