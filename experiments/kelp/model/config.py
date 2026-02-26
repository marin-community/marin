# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

import dataclasses
from dataclasses import dataclass

from levanter.grug.attention import RotaryConfig


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

    layer_norm_eps: float = 1e-5
    """Epsilon for layer normalization."""

    initializer_std: float = 0.02
    """Standard deviation for weight initialization."""

    rope: RotaryConfig = dataclasses.field(default_factory=RotaryConfig)
    """Rotary positional embedding configuration."""

    compute_dtype: str = "float32"
    """Dtype for forward/backward computation ('float32', 'bfloat16').
    Loss accumulation always uses float32 for numerical stability."""

    gradient_checkpointing: bool = False
    """If True, recompute activations during backward pass to save memory.
    Essential for large models (>1B params) but increases compute by ~33%."""

    prompt_tokens: bool = False
    """If True, the tokenizer uses PROMPT_START/PROMPT_END special tokens
    (IDs 3 and 4), shifting position and byte token offsets by +2.
    Old checkpoints deserialize with False, keeping the legacy layout."""

    pad_token_id: int = 0
    """Token ID for padding."""

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
