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

"""Configuration for AR-to-tree-diffusion transfer."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ARToTreeDiffusionTransferConfig:
    """Configuration for transferring AR weights to tree diffusion."""

    source_model_path: str
    """Path to source AR model (HuggingFace format)."""

    source_model_type: str = "llama"
    """Type of source model ('llama', 'gpt2', etc.)."""

    init_timestep_embed: str = "random"
    """How to initialize timestep embeddings: 'random', 'zeros', or 'sinusoidal'."""

    timestep_embed_std: float = 0.02
    """Standard deviation for random timestep embedding init."""

    freeze_embeddings: bool = False
    """Whether to freeze token embeddings during fine-tuning."""

    freeze_attention: bool = False
    """Whether to freeze attention weights during fine-tuning."""

    remove_causal_mask: bool = True
    """Whether to remove the causal attention mask (for bidirectional attention)."""


# Default path to the Marin 8b model
MARIN_8B_PATH = "gs://marin-us-central2/checkpoints/tootsie-8b-deeper-starling/hf/step-1419999"
