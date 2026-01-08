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

"""TPU hardware utilities for memory estimation and slice selection.

This module provides utilities for estimating memory requirements and
selecting appropriate TPU slice sizes for training runs.
"""

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from marin.scaling_laws.isoflop_analysis import CandidateConfig
    from marin.scaling_laws.recipe import ScalingRecipe

# ---------------- TPU v5p Hardware Constants ----------------
# These constants are specific to TPU v5p pods.

HBM_PER_CHIP_GIB = 95
"""High-bandwidth memory per TPU v5p chip in GiB."""

CORES_PER_CHIP = 2
"""Number of cores per TPU v5p chip."""

V5P_CORE_OPTIONS = [8, 16, 32, 128, 256, 512]
"""Available TPU v5p core configurations (slice sizes)."""


def estimate_memory_bytes(
    param_count: int,
    hidden_dim: int,
    num_layers: int,
    batch: int,
    seq_len: int,
    vocab: int,
    optim_mult: int = 3,
    dtype_size: int = 4,
    fudge_factor: float = 2,
) -> int:
    """Estimate float32 memory usage (in bytes) for one training step.

    This is a conservative estimate for LLaMA-style architectures with
    Adam optimizer. The fudge_factor provides a safety margin for
    additional memory overhead not captured in the simple model.

    Args:
        param_count: Number of model parameters.
        hidden_dim: Model hidden size.
        num_layers: Number of Transformer layers.
        batch: Training batch size.
        seq_len: Sequence length.
        vocab: Vocabulary size.
        optim_mult: Optimizer memory multiplier (default 3 for Adam with
            momentum and variance states).
        dtype_size: Bytes per float (default 4 for float32).
        fudge_factor: Safety margin multiplier (default 2x).

    Returns:
        Estimated total memory in bytes.

    Note:
        This assumes a LLaMA-style architecture with Adam optimizer in float32.
        Actual memory usage may vary based on specific model architecture,
        optimizer choice, and JAX/XLA memory optimizations.
    """
    param_bytes = param_count * optim_mult * dtype_size
    act_bytes = (batch * seq_len) * ((hidden_dim * num_layers) + vocab * fudge_factor)
    total_bytes = param_bytes + act_bytes
    return int(total_bytes * fudge_factor)


def pick_v5p_type(
    candidate: "CandidateConfig",
    vocab_size: int,
    seq_len: int,
    recipe: "ScalingRecipe | None" = None,
) -> str:
    """Select the smallest TPU v5p slice that fits the model in float32.

    Args:
        candidate: CandidateConfig with target_params and batch_size.
        vocab_size: Vocabulary size.
        seq_len: Sequence length.
        recipe: ScalingRecipe to determine architecture. If None, uses default.

    Returns:
        TPU slice name, e.g., "v5p-8" or "v5p-32".

    Raises:
        ValueError: If the model is too large for available v5p slices.
    """
    if recipe is None:
        from marin.scaling_laws.recipe import ScalingRecipe
        recipe = ScalingRecipe(name="default")

    hidden_size = recipe.hidden_size_for_params(candidate.target_params, vocab_size)
    num_layers = recipe.compute_num_layers(hidden_size)

    need_bytes = estimate_memory_bytes(
        candidate.target_params,
        hidden_size,
        num_layers,
        candidate.batch_size,
        seq_len,
        vocab_size,
    )
    chip_bytes = HBM_PER_CHIP_GIB * 1024**3
    chips = math.ceil(need_bytes / chip_bytes)
    cores_req = chips * CORES_PER_CHIP

    valid = [c for c in V5P_CORE_OPTIONS if c >= cores_req]
    if not valid:
        raise ValueError(f"Model too large for available v5p slices (need {cores_req} cores).")

    return f"v5p-{min(valid)}"
