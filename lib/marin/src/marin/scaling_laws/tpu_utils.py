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
    from marin.scaling_laws.isoflop_analysis import CandidateConfig, ScalingRecipe

# ---------------- TPU v5p Hardware Constants ----------------
# These constants are specific to TPU v5p pods.

HBM_PER_CHIP_GIB = 95
"""High-bandwidth memory per TPU v5p chip in GiB."""

CORES_PER_CHIP = 2
"""Number of cores per TPU v5p chip."""

V5P_CORE_OPTIONS = [8, 16, 32, 128, 256, 512]
"""Available TPU v5p core configurations (slice sizes)."""


def pick_v5p_type(
    candidate: "CandidateConfig",
    vocab_size: int,
    seq_len: int,
    recipe: "ScalingRecipe",
) -> str:
    """Select the smallest TPU v5p slice that fits the model in float32.

    Args:
        candidate: CandidateConfig with target_params and batch_size.
        vocab_size: Vocabulary size.
        seq_len: Sequence length.
        recipe: ScalingRecipe to determine architecture.

    Returns:
        TPU slice name, e.g., "v5p-8" or "v5p-32".

    Raises:
        ValueError: If the model is too large for available v5p slices.
    """
    model_config = recipe.build_model_config(candidate.target_params, vocab_size, seq_len)
    need_bytes = recipe.estimate_memory_bytes(model_config, candidate.batch_size, vocab_size)
    chip_bytes = HBM_PER_CHIP_GIB * 1024**3
    chips = math.ceil(need_bytes / chip_bytes)
    cores_req = chips * CORES_PER_CHIP

    valid = [c for c in V5P_CORE_OPTIONS if c >= cores_req]
    if not valid:
        raise ValueError(f"Model too large for available v5p slices (need {cores_req} cores).")

    return f"v5p-{min(valid)}"
