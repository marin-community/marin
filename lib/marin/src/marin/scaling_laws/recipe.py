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

"""Scaling recipes: named hyperparameter bundles for scaling law experiments.

A recipe makes "opinionated defaults" explicit and named, so users consciously
choose which set of hyperparameters to use rather than getting hidden defaults.

Usage:
    from marin.scaling_laws.recipe import MARIN_2025_RECIPE, ScalingRecipe

    # Use the default recipe
    recipe = MARIN_2025_RECIPE
    lr = recipe.compute_learning_rate(batch_size=256, hidden_dim=1024)
    beta2 = recipe.compute_beta2(batch_size=256)

    # Or create a custom recipe
    my_recipe = ScalingRecipe(
        name="my-experiment",
        lr_constant=0.25,
        weight_decay=0.05,
    )
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True)
class ScalingRecipe:
    """A named set of hyperparameters for scaling law experiments.

    The recipe controls:
    - Learning rate scaling formula
    - Beta2 scaling formula (for Adam)
    - Optimizer hyperparameters (weight decay, warmup, etc.)
    - Architecture ratios (MLP width, head size)
    """

    name: str
    """Name identifying this recipe (e.g., 'marin-2025')."""

    # Learning rate scaling: lr = lr_constant * sqrt(batch_size) / hidden_dim
    lr_constant: float = 0.33
    """Constant for learning rate calculation."""

    # Beta2 scaling for Adam: beta2 = beta2_base ** (batch_size / beta2_batch_divisor)
    # Reference: https://arxiv.org/pdf/2507.07101
    beta2_base: float = 0.98
    """Base for beta2 exponential scaling."""

    beta2_batch_divisor: float = 128
    """Divisor for beta2 batch size scaling."""

    # Optimizer hyperparameters
    weight_decay: float = 0.1
    min_lr_ratio: float = 0.0
    warmup: float = 0.1
    beta1: float = 0.95
    epsilon: float = 1e-15
    max_grad_norm: float = 1.0
    lr_schedule: str = "linear"
    decay: float = 0.2

    # Architecture ratios
    mlp_ratio: int = 4
    """MLP intermediate_dim = hidden_dim * mlp_ratio."""

    hidden_head_ratio: int = 128
    """num_heads = hidden_dim / hidden_head_ratio."""

    def compute_learning_rate(self, batch_size: int, hidden_dim: int) -> float:
        """Compute learning rate from batch size and hidden dim."""
        return (self.lr_constant * math.sqrt(batch_size)) / hidden_dim

    def compute_beta2(self, batch_size: int) -> float:
        """Compute beta2 from batch size."""
        return self.beta2_base ** (batch_size / self.beta2_batch_divisor)


# Named recipes
MARIN_2025_RECIPE = ScalingRecipe(name="marin-2025")
"""Default Marin scaling recipe based on 2025 best practices."""
