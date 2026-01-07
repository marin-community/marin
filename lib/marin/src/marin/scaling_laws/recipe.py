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

The recipe controls:
- Architecture formula (how to compute num_layers from hidden_size)
- Architecture ratios (MLP width, head size)
- Learning rate and optimizer hyperparameters
- Search bounds and constraints for isoflop sweeps

Usage:
    from marin.scaling_laws.recipe import MARIN_2025_RECIPE, ScalingRecipe

    # Use the default recipe
    recipe = MARIN_2025_RECIPE
    model_config = recipe.build_model_config(hidden_size=1024, seq_len=4096)
    lr = recipe.compute_learning_rate(batch_size=256, hidden_dim=1024)
    beta2 = recipe.compute_beta2(batch_size=256)

    # Or create a custom recipe with different architecture formula
    my_recipe = ScalingRecipe(
        name="my-experiment",
        lr_constant=0.25,
        base_hidden_layer_ratio=48,  # shallower models
    )
"""

import math
from dataclasses import dataclass

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig


@dataclass(frozen=True)
class ScalingRecipe:
    """A named set of hyperparameters for scaling law experiments.

    The recipe controls:
    - Architecture formula (num_layers from hidden_size)
    - Architecture ratios (MLP width, head size)
    - Learning rate scaling formula
    - Beta2 scaling formula (for Adam)
    - Optimizer hyperparameters (weight decay, warmup, etc.)
    - Search bounds and constraints for isoflop sweeps
    """

    name: str
    """Name identifying this recipe (e.g., 'marin-2025')."""

    # --- Learning rate scaling ---
    # lr = lr_constant * sqrt(batch_size) / hidden_dim
    lr_constant: float = 0.33
    """Constant for learning rate calculation."""

    # --- Beta2 scaling for Adam ---
    # beta2 = beta2_base ** (batch_size / beta2_batch_divisor)
    # Reference: https://arxiv.org/pdf/2507.07101
    beta2_base: float = 0.98
    """Base for beta2 exponential scaling."""

    beta2_batch_divisor: float = 128
    """Divisor for beta2 batch size scaling."""

    # --- Optimizer hyperparameters ---
    weight_decay: float = 0.1
    min_lr_ratio: float = 0.0
    warmup: float = 0.1
    beta1: float = 0.95
    epsilon: float = 1e-15
    max_grad_norm: float = 1.0
    lr_schedule: str = "linear"
    decay: float = 0.2

    # --- Architecture ratios ---
    mlp_ratio: int = 4
    """MLP intermediate_dim = hidden_dim * mlp_ratio."""

    hidden_head_ratio: int = 128
    """num_heads = hidden_dim / hidden_head_ratio."""

    # --- Architecture formula for depth-to-width scaling ---
    # num_layers = round(
    #     hidden_size
    #     / (
    #         base_hidden_layer_ratio
    #         + (log2(hidden_size) * layer_scaling_factor)
    #         - layer_formula_offset
    #     )
    # )
    base_hidden_layer_ratio: int = 64
    """Base divisor for depth-width formula."""

    layer_scaling_factor: float = 4.0
    """Multiplier for log2(hidden_size) in depth formula."""

    layer_formula_offset: int = 9
    """Offset (typically min_hidden_pow) in depth formula."""

    # --- Constraints ---
    max_learning_rate: float = 0.01
    """Maximum allowed learning rate (configs with higher LR are rejected)."""

    min_batch_size: int = 8
    """Minimum allowed batch size (configs with smaller batch are rejected)."""

    # --- Search bounds for isoflop sweeps ---
    min_hidden_pow: int = 9
    """Minimum hidden_size as power of 2 (2^9 = 512)."""

    max_hidden_pow: int = 12
    """Maximum hidden_size as power of 2 (2^12 = 4096)."""

    small_budget_step_size: int = 128
    """Step size for hidden_size search at smaller budgets."""

    large_budget_step_size: int = 256
    """Step size for hidden_size search at larger budgets."""

    budget_step_threshold: float = 9e18
    """Budget threshold for switching step sizes."""

    def compute_learning_rate(self, batch_size: int, hidden_dim: int) -> float:
        """Compute learning rate from batch size and hidden dim."""
        return (self.lr_constant * math.sqrt(batch_size)) / hidden_dim

    def compute_beta2(self, batch_size: int) -> float:
        """Compute beta2 from batch size."""
        return self.beta2_base ** (batch_size / self.beta2_batch_divisor)

    def compute_num_layers(self, hidden_size: int) -> int:
        """Compute number of layers from hidden size using the depth-width formula.

        This is an opinionated formula for balancing model depth and width.
        """
        hs_pow = math.log2(hidden_size)
        return round(
            hidden_size
            / (self.base_hidden_layer_ratio + (hs_pow * self.layer_scaling_factor) - self.layer_formula_offset)
        )

    def get_step_size(self, budget: float) -> int:
        """Get hidden_size search step size based on budget."""
        if budget > self.budget_step_threshold:
            return self.large_budget_step_size
        return self.small_budget_step_size

    def build_model_config(self, hidden_size: int, seq_len: int = 4096) -> Qwen3Config:
        """Build a model config from hidden_size using this recipe's architecture formula.

        This is the key interface: the recipe makes all architecture decisions
        and returns a fully-specified model config.

        Args:
            hidden_size: Model hidden dimension.
            seq_len: Maximum sequence length.

        Returns:
            A Qwen3Config with architecture determined by this recipe.
        """
        num_layers = self.compute_num_layers(hidden_size)
        intermediate_dim = hidden_size * self.mlp_ratio
        n_heads = max(1, hidden_size // self.hidden_head_ratio)

        return Qwen3Config(
            hidden_dim=hidden_size,
            intermediate_dim=intermediate_dim,
            num_layers=num_layers,
            num_heads=n_heads,
            num_kv_heads=n_heads,
            max_seq_len=seq_len,
            rope=Llama3RotaryEmbeddingsConfig(),
        )

    def build_optimizer_config(self, learning_rate: float, beta2: float) -> CautiousConfig:
        """Build optimizer config using this recipe's hyperparameters.

        This centralizes all optimizer configuration in the recipe, ensuring
        consistent hyperparameters across isoflop sweeps and optimal training runs.

        Args:
            learning_rate: Learning rate (typically from CandidateConfig).
            beta2: Adam beta2 (typically from CandidateConfig).

        Returns:
            A CautiousConfig with optimizer settings from this recipe.
        """
        return CautiousConfig(
            learning_rate=learning_rate,
            weight_decay=self.weight_decay,
            min_lr_ratio=self.min_lr_ratio,
            warmup=self.warmup,
            beta1=self.beta1,
            beta2=beta2,
            epsilon=self.epsilon,
            max_grad_norm=self.max_grad_norm,
            adamc_weight_decay=True,
            lr_schedule=self.lr_schedule,
            decay=self.decay,
        )


# Named recipes
MARIN_2025_RECIPE = ScalingRecipe(name="marin-2025")
"""Default Marin scaling recipe based on 2025 best practices."""
