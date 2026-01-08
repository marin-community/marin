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

"""Scaling recipes: model-specific hyperparameter bundles for scaling law experiments.

A ScalingRecipe encapsulates model-specific decisions for scaling experiments:
- Architecture formula (how to compute architecture from target param count)
- Architecture ratios (MLP width, head size)
- Model config building (returns LlamaConfig or subclass)
- Learning rate and optimizer hyperparameters
- Search bounds and constraints for isoflop sweeps
- Candidate generation for isoflop sweeps
"""

import math
import os
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.llama import LlamaConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.cautious import CautiousConfig
from levanter.optim.config import OptimizerConfig

if TYPE_CHECKING:
    from marin.scaling_laws.isoflop_analysis import CandidateConfig, IsoFlopTrainArgs

# TODO: LlamaConfig is used as our "abstract" model config base class.
# All model configs we use (Qwen3Config, etc.) inherit from LlamaConfig
# and provide flops_per_token() for FLOP calculations.

# Default constants
DEFAULT_SEQ_LEN = 4096
DEFAULT_STEPS_PER_RUN = 2**16  # Reference step count for hyperparameter tuning
DEFAULT_FLOP_TOLERANCE = 0.01  # Relative error tolerance for FLOP budget
DEFAULT_TOKENIZER = "stanford-crfm/marin-tokenizer"


def _round_to_power_of_two(x: float) -> int:
    """Round x UP to the nearest power of 2."""
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


@dataclass(frozen=True)
class ScalingRecipe:
    """A named set of hyperparameters for scaling law experiments.

    The recipe encapsulates ALL model-specific decisions:
    - Architecture formula (num_layers from hidden_size)
    - Architecture ratios (MLP width, head size)
    - Learning rate scaling formula
    - Beta2 scaling formula (for Adam)
    - Optimizer hyperparameters (weight decay, warmup, etc.)
    - Search bounds and constraints for isoflop sweeps
    - Candidate generation
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

    # --- Hyperparameter formulas ---

    def compute_learning_rate(self, batch_size: int, hidden_dim: int) -> float:
        """Compute learning rate from batch size and hidden dim."""
        return (self.lr_constant * math.sqrt(batch_size)) / hidden_dim

    def compute_beta2(self, batch_size: int) -> float:
        """Compute beta2 from batch size."""
        return self.beta2_base ** (batch_size / self.beta2_batch_divisor)

    def compute_num_layers(self, hidden_size: int) -> int:
        """Compute number of layers from hidden size using the depth-width formula."""
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

    # --- Parameter count estimation ---

    def compute_params_for_hidden_size(self, hidden_size: int, vocab_size: int) -> int:
        """Compute approximate parameter count for a given hidden size.

        This uses the standard transformer parameter formula for Llama/Qwen architectures.
        """
        num_layers = self.compute_num_layers(hidden_size)
        intermediate_dim = hidden_size * self.mlp_ratio
        n_heads = max(1, hidden_size // self.hidden_head_ratio)
        head_size = hidden_size // n_heads

        # Embeddings
        embed_params = vocab_size * hidden_size * 2  # input + output embeddings

        # Per-layer params: attention + mlp + layer norms
        q_proj = hidden_size * head_size * n_heads
        kv_proj = 2 * hidden_size * head_size * n_heads  # K and V
        o_proj = head_size * n_heads * hidden_size
        attn_params = q_proj + kv_proj + o_proj

        mlp_params = 3 * hidden_size * intermediate_dim  # gate, up, down
        norm_params = 2 * hidden_size  # 2 layer norms per layer

        layer_params = attn_params + mlp_params + norm_params
        total_layer_params = num_layers * layer_params

        # Final layer norm
        final_norm = hidden_size

        return embed_params + total_layer_params + final_norm

    def hidden_size_for_params(self, target_params: int, vocab_size: int) -> int:
        """Find the hidden size that gives approximately target_params.

        Uses binary search over valid hidden sizes.
        """
        min_hidden = 2**self.min_hidden_pow
        max_hidden = 2**self.max_hidden_pow

        best_hidden = min_hidden
        best_diff = abs(self.compute_params_for_hidden_size(min_hidden, vocab_size) - target_params)

        # Search in steps of 64 for efficiency
        for hidden_size in range(min_hidden, max_hidden + 1, 64):
            params = self.compute_params_for_hidden_size(hidden_size, vocab_size)
            diff = abs(params - target_params)
            if diff < best_diff:
                best_diff = diff
                best_hidden = hidden_size

        return best_hidden

    # --- Model config building ---

    def build_model_config(self, target_params: int, vocab_size: int, seq_len: int = DEFAULT_SEQ_LEN) -> LlamaConfig:
        """Build a model config for a target parameter count.

        The recipe determines the architecture (hidden_size, num_layers, etc.)
        that achieves approximately target_params parameters.

        Args:
            target_params: Target parameter count.
            vocab_size: Vocabulary size.
            seq_len: Maximum sequence length.

        Returns:
            A LlamaConfig (or subclass) with architecture determined by this recipe.
        """
        hidden_size = self.hidden_size_for_params(target_params, vocab_size)
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

    def _build_model_config_from_hidden_size(self, hidden_size: int, seq_len: int = DEFAULT_SEQ_LEN) -> LlamaConfig:
        """Internal: build model config from hidden_size directly.

        Used during candidate generation when we're iterating over hidden sizes.
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

    def build_optimizer_config(self, candidate: "CandidateConfig", vocab_size: int) -> OptimizerConfig:
        """Build optimizer config for a candidate.

        Computes learning rate and beta2 from the candidate's batch_size and target_params.

        Args:
            candidate: Model-agnostic candidate config.
            vocab_size: Vocabulary size (needed to determine hidden_size).

        Returns:
            An OptimizerConfig with settings from this recipe.
        """
        hidden_size = self.hidden_size_for_params(candidate.target_params, vocab_size)
        learning_rate = self.compute_learning_rate(candidate.batch_size, hidden_size)
        beta2 = self.compute_beta2(candidate.batch_size)

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

    # --- Candidate generation (model-specific search) ---

    def candidate_configs(
        self,
        budget: float,
        vocab_size: int,
        seq_len: int = DEFAULT_SEQ_LEN,
        steps_per_run: int = DEFAULT_STEPS_PER_RUN,
        flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
    ) -> "Iterator[CandidateConfig]":
        """Yield model-agnostic candidate configurations within the FLOP budget.

        This method encapsulates the model-specific search logic internally but
        returns model-agnostic CandidateConfig objects containing only:
        batch_size, train_steps, tokens, target_params, flops_budget.

        The caller uses recipe.build_model_config() and recipe.build_optimizer_config()
        to convert these to model-specific configs.

        Args:
            budget: Target FLOP budget.
            vocab_size: Vocabulary size for the tokenizer.
            seq_len: Sequence length for training.
            steps_per_run: Reference step count for FLOP budget calculation.
            flop_tolerance: Tolerance for matching FLOP budget (relative error).

        Yields:
            Model-agnostic CandidateConfig objects for each valid configuration.
        """
        # Import here to avoid circular dependency
        from marin.scaling_laws.isoflop_analysis import CandidateConfig, solve_for_batch_size, solve_for_train_steps

        step_size = self.get_step_size(budget)
        min_hidden = 2**self.min_hidden_pow
        max_hidden = 2**self.max_hidden_pow

        for hidden_size in range(min_hidden, max_hidden + 1, step_size):
            model_config = self._build_model_config_from_hidden_size(hidden_size, seq_len)

            batch_exact = solve_for_batch_size(model_config, vocab_size, budget, steps_per_run, seq_len)
            batch_size = _round_to_power_of_two(batch_exact)

            # Apply LR constraint
            lr = self.compute_learning_rate(batch_size, hidden_size)
            while lr > self.max_learning_rate:
                batch_size //= 2
                lr = self.compute_learning_rate(batch_size, hidden_size)

            if batch_size < self.min_batch_size:
                continue

            train_steps = round(solve_for_train_steps(model_config, vocab_size, budget, batch_size, seq_len))

            # Verify we hit the budget within tolerance
            # Training FLOPs = 3 * flops_per_token * batch * steps * seq_len
            achieved_flops = 3 * model_config.flops_per_token(vocab_size, seq_len) * batch_size * train_steps * seq_len
            if abs(achieved_flops - budget) / budget > flop_tolerance:
                continue

            tokens = batch_size * train_steps * seq_len
            target_params = self.compute_params_for_hidden_size(hidden_size, vocab_size)

            yield CandidateConfig(
                batch_size=batch_size,
                train_steps=train_steps,
                tokens=tokens,
                target_params=target_params,
                flops_budget=budget,
            )

    def generate_isoflop_train_args(
        self,
        budgets: Sequence[float],
        experiment_name: str,
        vocab_size: int,
        seq_len: int = DEFAULT_SEQ_LEN,
        steps_per_run: int = DEFAULT_STEPS_PER_RUN,
        flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
    ) -> "list[IsoFlopTrainArgs]":
        """Generate model-agnostic training arguments for each candidate in an isoflop sweep.

        Returns IsoFlopTrainArgs containing the model-agnostic CandidateConfig plus
        naming information. The caller uses recipe.build_model_config() and
        recipe.build_optimizer_config() to get model-specific configs.

        Args:
            budgets: Sequence of FLOP budgets to generate configs for.
            experiment_name: Name suffix for run names (e.g., 'nemo', 'dclm').
            vocab_size: Vocabulary size for the tokenizer.
            seq_len: Sequence length for training.
            steps_per_run: Reference step count for FLOP budget calculation.
            flop_tolerance: Tolerance for matching FLOP budget.

        Returns:
            List of IsoFlopTrainArgs, one per candidate config across all budgets.
        """
        from marin.scaling_laws.isoflop_analysis import IsoFlopTrainArgs

        results: list[IsoFlopTrainArgs] = []

        for budget in budgets:
            for candidate in self.candidate_configs(budget, vocab_size, seq_len, steps_per_run, flop_tolerance):
                run_name = (
                    f"isoflop-{budget:.0e}-N{candidate.target_params:.0e}-"
                    f"B{candidate.batch_size}-{experiment_name}"
                )

                tags = (
                    f"FLOPs={budget:.1e}",
                    f"N={candidate.target_params:.1e}",
                    f"B={candidate.batch_size}",
                    f"steps={candidate.train_steps}",
                    f"tokens={candidate.tokens:.1e}",
                )

                # Static output path for checkpoint reuse
                output_path = os.path.join("checkpoints", "isoflop", run_name)

                results.append(
                    IsoFlopTrainArgs(
                        candidate=candidate,
                        run_name=run_name,
                        tags=tags,
                        output_path=output_path,
                    )
                )

        return results

    def predict_optimal_config(
        self,
        scaling_fits: "dict[str, tuple[float, float]]",
        target_flops: float,
        label: str,
        vocab_size: int,
        seq_len: int = DEFAULT_SEQ_LEN,
        steps_per_run: int = DEFAULT_STEPS_PER_RUN,
        flop_tolerance: float = DEFAULT_FLOP_TOLERANCE,
    ) -> "CandidateConfig | None":
        """Predict optimal training config for a target compute budget using fitted scaling laws.

        This implements IsoFLOP Approach 2 from the Chinchilla paper:
        1. Uses the scaling fit (N* ~ A * C^alpha) to predict optimal tokens for target_flops
        2. Generates candidate configs for the target budget using this recipe
        3. Selects the candidate whose token count is closest to the predicted optimal

        Args:
            scaling_fits: Dict of {label: (alpha, A)} from scaling ladder result.
            target_flops: Target compute budget in FLOPs.
            label: Dataset/experiment label to use for scaling fit.
            vocab_size: Vocabulary size.
            seq_len: Sequence length for training.
            steps_per_run: Reference step count for FLOP budget calculation.
            flop_tolerance: Tolerance for matching FLOP budget.

        Returns:
            CandidateConfig for the predicted optimal, or None if label not in fits
            or no valid candidates found.
        """
        import logging

        logger = logging.getLogger(__name__)

        if label not in scaling_fits:
            logger.warning(f"Label '{label}' not found in scaling fits")
            return None

        alpha, A = scaling_fits[label]
        optimal_tokens = A * (target_flops**alpha)

        logger.info(f"Predicted optimal tokens for {target_flops:.2e} FLOPs: {optimal_tokens:.2e}")

        candidates = list(self.candidate_configs(target_flops, vocab_size, seq_len, steps_per_run, flop_tolerance))

        if not candidates:
            logger.warning(f"No valid candidates found for budget {target_flops:.2e}")
            return None

        best = min(candidates, key=lambda c: c.tokens - optimal_tokens if c.tokens >= optimal_tokens else float("inf"))
        # If all candidates have fewer tokens than optimal, pick the one with the most tokens
        if best.tokens < optimal_tokens:
            best = max(candidates, key=lambda c: c.tokens)

        logger.info(
            f"Selected config: N={best.target_params:.2e}, "
            f"B={best.batch_size}, tokens={best.tokens:.2e} (optimal: {optimal_tokens:.2e})"
        )

        return best
