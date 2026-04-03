# Copyright The Marin Authors
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

"""Compute-scaling AdamH heuristic for ISOFlop sweeps.

Formulas (from v10/v11/v12 LR sweep, 102 runs, filtered):
- Adam LR: adam_lr = 2943.27 * C^(-0.334) * sqrt(B/32)
- AdamH LR: lr = (13/3) * adam_lr
- C = 3 * flops_per_token(no_lm_head) * tokens
- Epsilon: epsilon = epsilon0 * sqrt(r0/r)
- Beta1: fixed at 0.9062
- Beta2: beta2 = clip(beta2_0^(B/B0), min_beta2, max_beta2)
"""

import math
from dataclasses import dataclass

from experiments.grug.moe_apr2.model import GrugModelConfig
from experiments.grug.moe_apr2.optimizer import GrugMoeAdamHConfig

SEQ_LEN: int = 4096


@dataclass(frozen=True)
class CompletedAdamHHeuristic:
    """Compute-scaling AdamH heuristic.

    adam_lr = adam_lr_coeff * C^(adam_lr_exponent) * sqrt(B/32)
    adamh_lr = adamh_ratio * adam_lr
    C = 3 * flops_per_token * tokens  (flops_per_token excludes lm_head)
    """

    # --- LR scaling (from empirical fit) ---
    adam_lr_coeff: float = 2943.27
    adam_lr_exponent: float = -0.334
    adamh_ratio: float = 13 / 3

    # --- Base hyperparameters ---
    reference_batch_size: int = 32
    reference_tokens: float = 1.4e9
    epsilon_base: float = 1e-15
    beta1: float = 0.9062
    beta2_base: float = 0.999

    # --- Fixed hyperparameters ---
    max_grad_norm: float = 1.0
    z_loss_weight: float = 0.0001

    # --- Schedule ---
    min_lr_ratio: float = 0.0
    warmup: float = 0.1
    lr_schedule: str = "linear"
    decay: float | None = None

    # --- Architecture ---
    vocab_size: int = 128_256
    hidden_head_ratio: int = 128
    base_hidden_layer_ratio: int = 64
    layer_scaling_factor: float = 4.0
    layer_formula_offset: int = 9

    # --- Constraints ---
    max_learning_rate: float = 0.05
    min_beta2: float = 0.95
    max_beta2: float = 0.9999

    def _compute_scaling_ratio(self, batch_size: int, tokens: float) -> float:
        """Compute r/r0 = (B * T0) / (B0 * T)."""
        return (batch_size * self.reference_tokens) / (self.reference_batch_size * tokens)

    def _compute_adam_lr(self, batch_size: int, tokens: float, flops_per_token: float) -> float:
        """adam_lr = adam_lr_coeff * C^(exponent) * sqrt(B/32)"""
        C = 3 * flops_per_token * tokens
        adam_lr = self.adam_lr_coeff * (C**self.adam_lr_exponent) * math.sqrt(batch_size / 32)
        return min(self.max_learning_rate, adam_lr)

    def _compute_learning_rate(self, batch_size: int, tokens: float, flops_per_token: float) -> float:
        """adamh_lr = (13/3) * adam_lr"""
        adam_lr = self._compute_adam_lr(batch_size, tokens, flops_per_token)
        return min(self.max_learning_rate, self.adamh_ratio * adam_lr)

    def _compute_epsilon(self, batch_size: int, tokens: float) -> float:
        """epsilon = epsilon0 * sqrt(r0/r)"""
        ratio = self._compute_scaling_ratio(batch_size, tokens)
        return self.epsilon_base * math.sqrt(1.0 / ratio)

    def _compute_beta2(self, batch_size: int) -> float:
        """beta2 = clip(beta2_0^(B/B0), min_beta2, max_beta2). Constant token half-life."""
        exponent = batch_size / self.reference_batch_size
        return max(self.min_beta2, min(self.max_beta2, self.beta2_base**exponent))

    def build_optimizer_config(self, batch_size: int, tokens: float, flops_per_token: float) -> GrugMoeAdamHConfig:
        lr = self._compute_learning_rate(batch_size, tokens, flops_per_token)
        adam_lr = self._compute_adam_lr(batch_size, tokens, flops_per_token)
        epsilon = self._compute_epsilon(batch_size, tokens)
        beta2 = self._compute_beta2(batch_size)
        return GrugMoeAdamHConfig(
            learning_rate=lr,
            adam_lr=adam_lr,
            min_lr_ratio=self.min_lr_ratio,
            warmup=self.warmup,
            beta1=self.beta1,
            beta2=beta2,
            epsilon=epsilon,
            max_grad_norm=self.max_grad_norm,
            lr_schedule=self.lr_schedule,
            decay=self.decay,
        )

    def _compute_num_layers(self, hidden_size: int) -> int:
        hs_pow = math.log2(hidden_size)
        return round(
            hidden_size
            / (self.base_hidden_layer_ratio + (hs_pow * self.layer_scaling_factor) - self.layer_formula_offset)
        )

    def _get_step_size(self, budget: float) -> int:
        if budget > self.budget_step_threshold:
            return self.large_budget_step_size
        return self.small_budget_step_size

    def _max_params_for_budget(self, budget: float) -> float:
        scaling = self.base_max_params * math.sqrt(budget / self.base_max_params_budget)
        return min(max(self.base_max_params, scaling), self.global_max_params)

    def build_model_config(self, hidden_size: int) -> GrugModelConfig:
        if hidden_size % self.hidden_head_ratio != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by hidden_head_ratio ({self.hidden_head_ratio})."
            )
        num_layers = self._compute_num_layers(hidden_size)
        num_heads = max(1, hidden_size // self.hidden_head_ratio)

        return GrugModelConfig(
            vocab_size=self.vocab_size,
            hidden_dim=hidden_size,
            intermediate_dim=hidden_size // 2,
            shared_expert_intermediate_dim=hidden_size,
            num_experts=64,
            num_experts_per_token=4,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_heads,
            max_seq_len=SEQ_LEN,
            sliding_window=SEQ_LEN,
            initializer_std=0.5 / math.sqrt(hidden_size),
            qk_mult=1.3,
        )


completed_adamh_heuristic = CompletedAdamHHeuristic()
