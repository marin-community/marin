# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Complete(d) AdamH scaling heuristic for Delphi ISOFlop sweeps.

This module intentionally contains only the heuristic used by existing Delphi
mixture launchers. The old sweep-construction helpers depended on legacy
Executor-era eval code and were removed during the lazy ArtifactStep migration.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from dataclasses import dataclass

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim.adamh import AdamHConfig
from marin.processing.tokenize import get_vocab_size_for_tokenizer
from marin.scaling_laws import CandidateConfig

SEQ_LEN: int = 4096
STEPS_PER_RUN: int = 2**16


def _round_to_power_of_two(x: float) -> int:
    """Round ``x`` up to the nearest power of two."""
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


@dataclass(frozen=True)
class CompletedAdamHHeuristic:
    """Complete(d) AdamH scaling heuristic used by Delphi validation launchers."""

    name: str = "completed-adamh"
    tokenizer: str = "marin-community/marin-tokenizer"

    reference_batch_size: int = 64
    reference_tokens: float = 2.5e9
    lr_base: float = 0.00630
    adam_lr_base: float = 0.000656
    epsilon_base: float = 1.85e-08
    beta1: float = 0.9
    beta2_base: float = 0.9999

    max_grad_norm: float = 0.1
    z_loss_weight: float = 1.0e-07
    nesterov: bool = False

    min_lr_ratio: float = 0.0
    warmup: float = 0.1
    lr_schedule: str = "linear"
    decay: float = 0.2

    mlp_ratio: int = 4
    hidden_head_ratio: int = 128
    base_hidden_layer_ratio: int = 64
    layer_scaling_factor: float = 4.0
    layer_formula_offset: int = 9

    max_learning_rate: float = 0.01
    min_batch_size: int = 8
    max_batch_size: int = 8192
    base_max_params: float = 12e9
    base_max_params_budget: float = 3e20
    global_max_params: float = 1e12

    min_beta2: float = 0.9
    max_beta2: float = 0.9999
    max_tokens_per_param: float = 250

    small_budget_step_size: int = 128
    large_budget_step_size: int = 256
    budget_step_threshold: float = 2e19

    @property
    def vocab_size(self) -> int:
        return get_vocab_size_for_tokenizer(self.tokenizer)

    def _compute_scaling_ratio(self, batch_size: int, tokens: float) -> float:
        return (batch_size * self.reference_tokens) / (self.reference_batch_size * tokens)

    def _compute_learning_rate(self, batch_size: int, tokens: float) -> float:
        batch_ratio = batch_size / self.reference_batch_size
        token_ratio = self.reference_tokens / tokens
        learning_rate = self.lr_base * math.sqrt(batch_ratio) * (token_ratio**0.3)
        return min(self.max_learning_rate, learning_rate)

    def _compute_adam_lr(self, batch_size: int, tokens: float) -> float:
        ratio = self._compute_scaling_ratio(batch_size, tokens)
        adam_lr = self.adam_lr_base * math.sqrt(ratio)
        return min(self.max_learning_rate, adam_lr)

    def _compute_epsilon(self, batch_size: int, tokens: float) -> float:
        ratio = self._compute_scaling_ratio(batch_size, tokens)
        return self.epsilon_base * math.sqrt(1.0 / ratio)

    def _compute_beta2(self, batch_size: int) -> float:
        exponent = batch_size / self.reference_batch_size
        return max(self.min_beta2, min(self.max_beta2, self.beta2_base**exponent))

    def build_optimizer_config(self, batch_size: int, tokens: float) -> AdamHConfig:
        return AdamHConfig(
            learning_rate=self._compute_learning_rate(batch_size, tokens),
            adam_lr=self._compute_adam_lr(batch_size, tokens),
            min_lr_ratio=self.min_lr_ratio,
            warmup=self.warmup,
            beta1=self.beta1,
            beta2=self._compute_beta2(batch_size),
            epsilon=self._compute_epsilon(batch_size, tokens),
            max_grad_norm=self.max_grad_norm,
            lr_schedule=self.lr_schedule,
            decay=self.decay,
            nesterov=self.nesterov,
        )

    def _compute_num_layers(self, hidden_size: int) -> int:
        hidden_power = math.log2(hidden_size)
        return round(
            hidden_size
            / (self.base_hidden_layer_ratio + hidden_power * self.layer_scaling_factor - self.layer_formula_offset)
        )

    def _get_step_size(self, budget: float) -> int:
        if budget > self.budget_step_threshold:
            return self.large_budget_step_size
        return self.small_budget_step_size

    def _max_params_for_budget(self, budget: float) -> float:
        scaling = self.base_max_params * math.sqrt(budget / self.base_max_params_budget)
        return min(max(self.base_max_params, scaling), self.global_max_params)

    def _build_model_config(self, hidden_size: int, seq_len: int = SEQ_LEN) -> Qwen3Config:
        if hidden_size % self.hidden_head_ratio != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by hidden_head_ratio ({self.hidden_head_ratio})."
            )
        num_layers = self._compute_num_layers(hidden_size)
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

    def _build_model_configs(self, budget: float, seq_len: int = SEQ_LEN) -> Iterator[Qwen3Config]:
        step_size = self._get_step_size(budget)
        for hidden_size in range(2**9, 2**17, step_size):
            yield self._build_model_config(hidden_size, seq_len)

    def _build_candidate_config(
        self,
        model_config: Qwen3Config,
        tokens: float,
        flops_budget: float,
        seq_len: int = SEQ_LEN,
    ) -> CandidateConfig | None:
        batch_exact = tokens / (STEPS_PER_RUN * seq_len)
        batch_size = _round_to_power_of_two(batch_exact)

        while batch_size > self.min_batch_size:
            learning_rate = self._compute_learning_rate(batch_size, tokens)
            adam_lr = self._compute_adam_lr(batch_size, tokens)
            beta2 = self._compute_beta2(batch_size)
            if learning_rate < self.max_learning_rate and adam_lr < self.max_learning_rate and beta2 > self.min_beta2:
                break
            batch_size //= 2

        if batch_size < self.min_batch_size or batch_size > self.max_batch_size:
            return None

        train_steps = round(tokens / (batch_size * seq_len))
        actual_tokens = batch_size * train_steps * seq_len

        return CandidateConfig(
            model_config=model_config,
            optimizer_config=self.build_optimizer_config(batch_size, tokens),
            batch_size=batch_size,
            train_steps=train_steps,
            tokens=actual_tokens,
            flops_budget=flops_budget,
        )

    def candidates_for_budget(self, budget: float, seq_len: int = SEQ_LEN) -> Iterator[CandidateConfig]:
        max_params = self._max_params_for_budget(budget)
        for model_config in self._build_model_configs(budget, seq_len):
            params = model_config.total_trainable_params(self.vocab_size)
            if params > max_params:
                continue
            flops_per_token = model_config.flops_per_token(self.vocab_size, seq_len)
            tokens = budget / (3 * flops_per_token)
            if tokens / params > self.max_tokens_per_param:
                continue
            candidate = self._build_candidate_config(model_config, tokens, budget, seq_len)
            if candidate is not None:
                yield candidate


completed_adamh_heuristic = CompletedAdamHHeuristic()
