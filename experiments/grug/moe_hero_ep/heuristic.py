# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute-scaling LR heuristic and the EP64 hero config builder.

``MoeHeuristic`` is the May Recipe refit (issue #5951, R^2=0.996): it sets compute-optimal MuonH /
Adam learning rates, epsilon, and beta2 from the token budget and batch size. ``build_hero_configs``
pairs it with the fixed hero model spec so a launcher gets both configs back from a single
``(num_train_steps, batch_size)`` call, keeping the hero self-contained.
"""

import math
from dataclasses import dataclass

from levanter.grug.grug_moe import MoeImplementation, MoonEPConfig

from experiments.grug.moe_hero_ep.model import GrugModelConfig
from experiments.grug.moe_hero_ep.optimizer import GrugMoeMuonHConfig
from experiments.grug.moe_hero_ep.quantile_balancing import QuantileBalancingMethod


@dataclass(frozen=True)
class MoeHeuristic:
    """May Recipe MuonH LR-scaling refit (issue #5951, seq_len=4096 fits).

    adam_lr  = lr_coeff * tokens^lr_tokens_exp * hidden_dim^lr_dim_exp * sqrt(tokens_per_batch)
    muonh_lr = muonh_ratio * adam_lr
    epsilon  = epsilon_coeff * sqrt(tokens / tokens_per_batch)
    beta2    = clip(beta2_base^(tokens_per_batch / beta2_reference_tpb), min_beta2, max_beta2)
    """

    lr_coeff: float = 0.06602
    lr_tokens_exp: float = -0.395
    lr_dim_exp: float = -0.150
    muonh_ratio: float = 13 / 3
    epsilon_coeff: float = 9.676e-18
    beta1: float = 0.9062
    beta2_base: float = 0.999
    beta2_reference_tpb: int = 131_072
    min_lr_ratio: float = 0.05
    lr_schedule: str = "linear"
    max_learning_rate: float = 0.05
    min_beta2: float = 0.95
    max_beta2: float = 0.9999

    def _adam_lr(self, tokens_per_batch: int, tokens: float, hidden_dim: int) -> float:
        adam_lr = (
            self.lr_coeff * (tokens**self.lr_tokens_exp) * (hidden_dim**self.lr_dim_exp) * math.sqrt(tokens_per_batch)
        )
        return min(self.max_learning_rate, adam_lr)

    def _learning_rate(self, tokens_per_batch: int, tokens: float, hidden_dim: int) -> float:
        return min(self.max_learning_rate, self.muonh_ratio * self._adam_lr(tokens_per_batch, tokens, hidden_dim))

    def _epsilon(self, tokens_per_batch: int, tokens: float) -> float:
        return self.epsilon_coeff * math.sqrt(tokens / tokens_per_batch)

    def _beta2(self, tokens_per_batch: int) -> float:
        exponent = tokens_per_batch / self.beta2_reference_tpb
        return max(self.min_beta2, min(self.max_beta2, self.beta2_base**exponent))

    def build_optimizer_config(
        self, *, num_train_steps: int, batch_size: int, hidden_dim: int, seq_len: int
    ) -> GrugMoeMuonHConfig:
        """MuonH optimizer with LR / beta2 / epsilon scaled to this token budget (1pct-noclip schedule)."""
        tokens_per_batch = batch_size * seq_len
        tokens = float(num_train_steps * tokens_per_batch)
        return GrugMoeMuonHConfig(
            learning_rate=self._learning_rate(tokens_per_batch, tokens, hidden_dim),
            adam_lr=self._adam_lr(tokens_per_batch, tokens, hidden_dim),
            min_lr_ratio=self.min_lr_ratio,
            lr_schedule=self.lr_schedule,
            beta1=self.beta1,
            beta2=self._beta2(tokens_per_batch),
            epsilon=self._epsilon(tokens_per_batch, tokens),
        )


def build_hero_configs(
    *,
    num_train_steps: int,
    batch_size: int,
    moe_implementation: MoeImplementation = "fixed_all_to_all",
    moonep_config: MoonEPConfig | None = None,
    qb_method: QuantileBalancingMethod = QuantileBalancingMethod.LOCAL_EXACT,
    qb_histogram_bins: int = 1000,
) -> tuple[GrugModelConfig, GrugMoeMuonHConfig]:
    """The fixed EP64 hero model plus its compute-scaled MuonH optimizer."""
    model = GrugModelConfig(
        vocab_size=128_256,
        hidden_dim=5120,
        intermediate_dim=1280,
        shared_expert_intermediate_dim=5120,
        num_shared_experts=1,
        num_experts=256,
        num_experts_per_token=8,
        num_layers=48,
        num_heads=40,
        num_kv_heads=10,
        head_dim=128,
        max_seq_len=4096,
        sliding_window=2048,
        global_every=4,
        capacity_factor=1.0,
        initializer_std=0.5 / math.sqrt(5120),
        qk_mult=1.3,
        attention_implementation="gpu_fa4_cute",
        moe_implementation=moe_implementation,
        moonep_config=moonep_config,
        qb_method=qb_method,
        qb_histogram_bins=qb_histogram_bins,
        expert_chunks=1,
        report_capacity_overflow=True,
    )
    optimizer = MoeHeuristic().build_optimizer_config(
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        hidden_dim=model.hidden_dim,
        seq_len=model.max_seq_len,
    )
    return model, optimizer
