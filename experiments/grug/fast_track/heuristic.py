# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute-scaling LR heuristic.

``MoeHeuristic`` is the Aug LR-sweep refit (issues #7856 / #8003, R^2=0.978): it sets compute-optimal
MuonH / Adam learning rates, epsilon, and beta2 from the token budget and batch size.
"""

import math
from dataclasses import dataclass

from experiments.grug.fast_track.optimizer import GrugMoeMuonHConfig


@dataclass(frozen=True)
class MoeHeuristic:
    """Aug LR-sweep MuonH refit (issues #7856 / #8003, seq_len=8192, R^2=0.978).

    adam_lr  = lr_coeff * tokens^lr_tokens_exp * hidden_dim^lr_dim_exp * sqrt(tokens_per_batch)
    muonh_lr = muonh_ratio * adam_lr
    epsilon  = epsilon_coeff * sqrt(tokens / tokens_per_batch)
    beta2    = clip(beta2_base^(tokens_per_batch / beta2_reference_tpb), min_beta2, max_beta2)

    LR exponents/coefficient are the per-cell paloma-optimal fit: muonh_lr =
    34.35 * tokens^-0.346 * hidden^-0.345 * batch^0.5 at seq_len=8192, folded into the
    sqrt(tokens_per_batch) form (lr_coeff = 34.35 / (muonh_ratio * sqrt(8192))). Prior
    May-Recipe fit (#5951): lr_coeff=0.06602, lr_tokens_exp=-0.395, lr_dim_exp=-0.150.
    """

    lr_coeff: float = 0.087571
    lr_tokens_exp: float = -0.3461
    lr_dim_exp: float = -0.3448
    muonh_ratio: float = 13 / 3
    epsilon_coeff: float = 9.676e-18
    beta1: float = 0.90
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
