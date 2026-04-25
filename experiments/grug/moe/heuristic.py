# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compute-scaling AdamH heuristic for MoE ISOFlop sweeps.

All empirical fits below were measured on runs with seq_len=4096. The formulas
use tokens_per_batch (= batch_size * seq_len) so they generalize to other
sequence lengths, though the coefficients are an extrapolation beyond 4096.

Formulas (fit on v16 LR sweep, 186 runs, R²=0.995):
- Adam LR: adam_lr = lr_coeff * tokens^lr_tokens_exp * dim^lr_dim_exp * sqrt(B)
  (with lr_coeff=1.63, lr_tokens_exp=-0.2813, lr_dim_exp=-0.3678)
- AdamH LR: lr = (13/3) * adam_lr
- Compute budget convention: C = 3 * flops_per_token(no_lm_head) * tokens
- Epsilon: epsilon = epsilon_base * sqrt(r0/r), where r = (B*T0)/(B0*T)
- Beta1: fixed at 0.9062
- Beta2: beta2 = clip(beta2_base^(B/B0), min_beta2, max_beta2)
"""

import math
from dataclasses import dataclass

from levanter.utils.flop_utils import lm_flops_per_token

from experiments.grug.moe.model import GrugModelConfig
from experiments.grug.moe.optimizer import GrugMoeAdamHConfig

SEQ_LEN: int = 4096
MIN_BATCH_SIZE: int = 32
DEFAULT_TARGET_STEPS: int = 2**14


def _round_to_power_of_two(x: float) -> int:
    if x <= 1:
        return 1
    return 2 ** math.ceil(math.log2(x))


def compute_flops_per_token(cfg: GrugModelConfig) -> float:
    """Non-embedding FLOPs per token (excludes lm_head)."""
    fpt_with_lm_head = lm_flops_per_token(
        hidden_dim=cfg.hidden_dim,
        intermediate_dim=cfg.intermediate_dim,
        num_layers=cfg.num_layers,
        num_kv_heads=cfg.num_kv_heads,
        num_heads=cfg.num_heads,
        seq_len=cfg.max_seq_len,
        vocab_size=cfg.vocab_size,
        glu=True,
        num_experts=cfg.num_experts,
        num_shared_experts=1 if cfg.shared_expert_intermediate_dim > 0 else 0,
        num_experts_per_tok=cfg.num_experts_per_token,
        shared_intermediate_dim=cfg.shared_expert_intermediate_dim,
    )
    return fpt_with_lm_head - 2 * cfg.hidden_dim * cfg.vocab_size


def compute_tokens_and_batch(
    budget: float,
    flops_per_token: float,
    target_steps: int = DEFAULT_TARGET_STEPS,
    min_batch_size: int = MIN_BATCH_SIZE,
    seq_len: int = SEQ_LEN,
) -> tuple[float, int, int]:
    """Derive (tokens, batch_size, num_steps) from a compute budget and FLOPs-per-token.

    ``seq_len`` controls the sequence length used to convert between batch_size
    (sequences) and tokens_per_batch (tokens per step). All downstream formulas
    that depend on batch size use ``tokens_per_batch = batch_size * seq_len``
    so they work correctly at any sequence length.
    """
    tokens = budget / (3 * flops_per_token)
    batch_exact = tokens / (target_steps * seq_len)
    batch_size = max(min_batch_size, _round_to_power_of_two(batch_exact))
    train_steps = max(1, round(tokens / (batch_size * seq_len)))
    return tokens, batch_size, train_steps


@dataclass(frozen=True)
class MoeAdamHHeuristic:
    """Compute-scaling AdamH heuristic for MoE models.

    adam_lr = lr_coeff * tokens^lr_tokens_exp * dim^lr_dim_exp * sqrt(batch_size)
    adamh_lr = adamh_ratio * adam_lr
    C = 3 * flops_per_token * tokens  (flops_per_token excludes lm_head)
    """

    # --- LR scaling ---
    # adam_lr = lr_coeff * tokens^lr_tokens_exp * dim^lr_dim_exp * sqrt(tokens_per_batch)
    # Original (186 runs, R²=0.995) — used for v16 sweep:
    lr_coeff: float = 0.025469  # 1.63 / sqrt(4096)
    lr_tokens_exp: float = -0.2813
    lr_dim_exp: float = -0.3678
    adamh_ratio: float = 13 / 3

    # --- Base hyperparameters ---
    epsilon_coeff: float = 9.676e-18
    beta1: float = 0.9062
    beta2_base: float = 0.999
    beta2_reference_tpb: int = 131_072  # beta2 = beta2_base^(tpb / beta2_reference_tpb)

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
    gqa_ratio: int | None = 4  # None = MHA, 4 = 4:1 GQA, etc.
    base_hidden_layer_ratio: int = 64
    layer_scaling_factor: float = 4.0
    layer_formula_offset: int = 9
    depth_mup_residual_scaling: bool = False

    # --- Constraints ---
    max_learning_rate: float = 0.05
    min_beta2: float = 0.95
    max_beta2: float = 0.9999

    def _compute_adam_lr(self, tokens_per_batch: int, tokens: float, hidden_dim: int) -> float:
        """adam_lr = lr_coeff * tokens^lr_tokens_exp * dim^lr_dim_exp * sqrt(tokens_per_batch)"""
        adam_lr = (
            self.lr_coeff * (tokens**self.lr_tokens_exp) * (hidden_dim**self.lr_dim_exp) * math.sqrt(tokens_per_batch)
        )
        return min(self.max_learning_rate, adam_lr)

    def _compute_learning_rate(self, tokens_per_batch: int, tokens: float, hidden_dim: int) -> float:
        """adamh_lr = (13/3) * adam_lr"""
        adam_lr = self._compute_adam_lr(tokens_per_batch, tokens, hidden_dim)
        return min(self.max_learning_rate, self.adamh_ratio * adam_lr)

    def _compute_epsilon(self, tokens_per_batch: int, tokens: float) -> float:
        """epsilon = epsilon_coeff * sqrt(tokens / tokens_per_batch)"""
        return self.epsilon_coeff * math.sqrt(tokens / tokens_per_batch)

    def _compute_beta2(self, tokens_per_batch: int) -> float:
        """beta2 = clip(beta2_0^(tpb/tpb0), min_beta2, max_beta2). Constant token half-life."""
        exponent = tokens_per_batch / self.beta2_reference_tpb
        return max(self.min_beta2, min(self.max_beta2, self.beta2_base**exponent))

    def build_optimizer_config(
        self, batch_size: int, tokens: float, hidden_dim: int, seq_len: int = SEQ_LEN
    ) -> GrugMoeAdamHConfig:
        tokens_per_batch = batch_size * seq_len
        lr = self._compute_learning_rate(tokens_per_batch, tokens, hidden_dim)
        adam_lr = self._compute_adam_lr(tokens_per_batch, tokens, hidden_dim)
        epsilon = self._compute_epsilon(tokens_per_batch, tokens)
        beta2 = self._compute_beta2(tokens_per_batch)
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

    @staticmethod
    def _compute_kv_heads(num_heads: int, gqa_ratio: int | None) -> int:
        """Compute num_kv_heads for a given GQA ratio.

        If gqa_ratio is None, returns num_heads (MHA).
        Otherwise returns the largest divisor of num_heads <= num_heads // gqa_ratio.
        """
        if gqa_ratio is None:
            return num_heads
        target = num_heads // gqa_ratio
        for k in range(target, 0, -1):
            if num_heads % k == 0:
                return k
        return 1

    def build_model_config(self, hidden_size: int, seq_len: int = SEQ_LEN) -> GrugModelConfig:
        if hidden_size % self.hidden_head_ratio != 0:
            raise ValueError(
                f"hidden_size ({hidden_size}) must be divisible by hidden_head_ratio ({self.hidden_head_ratio})."
            )
        num_layers = self._compute_num_layers(hidden_size)
        num_heads = max(1, hidden_size // self.hidden_head_ratio)
        num_kv_heads = self._compute_kv_heads(num_heads, self.gqa_ratio)

        return GrugModelConfig(
            vocab_size=self.vocab_size,
            hidden_dim=hidden_size,
            # Round up to nearest 128 for Pallas TPU MoE kernel compatibility
            intermediate_dim=math.ceil(hidden_size / 2 / 128) * 128,
            shared_expert_intermediate_dim=hidden_size,
            num_experts=64,
            num_experts_per_token=4,
            num_layers=num_layers,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            max_seq_len=seq_len,
            sliding_window=seq_len,
            initializer_std=0.5 / math.sqrt(hidden_size),
            qk_mult=1.3,
            depth_mup_residual_scaling=self.depth_mup_residual_scaling,
        )


moe_adamh_heuristic = MoeAdamHHeuristic()


def build_from_heuristic(
    *,
    budget: float,
    hidden_dim: int,
    heuristic: MoeAdamHHeuristic | None = None,
    target_steps: int = DEFAULT_TARGET_STEPS,
    min_batch_size: int = MIN_BATCH_SIZE,
    seq_len: int = SEQ_LEN,
) -> tuple[GrugModelConfig, GrugMoeAdamHConfig, int, int]:
    """Construct (model, optimizer, batch_size, num_steps) for a compute budget.

    Uses `MoeAdamHHeuristic` to size the model (from `hidden_dim`) and to set
    the AdamH hyperparameters (scaled by tokens_per_batch = batch_size * seq_len).
    Callers who want manual control should continue passing `GrugModelConfig` /
    `GrugMoeAdamHConfig` directly to `GrugMoeLaunchConfig`.
    """
    h = heuristic or MoeAdamHHeuristic()
    model_cfg = h.build_model_config(hidden_dim, seq_len=seq_len)
    fpt = compute_flops_per_token(model_cfg)
    tokens, batch_size, num_steps = compute_tokens_and_batch(
        budget,
        fpt,
        target_steps=target_steps,
        min_batch_size=min_batch_size,
        seq_len=seq_len,
    )
    optimizer_cfg = h.build_optimizer_config(batch_size, tokens, hidden_dim, seq_len=seq_len)
    return model_cfg, optimizer_cfg, batch_size, num_steps
