# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Mixtral MoE with null expert data sparsity (arXiv:2601.15370) on v4-64.

Trains a ~300M Mixtral-style MoE with data_sparsity=0.5, meaning half of the
top-k routing slots are expected to land on null (zero-output) experts.
This composes weight sparsity (standard top-k MoE) with data sparsity
(variable tokens per expert) for a more compute-efficient frontier.
"""

# nodryrun

import logging

from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.custom_mixtral import MixtralConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

# ~300M Mixtral with 8 routed experts, top-4, and 50% data sparsity.
# At rho=0.5 with k_max=4, the expected real experts per token is ~2.
# M = N*(1-rho)/rho = 8 null copies, so the router sees 16 slots total.
# Based on the proven pranshu_mixtral_moe_run 300M config that runs on v4-8.
moe_300m_null_expert = MixtralConfig(
    seq_len=1024,
    hidden_dim=768,
    intermediate_dim=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    gradient_checkpointing=True,
    scan_layers=True,
    n_routed_experts=8,
    num_experts_per_tok=4,
    use_gmm=False,
    # Null expert data sparsity
    data_sparsity=0.5,
    # Aux losses (important for routing null experts uniformly)
    lbl_coef=0.02,
    rzl_coef=0.001,
)

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Marin Team",
        affiliation="Marin Project",
        url=None,
    ),
    description=(
        "Mixtral ~300M MoE with null expert data sparsity (rho=0.5). "
        "8 real experts, top-4 routing, 8 null copies (16 total routing slots). "
        "Expected ~2 real experts per token."
    ),
    model_config=moe_300m_null_expert,
    train_config=SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v4-64"),
        train_batch_size=256,
        num_train_steps=4000,
        learning_rate=5e-4,
        weight_decay=0.1,
        steps_per_eval=500,
    ),
)

if __name__ == "__main__":
    cfg = moe_300m_null_expert
    logger.info(
        "Launching Mixtral null-expert run: hidden=%d, layers=%d, experts=%d, "
        "top_k=%d, data_sparsity=%.2f, null_experts=%d",
        cfg.hidden_dim,
        cfg.num_layers,
        cfg.n_routed_experts,
        cfg.num_experts_per_tok,
        cfg.data_sparsity,
        cfg.NullExperts.size,
    )
    executor_main(steps=default_speedrun("mixtral_300m_null_expert_rho05_v4-64", speedrun_config))
