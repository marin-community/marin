# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Speedrun code for a 300M activated parameter MoE model based on the Mixtral architecture.
This model has 32 experts and only activates 4 of them.
"""

import logging

from levanter.models.mixtral import MixtralConfig

from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from fray.cluster import ResourceConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

moe_300m = MixtralConfig(
    max_seq_len=1024,
    hidden_dim=768,
    intermediate_dim=768,
    num_heads=12,
    num_kv_heads=12,
    num_layers=12,
    n_routed_experts=32,
    num_experts_per_tok=4,
)

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Jason Wang",
        affiliation="Stanford University",
        url="https://www.linkedin.com/in/jason-wang-468117193/",
    ),
    description=(
        "300M activated parameter MoE model based on the Mixtral architecture. "
        "Has 32 experts and only activates 4 of them."
    ),
    model_config=moe_300m,
    train_config=SimpleTrainConfig(
        ResourceConfig.with_tpu("v4-256"),
        train_batch_size=1024,
        num_train_steps=3000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("moe_mixtral_300m_activated", speedrun_config))
