"""
Speedrun code for a 300M activated parameter MoE model based on the Mixtral architecture.
This model has 32 experts and only activates 4 of them.
"""

import logging

from levanter.models.mixtral import MixtralConfig

from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import HardwareConfig, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

moe_300m = MixtralConfig(
    seq_len=1024,
    hidden_dim=768,
    intermediate_dim=768,
    num_heads=12,
    num_kv_heads=12,
    num_layers=12,
    n_routed_experts=32,
    num_experts_per_tok=4,
)

speedrun_config = SpeedrunConfig(
    model_config=moe_300m,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v4-256"),
        train_batch_size=1024,
        num_train_steps=3000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
    hardware_config=HardwareConfig(
        device_type="v4-256",
        num_devices=128,
        device_flops=275e12,  # from https://cloud.google.com/tpu/docs/v4
    ),
)

if __name__ == "__main__":
    executor_main(steps=default_speedrun("300M_moe-2", speedrun_config))
