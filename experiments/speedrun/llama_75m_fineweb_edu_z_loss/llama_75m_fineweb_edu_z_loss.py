"""
Speedrun code for a 75M parameter model based on the LLaMA architecture.
The run also experiments with setting z_loss_weight to 1e-4.
"""

import logging

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import HardwareConfig, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v4-128"),
        train_batch_size=512,
        num_train_steps=3000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1000,
        z_loss_weight=1e-4,
    ),
    hardware_config=HardwareConfig(
        device_type="v4-128",
        num_devices=64,
        device_flops=275e12,  # from https://cloud.google.com/tpu/docs/v4
    ),
)

if __name__ == "__main__":
    executor_main(steps=default_speedrun("75M_llama_fineweb_edu_z_loss", speedrun_config))
