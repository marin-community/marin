"""
Speedrun code for a 300M parameter model based on the LLaMA architecture.
"""

import logging

from experiments.llama import llama_300m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import HardwareConfig, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    model_config=llama_300m,
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
    executor_main(steps=default_speedrun("300M_llama", speedrun_config))