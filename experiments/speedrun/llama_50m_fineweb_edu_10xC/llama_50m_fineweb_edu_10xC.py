"""
Sample speedrun with an 50M LLaMA model.
"""

import logging

from experiments.llama import llama_50m
from experiments.simple_train_config import SimpleTrainConfig
from marin.speedrun.speedrun import HardwareConfig, SpeedrunConfig, default_speedrun
from marin.execution.executor import executor_main

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    model_config=llama_50m,
    train_config=SimpleTrainConfig(
        tpu_type="v4-128",
        train_batch_size=512,
        num_train_steps=20000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1500,
    ),
    hardware_config=HardwareConfig(
        device_type="v4-128",
        num_devices=64,
        device_flops=275e12,  # from https://cloud.google.com/tpu/docs/v4
    ),
)

if __name__ == "__main__":
    override_output_path = "checkpoints/speedrun/50M_llama_fineweb_edu_10xC-5f74d4"
    executor_main(steps=default_speedrun("50M_llama_fineweb_edu_10xC", speedrun_config, override_output_path=override_output_path))
