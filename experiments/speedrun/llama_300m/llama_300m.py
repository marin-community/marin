"""
Sample speedrun with an 75M LLaMA model.
"""

import logging

from experiments.llama import llama_300m
from experiments.simple_train_config import SimpleTrainConfig
from marin.speedrun.speedrun import ComputeBudget, HardwareConfig, SpeedrunConfig, default_speedrun
from marin.execution.executor import executor_main

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    compute_budget=ComputeBudget.MEDIUM,
    model_config=llama_300m,
    train_config=SimpleTrainConfig(
        tpu_type="v4-256",
        train_batch_size=1024,
        num_train_steps=3000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
    hardware_config=HardwareConfig(
        device_type="v4-256",
        num_devices=128,
        device_flops=275e12,  # one v4 chip is capable of 275 TFLOPs of peak compute: https://cloud.google.com/tpu/docs/v4
    ),
)

# can choose to validate configuration before training
is_valid, error = speedrun_config.validate()
logger.info(f"Speedrun validation: {is_valid}, {error}")

if __name__ == "__main__":
    executor_main(steps=default_speedrun("300M_llama", speedrun_config))
