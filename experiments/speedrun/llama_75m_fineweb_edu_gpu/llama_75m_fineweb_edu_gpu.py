"""
Speedrun code for a 75M parameter model based on the LLaMA architecture on a 2 A100s.
"""

import logging

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import HardwareConfig, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=2),
        train_batch_size=64,
        num_train_steps=1000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
    ),
    hardware_config=HardwareConfig(
        device_type="a100",
        num_devices=2,
        device_flops=195e11,
    ),
)

if __name__ == "__main__":
    executor_main(steps=default_speedrun("75M_llama_fineweb_edu_speedrun_gpu", speedrun_config))
