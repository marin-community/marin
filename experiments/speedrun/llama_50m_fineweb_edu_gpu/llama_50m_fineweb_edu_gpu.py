"""
Speedrun code for a 50M parameter model based on the LLaMA architecture. The model is trained on the Fineweb-Edu dataset
(the default dataset for speedruns) on one A100.
"""

import logging

from experiments.llama import llama_50m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import HardwareConfig, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    model_config=llama_50m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1),
        train_batch_size=128,
        num_train_steps=1000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
        steps_per_task_eval=500,
    ),
    hardware_config=HardwareConfig(
        device_type="a100",
        num_devices=1,
        device_flops=312e12,
    ),
)

if __name__ == "__main__":
    executor_main(steps=default_speedrun("50M_llama_fineweb_edu_gpu", speedrun_config))
