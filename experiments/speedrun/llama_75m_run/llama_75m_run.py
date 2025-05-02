"""
Sample speedrun with an 75M LLaMA model.
"""

import logging

from experiments.dclm.tokenize_dclm import dclm_components_llama3
from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from marin.speedrun.speedrun import HardwareConfig, SpeedrunConfig, default_speedrun
from marin.execution.executor import executor_main

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        tpu_type="v4-128",
        train_batch_size=512,
        num_train_steps=3000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1000,
        steps_per_task_eval=1000,
    ),
    tokenized_dataset=dclm_components_llama3["dclm_baseline"],
    hardware_config=HardwareConfig(
        device_type="v4-128",
        num_devices=64,
        device_flops=275e12,
    ),
)

if __name__ == "__main__":
    override_output_path = "checkpoints/speedrun/75M_llama_dclm_baseline-b7a5f5"
    executor_main(steps=default_speedrun("75M_llama_dclm_baseline", speedrun_config, override_output_path=override_output_path))
