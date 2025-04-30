"""
Sample speedrun run using the custom Muon optimizer.
"""

import logging

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from experiments.llama import llama_150m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.speedrun import ComputeBudget, HardwareConfig, SpeedrunConfig, default_speedrun
from marin.execution.executor import executor_main
from experiments.speedrun.custom_muon_optimizer import MuonConfig

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    compute_budget=ComputeBudget.SMALL,
    model_config=llama_150m,
    train_config=SimpleTrainConfig(
        tpu_type="v4-128",
        train_batch_size=512,
        num_train_steps=6000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=2000,
        steps_per_task_eval=2000,
        optimizer_config=MuonConfig(learning_rate=3e-3, weight_decay=0.1),
    ),
    tokenized_dataset=dclm_mixture_config_llama3,
    hardware_config=HardwareConfig(
        device_type="v4-128",
        num_devices=64,
        device_flops=275e12,
    ),
)

is_valid, error = speedrun_config.validate()
logger.info(f"Speedrun validation: {is_valid}, {error}")

if __name__ == "__main__":
    executor_main(steps=default_speedrun("150M_llama_muon", speedrun_config))
