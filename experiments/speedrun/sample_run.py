"""
Sample speedrun script demonstrating how to configure and run a model within compute constraints.
"""

from experiments.dclm.tokenize_dclm import dclm_mixture_config_llama3
from experiments.llama import llama_150m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.speedrun import ComputeBudget, SpeedrunConfig, default_speedrun
from marin.execution.executor import executor_main


# Configure speedrun with 150M LLaMA model
speedrun_config = SpeedrunConfig(
    compute_budget=ComputeBudget.SMALL,
    model_config=llama_150m,
    train_config=SimpleTrainConfig(
        tpu_type="v4-128",
        train_batch_size=512,
        num_train_steps=6000,  # 512 * 1024 * 6000 = ~3B tokens (3.1457B tokens)
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=2000,
        steps_per_task_eval=2000,
    ),
    tokenized_dataset=dclm_mixture_config_llama3,
    hardware_config=HardwareConfig(
        device_type="v4-128",
        num_devices=1,
        device_flops=1e18
    ),
)

speedrun_steps = default_speedrun(
    name="speedrun/150M_llama_dclm_mix_Apr2",
    config=speedrun_config,
)

if __name__ == "__main__":
    executor_main(steps=speedrun_steps)
