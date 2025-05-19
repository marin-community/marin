"""
Speedrun code for a 75M parameter model based on the Llama architecture on a 2 A100s.
"""

import logging

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Herumb Shandilya",
        affiliation="Stanford University",
        url="https://www.linkedin.com/in/herumb-shandilya/",
    ),
    description="75M parameter model based on Llama architecture.",
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=2, accelerator_type="A100"),
        train_batch_size=64,
        num_train_steps=1000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_75m_gpu", speedrun_config))
