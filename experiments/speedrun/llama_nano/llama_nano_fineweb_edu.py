"""
Speedrun code for a nano model based on the LLaMA architecture.
So tiny that you can run it on your laptop. Used to test the speedrun framework.
"""

import logging

from experiments.llama import llama_nano
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Joel Niklaus  ",
        affiliation="Niklaus AI",
        url="https://niklaus.ai",
    ),
    description="LLaMA-Nano on Fineweb-edu",
    model_config=llama_nano,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v4-32"),
        train_batch_size=512,
        num_train_steps=3000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)

if __name__ == "__main__":
    executor_main(steps=default_speedrun("nano_llama_fineweb_edu", speedrun_config))
