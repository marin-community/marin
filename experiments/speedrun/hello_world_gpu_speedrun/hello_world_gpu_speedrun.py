"""
This is a hello-world speedrun that trains a tiny model, and is designed to run on a single **GPU**.
See https://github.com/marin-community/marin/tree/main/docs/how-to-guides/submitting-speedrun.md for
further documentation around submitting speedrun, and for a more detailed walkthrough of the code
and process.
"""

import logging

from experiments.llama import llama_nano
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.prebuilt_caches import fineweb_edu_subcache_10B
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Herumb Shandilya",
        affiliation="Stanford University",
        url="https://www.x.com/krypticmouse",
    ),
    description="Nano model based on Llama architecture.",
    model_config=llama_nano,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="A100"),
        train_batch_size=32,
        num_train_steps=100,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
    ),
    tokenized_dataset=fineweb_edu_subcache_10B,
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_nano_gpu_speedrun", speedrun_config))
