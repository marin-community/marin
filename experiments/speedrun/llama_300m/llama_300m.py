"""
Speedrun code for a 300M parameter model based on the Llama architecture.
"""

import logging

from experiments.llama import llama_300m
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.prebuilt_caches import fineweb_edu_subcache_10B
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Nikil Ravi",
        affiliation="Stanford University",
        url="https://www.linkedin.com/in/nikilravi/",
    ),
    description="300M param model based on Llama architecture.",
    model_config=llama_300m,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v4-256"),
        train_batch_size=1024,
        num_train_steps=6000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
    tokenized_dataset=fineweb_edu_subcache_10B,
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(
        steps=default_speedrun(
            "llama_300m_run", config=speedrun_config, override_output_path="checkpoints/speedrun/llama_300m_run-e76a8f"
        )
    )
