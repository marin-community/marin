"""
Speedrun code for a 30M parameter model based on the LLaMA architecture.
"""

import logging

from experiments.llama import llama_30m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    model_config=llama_30m,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(
            tpu_type="v4-128",
            slice_count=64,
        ),
        train_batch_size=512,
        num_train_steps=1500,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
        steps_per_task_eval=500,
    ),
)

if __name__ == "__main__":
    executor_main(steps=default_speedrun("30M_llama_fineweb", speedrun_config))
