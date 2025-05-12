"""
This is a hello-world speedrun that trains a tiny model, and is designed to run on a single **GPU**.
See https://github.com/marin-community/marin/tree/main/docs/how-to-guides/submitting-speedrun.md for further documentation
around submitting speedrun, and for a more detailed walkthrough of the code and process.
"""

import logging

from experiments.llama import llama_nano
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import HardwareConfig, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    model_config=llama_nano,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v4-8"),
        train_batch_size=32,
        num_train_steps=100,
        learning_rate=6e-4,
        weight_decay=0.1,
    ),
    hardware_config=HardwareConfig(
        device_type="v4-8",  # a str used only for metadata/logging purposes (examples: 'h100', 'v4-128', etc)
        num_devices=4,
        device_flops=275e12,
    ),
)

if __name__ == "__main__":
    executor_main(steps=default_speedrun("hello_world_tpu_speedrun_sanity", speedrun_config))
