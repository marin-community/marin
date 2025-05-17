"""
Speedrun code for a 75M parameter model based on the LLaMA architecture.
"""

import logging

from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    author="Nikil Ravi",
    affiliation="Marin Community",
    description="75M parameter model based on LLaMA architecture.",
    model_config=llama_75m,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type="v4-128", slice_count=2, device_flops_override=1e15),
        train_batch_size=512,
        num_train_steps=3000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)

if __name__ == "__main__":
    # Debug info about devices
    print(f"Number of devices: {speedrun_config.num_devices}")
    print(f"Device FLOPs: {speedrun_config.device_flops}")
    print(f"Total peak FLOPs: {speedrun_config.device_flops * speedrun_config.num_devices}")

    print(f"Estimated model flops: {speedrun_config.estimate_model_flops()}")

    # Comment out the actual run for now
    # executor_main(steps=default_speedrun("75M_llama_fineweb_edu", speedrun_config))
