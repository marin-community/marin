"""
Speedrun code for a 50M parameter model based on the LLaMA architecture. The model is trained on the Fineweb-Edu dataset
(the default dataset for speedruns).
"""
import logging
from experiments.llama import llama_50m
 from experiments.simple_train_config import SimpleTrainConfig
 from marin.execution.executor import executor_main
 from marin.resources import TpuPodConfig
 from marin.speedrun.speedrun import HardwareConfig, SpeedrunConfig, default_speedrun

 logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    model_config=llama_50m,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v4-128"),
        train_batch_size=512,
        num_train_steps=4500,
        learning_rate=3e-3,
        weight_decay=0.1,
         steps_per_eval=1500,
         steps_per_task_eval=1500,
     ),
     hardware_config=HardwareConfig(
         device_type="v4-128",
         num_devices=64,
         device_flops=275e12,  # from https://cloud.google.com/tpu/docs/v4
     ),
 )

 if __name__ == "__main__":
    executor_main(steps=default_speedrun("50M_llama_fineweb_edu", speedrun_config))