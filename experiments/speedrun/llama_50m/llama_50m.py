"""
Speedrun code for a 50M parameter model based on the Llama architecture. The model is trained on the Fineweb-Edu dataset
(the default dataset for speedruns).
"""
import logging
from experiments.llama import llama_50m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Nikil Ravi",
        affiliation="Marin Community",
        url="https://www.linkedin.com/in/nikilravi/",
    ),
    description="50M parameter model based on Llama architecture.",
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
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("llama_50m", speedrun_config))