# nodryrun
import logging
import dataclasses
from experiments.llama import llama_75m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

# Create a copy of llama_75m with cross_entropy_block_size=32000 for TPU v4-8
llama_75m_tpu_v4_8 = dataclasses.replace(
    llama_75m,
    cross_entropy_block_size=32000
)

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Pranshu Chaturvedi",
        affiliation="Stanford University",
        url="https://stanford.edu/~pranshu",
    ),
    description="Training Llama 75M on a TPU v4-8 for the speedrun.",
    model_config=llama_75m_tpu_v4_8,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v4-8"),
        train_batch_size=512,
        num_train_steps=3000,
        learning_rate=3e-4,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("pranshu_llama_75m_speedrun", speedrun_config))
