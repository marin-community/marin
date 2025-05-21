"""
Speedrun code for a 50M parameter model based on the OLMo architecture. The model is trained on the Fineweb-Edu dataset
(the default dataset for speedruns) on one H200.
"""

import logging

from levanter.models.olmo import Olmo2Config

from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

olmo_50m = Olmo2Config(
    seq_len=1024,
    hidden_dim=192,
    intermediate_dim=448,
    num_heads=2,
    num_kv_heads=2,
    num_layers=4,
)

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Will Held",
        affiliation="Georgia Institute of Technology",
        url="https://WilliamHeld.com",
    ),
    description="50M param model based on OLMo architecture.",
    model_config=olmo_50m,
    train_config=SimpleTrainConfig(
        GpuConfig(gpu_count=1, accelerator_type="H200"),
        train_batch_size=128,
        num_train_steps=7600,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=500,
    ),
)

speedrun_config.print_run_info()

if __name__ == "__main__":
    executor_main(steps=default_speedrun("olmo_50m_gpu_1xH200_run", speedrun_config))
