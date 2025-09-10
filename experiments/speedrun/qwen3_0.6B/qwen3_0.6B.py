from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun
from marin.resources import TpuPodConfig
from experiments.simple_train_config import SimpleTrainConfig
from experiments.qwen3 import qwen3_06b


speedrun_config = SpeedrunConfig(
    author=Author(
        name="Calvin Xu",
        affiliation="Stanford University",
        url="https://pinlinxu.com",
    ),
    description="Qwen3-0.6B with QK-Norm",
    model_config=qwen3_06b,
    train_config=SimpleTrainConfig(
        resources=TpuPodConfig(tpu_type="v4-128"),
        train_batch_size=256,
        num_train_steps=3000,
        learning_rate=3e-3,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)


if __name__ == "__main__":
    speedrun_config.print_run_info()
    executor_main(steps=default_speedrun("qwen3_0.6B_tpu_qk_norm", speedrun_config))
