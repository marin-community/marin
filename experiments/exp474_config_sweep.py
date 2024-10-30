# https://github.com/stanford-crfm/marin/issues/474
# Sweep to determine optimal training config
from typing import Sequence

import math
import numpy as np
from levanter.models.llama import LlamaConfig

from experiments.exp72_baselines import fineweb_edu_tokenized
from experiments.simple_train_config import SimpleTrainConfig
from experiments.llama import llama_150m

from marin.execution.executor import ExecutorStep, this_output_path, versioned, executor_main
from experiments.defaults import default_train


# TODO: might be nice to do use wandb sweeps, but not today.
# TODO: redo with mup

# Sweep to determine optimal training config
LR_CHOICES = [1e-4, 3e-4, 1e-3, 3e-3]
WD_CHOICES = [0.05, 0.1, 0.25, 0.33]
TPU_TYPES = ["v4-32", "v4-64"]
TOKEN_TARGETS = np.array([1, 3, 10, 30, 50]) * 1_000_000_000
BATCH_SIZE = [256, 512, 1024, 2048, 4096]
SEQ_LEN = 4096

def step_target(token_target, batch_size):
    actual_step_count = math.ceil(token_target / (batch_size * SEQ_LEN))
    nice_round_step_count = math.ceil(actual_step_count / 1000) * 1000
    return nice_round_step_count


train_configs = []

try:
    for lr in LR_CHOICES:
        for wd in WD_CHOICES:
            for tpu_type in TPU_TYPES:
                for batch_size in BATCH_SIZE:
                    for token_target in TOKEN_TARGETS:
                        num_train_steps = step_target(token_target, batch_size)
                        if num_train_steps <= 1000:
                            print(f"Skipping comically small number of steps: {num_train_steps}: {token_target=}, {batch_size=}")
                            continue

                        train_configs.append(
                            SimpleTrainConfig(
                                tpu_type=tpu_type,
                                train_batch_size=batch_size,
                                num_train_steps=num_train_steps,
                                learning_rate=lr,
                                weight_decay=wd,
                            )
                        )
                    raise ValueError("Stop")
except ValueError:
    pass


def format_train_config(prefix: str, config: SimpleTrainConfig):
    return f"{prefix}-bs={config.train_batch_size}-step={config.num_train_steps}-lr={config.learning_rate}-wd{config.weight_decay}"


def make_sweep_steps(
        prefix: str,
        model_config: LlamaConfig,
        train_configs: list[SimpleTrainConfig],
        tokenized_data: ExecutorStep,
        tags: Sequence[str] = (),
):
    steps = []
    for i, train_config in enumerate(train_configs):
        name = format_train_config(prefix, train_config)

        steps.append(
            default_train(
                name=name,
                train_config=train_config,
                model_config=model_config,
                tokenized=tokenized_data,
                tags=tags,
            )
        )
    return steps


steps_150m = make_sweep_steps(
    prefix="sweep474-150m",
    model_config=llama_150m,
    train_configs=train_configs,
    tokenized_data=fineweb_edu_tokenized,
    tags=("llama", "150m", "474_config_sweep", "fineweb_edu"),
)

if __name__ == "__main__":
    print(len(steps_150m))
    executor_main(steps_150m)