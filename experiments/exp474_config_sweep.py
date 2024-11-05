# https://github.com/stanford-crfm/marin/issues/474
# Sweep to determine optimal training config
import dataclasses
import functools
import logging
import math
from collections.abc import Sequence

import numpy as np
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_train
from experiments.exp72_baselines import fineweb_edu_tokenized
from experiments.llama import llama_150m, llama_300m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, executor_main

# TODO: might be nice to do use wandb sweeps, but not today.
# TODO: redo with mup

logger = logging.getLogger("ray")

# Sweep to determine optimal training config
LR_CHOICES = [1e-3, 3e-3, 1e-2]
WD_CHOICES = [0.05, 0.1, 0.25]
TPU_TYPES_150m = ["v4-32", "v4-64"]
TPU_TYPES_300m = ["v4-64"]
TOKEN_TARGETS = np.array([3, 6, 10, 18, 30]) * 1_000_000_000
BATCH_SIZE = [256, 512, 1024, 2048]


def step_target(token_target, batch_size, seq_len):
    actual_step_count = math.ceil(token_target / (batch_size * seq_len))
    nice_round_step_count = math.ceil(actual_step_count / 1000) * 1000
    return nice_round_step_count


train_configs_150m = []

for lr in LR_CHOICES:
    for wd in WD_CHOICES:
        for tpu_type in TPU_TYPES_150m:
            for batch_size in BATCH_SIZE:
                for token_target in TOKEN_TARGETS:
                    num_train_steps = step_target(token_target, batch_size, llama_150m.seq_len)
                    if num_train_steps <= 1000:
                        print(
                            f"Skipping comically small number of steps: {num_train_steps}: {token_target=}, {batch_size=}"
                        )
                        continue

                    train_configs_150m.append(
                        SimpleTrainConfig(
                            tpu_type=tpu_type,
                            train_batch_size=batch_size,
                            num_train_steps=num_train_steps,
                            learning_rate=lr,
                            weight_decay=wd,
                        )
                    )


train_configs_300m = []

for lr in LR_CHOICES:
    for wd in WD_CHOICES:
        for tpu_type in TPU_TYPES_300m:
            for batch_size in BATCH_SIZE:
                for token_target in TOKEN_TARGETS:
                    num_train_steps = step_target(token_target, batch_size, llama_300m.seq_len)
                    if num_train_steps <= 1000:
                        print(
                            f"Skipping comically small number of steps: {num_train_steps}: {token_target=}, {batch_size=}"
                        )
                        continue

                    train_configs_300m.append(
                        SimpleTrainConfig(
                            tpu_type=tpu_type,
                            train_batch_size=batch_size,
                            num_train_steps=num_train_steps,
                            learning_rate=lr,
                            weight_decay=wd,
                        )
                    )


def format_train_config(prefix: str, config: SimpleTrainConfig):
    return f"{prefix}-bs={config.train_batch_size}-step={config.num_train_steps}-lr={config.learning_rate}-wd{config.weight_decay}"


def _failure_ok(fn):
    """
    Decorator to catch exceptions and log them, but not fail the whole sweep.
    """

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception:
            logger.exception(f"Failed to run {fn.__name__} with {args=} {kwargs=}")
            return None

    return wrapper


def make_sweep_steps(
    prefix: str,
    model_config: LlamaConfig,
    train_configs: list[SimpleTrainConfig],
    tokenized_data: ExecutorStep,
    tags: Sequence[str] = (),
):
    steps = []
    for train_config in train_configs:
        name = format_train_config(prefix, train_config)

        step = default_train(
            name=name,
            train_config=train_config,
            model_config=model_config,
            tokenized=tokenized_data,
            tags=tags,
        )

        # because a lot of batch sizes are too big (for now) and we don't want to fail the whole sweep
        step = dataclasses.replace(step, fn=_failure_ok(step.fn))

        steps.append(step)
    return steps


steps_150m = make_sweep_steps(
    prefix="sweep474-150m",
    model_config=llama_150m,
    train_configs=train_configs_150m,
    tokenized_data=fineweb_edu_tokenized,
    tags=("llama", "150m", "474_config_sweep", "fineweb_edu"),
)

steps_300m = make_sweep_steps(
    prefix="sweep474-300m",
    model_config=llama_300m,
    train_configs=train_configs_300m,
    tokenized_data=fineweb_edu_tokenized,
    tags=("llama", "300m", "474_config_sweep", "fineweb_edu"),
)

if __name__ == "__main__":
    executor_main(steps_150m + steps_300m)
