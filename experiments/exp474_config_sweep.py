# https://github.com/stanford-crfm/marin/issues/474
# Sweep to determine optimal training configs for small models
import dataclasses
import itertools
import logging
import math
from collections.abc import Sequence

import numpy as np
import ray
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_train
from experiments.exp72_baselines import fineweb_edu_tokenized
from experiments.llama import llama_150m, llama_300m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, executor_main, versioned

# TODO: might be nice to do use wandb sweeps, but not today.
# TODO: redo with mup

logger = logging.getLogger("ray")

# Sweep to determine optimal training config
LR_CHOICES = [1e-4, 1e-3, 3e-3, 1e-2]  # 3e-3 is best, 1e-2 diverges at 300m
WD_CHOICES = [0.1]
TPU_TYPES_150m = ["v4-32"]
TPU_TYPES_300m = ["v4-64"]
TOKEN_TARGETS = np.array([1, 3, 6, 10, 18, 30]) * 1_000_000_000
# shockingly, 512 and 256 have highest throughput. I think this is because of
# Levanter's naive FA implementation. (Splash attention doesn't work with headsize 64)
BATCH_SIZE = [256, 512, 1024, 2048, 4096]
# ATM None is best, but 512 is best of the non-None
CE_BLOCK_SIZE = [256, 512, 2048, 4096, 32000, 64000, None]
HEAD_SIZE = [128]


def all_combos(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo, strict=False))


def step_target(token_target, batch_size, seq_len):
    actual_step_count = math.ceil(token_target / (batch_size * seq_len))
    nice_round_step_count = math.ceil(actual_step_count / 1000) * 1000
    return nice_round_step_count


train_configs_150m = []

for combo in all_combos(
    lr=LR_CHOICES, wd=WD_CHOICES, tpu_type=TPU_TYPES_150m, batch_size=BATCH_SIZE, token_target=TOKEN_TARGETS
):
    num_train_steps = step_target(combo["token_target"], combo["batch_size"], llama_150m.seq_len)
    if num_train_steps <= 1000:
        print(
            "Skipping comically small number of steps:"
            f" {num_train_steps} {combo['token_target']=}, {combo['batch_size']=}"
        )
        continue

    train_configs_150m.append(
        SimpleTrainConfig(
            tpu_type=versioned(combo["tpu_type"]),
            train_batch_size=combo["batch_size"],
            num_train_steps=num_train_steps,
            learning_rate=combo["lr"],
            weight_decay=combo["wd"],
        )
    )


train_configs_300m = []

for combo in all_combos(
    lr=LR_CHOICES, wd=WD_CHOICES, tpu_type=TPU_TYPES_300m, batch_size=BATCH_SIZE, token_target=TOKEN_TARGETS
):
    num_train_steps = step_target(combo["token_target"], combo["batch_size"], llama_300m.seq_len)
    if num_train_steps <= 1000:
        print(
            "Skipping comically small number of steps:"
            f" {num_train_steps} {combo['token_target']=}, {combo['batch_size']=}"
        )
        continue

    train_configs_300m.append(
        SimpleTrainConfig(
            tpu_type=versioned(combo["tpu_type"]),
            train_batch_size=combo["batch_size"],
            num_train_steps=num_train_steps,
            learning_rate=combo["lr"],
            weight_decay=combo["wd"],
        )
    )


def format_train_config(prefix: str, config: SimpleTrainConfig, ce_bs, hs):
    return (
        f"{prefix}-hs={hs}-ce={ce_bs}-bs={config.train_batch_size}-step={config.num_train_steps}"
        f"-lr={config.learning_rate}-wd{config.weight_decay}"
    )


def _failure_ok_train(*args, **kwargs):
    """
    Wrapper to catch exceptions and log them, but not fail the whole sweep. We do this because some batch sizes are too
    big.
    """
    from marin.training.training import run_levanter_train_lm

    try:
        return ray.get(run_levanter_train_lm.remote(*args, **kwargs))
    except Exception as e:
        logger.exception("Failed to run training", exc_info=e)
        return None


def make_sweep_steps(
    prefix: str,
    model_config: LlamaConfig,
    train_configs: list[SimpleTrainConfig],
    tokenized_data: ExecutorStep,
    tags: Sequence[str] = (),
):
    steps = []
    for train_config in train_configs:
        for ce_block_size in CE_BLOCK_SIZE:
            for head_size in HEAD_SIZE:
                num_heads = model_config.hidden_dim // head_size
                model_config = dataclasses.replace(
                    model_config,
                    cross_entropy_block_size=ce_block_size,
                    num_heads=num_heads,
                    num_kv_heads=num_heads,
                )
                name = format_train_config(prefix, train_config, ce_block_size, head_size)

                step = default_train(
                    name=name,
                    train_config=train_config,
                    model_config=model_config,
                    tokenized=tokenized_data,
                    tags=tags,
                )

                step = dataclasses.replace(step, fn=_failure_ok_train)

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
