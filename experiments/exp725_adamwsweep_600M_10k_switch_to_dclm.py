# https://github.com/stanford-crfm/marin/issues/725
# Sweep to determine optimal hyperparameters for Adam on small scale
import itertools
import logging
from collections.abc import Sequence

import ray
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_train
from experiments.exp600_tootsie import dclm_mixture_config_llama3
from experiments.exp725_adamwsweep_150M_50k_switch_to_dclm import format_train_config
from experiments.llama import llama_600m
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, executor_main, versioned

logger = logging.getLogger("ray")

# Sweep to determine optimal training config
BATCH_SIZE = 4096
target_steps = [10000]
TPU_TYPES_600m = "v4-256"


def all_combos(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo, strict=False))


sweep_grids = {
    "learning_rate": [1e-3, 2e-3, 4e-3, 8e-3, 1.6e-2, 3.2e-2],
    "weight_decay": [0, 0.1, 0.2],
    "min_lr_ratio": [0, 0.05, 0.1],
    "warmup": [1000, 2000, 4000, 8000],
    "beta1": [0.9, 0.95, 0.98, 0.99],
    "beta2": [0.9, 0.95, 0.98, 0.99],
    "epsilon": [1e-15, 1e-10, 1e-5],
    "max_grad_norm": [0, 1.0, 2.0],
}

baseline_config = {
    "learning_rate": 8e-3,
    "weight_decay": 0.1,
    "min_lr_ratio": 0,
    "warmup": 2000,
    "beta1": 0.95,
    "beta2": 0.95,
    "epsilon": 1e-15,
    "max_grad_norm": 1.0,
}


versioned_sweep_grids = {}
versioned_baseline_config = {}

for key in sweep_grids:
    versioned_sweep_grids[key] = [versioned(v) for v in sweep_grids[key]]
    versioned_baseline_config[key] = versioned(baseline_config[key])

sweep_grids = versioned_sweep_grids
baseline_config = versioned_baseline_config

train_configs_600m = []

import copy

for step in target_steps:
    train_configs_600m.append(
        SimpleTrainConfig(
            tpu_type=versioned(TPU_TYPES_600m),
            train_batch_size=BATCH_SIZE,
            steps_per_eval=1000,
            num_train_steps=step,
            **baseline_config,
        )
    )
    for key in sweep_grids:
        for value in sweep_grids[key]:
            new_config = copy.copy(baseline_config)
            if baseline_config[key] != value:
                new_config[key] = value
                train_configs_600m.append(
                    SimpleTrainConfig(
                        tpu_type=versioned(TPU_TYPES_600m),
                        train_batch_size=BATCH_SIZE,
                        steps_per_eval=1000,
                        num_train_steps=step,
                        **new_config,
                    )
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
        name = format_train_config(prefix, train_config)
        step = default_train(
            name=name,
            train_config=train_config,
            model_config=model_config,
            tokenized=tokenized_data,
            tags=tags,
        )

        # step = dataclasses.replace(step, fn=_failure_ok_train)

        steps.append(step)
    return steps


steps_600m = make_sweep_steps(
    prefix="sweep-725-600m-10k",
    model_config=llama_600m,
    train_configs=train_configs_600m,
    tokenized_data=dclm_mixture_config_llama3,
    tags=("llama", "600m", "725_adamw_sweep", "dclm", "10k"),
)


if __name__ == "__main__":
    executor_main(steps_600m)
