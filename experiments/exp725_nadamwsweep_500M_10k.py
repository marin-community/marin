# https://github.com/stanford-crfm/marin/issues/725
# Sweep to determine optimal hyperparameters for Adam on small scale
import dataclasses
import itertools
import logging
import math
from collections.abc import Sequence
from experiments.exp600_tootsie import dclm_mixture_config_llama3

import numpy as np
import ray
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_train
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, executor_main, versioned, unwrap_versioned_value


def format_train_config(prefix: str, config: SimpleTrainConfig):
    name = (
        f"lr{unwrap_versioned_value(config.learning_rate)}-"
        f"wd{unwrap_versioned_value(config.weight_decay)}-"
        f"minlr{unwrap_versioned_value(config.min_lr_ratio)}-"
        f"warmup{unwrap_versioned_value(config.warmup)}-"
        f"b1{unwrap_versioned_value(config.beta1)}-"
        f"b2{unwrap_versioned_value(config.beta2)}-"
        f"gn{unwrap_versioned_value(config.max_grad_norm)}-"
        f"steps{unwrap_versioned_value(config.num_train_steps)}"
        f"eps{unwrap_versioned_value(config.epsilon)}-"
    )
    hashed_name = str(hash(name))[:6]
    return (prefix + '_' + hashed_name + name)[:64]



logger = logging.getLogger("ray")

# Sweep to determine optimal training config
BATCH_SIZE = 4096
target_steps = [10000]
TPU_TYPES_500m = ["v4-128"]



def all_combos(**kwargs):
    keys = kwargs.keys()
    values = kwargs.values()
    for combo in itertools.product(*values):
        yield dict(zip(keys, combo, strict=False))

# sweep_grids = {
#     'learning_rate': [4e-3, 8e-3, 1.6e-2, 3.2e-2],
#     'weight_decay': [0, 0.1, 0.2],
#     'min_lr_ratio': [0, 0.05, 0.1],
#     'warmup': [1000, 2000, 4000, 8000, 16000],
#     'beta1': [0.8, 0.9, 0.95, 0.98],
#     'beta2': [0.9, 0.95, 0.98],
#     'epsilon': [1e-20, 1e-15, 1e-10],
#     'max_grad_norm': [0, 1.0, 2.0],
# }
# baseline_config = {
#     'learning_rate': 8e-3, 
#     'weight_decay': 0.1,
#     'min_lr_ratio': 0,
#     'warmup': 4000,
#     'beta1': 0.95,
#     'beta2': 0.95,
#     'epsilon': 1e-15,
#     'max_grad_norm': 1.0
# }

sweep_grids = {
    'learning_rate': [4e-3, 8e-3, 1.6e-2, 3.2e-2],
    'weight_decay': [0, 0.1, 0.2],
    'min_lr_ratio': [0, 0.05, 0.1],
    'warmup': [1000, 2000, 4000, 8000],
    'beta1': [0.8, 0.9, 0.95, 0.98],
    'beta2': [0.9, 0.95, 0.98],
    'epsilon': [1e-20, 1e-15, 1e-10],
    'max_grad_norm': [0, 1.0, 2.0],
}
# baseline_config = {
#     'learning_rate': 8e-3, 
#     'weight_decay': 0.1,
#     'min_lr_ratio': 0,
#     'warmup': 2000,
#     'beta1': 0.95,
#     'beta2': 0.95,
#     'epsilon': 1e-15,
#     'max_grad_norm': 1.0,
#     'nesterov': True
# }


baseline_config = {
    'learning_rate': 8e-3, 
    'weight_decay': 0.1,
    'min_lr_ratio': 0,
    'warmup': 1000,
    'beta1': 0.95,
    'beta2': 0.95,
    'epsilon': 1e-15,
    'max_grad_norm': 1.0,
    'nesterov': True
}

versioned_sweep_grids = {}
versioned_baseline_config = {}

for key in sweep_grids:
    versioned_sweep_grids[key] = [versioned(v) for v in sweep_grids[key]]

for key in baseline_config:
    versioned_baseline_config[key] = versioned(baseline_config[key])

sweep_grids = versioned_sweep_grids
baseline_config = versioned_baseline_config



train_configs_500m = []

import copy

for step in target_steps:
    train_configs_500m.append(
        SimpleTrainConfig(
                        tpu_type=versioned("v4-128"),
                        train_batch_size=BATCH_SIZE,
                        steps_per_eval=1000,
                        num_train_steps=step,
                        **baseline_config
                    )
    )
    for key in sweep_grids:
        for value in sweep_grids[key]:
            new_config = copy.copy(baseline_config)     
            if(baseline_config[key] != value):   
                new_config[key] = value
                train_configs_500m.append(
                    SimpleTrainConfig(
                        tpu_type=versioned("v4-128"),
                        train_batch_size=BATCH_SIZE,
                        steps_per_eval=1000,
                        num_train_steps=step,
                        **new_config
                    )
                )

from levanter.models.llama import LlamaConfig
llama_500m = LlamaConfig(
    seq_len=1024,
    hidden_dim=1024,
    intermediate_dim=4096,
    num_heads=8,
    num_kv_heads=8,
    num_layers=32,
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

        step = dataclasses.replace(step, fn=_failure_ok_train)

        steps.append(step)
    return steps


steps_500m = make_sweep_steps(
    prefix="sweep-725-500m-10k",
    model_config=llama_500m,
    train_configs=train_configs_500m,
    tokenized_data=dclm_mixture_config_llama3,
    tags=("llama", "500m", "725_nadamw_sweep", "dclm", "10k", "nadamw"),
)


if __name__ == "__main__":
    executor_main(steps_500m)
