# https://github.com/stanford-crfm/marin/issues/764
# Sweep to determine optimal learning rate schedule shape for Adam on 0.5B model x 160B tokens
import dataclasses
import itertools
import logging
import math
from collections.abc import Sequence
from experiments.exp600_tootsie import dclm_mixture_config_llama3

import numpy as np
import ray
from levanter.models.llama import LlamaConfig

from experiments.defaults_hack import default_train
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, executor_main, versioned, unwrap_versioned_value

logger = logging.getLogger("ray")

# Sweep to determine optimal training config
BATCH_SIZE = 4096
target_steps = [50000]
TPU_TYPES_500m = ["v4-128"]


def format_train_config(prefix: str, key: str):
    name = key
    hashed_name = str(hash(name))[:6]
    return (prefix + '_' + hashed_name + name)[:64]


baseline_config = {
    'weight_decay': 0.1,
    'min_lr_ratio': 0,
    'warmup': 2000,
    'beta1': 0.9,
    'beta2': 0.95,
    'epsilon': 1e-15,
    'max_grad_norm': 1.0,
}


configs = {
#     'cosine_8e-3': {
#     'learning_rate': 8e-3, 
#     'lr_schedule': 'cosine'
# },
#     'linear_8e-3': {
#     'learning_rate': 8e-3, 
#     'lr_schedule': 'linear'        
# },
#     'wsd_8e-3_constant_20d': {
#         'learning_rate': 8e-3, 
#         'lr_schedule': 'linear',
#         'decay': 0.2
#     },
#     'wsd_8e-3_constant_40d': {
#         'learning_rate': 8e-3, 
#         'lr_schedule': 'linear',
#         'decay': 0.4
#     },
#     'wsd_8e-3_inv_sqrt_20d': {
#         'learning_rate': 8e-3, 
#         'lr_schedule': 'linear',
#         'stable_lr_schedule': 'inv_sqrt',
#         'decay': 0.2
#     },
    'wsd_8e-3_inv_sqrt_40d': {
        'learning_rate': 8e-3, 
        'lr_schedule': 'linear',
        'stable_lr_schedule': 'inv_sqrt',
        'decay': 0.4,
        'id': "sweep-764-lr_schedule_-25992wsd_8e-3_inv_sqrt_40d-9299e4",
        'ckpt_path': "gs://marin-us-central2/checkpoints/sweep-764-lr_schedule_-25992wsd_8e-3_inv_sqrt_40d-9299e4/checkpoints/step-33622"
    },
#     'cosine_8e-4': {
#     'learning_rate': 8e-4, 
#     'lr_schedule': 'cosine'
# },
#     'wsd_8e-4_constant_20d': {
#         'learning_rate': 8e-4, 
#         'lr_schedule': 'linear',
#         'decay': 0.2
#     },
}







train_configs_500m = {}

import copy

for step in target_steps:
    for key in configs:
        config = configs[key]
        train_configs_500m[key] = SimpleTrainConfig(
                            tpu_type=versioned("v4-128"),
                            train_batch_size=BATCH_SIZE,
                            steps_per_eval=1000,
                            num_train_steps=step,
                            **baseline_config,
                            **config
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
    train_configs,
    tokenized_data: ExecutorStep,
    tags: Sequence[str] = (),
):
    steps = []
    for key in train_configs:
        train_config = train_configs[key]
        name = format_train_config(prefix, key)
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
    prefix="sweep-764-lr_schedule",
    model_config=llama_500m,
    train_configs=train_configs_500m,
    tokenized_data=dclm_mixture_config_llama3,
    tags=("llama", "500m", "764_lr_schedule", "dclm", "50k"),
)


if __name__ == "__main__":
    executor_main(steps_500m)
