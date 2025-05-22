import wandb

# Initialize the WandB API
api = wandb.Api()
# Define your details
username = "marin-community"
project = "optimizer-scaling"
thshold = 3e-3
from marin.optimizer_sweep.utils_simp import approximate, create_configs, check_baseline_run, grab_best_run, convert_run_to_config, bad_number
import copy
import dataclasses
import logging
from collections.abc import Sequence
from marin.resources import ResourceConfig, TpuPodConfig

import ray
from levanter.models.llama import LlamaConfig

from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, versioned
from marin.optimizer_sweep.utils_simp import create_configs

logger = logging.getLogger("ray")


def create_configs(baseline_config, sweep_grids, target_data=5120000):
    # train_configs = []
    config_in_dict = []
    target_steps = []
    batch_size = baseline_config["train_batch_size"]
    target_step = target_data // (4096 * batch_size)
    target_steps.append(target_step)
    # train_configs.append(
    #     config_class(
    #                     tpu_type=versioned(tpu_type),
    #                     steps_per_eval=min(1000, target_step - 1),
    #                     num_train_steps=target_step,
    #                     **baseline_config
    #                 )
    # )
    config_in_dict.append(baseline_config)
    for key in sweep_grids:
        for value in sweep_grids[key]:
            new_config = copy.copy(baseline_config)     
            if(baseline_config[key] != (value)):   
                new_config[key] = (value)
                batch_size = (new_config['train_batch_size'])
                target_step  = target_data // (4096 * batch_size)
                target_steps.append(target_step)
                if (new_config['warmup']) <= target_step:
                    config_in_dict.append(new_config)
    return target_steps, config_in_dict


def config_to_train_config(
    config_in_dict,
    target_steps,
    config_generator,
    tpu_type="v4-128",
):
    train_configs = []
    for config, target_step in zip(config_in_dict, target_steps, strict=False):
        train_batch_size = config.pop('train_batch_size')
        optimizer_config = config_generator(**config)
        train_configs.append(
            SimpleTrainConfig(
                TpuPodConfig(versioned(tpu_type)),
                steps_per_eval=min(1000, target_step - 1),
                num_train_steps=target_step,
                train_batch_size=train_batch_size,
                optimizer_config=optimizer_config,
            )
        )
    return train_configs


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
    format_train_config,
    default_train,
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

