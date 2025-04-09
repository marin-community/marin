import wandb
# Initialize the WandB API
api = wandb.Api()
# Define your details
username = "stanford-mercury"
project = "optimizer-scaling"
thshold = 3e-3
# Retrieve the run directly using its full path

from optimizer_sweep.utils_simp import approximate, create_configs, check_baseline_run, grab_best_run, convert_run_to_config, bad_number



import copy
from marin.execution.executor import ExecutorStep, executor_main, versioned, unwrap_versioned_value
from experiments.simple_train_config import SimpleTrainConfig
import ray
from levanter.models.llama import LlamaConfig
from collections.abc import Sequence
import logging
import dataclasses
logger = logging.getLogger("ray")

def create_configs(baseline_config, sweep_grids, target_data = 5120000):
    # train_configs = []
    config_in_dict = []
    target_steps = []
    batch_size = (baseline_config['train_batch_size'])
    target_step  = target_data // (4096 * batch_size)
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

def config_to_train_config(config_in_dict, target_steps, config_class = SimpleTrainConfig, tpu_type = 'v4-128',):
    train_configs = []
    for config, target_step in zip(config_in_dict, target_steps):
        train_configs.append(
                    config_class(
                        tpu_type=versioned(tpu_type),
                        steps_per_eval=min(1000, target_step - 1),
                        num_train_steps=target_step,
                        **config
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
print(77210)
print(53952)
print(89002)
print(78383)
print(4430)
print(85385)
print(53871)
print(36916)
print(39699)
print(99943)
print(80247)
print(4928)
print(24425)
print(88476)
print(82643)
print(53936)
print(65780)
print(84238)
print(97349)
print(95680)
print(43624)
print(55822)
print(80982)
print(18973)
print(12457)
print(77816)
print(24541)
print(2137)
print(5952)
print(24837)
print(43902)
print(96750)