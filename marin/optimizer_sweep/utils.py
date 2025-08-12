import wandb
import dataclasses
import logging
from collections.abc import Sequence
from marin.resources import TpuPodConfig
import ray
from levanter.models.llama import LlamaConfig

from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import ExecutorStep, versioned

logger = logging.getLogger("ray")

# Initialize the WandB API
api = wandb.Api()
username = "marin-community"
project = "optimizer-scaling"
thshold = 3e-3


def config_to_train_config(
    config_in_dict,
    target_steps,
    config_generator,
    tpu_type="v4-128",
):
    train_configs = []
    for config, target_step in zip(config_in_dict, target_steps, strict=False):
        new_config = config.copy()
        train_batch_size = new_config.pop("train_batch_size")
        optimizer_config = config_generator(**new_config)
        if isinstance(tpu_type, str):
            tpu_type = TpuPodConfig(versioned(tpu_type))
        train_configs.append(
            SimpleTrainConfig(
                resources=tpu_type,
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
