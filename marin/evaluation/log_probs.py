"""
This file uses Levanter to compute validation losses and entropies.
"""

import dataclasses
import os
from dataclasses import dataclass

import ray
from levanter.data.text import LMMixtureDatasetConfig
from levanter.distributed import RayConfig
from levanter.main.eval_lm import EvalLmConfig as LevanterEvalLmConfig
from levanter.main.eval_lm import main as eval_lm_main
from levanter.models.lm_model import LmConfig
from levanter.tracker.wandb import WandbConfig
from levanter.trainer import TrainerConfig

from marin.execution.executor import ExecutorStep, InputName, this_output_path
from marin.utilities.executor_utils import ckpt_path_to_step_name


@dataclass
class EvalLmConfig:
    """
    Configuration for visualizing log probabilities of a language model.
    """

    checkpoint_path: str
    model: LmConfig
    datasets: LMMixtureDatasetConfig
    per_device_batch_size: int = 4
    output_path: str = dataclasses.field(default_factory=this_output_path)  # type: ignore
    checkpoint_is_hf: bool = False
    """Whether the checkpoint is in HF format."""

    log_entropy: bool = True
    """ Whether to log entropies of the model. """

    max_samples_per_dataset: int | None = None


def default_lm_log_probs(
    checkpoint: str | InputName,
    model: LmConfig,
    data: LMMixtureDatasetConfig,
    checkpoint_is_hf: bool,
    per_device_batch_size: int = 4,
    max_samples_per_dataset: int | None = None,
) -> ExecutorStep:
    """
    Creates a step to evaluate log probabilities of a language model.
    Args:
        checkpoint:  The checkpoint to evaluate.
        model:  The model configuration.
        data: The data to evaluate on.
        checkpoint_is_hf:  Whether the checkpoint is in HF format.
    """
    name = ckpt_path_to_step_name(checkpoint)
    name = f"analysis/log_probs/{name}"
    return ExecutorStep(
        name=name,
        fn=evaluate_lm_log_probs,
        config=EvalLmConfig(
            checkpoint_path=checkpoint,  # type: ignore
            model=model,
            datasets=data,
            log_entropy=True,
            checkpoint_is_hf=checkpoint_is_hf,
            per_device_batch_size=per_device_batch_size,
            max_samples_per_dataset=max_samples_per_dataset,
        ),
    )


@ray.remote(memory=64 * 1024 * 1024 * 1024, resources={"TPU": 4, "TPU-v4-8-head": 1}, max_calls=1)
def do_eval_lm(config: LevanterEvalLmConfig) -> None:
    """
    Visualizes log probabilities of a language model.

    Args:
        config (EvalLmConfig): The configuration for visualizing log probabilities.
    """
    # _separate_process_fn(eval_lm_main, (config,), {})
    eval_lm_main(config)


def evaluate_lm_log_probs(config: EvalLmConfig) -> None:
    """
    Evaluate log probabilities of a language model on a mixture, and optionally entropies.

    Args:
        config (EvalLmConfig): The configuration for visualizing log probabilities.
    """

    name = os.path.basename(config.output_path)
    name = f"ppl-eval-{name}"

    if config.max_samples_per_dataset is None:
        max_eval_batches = None
    else:
        max_eval_batches = config.max_samples_per_dataset // config.per_device_batch_size

    levanter_config = LevanterEvalLmConfig(
        checkpoint_path=config.checkpoint_path if not config.checkpoint_is_hf else None,
        hf_checkpoint=config.checkpoint_path if config.checkpoint_is_hf else None,
        model=config.model,
        data=config.datasets,
        trainer=TrainerConfig(
            tracker=WandbConfig(project="marin", tags=["eval_lm"], name=name, id=name[:64]),
            ray=RayConfig(auto_start_cluster=False),
            per_device_eval_parallelism=config.per_device_batch_size,
            max_eval_batches=max_eval_batches,
        ),
        log_entropy=config.log_entropy,
    )
    ray.get(do_eval_lm.remote(levanter_config))
