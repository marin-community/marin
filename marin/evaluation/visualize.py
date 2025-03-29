"""
Uses Levanter's viz_lm functionality to visualize log probabilities of a language model.
"""

import dataclasses
from dataclasses import dataclass

import ray
from levanter.data.text import LMMixtureDatasetConfig
from levanter.distributed import RayConfig
from levanter.infra.ray_tpu import _separate_process_fn
from levanter.main.viz_logprobs import VizLmConfig as LevanterVizLmConfig
from levanter.main.viz_logprobs import main as viz_lm_main
from levanter.models.lm_model import LmConfig
from levanter.trainer import TrainerConfig

from marin.execution.executor import ExecutorStep, this_output_path
from marin.processing.tokenize import lm_mixture_data_config
from marin.utils import remove_tpu_lockfile_on_exit


@dataclass
class VizLmConfig:
    """
    Configuration for visualizing log probabilities of a language model.
    """

    checkpoint_path: str
    model: LmConfig
    datasets: LMMixtureDatasetConfig
    num_docs_per_dataset: int = 32
    per_device_batch_size: int = 4
    output_path: str = dataclasses.field(default_factory=this_output_path)  # type: ignore

    comparison_model_path: str | None = None
    comparison_is_hf: bool = False


@ray.remote(memory=64 * 1024 * 1024 * 1024, resources={"TPU": 4, "TPU-v4-8-head": 1})
@remove_tpu_lockfile_on_exit
def do_viz_lm(config: LevanterVizLmConfig) -> None:
    """
    Visualizes log probabilities of a language model.

    Args:
        config (VizLmConfig): The configuration for visualizing log probabilities.
    """
    # remove_tpu_lockfile_on_exit() isn't sufficient now?
    _separate_process_fn(viz_lm_main, (config,), {})


def mixture_for_visualization(inputs: dict[str, ExecutorStep]) -> LMMixtureDatasetConfig:
    """
    Creates a mixture of datasets for visualizing log probabilities of a language model.

    Args:
        inputs (dict[str, ExecutorStep]): The inputs to the mixture.

    Returns:
        LMMixtureDatasetConfig: The mixture of datasets.
    """
    return lm_mixture_data_config(
        {name: step for name, step in inputs.items()},
        {name: 1.0 for name in inputs},
        shuffle=False,
        missing_weights_are_validation=True,
    )


def visualize_lm_log_probs(config: VizLmConfig) -> None:
    """
    Visualizes log probabilities of a language model.

    Args:
        config (VizLmConfig): The configuration for visualizing log probabilities.
    """
    levanter_config = LevanterVizLmConfig(
        checkpoint_path=config.checkpoint_path,
        model=config.model,
        num_docs=config.num_docs_per_dataset,
        path=config.output_path,
        data=config.datasets,
        trainer=TrainerConfig(
            ray=RayConfig(auto_start_cluster=False), per_device_eval_parallelism=config.per_device_batch_size
        ),
        comparison_model_path=config.comparison_model_path,
        comparison_is_hf=config.comparison_is_hf,
    )
    ray.get(do_viz_lm.remote(levanter_config))
