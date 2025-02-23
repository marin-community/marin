"""
Uses Levanter's viz_lm functionality to visualize log probabilities of a language model.
"""
from dataclasses import dataclass

import ray
from dolma.cli.tokenizer import TokenizerConfig

from levanter.data.text import LMMixtureDatasetConfig
from levanter.main.viz_logprobs import VizLmConfig as LevanterVizLmConfig, main as viz_lm_main
from levanter.models.lm_model import LmConfig
from marin.execution.executor import ExecutorStep, this_output_path
from marin.processing.tokenize import TokenizerStep, lm_mixture_data_config
from marin.utils import remove_tpu_lockfile_on_exit

@dataclass
class VizLmConfig:
    """
    Configuration for visualizing log probabilities of a language model.
    """
    checkpoint_path: str
    model: LmConfig
    datasets: dict[str, TokenizerStep | TokenizerConfig]
    num_docs_per_dataset: int = 256
    output_path: str = this_output_path()  # type: ignore


@ray.remote(
    memory=64 * 1024 * 1024 * 1024, resources={"TPU": 1, "TPU-v4-8-head": 1}
)
@remove_tpu_lockfile_on_exit
def do_viz_lm(config: LevanterVizLmConfig) -> None:
    """
    Visualizes log probabilities of a language model.

    Args:
        config (VizLmConfig): The configuration for visualizing log probabilities.
    """
    viz_lm_main(config)


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


def visualize_lm_lob_probs(config: VizLmConfig) -> None:
    """
    Visualizes log probabilities of a language model.

    Args:
        config (VizLmConfig): The configuration for visualizing log probabilities.
    """
    mixture_config = mixture_for_visualization(config.datasets)
    levanter_config = LevanterVizLmConfig(
        checkpoint_path=config.checkpoint_path,
        model=config.model,
        num_docs=config.num_docs_per_dataset,
        path=config.output_path,
        data=mixture_config,
    )
    ray.get(do_viz_lm.remote(levanter_config))