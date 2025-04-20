"""Filter training data based on LZ4 compression ratios.

Using pre-filtered data from C4 that has been filtered based on
compression ratios between 0.6 and 0.85.

Model: llama_1_4b

Pipeline: tokenization -> training (using pre-filtered data)
"""

import logging
from dataclasses import dataclass

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
)
from marin.processing.tokenize import lm_mixture_data_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")


@dataclass
class ExperimentConfig:
    experiment_name: str
    filtered_data_path: str
    input_data_source: str = "c4-filtered-new"


def create_steps(config: ExperimentConfig) -> list[ExecutorStep]:
    """Create steps for a single experiment using pre-filtered data."""
    assert config.experiment_name is not None

    steps = []
    tokenized: dict[str, ExecutorStep] = {}
    weights: dict[str, float] = {}

    # Tokenize pre-filtered data
    tokenize_step = default_tokenize(
        name=f"compel/{config.experiment_name}/{config.input_data_source}",
        dataset=config.filtered_data_path,
        tokenizer=llama3_tokenizer,
    )
    steps.append(tokenize_step)

    tokenized[config.input_data_source] = tokenize_step
    weights[config.input_data_source] = 1.0

    # Create training data config from tokenized dataset
    data_config = lm_mixture_data_config(components=tokenized, weights=weights)

    # Set up model training
    train_step = default_train(
        name=f"compel/{config.experiment_name}",
        tokenized=data_config,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
    )
    steps.append(train_step)

    return steps


def create_experiment_configs() -> list[ExperimentConfig]:
    experiment_config = ExperimentConfig(
        experiment_name="c4-compression-filtered-training",
        filtered_data_path="gs://marin-us-central2/documents/compel/compression-ratio-filter-c4-filtered/c4-elyas-new-run-751cd3/v1.7/",
    )
    return [experiment_config]


def main():
    steps = []
    for experiment_config in create_experiment_configs():
        steps.extend(create_steps(experiment_config))
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
