"""Filter training data based on LZ4 compression ratios.

Filter documents based on their LZ4 compression ratios (compressed_size/original_size),
keeping only those with ratios between 0.6 and 0.9.

Dataset: fineweb_2024_18 (CommonCrawl 2024-18)
Model: llama_1_4b

Pipeline: compression ratio calculation -> filtering -> tokenization -> training
"""

import logging
import os
from dataclasses import dataclass, field

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
from marin.core.runtime import TaskConfig
from marin.execution.executor import (
    ExecutorStep,
    executor_main,
    output_path_of,
    this_output_path,
    versioned,
)
from marin.processing.classification.config.inference_config import RuntimeConfig
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.inference import InferenceConfig, run_inference
from marin.processing.tokenize import lm_mixture_data_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ray")


@dataclass
class ExperimentConfig:
    experiment_name: str
    input_data_source_to_path: dict[str, str] = field(
        default_factory=lambda: {
            "fineweb_2024_18": (
                "gs://marin-us-central2/documents/fineweb-small-resiliparse-preserve-formatting-e8c6ec/md/CC-MAIN-2024-18"
            ),
        }
    )


def create_steps(config: ExperimentConfig) -> list[ExecutorStep]:
    """Create the steps for a single experiment with compression ratio filtering."""
    assert config.experiment_name is not None

    steps = []
    tokenized: dict[str, ExecutorStep] = {}
    weights: dict[str, float] = {}

    for input_data_source, input_data_path in config.input_data_source_to_path.items():
        input_basename = os.path.basename(os.path.normpath(input_data_path))

        # Calculate compression ratios
        compression_step = ExecutorStep(
            name=f"attributes/compression_filtering/{config.experiment_name}/{input_data_source}",
            fn=run_inference,
            config=InferenceConfig(
                input_path=input_data_path,
                output_path=this_output_path(input_basename),
                model_type="compression",  # Use our new compression classifier
                model_name=None,  # This doesn't matter for compression
                attribute_name=versioned("compression_ratio"),
                runtime=RuntimeConfig(
                    memory_limit_gb=12,
                ),
                task=TaskConfig(max_in_flight=500),
            ),
            pip_dependency_groups=["lz4", "datasets", "filelock"],
        )

        # Filter based on compression ratios
        consolidate_step = ExecutorStep(
            name=f"documents/compression_filtering/{config.experiment_name}/{input_data_source}",
            fn=consolidate,
            config=ConsolidateConfig(
                input_path=input_data_path,
                output_path=this_output_path(input_basename),
                filters=[
                    FilterConfig(
                        type=versioned("classify"),
                        attribute_path=output_path_of(compression_step, input_basename),
                        name=versioned("compression_ratio"),
                        threshold=versioned(0.6),  # Lower bound
                        upper_threshold=versioned(0.9),  # Upper bound
                    ),
                ],
                ray_memory_limit_gb=12,
            ),
            pip_dependency_groups=["ddsketch", "lz4"],
        )

        tokenize_step = default_tokenize(
            name=f"compression_filtering/{config.experiment_name}/{input_data_source}",
            dataset=output_path_of(consolidate_step),
            tokenizer=llama3_tokenizer,
        )

        steps.append(compression_step)
        steps.append(consolidate_step)
        steps.append(tokenize_step)
        tokenized[input_data_source] = tokenize_step
        weights[input_data_source] = 1.0

    data_config = lm_mixture_data_config(components=tokenized, weights=weights)

    train_step = default_train(
        name=f"compression_filtering/{config.experiment_name}",
        tokenized=data_config,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
    )

    steps.append(train_step)
    return steps


def create_experiment_configs() -> list[ExperimentConfig]:
    compression_filter_config = ExperimentConfig(
        experiment_name="compression-ratio-filter",
    )
    return [compression_filter_config]


def main():
    steps = []
    for experiment_config in create_experiment_configs():
        steps.extend(create_steps(experiment_config))
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
