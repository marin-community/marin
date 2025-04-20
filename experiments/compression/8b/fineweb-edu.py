"""Filter training data based on LZ4 compression ratios.

Filter documents based on their LZ4 compression ratios (compressed_size/original_size),
keeping only those with ratios between 0.6 and 0.9.

Dataset: fineweb_edu
Model: llama_8b

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
            "fineweb-edu-2024-10": "gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2024-10/",
            "fineweb-edu-2023-50": "gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2023-50/",
            "fineweb-edu-2023-40": "gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2023-40/",
            "fineweb-edu-2023-23": "gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2023-23/",
            "fineweb-edu-2023-14": "gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2023-14/",
            "fineweb-edu-2023-06": "gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2023-06/",
            "fineweb-edu-2022-49": "gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2022-49/",
            "fineweb-edu-2022-40": "gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2022-40/",
            "fineweb-edu-2022-33": "gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2022-33/",
            "fineweb-edu-2022-27": "gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2022-27/",
            "fineweb-edu-2022-21": "gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2022-21/",
            "fineweb-edu-2022-05": "gs://marin-us-central2/raw/fineweb-edu/CC-MAIN-2022-05/",
        }
    )

def create_steps(config: ExperimentConfig) -> list[ExecutorStep]:
    """Create the steps for a single experiment with compression ratio filtering."""
    assert config.experiment_name is not None

    steps = []
    tokenized: dict[str, ExecutorStep] = {}
    weights: dict[str, float] = {}

    for input_data_source, input_data_path in config.input_data_source_to_path.items():
        input_basename = os.path.basename(os.path.normpath(input_data_path.split("{")[0]))  # Extract base name without pattern

        # Calculate compression ratios
        compression_step = ExecutorStep(
            name=f"attributes/compel/{config.experiment_name}/{input_data_source}",
            fn=run_inference,
            config=InferenceConfig(
                input_path=input_data_path,  # This should handle the brace expansion pattern
                output_path=this_output_path(input_basename),
                model_type="compression",
                model_name=None,
                attribute_name=versioned("compression_ratio"),
                runtime=RuntimeConfig(
                    memory_limit_gb=12,
                ),
                task=TaskConfig(max_in_flight=500),
            ),
            pip_dependency_groups=["lz4", "datasets", "filelock"],
        )
        steps.append(compression_step)

        # Filter based on compression ratios
        consolidate_step = ExecutorStep(
            name=f"documents/compel/{config.experiment_name}/{input_data_source}",
            fn=consolidate,
            config=ConsolidateConfig(
                input_path=input_data_path,  # Use the same brace expansion pattern
                output_path=this_output_path(input_basename),
                filters=[
                    FilterConfig(
                        type=versioned("classify"),
                        attribute_path=output_path_of(compression_step, input_basename),
                        name=versioned("compression_ratio"),
                        threshold=versioned(0.6),  # Lower bound
                        upper_threshold=versioned(0.85),  # Upper bound
                    ),
                ],
                ray_memory_limit_gb=12,
            ),
            pip_dependency_groups=["ddsketch", "lz4"],
        )
        steps.append(consolidate_step)

        # Tokenize filtered data
        tokenize_step = default_tokenize(
            name=f"compel/{config.experiment_name}/{input_data_source}",
            dataset="output_path_of(consolidate_step)",
            tokenizer=llama3_tokenizer,
        )
        steps.append(tokenize_step)

        tokenized[input_data_source] = tokenize_step
        weights[input_data_source] = 1.0

    # Create training data config from tokenized datasets
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
    compression_filter_config = ExperimentConfig(
        experiment_name="compression-ratio-filter-fineweb-edu",
    )
    return [compression_filter_config]


def main():
    steps = []
    for experiment_config in create_experiment_configs():
        steps.extend(create_steps(experiment_config))
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
