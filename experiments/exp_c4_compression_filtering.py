"""Filter C4 training data based on LZ4 compression ratios.

Filter C4 documents based on their LZ4 compression ratios (compressed_size/original_size),
keeping only those with ratios between 0.65 and 0.8.

Dataset: C4 (allenai/c4, en subset) from Dolma v1.7
Model: llama_1_4b

Pipeline: compression ratio calculation -> filtering -> tokenization -> training

Usage:
    python marin/run/ray_run.py --env_vars WANDB_API_KEY <your_key> -- python experiments/exp_c4_compression_filtering.py

    For forced re-run of specific steps:
    python marin/run/ray_run.py --env_vars WANDB_API_KEY <your_key> -- python experiments/exp_c4_compression_filtering.py --force_run_failed

Note: 
    - C4 data is sourced from the pre-downloaded Dolma v1.7 dataset
    - Compression filtering uses LZ4 with thresholds: 0.65 ≤ ratio ≤ 0.8
    - Final model will be saved to: gs://marin-us-central2/checkpoints/compression_filtering/c4-compression-ratio-filter-065-08/
"""

import logging
import os
from dataclasses import dataclass, field

from experiments.defaults import default_tokenize, default_train, default_download
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
            # Option 1: Use pre-downloaded C4 from Dolma v1.7 (recommended)
            # Contains c4-{0000..0170}.json.gz files
            "c4_en": "gs://marin-us-central2/raw/dolma/v1.7/",
            
            # Option 2: Alternative - download C4 directly from HuggingFace (uncomment if needed)
            # Note: This would require adding a download step in create_steps()
            # "c4_en_hf": "allenai/c4:en",
        }
    )


def create_steps(config: ExperimentConfig) -> list[ExecutorStep]:
    """Create the steps for a single experiment with compression ratio filtering on C4."""
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
                model_type="compression",  # Use our compression classifier
                model_name=None,  # This doesn't matter for compression
                attribute_name=versioned("compression_ratio"),
                runtime=RuntimeConfig(
                    memory_limit_gb=12,
                ),
                task=TaskConfig(max_in_flight=500),
            ),
            pip_dependency_groups=["lz4", "datasets", "filelock"],
        )

        # Filter based on compression ratios (0.65-0.8 range)
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
                        threshold=versioned(0.65),  # Lower bound (increased from 0.6)
                        upper_threshold=versioned(0.8),  # Upper bound (decreased from 0.9)
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
    c4_compression_filter_config = ExperimentConfig(
        experiment_name="c4-compression-ratio-filter-065-08",
    )
    return [c4_compression_filter_config]


def main():
    steps = []
    for experiment_config in create_experiment_configs():
        steps.extend(create_steps(experiment_config))
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
