import logging
import os
from dataclasses import dataclass, field

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    experiment_name: str
    quality_classifier_model_path: str
    input_data_source_to_path: dict[str, str] = field(
        default_factory=lambda: {
            "fineweb_2020_10": "gs://marin-data/processed/fineweb/fw-v1.0/text_fw/CC-MAIN-2020-10/",
            "fineweb_2020_16": "gs://marin-data/processed/fineweb/fw-v1.0/text_fw/CC-MAIN-2020-16/",
            "fineweb_2020_24": "gs://marin-data/processed/fineweb/fw-v1.0/text_fw/CC-MAIN-2020-24/",
        }
    )
    keep_fraction: float = 0.2  # Keep 20% of the documents


def create_steps(config: ExperimentConfig) -> list[ExecutorStep]:
    assert config.experiment_name is not None and config.quality_classifier_model_path is not None

    steps = []
    tokenized: dict[str, ExecutorStep] = {}
    weights: dict[str, float] = {}
    for input_data_source, input_data_path in config.input_data_source_to_path.items():
        # Get the basename of the input directory
        input_basename = os.path.basename(os.path.normpath(input_data_path))
        inference_step = ExecutorStep(
            name=f"attributes/{config.experiment_name}/{input_data_source}",
            fn=run_inference,
            config=InferenceConfig(
                input_path=input_data_path,
                output_path=this_output_path(input_basename),
                model_name=versioned(config.quality_classifier_model_path),
                model_type="fasttext",
                attribute_name=versioned(f"{config.experiment_name}-quality"),
                runtime=RuntimeConfig(
                    requirements_filepath="marin/processing/classification/config/dclm_fasttext_requirements.txt",
                    memory_limit_gb=6,
                ),
            ),
        )

        consolidate_step = ExecutorStep(
            name=f"documents/{config.experiment_name}/{input_data_source}",
            fn=consolidate,
            config=ConsolidateConfig(
                input_path=input_data_path,
                # Can't use the versioned output path here because version
                #  string also takes into account the dependencies
                # This means that the hash at the end of the path will be different
                # based on the attribute_path. However,
                # we ultimately want the output path to be under the same directory for each fineweb dump.
                output_path=this_output_path(input_basename),
                filters=[
                    FilterConfig(
                        type=versioned("classify"),
                        attribute_path=output_path_of(inference_step, input_basename),
                        name=versioned(f"{config.experiment_name}-quality"),
                        label="__label__hq",
                        threshold=versioned(None),
                        keep_fraction=versioned(config.keep_fraction),
                    ),
                ],
            ),
        )

        tokenize_step = default_tokenize(
            name=f"{config.experiment_name}/{input_data_source}",
            dataset=output_path_of(consolidate_step),
            tokenizer=llama3_tokenizer,
        )

        steps.append(inference_step)
        steps.append(consolidate_step)
        steps.append(tokenize_step)
        tokenized[input_data_source] = tokenize_step
        weights[input_data_source] = 1.0

    train_step = default_train(
        name=config.experiment_name,
        tokenized=tokenized,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
        weights=weights,
    )

    steps.append(train_step)

    return steps


def create_experiment_configs() -> list[ExperimentConfig]:
    marin_eli5_100k_oh_100k_rw_200k_config = ExperimentConfig(
        experiment_name="eli5-100k-oh-100k-rw-200k",
        quality_classifier_model_path="gs://marin-us-central2/classifiers/dclm_eli5_100k_oh_100k_rw_200k-4a11ec/model.bin",
    )

    marin_eli5_200k_rw_200k_config = ExperimentConfig(
        experiment_name="eli5-200k-rw-200k",
        quality_classifier_model_path="gs://marin-us-central2/classifiers/dclm_eli5_200k_rw_200k-139db0/model.bin",
    )

    marin_oh_200k_rw_200k_config = ExperimentConfig(
        experiment_name="oh-200k-rw-200k",
        quality_classifier_model_path="gs://marin-us-central2/classifiers/dclm_oh_200k_rw_200k-725b11/model.bin",
    )

    original_dclm_quality_classifier_config = ExperimentConfig(
        experiment_name="original-dclm-quality-classifier",
        quality_classifier_model_path="mlfoundations/fasttext-oh-eli5",
    )

    return [
        marin_eli5_100k_oh_100k_rw_200k_config,
        marin_eli5_200k_rw_200k_config,
        marin_oh_200k_rw_200k_config,
        original_dclm_quality_classifier_config,
    ]


def main():
    steps = []
    for experiment_config in create_experiment_configs():
        steps.extend(create_steps(experiment_config))
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
