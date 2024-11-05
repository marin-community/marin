import logging
import os
from dataclasses import dataclass, field

from experiments.defaults import default_tokenize, default_train
from experiments.exp164_quality_classifiers import (
    dclm_eli5_100k_oh_100k_rw_200k,
    dclm_eli5_100k_oh_100k_rw_200k_seed_1,
    dclm_eli5_100k_oh_100k_rw_200k_seed_2,
    dclm_eli5_200k_rw_200k,
    teknium_oh_200k_rw_200k,
)
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

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


@dataclass
class ExperimentConfig:
    experiment_name: str
    quality_classifier_model_path: str | ExecutorStep
    input_data_source_to_path: dict[str, str] = field(
        default_factory=lambda: {
            "fineweb_2020_10": "gs://marin-us-central2/raw/fineweb/cd85054/CC-MAIN-2020-10/",
        }
    )
    keep_fraction: float = 0.2  # Keep 20% of the documents


def get_model_path(model_path: str | ExecutorStep):
    if isinstance(model_path, ExecutorStep):
        return output_path_of(model_path, "model.bin")
    return versioned(model_path)


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
                model_name=get_model_path(config.quality_classifier_model_path),
                model_type="fasttext",
                attribute_name=versioned(f"{config.experiment_name}-quality"),
                filetype="parquet",
                runtime=RuntimeConfig(
                    requirements_filepath="marin/processing/classification/config/dclm_fasttext_requirements.txt",
                    memory_limit_gb=40,
                ),
                task=TaskConfig(max_in_flight=500),
            ),
        )

        consolidate_step = ExecutorStep(
            name=f"documents/{config.experiment_name}/{input_data_source}",
            fn=consolidate,
            config=ConsolidateConfig(
                input_path=input_data_path,
                output_path=this_output_path(input_basename),
                filetype="parquet",
                filters=[
                    FilterConfig(
                        type=versioned("classify"),
                        attribute_path=output_path_of(inference_step, input_basename),
                        name=versioned(f"{config.experiment_name}-quality"),
                        label="__label__hq",
                        threshold=versioned(None),
                        keep_fraction=versioned(config.keep_fraction),
                        filetype="parquet",
                    ),
                ],
                max_tasks_in_flight=500,
                memory_limit_gb=40,
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

    # TODO: Uncomment this when WANDBAPI error fixed
    steps.append(train_step)

    return steps


def create_experiment_configs() -> list[ExperimentConfig]:
    marin_eli5_100k_oh_100k_rw_200k_config = ExperimentConfig(
        experiment_name="eli5-100k-oh-100k-rw-200k",
        quality_classifier_model_path=dclm_eli5_100k_oh_100k_rw_200k,
    )

    marin_eli5_100k_oh_100k_rw_200k_seed_1_config = ExperimentConfig(
        experiment_name="eli5-100k-oh-100k-rw-200k-seed-1",
        quality_classifier_model_path=dclm_eli5_100k_oh_100k_rw_200k_seed_1,
    )

    marin_eli5_100k_oh_100k_rw_200k_seed_2_config = ExperimentConfig(
        experiment_name="eli5-100k-oh-100k-rw-200k-seed-2",
        quality_classifier_model_path=dclm_eli5_100k_oh_100k_rw_200k_seed_2,
    )

    marin_eli5_200k_rw_200k_config = ExperimentConfig(
        experiment_name="eli5-200k-rw-200k",
        quality_classifier_model_path=dclm_eli5_200k_rw_200k,
    )

    marin_oh_200k_rw_200k_config = ExperimentConfig(
        experiment_name="oh-200k-rw-200k",
        quality_classifier_model_path=teknium_oh_200k_rw_200k,
    )

    original_dclm_quality_classifier_config = ExperimentConfig(
        experiment_name="original-dclm-quality-classifier",
        quality_classifier_model_path="mlfoundations/fasttext-oh-eli5",
    )

    return [
        marin_eli5_100k_oh_100k_rw_200k_config,
        marin_eli5_100k_oh_100k_rw_200k_seed_1_config,
        marin_eli5_100k_oh_100k_rw_200k_seed_2_config,
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
