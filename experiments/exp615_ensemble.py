import os
from dataclasses import dataclass, field

from experiments.defaults import default_tokenize, default_train
from experiments.exp164_quality_classifiers import (
    dclm_eli5_100k_oh_100k_rw_200k,
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
from marin.processing.classification.custom.custom_attribute import CustomAttributeConfig, create_custom_attribute
from marin.processing.classification.inference import InferenceConfig, run_inference
from marin.processing.tokenize import lm_mixture_data_config


@dataclass
class ExperimentConfig:
    experiment_name: str
    quality_classifier_model_paths: list[str | ExecutorStep]
    input_data_source_to_path: dict[str, str] = field(
        default_factory=lambda: {
            "test": "gs://marin-us-central2/documents/quick-start-tests"
            # "fineweb_2024_18": "gs://marin-us-central2/documents/fineweb-small-resiliparse-preserve-formatting-e8c6ec/md/CC-MAIN-2024-18",
        }
    )
    keep_fraction: float = 0.2  # Keep 20% of the documents


def get_model_path(model_path: str | ExecutorStep):
    if isinstance(model_path, ExecutorStep):
        return output_path_of(model_path, "model.bin")
    return versioned(model_path)


def create_steps(config: ExperimentConfig) -> list[ExecutorStep]:
    """Create the steps for a single experiment.

    The experiment consists of taking a fineweb dump, filtering for high-quality documents,
    tokenizing the documents, and training a model. The only difference between the experiments
    is the quality classifier used to filter the documents.
    """

    steps = []
    tokenized: dict[str, ExecutorStep] = {}
    weights: dict[str, float] = {}
    for input_data_source, input_data_path in config.input_data_source_to_path.items():
        # Get the basename of the input directory
        input_basename = os.path.basename(os.path.normpath(input_data_path))
        inference_steps = []
        for classifier_id, quality_classifier_model_path in enumerate(config.quality_classifier_model_paths):
            inference_step = ExecutorStep(
                name=f"attributes/quality_filtering/{config.experiment_name}/{input_data_source}",
                fn=run_inference,
                config=InferenceConfig(
                    input_path=input_data_path,
                    output_path=this_output_path(input_basename),
                    model_name=get_model_path(quality_classifier_model_path),
                    model_type="fasttext",
                    attribute_name=versioned(f"{config.experiment_name}-quality_classifier-{classifier_id}"),
                    runtime=RuntimeConfig(
                        memory_limit_gb=12,
                    ),
                    task=TaskConfig(max_in_flight=500),
                ),
                pip_dependency_groups=["fasttext", "datasets", "filelock"],
            )
            inference_steps.append(inference_step)

        def label_func(doc, attrs):
            return {
                f"{config.experiment_name}-quality": {
                    "score": max(
                        attr["attributes"][f"{config.experiment_name}-quality_classifier-{classifier_id}"]["__label__hq"]
                        for classifier_id, attr in enumerate(attrs)
                    )
                }
            }

        ensemble_step = ExecutorStep(
            name=f"attributes/quality_filtering/{config.experiment_name}/{input_data_source}",
            fn=create_custom_attribute,
            config=CustomAttributeConfig(
                input_doc_path=input_data_path,
                output_attr_path=this_output_path(input_basename),
                label_func=versioned(label_func),
                input_attr_paths=[output_path_of(inference_step, input_basename) for inference_step in inference_steps],
            ),
        )

        consolidate_step = ExecutorStep(
            name=f"documents/quality_filtering/{config.experiment_name}/{input_data_source}",
            fn=consolidate,
            config=ConsolidateConfig(
                input_path=input_data_path,
                output_path=this_output_path(input_basename),
                filters=[
                    FilterConfig(
                        type=versioned("classify"),
                        attribute_path=output_path_of(ensemble_step, input_basename),
                        name=versioned(f"{config.experiment_name}-quality"),
                        label="score",
                        threshold=versioned(None),
                        keep_fraction=versioned(config.keep_fraction),
                    ),
                ],
                ray_memory_limit_gb=12,
            ),
            pip_dependency_groups=["ddsketch"],
        )

        tokenize_step = default_tokenize(
            name=f"quality_filtering/{config.experiment_name}/{input_data_source}",
            dataset=output_path_of(consolidate_step),
            tokenizer=llama3_tokenizer,
        )

        steps.append(inference_step)
        steps.append(consolidate_step)
        steps.append(tokenize_step)
        tokenized[input_data_source] = tokenize_step
        weights[input_data_source] = 1.0

    data_config = lm_mixture_data_config(components=tokenized, weights=weights)

    train_step = default_train(
        name=f"quality_filtering/{config.experiment_name}",
        tokenized=data_config,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
    )

    steps.append(train_step)

    return steps


def main():
    experiment_config = ExperimentConfig(
        experiment_name="exp615_ensemble",
        quality_classifier_model_paths=[dclm_eli5_100k_oh_100k_rw_200k, teknium_oh_200k_rw_200k],
    )
    steps = create_steps(experiment_config)
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
