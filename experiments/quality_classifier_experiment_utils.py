# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from dataclasses import dataclass, field

from marin.core.runtime import TaskConfig
from marin.execution import (
    ExecutorStep,
    StepContext,
    StepRef,
    step,
    versioned,
)
from marin.processing.classification.config.inference_config import RuntimeConfig
from marin.processing.classification.consolidate import ConsolidateConfig, FilterConfig, consolidate
from marin.processing.classification.inference import InferenceConfig, run_inference
from marin.processing.tokenize import lm_mixture_data_config

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_1_4b, llama_1_4b_train_config


@dataclass
class ExperimentConfig:
    experiment_name: str
    quality_classifier_model_path: str | ExecutorStep
    input_data_source_to_path: dict[str, str] = field(
        # TODO(chris): Change this to an ExecutorStep. This is currently hardcoded because we would need to
        # re-run the downloading of fineweb and run the extraction script on it.
        default_factory=lambda: {
            "fineweb_2024_18": (
                "gs://marin-us-central2/documents/fineweb-small-resiliparse-preserve-formatting-e8c6ec/md/CC-MAIN-2024-18"
            ),
        }
    )
    keep_fraction: float = 0.2  # Keep 20% of the documents


def get_model_path(model_path: str | ExecutorStep):
    if isinstance(model_path, ExecutorStep):
        return model_path / "model.bin"
    return versioned(model_path)


def create_inference_step(
    experiment_name: str,
    quality_classifier_model_path: str | ExecutorStep,
    input_data_source: str,
    input_data_path: str,
) -> ExecutorStep:
    """Create an inference step for quality filtering."""
    input_basename = os.path.basename(os.path.normpath(input_data_path))

    @step(
        name=f"attributes/quality_filtering/{experiment_name}/{input_data_source}",
        fn=run_inference,
        pip_dependency_groups=["filelock"],
    )
    def _inference_step(ctx: StepContext):
        return InferenceConfig(
            input_path=input_data_path,
            output_path=ctx.output / input_basename,
            model_name=get_model_path(quality_classifier_model_path),
            model_type="fasttext",
            attribute_name=versioned(f"{experiment_name}-quality"),
            runtime=RuntimeConfig(
                memory_limit_gb=12,
            ),
            task=TaskConfig(max_in_flight=500),
        )

    return _inference_step()


def create_consolidate_step(
    experiment_name: str,
    input_data_source: str,
    input_data_path: str,
    keep_fraction: float,
    inference_step: ExecutorStep,
) -> ExecutorStep:
    """Create a consolidate step for quality filtering."""
    input_basename = os.path.basename(os.path.normpath(input_data_path))

    @step(
        name=f"documents/quality_filtering/{experiment_name}/{input_data_source}",
        fn=consolidate,
        pip_dependency_groups=["ddsketch"],
    )
    def _consolidate_step(ctx: StepContext):
        inference_ref = ctx.require(inference_step)
        return ConsolidateConfig(
            input_path=input_data_path,
            output_path=ctx.output / input_basename,
            filters=[
                FilterConfig(
                    type=versioned("classify"),
                    attribute_path=inference_ref / input_basename,
                    name=versioned(f"{experiment_name}-quality"),
                    label="__label__hq",
                    lower_threshold=versioned(None),
                    keep_fraction=versioned(keep_fraction),
                ),
            ],
        )

    return _consolidate_step()


def create_steps(config: ExperimentConfig) -> list[ExecutorStep]:
    """Create the steps for a single experiment.

    The experiment consists of taking a fineweb dump, filtering for high-quality documents,
    tokenizing the documents, and training a model. The only difference between the experiments
    is the quality classifier used to filter the documents.
    """
    assert config.experiment_name is not None and config.quality_classifier_model_path is not None

    steps = []
    tokenized: dict[str, ExecutorStep] = {}
    weights: dict[str, float] = {}
    for input_data_source, input_data_path in config.input_data_source_to_path.items():
        # Create the inference step using the factory function
        inference_step = create_inference_step(
            experiment_name=config.experiment_name,
            quality_classifier_model_path=config.quality_classifier_model_path,
            input_data_source=input_data_source,
            input_data_path=input_data_path,
        )

        # Create the consolidate step using the factory function
        consolidate_step = create_consolidate_step(
            experiment_name=config.experiment_name,
            input_data_source=input_data_source,
            input_data_path=input_data_path,
            keep_fraction=config.keep_fraction,
            inference_step=inference_step,
        )

        # Create the tokenize step
        tokenize_step = default_tokenize(
            name=f"quality_filtering/{config.experiment_name}/{input_data_source}",
            dataset=consolidate_step,
            tokenizer=llama3_tokenizer,
        )

        steps.append(inference_step)
        steps.append(consolidate_step)
        steps.append(tokenize_step)
        tokenized[input_data_source] = tokenize_step
        weights[input_data_source] = 1.0

    data_config = lm_mixture_data_config(
        components=tokenized,
        weights=weights,
        permutation_type="linear",
    )

    train_step = default_train(
        name=f"quality_filtering/{config.experiment_name}",
        tokenized=data_config,
        model_config=llama_1_4b,
        train_config=llama_1_4b_train_config,
    )

    steps.append(train_step)

    return steps
