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

"""
Experiment 615: Ensemble quality classifiers (via max score).

This experiment ensembles a list of quality classifiers (in this case, the MMLU and DCLM classifiers)
by taking their maximum score.

See https://github.com/marin-community/marin/issues/615 for more details.
"""

import os
from dataclasses import dataclass, field

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

from experiments.cooldown_quality import QualityAblationConfig, default_quality_ablation
from experiments.defaults import default_tokenize
from experiments.exp164_quality_classifiers import dclm_eli5_100k_oh_100k_rw_200k
from experiments.exp274_mmlu_quality_classifier import marin_mmlu_100k_rw_100k
from experiments.llama import llama3_tokenizer

# mapping of data source names to their GCS paths
INPUT_DATA_SOURCE_TO_PATH: dict[str, str] = {
    "fineweb_2024_18": (
        "gs://marin-us-central2/documents/fineweb-small-resiliparse-preserve-formatting-v2-e72837/md/CC-MAIN-2024-18/"
    ),
}


@dataclass
class ExperimentConfig:
    """Configuration for an ensemble quality filtering experiment.

    This config defines parameters for an experiment that:
    1. Takes input documents from specified data sources
    2. Runs multiple quality classifiers on these documents
    3. Combines the classifier scores using an ensemble approach (e.g., taking the max score)
    4. Filters documents to keep only the highest quality ones

    Args:
        experiment_name: Identifier for this experiment
        quality_classifier_model_paths: List of paths to quality classifier models to ensemble
        keep_fraction: Fraction of highest-quality documents to keep after filtering
    """

    experiment_name: str
    quality_classifier_model_paths: list[str | ExecutorStep]
    keep_fraction: float = 0.1  # Keep 20% of the documents
    cooldown_config: QualityAblationConfig = field(
        default_factory=lambda: QualityAblationConfig(permutation_type="linear")
    )


def get_model_path(model_path: str | ExecutorStep):
    if isinstance(model_path, ExecutorStep):
        return output_path_of(model_path, "model.bin")
    return versioned(model_path)


def create_steps(config: ExperimentConfig) -> list[ExecutorStep]:
    """Create the steps for a single experiment.

    Variation of exp614_quality_filtering.py, but uses an ensemble of quality classifiers.
    """

    steps = []
    for input_data_source, input_data_path in INPUT_DATA_SOURCE_TO_PATH.items():
        input_basename = os.path.basename(os.path.normpath(input_data_path))
        inference_steps = []
        for classifier_id, quality_classifier_model_path in enumerate(config.quality_classifier_model_paths):
            # run inference with each quality classifier
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
                pip_dependency_groups=["filelock"],
            )
            inference_steps.append(inference_step)

        ensemble_step = ExecutorStep(
            name=f"attributes/quality_filtering/{config.experiment_name}/{input_data_source}",
            fn=create_custom_attribute,
            config=CustomAttributeConfig(
                input_doc_path=input_data_path,
                output_attr_path=this_output_path(input_basename),
                attribute_func_name="max_quality_score",
                attribute_func_kwargs=versioned(
                    {
                        "score_name": "__label__hq",
                        "output_attr_name": f"{config.experiment_name}-quality",
                        "input_attr_names": [
                            f"{config.experiment_name}-quality_classifier-{classifier_id}"
                            for classifier_id in range(len(config.quality_classifier_model_paths))
                        ],
                    }
                ),
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
                        lower_threshold=versioned(None),
                        keep_fraction=versioned(config.keep_fraction),
                    ),
                ],
            ),
            pip_dependency_groups=["ddsketch"],
        )

        tokenize_step = default_tokenize(
            name=f"quality_filtering/{config.experiment_name}/{input_data_source}",
            dataset=output_path_of(consolidate_step),
            tokenizer=llama3_tokenizer,
        )

        cooldown_step = default_quality_ablation(tokenize_step, config.cooldown_config)

        steps.append(consolidate_step)
        steps.append(tokenize_step)
        steps.append(cooldown_step)

    return steps


def main():
    experiment_config = ExperimentConfig(
        experiment_name="exp615_ensemble",
        quality_classifier_model_paths=[dclm_eli5_100k_oh_100k_rw_200k, marin_mmlu_100k_rw_100k],
    )
    steps = create_steps(experiment_config)
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
