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

"""DatashopRunner is the entrypoint to running an experiment using the Datashop pipeline.

Currently, the Datashop pipeline requires two different clusters depending on the step:
1. Initially we need to use the vLLM cluster to annotate the data that is distilled into the encoder model.
2. We then use the training cluster to filter the large pretraining data pool and train the final model.

See experiments/exp923_medu_mmlu.py for an example.
1. First connect to both vLLM cluster and training cluster. Let the RAY_ADDRESS of vLLM cluster be
$RAY_ADDRESS_VLLM and the RAY_ADDRESS of the training cluster be $RAY_ADDRESS_TRAINING.
2. Run `run_eval_cluster_steps()` to annotate the data that is distilled into the encoder model with marin ray_run
and RAY_ADDRESS=$RAY_ADDRESS_VLLM.
3. After step (2) finishes, run `run_all_steps()` to train the final model with marin ray_run
and RAY_ADDRESS=$RAY_ADDRESS_TRAINING.
"""

# nodryrun

from dataclasses import dataclass, field

from experiments.datashop.defaults import (
    default_candidate_anneal,
    default_consolidate,
    default_label,
    default_quality_filter,
    default_train_quality_model,
)
from experiments.evals.evals import default_eval
from fray.cluster import ResourceConfig
from experiments.evals.task_configs import MMLU_5_SHOT
from marin.datashop.pipeline import CorpusContent
from marin.execution.executor import executor_main


@dataclass
class DatashopRunnerConfig:
    # Defines the name of the experiment
    experiment_name: str

    # Defines the model that will be used to annotate the corpus content (e.g. Llama-3.3-70B-Instruct)
    annotator_model_name: str

    # Defines the large-scale pretraining data that will be used to train the final model
    pretraining_data_path: str

    # Defines the data that will be annotated to be used as the encoder model's training data
    annotator_data_path: str

    # Defines the user's prompt for the annotator model. The prompt should include the
    # `{example}` placeholder for where the example will be inserted.
    # Default: None to represent that the user does not specify a prompt to filter the data pool
    data_filter_prompt: str | None = None

    # Defines the corpus content that we will create annotator prompts for
    # In other words, the data that we "care" about finding more of
    # Default: Empty list to represent that the user does not specify a targeted dataset
    corpus_content_paths: list[CorpusContent] = field(default_factory=list)

    # Name of the pretraining data path to group the output by in a single directory
    pretraining_data_path_name: str = "datashop-dclm-pretraining-subset"

    # How to schedule the TPUs (what hardware to use and how to pack them) specifically for labeling
    labeler_resource_config: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_tpu("v6e-8"))

    # What hardware to use for training the final model
    training_tpu_type: str = "v6e-128"

    # Config for MEDU Pipeline that generates the prompt from benchmark automatically
    medu_pipeline_config_kwargs: dict | None = None

    # Config for dataset output processor to use
    dataset_output_processor_config_kwargs: dict | None = None

    # Config for the quality filter model training
    quality_train_config_kwargs: dict | None = None

    # Config for generating text
    text_generation_inference_config_kwargs: dict | None = None

    # Config for quality filter model inference
    inference_config_kwargs: dict | None = None

    # Config for filter config to allow for changing things like score threshold
    filter_config_kwargs: dict | None = None

    # Config for consolidate to allow for changing things like filetype or ray memory usage
    consolidate_config_kwargs: dict | None = None

    # What hardware to use for evaluating the model
    eval_resource_config: ResourceConfig = field(default_factory=lambda: ResourceConfig.with_tpu("v6e-8"))


class DatashopRunner:
    def __init__(self, config: DatashopRunnerConfig):
        self.config = config

        # TODO(chris): In a later PR, support other TPU types
        self.labeled_documents = default_label(
            self.config.annotator_data_path,
            self.config.corpus_content_paths,
            self.config.experiment_name,
            self.config.labeler_resource_config,
            self.config.annotator_model_name,
            self.config.data_filter_prompt,
            self.config.medu_pipeline_config_kwargs,
            self.config.text_generation_inference_config_kwargs,
        )
        self.encoder_model = default_train_quality_model(
            self.labeled_documents,
            self.config.experiment_name,
            self.config.labeler_resource_config,
            self.config.dataset_output_processor_config_kwargs,
            self.config.quality_train_config_kwargs,
        )
        self.attributes = default_quality_filter(
            self.encoder_model,
            self.config.pretraining_data_path,
            self.config.pretraining_data_path_name,
            self.config.experiment_name,
            self.config.inference_config_kwargs,
        )
        self.filtered_documents = default_consolidate(
            self.attributes,
            self.config.pretraining_data_path,
            self.config.pretraining_data_path_name,
            self.config.experiment_name,
            self.config.filter_config_kwargs,
            self.config.consolidate_config_kwargs,
        )
        self.control_model = default_candidate_anneal(
            None,
            self.config.training_tpu_type,
            self.config.experiment_name,
        )
        self.quality_ablation_model = default_candidate_anneal(
            self.filtered_documents,
            self.config.training_tpu_type,
            self.config.experiment_name,
        )
        self.evals = [
            default_eval(self.control_model, self.config.eval_resource_config, MMLU_5_SHOT),
            default_eval(self.quality_ablation_model, self.config.eval_resource_config, MMLU_5_SHOT),
        ]

    def get_eval_cluster_steps(self):
        return [self.encoder_model]

    def get_all_steps(self):
        return self.evals

    # NOTE(chris): Run this in the vLLM Cluster
    def run_eval_cluster_steps(self):
        executor_main(self.get_eval_cluster_steps())

    # NOTE(chris): Run this in the training cluster
    def run_all_steps(self):
        executor_main(self.get_all_steps())
