from dataclasses import dataclass, field

from experiments.datashop.defaults import (
    default_candidate_anneal,
    default_label,
    default_quality_filter_and_consolidate,
    default_train_quality_model,
)
from experiments.evals.resource_configs import TPU_V6E_8_STRICT_PACK, ResourceConfig
from marin.execution.executor import executor_main


@dataclass
class DatashopRunnerConfig:
    # Defines the name of the experiment
    experiment_name: str

    # Defines the model that will be used to annotate the corpus content
    # Default: Llama-3.3-70B-Instruct,
    # TODO(chris): Support more models later
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
    corpus_content_paths: list[str] = field(default_factory=list)

    # Name of the pretraining data path to group the output by in a single directory
    pretraining_data_path_name: str = "medu-dclm-pretraining-subset"

    # How to schedule the TPUs (what hardware to use and how to pack them) specifically for labeling
    labeler_resource_config: ResourceConfig = field(default_factory=lambda: TPU_V6E_8_STRICT_PACK)

    # What hardware to use for training the final model
    training_tpu_type: str = "v6e-128"


class DatashopRunner:
    def __init__(self, config: DatashopRunnerConfig):
        self.config = config

        # TODO(chris): In a later PR, support other TPU types
        self.labeled_documents = default_label(
            self.config.annotator_data_path,
            self.config.corpus_content_paths,
            self.config.experiment_name,
            self.config.labeler_resource_config,
            self.config.data_filter_prompt,
        )
        self.encoder_model = default_train_quality_model(
            self.labeled_documents, self.config.experiment_name, self.config.labeler_resource_config
        )
        self.filtered_documents = default_quality_filter_and_consolidate(
            self.encoder_model,
            self.config.pretraining_data_path,
            self.config.pretraining_data_path_name,
            self.config.experiment_name,
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

    def get_eval_cluster_steps(self):
        return [self.encoder_model]

    def get_all_steps(self):
        return [self.quality_ablation_model, self.control_model]

    # NOTE(chris): Run this in the vLLM Cluster
    def run_eval_cluster_steps(self):
        executor_main(self.get_eval_cluster_steps())

    # NOTE(chris): Run this in the training cluster
    def run_all_steps(self):
        executor_main(self.get_all_steps())
