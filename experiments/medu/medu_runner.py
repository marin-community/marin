import os
from dataclasses import dataclass

from experiments.medu.defaults import (
    default_candidate_anneal,
    default_control_experiment,
    default_label,
    default_quality_filter_and_consolidate,
    default_quality_filter_model,
)
from marin.execution.executor import executor_main


@dataclass
class MEDURunnerConfig:
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

    # Defines the corpus content that we will create annotator prompts for
    # In other words, the data that we "care" about finding more of
    corpus_content_paths: list[str]

    # Name of the pretraining data path to group the output by in a single directory
    pretraining_data_path_name: str = "medu-dclm-pretraining-subset"


REGION_TO_TPU_TYPES = {
    "marin-us-east5": {
        "labeler_tpu_type": "TPU-v6e-8",
        "training_tpu_type": "TPU-v6e-128",
    },
    "marin-us-east1": {
        "labeler_tpu_type": "TPU-v6e-8",
        "training_tpu_type": "TPU-v6e-128",
    },
}


class MEDURunner:
    def __init__(self, config: MEDURunnerConfig):
        self.config = config
        region = os.getenv("BUCKET")
        labeler_tpu_type = REGION_TO_TPU_TYPES[region]["labeler_tpu_type"]
        training_tpu_type = REGION_TO_TPU_TYPES[region]["training_tpu_type"]

        # TODO(chris): In a later PR, support other TPU types
        self.labeled_documents = default_label(
            self.config.annotator_data_path, self.config.corpus_content_paths, self.config.experiment_name
        )
        self.encoder_model = default_quality_filter_model(
            self.labeled_documents, self.config.experiment_name, labeler_tpu_type
        )
        self.filtered_documents = default_quality_filter_and_consolidate(
            self.encoder_model,
            self.config.pretraining_data_path,
            self.config.pretraining_data_path_name,
            self.config.experiment_name,
        )
        self.control_model = default_control_experiment(
            training_tpu_type,
        )
        self.quality_ablation_model = default_candidate_anneal(
            self.filtered_documents,
            training_tpu_type,
            self.config.experiment_name,
        )

    # NOTE(chris): Run this in the vLLM Cluster
    def run_eval_cluster_steps(self):
        executor_main([self.encoder_model])

    # NOTE(chris): Run this in the training cluster
    def run_all_steps(self):
        executor_main([self.quality_ablation_model])
