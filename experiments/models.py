from dataclasses import dataclass

from experiments.instruction_datasets import get_directory_friendly_dataset_name
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from operations.download.huggingface.download import DownloadConfig
from operations.download.huggingface.download_hf import download_hf


@dataclass(frozen=True)
class ModelConfig:
    hf_repo_id: str
    hf_revision: str


# We utilize GCSFuse because our disk space is limited on TPUs.
# This means that for certain large models (e.g. Llama 70B), we will not be able
# to fit the models on local disk. We use GCSFuse to mount the GCS bucket to the local filesystem
# to be able to download and use these large models.
GCS_FUSE_MOUNT_PATH = "gcsfuse_mount/models"

MODEL_NAME_TO_CONFIG = {
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": ModelConfig(
        hf_repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        hf_revision="450ff1f",
    ),
}


def download_model_step(model_config: ModelConfig) -> ExecutorStep:
    model_name = get_directory_friendly_dataset_name(model_config.hf_repo_id)
    download_step = ExecutorStep(
        name=f"{GCS_FUSE_MOUNT_PATH}/{model_name}",
        fn=download_hf,
        config=DownloadConfig(
            hf_dataset_id=model_config.hf_repo_id,
            revision=versioned(model_config.hf_revision),
            gcs_output_path=this_output_path(),
            wait_for_completion=True,
            hf_repo_type_prefix="",
        ),
    )

    return download_step


def get_model(model_name: str) -> ExecutorStep:
    model_config = MODEL_NAME_TO_CONFIG[model_name]
    download_step = download_model_step(model_config)
    return download_step


if __name__ == "__main__":
    steps = []
    for model_name in MODEL_NAME_TO_CONFIG.keys():
        steps.append(get_model(model_name))
    executor_main(steps)
