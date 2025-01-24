import os
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
LOCAL_PREFIX = "/opt"
GCS_FUSE_MOUNT_PATH = "gcsfuse_mount/models"

MODEL_NAME_TO_CONFIG = {
    "HuggingFaceTB/SmolLM2-1.7B-Instruct": ModelConfig(
        hf_repo_id="HuggingFaceTB/SmolLM2-1.7B-Instruct",
        hf_revision="450ff1f",
    ),
    "Qwen/Qwen2.5-7B-Instruct": ModelConfig(
        hf_repo_id="Qwen/Qwen2.5-7B-Instruct",
        hf_revision="a09a354",
    ),
    "Qwen/Qwen2.5-72B-Instruct": ModelConfig(
        hf_repo_id="Qwen/Qwen2.5-72B-Instruct",
        hf_revision="495f393",
    ),
    "meta-llama/Llama-3.3-70B-Instruct": ModelConfig(
        hf_repo_id="meta-llama/Llama-3.3-70B-Instruct",
        hf_revision="6f6073b",
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
        # must override because it because if we don't then it will end in a hash
        # if it ends in a hash, then we cannot determine the local path
        override_output_path=f"{GCS_FUSE_MOUNT_PATH}/{model_name}",
    )

    return download_step


def get_model_local_path(model_name: str) -> str:
    step = download_model_step(MODEL_NAME_TO_CONFIG[model_name])
    return os.path.join(LOCAL_PREFIX, step.name)


if __name__ == "__main__":
    steps = []
    for model_name in MODEL_NAME_TO_CONFIG.keys():
        steps.append(download_model_step(MODEL_NAME_TO_CONFIG[model_name]))
    executor_main(steps)
