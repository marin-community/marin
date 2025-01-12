import subprocess
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
    # While the above operation downloads the model to the GCS, the model
    # is not available on the local filesystem since GCS does not handle
    # implicitly defined directories that are not created through GCSFuse.
    # The previous operation was created using Cloud Storage but not FUSE
    # so it will no appear to be available. The following operation creates
    # the directory on the local filesystem.
    # Source: https://arc.net/l/quote/dcgeuozs
    create_local_filesystem_link(model_name)

    return download_step


def create_local_filesystem_link(model_name: str):
    # Create /opt/gcsfuse_mount/models
    subprocess.run(["mkdir", f"{LOCAL_PREFIX}/{GCS_FUSE_MOUNT_PATH}"], capture_output=True, text=True)

    # TODO: Link the actual model folder to the local filesystem as well
    # Right now it is abstracted away because no access to the hashed version


def get_model(model_name: str) -> ExecutorStep:
    model_config = MODEL_NAME_TO_CONFIG[model_name]
    download_step = download_model_step(model_config)
    return download_step


if __name__ == "__main__":
    steps = []
    for model_name in MODEL_NAME_TO_CONFIG.keys():
        steps.append(get_model(model_name))
    executor_main(steps)
