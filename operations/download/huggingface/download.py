"""
huggingface/download.py

Download script for arbitrary datasets hosted on HuggingFace (supports public HF Datasets); this script
requires a pointer to the dataset repository URL, and a revision (Git SHA from the HF Dataset page). Given this, we
will automatically launch a Google Storage Transfer Service (STS) job to download the appropriate files to GCS.

Run with:
    - [Local] python operations/download/huggingface/download.py \
        --gcs_output_path="gs://marin-us-central2/raw/hello_world_fw" \
        --hf_dataset_id="skaramcheti/hello_world_fw" \
        --revision="8fd6e8e"
"""

import dataclasses
import logging
from dataclasses import dataclass

import draccus
import ray

from marin.utilities.huggingface_hub_utils import download_hf_dataset
from marin.utilities.storage_transfer_utils import wait_for_transfer_job

logger = logging.getLogger("ray")


@dataclass
class DownloadConfig:
    # fmt: off
    gcs_output_path: str
    """
    Path to store raw data in persistent storage (e.g. gs://$BUCKET/...).
    This works with any fsspec-compatible path, but for backwards compatibility, we call it gcs_output_path.
    """

    # HuggingFace Dataset Parameters
    hf_dataset_id: str                                      # HF Dataset to Download (as `$ORG/$DATASET` on HF Hub)

    revision: str  # (Short) Commit Hash (from HF Dataset Repo; 7 characters)
    hf_urls_glob: list[str] = dataclasses.field(default_factory=list)
    # List of Glob Patterns to Match Files in HF Dataset, If empty we get all the files in a hf repo

    # Additional GCS Parameters
    public_gcs_path: str = (                                # Path to Publicly Readable Bucket (for Storage Transfer)
        "gs://hf_dataset_transfer_bucket"
    )

    # Job Control Parameters, used only for non-gated dataset transfers done via STS
    wait_for_completion: bool = True                        # if True, will block until job completes
    timeout: int = 1800                                     # Maximum time to wait for job completion (in seconds)
    poll_interval: int = 10                                 # Time to wait between polling job status (in seconds)

    # fmt: on
    hf_repo_type_prefix: str = (
        "datasets"  # The repo_type_prefix is datasets/ for datasets,
        # spaces/ for spaces, and models do not need a prefix in the URL.
    )


@ray.remote
def _wait_for_job_completion(job_name: str, timeout: int, poll_interval: int) -> str:
    """Wait for a Transfer Job to complete.
    Parameters:
        job_name (str): Name of the Transfer Job to wait for.
        timeout (int): Maximum time to wait for the job to complete (in seconds).
        poll_interval (int): Time to wait between polling the job status (in seconds

    Raises:
        TimeoutError: If the job does not complete within the specified `timeout`.
    """

    wait_for_transfer_job(job_name, timeout=timeout, poll_interval=poll_interval)
    return f"Transfer job completed: {job_name}"


def download(cfg: DownloadConfig) -> None | ray.ObjectRef:
    logging.warning(
        "DEPRECATED: This function is deprecated and will be removed in a future release." "Consider using download_hf"
    )

    logger.info(f"[*] Downloading HF Dataset `{cfg.hf_dataset_id}` to `{cfg.gcs_output_path}`")

    job_name, job_url = download_hf_dataset(
        cfg.hf_dataset_id, cfg.revision, cfg.hf_urls_glob, cfg.gcs_output_path, cfg.public_gcs_path
    )

    if cfg.wait_for_completion:
        logger.info(f"[*] Waiting for Job Completion :: {job_url}")
        future = _wait_for_job_completion.remote(job_name, cfg.timeout, cfg.poll_interval)

        logger.info(f"[*] Launched Job Completion Waiter :: {future}")
        result = ray.get(future)
        logger.info(f"[*] Job Completion Waiter Result :: {result}")

        return future

    # Finalize
    logger.info(f"[*] Launched Transfer Job & wrote `provenance.json`; check Transfer Job status at:\n\t=> {job_url}")


@draccus.wrap()
def main(cfg: DownloadConfig):
    download(cfg)


if __name__ == "__main__":
    main()
