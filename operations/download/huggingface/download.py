"""
huggingface/download.py

Download script for arbitrary datasets hosted on HuggingFace (supports both public and gated HF Datasets); this script
requires a pointer to the dataset repository URL, and a revision (Git SHA from the HF Dataset page). Given this, we
will automatically launch a Google Storage Transfer Service (STS) job to download the appropriate files to GCS.

Run with:
    - [Local] python operations/download/huggingface/download.py \
        --gcs_output_path="gs://marin-us-central2/raw/hello_world_fw" \
        --hf_dataset_id="skaramcheti/hello_world_fw" \
        --revision="8fd6e8e"
"""

from dataclasses import dataclass

import draccus

from marin.utilities.huggingface_hub_utils import download_hf_dataset


@dataclass
class DownloadConfig:
    # fmt: off
    gcs_output_path: str                                    # Path to store raw data on GCS (includes gs://$BUCKET/...)

    # HuggingFace Dataset Parameters
    hf_dataset_id: str                                      # HF Dataset to Download (as `$ORG/$DATASET` on HF Hub)
    revision: str = "main"                                  # (Short) Commit Hash (from HF Dataset Repo; 7 characters)
    hf_url_glob: str = "*"                                  # Glob Pattern to Match Files in HF Dataset

    # Additional GCS Parameters
    public_gcs_path: str = (                                # Path to Publicly Readable Bucket (for Storage Transfer)
        "gs://hf_dataset_transfer_bucket"
    )

    def __post_init__(self) -> None:
        if not self.gcs_output_path.startswith("gs://"):
            raise ValueError(
                f"Invalid `{self.gcs_output_path = }`; expected URI of form `gs://BUCKET/path/to/resource`"
            )

        if not self.public_gcs_path.startswith("gs://"):
            raise ValueError(
                f"Invalid `{self.public_gcs_path = }`; expected URI of form `gs://BUCKET/...`"
            )

    # fmt: on


@draccus.wrap()
def download(cfg: DownloadConfig) -> None:
    print(f"[*] Downloading HF Dataset `{cfg.hf_dataset_id}` to `{cfg.gcs_output_path}`")
    job_url = download_hf_dataset(
        cfg.hf_dataset_id, cfg.revision, cfg.hf_url_glob, cfg.gcs_output_path, cfg.public_gcs_path
    )

    # Finalize
    print(f"[*] Launched Transfer Job & wrote `provenance.json`; check Transfer Job status at:\n\t=> {job_url}")


if __name__ == "__main__":
    download()
