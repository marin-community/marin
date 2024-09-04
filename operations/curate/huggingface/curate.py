"""
hf/curate.py

Curation/Download script for arbitrary datasets hosted on HuggingFace (supports both public and gated HF Datasets); this
script requires a pointer to the dataset repository URL, a revision (Git SHA from the HF Dataset page), and an optional
authentication token (for gated datasets).

We will automatically generate a Google Storage Transfer Service (STS) job to download the appropriate files to GCS.

Run with:
    - [Local] python operations/curate/hf/curate.py \
        --gcs_output_path="raw/hello_world_fw" \
        --hf_dataset_id="skaramcheti/hello_world_fw" \
        --revision="8fd6e8e"
"""

import os
from dataclasses import dataclass
from pathlib import Path

import draccus

from marin.utilities.huggingface_hub_utils import download_hf_dataset


@dataclass
class CurationConfig:
    # fmt: off
    gcs_output_path: Path = Path("raw/hello_world_fw")              # Path to store (versioned) raw data on GCS
    gcs_bucket: str | None = None                                   # Default GCS Bucket (if None: os.environ["MARIN"])

    # HuggingFace Dataset Parameters
    hf_dataset_id: str = "skaramcheti/hello_world_fw"               # HF Dataset to Download (as Repository ID)
    revision: str = "8fd6e8e"                                       # (Short) Git Commit Hash (from HF Dataset Repo)

    def __post_init__(self) -> None:
        self.gcs_bucket = os.environ["MARIN"] if self.gcs_bucket is None else self.gcs_bucket
        if "gs://" in str(self.gcs_output_path):
            raise ValueError(f"Unexpected GCS Bucket Prefix in `{self.gcs_output_path = }`")

    # fmt: on


@draccus.wrap()
def curate(cfg: CurationConfig) -> None:
    print(f"[*] Downloading HF Dataset `{cfg.hf_dataset_id}` to `gs://{cfg.gcs_bucket}/{cfg.gcs_output_path}`")
    job_url = download_hf_dataset(cfg.hf_dataset_id, cfg.revision, cfg.gcs_output_path, cfg.gcs_bucket)

    print(f"[*] Launched Transfer Job & wrote `provenance.json`; check Transfer Job status at:\n\t=> {job_url}")


if __name__ == "__main__":
    curate()
