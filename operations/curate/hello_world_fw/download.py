"""
hello_world_fw/download.py

Download script for the `hello_world_fw` data, provided through HuggingFace Datasets; this script is a generic
template for ingesting any HuggingFace dataset (raw files, preserving paths and metadata) into GCS.

Home Page (Placeholder): https://huggingface.co/datasets/skaramcheti/hello_world_fw

Run with:
    - [Local] python operations/curate/hello_world_fw/download.py --gcs_output_path="scratch/raw/hello_world_fw"
"""

import os
from dataclasses import dataclass
from pathlib import Path

import draccus

from marin.utilities.storage_transfer_utils import (
    create_gcs_transfer_job_from_tsv,
    create_url_list_tsv_on_gcs,
    get_hf_dataset_urls,
)
from marin.utilities.validation_utils import write_provenance_json


@dataclass
class DownloadConfig:
    # fmt: off
    gcs_output_path: Path = Path("scratch/raw/hello_world_fw")  # Path to store (versioned) raw data on GCS
    gcs_bucket: str | None = None                               # Default GCS Bucket (default: `os.environ["MARIN"]`)

    # HuggingFace Dataset Parameters
    hf_dataset_id: str = "skaramcheti/hello_world_fw"           # HF Dataset to Download (as Repo ID)
    revision: str = "8fd6e8e"                                   # (Short) Git Commit Hash (from HF Dataset Repo)

    def __post_init__(self) -> None:
        self.gcs_bucket = os.environ["MARIN"] if self.gcs_bucket is None else self.gcs_bucket

    # fmt: on


@draccus.wrap()
def download(cfg: DownloadConfig) -> None:
    print(f"[*] Downloading hello_world_fw Dataset to `gs://{cfg.gcs_bucket}/{cfg.gcs_output_path}`")

    # Parse Version (use `revision`) =>> update `gcs_output_path`
    gcs_output_path = cfg.gcs_output_path / cfg.revision

    # Get File URLs from HF Dataset Repo =>> Create URL Transfer Job =>> Launch Transfer Job =>> Write Provenance
    hf_urls = get_hf_dataset_urls(cfg.hf_dataset_id, cfg.revision)
    tsv_url = create_url_list_tsv_on_gcs(hf_urls, gcs_output_path, return_url=True)
    job_url = create_gcs_transfer_job_from_tsv(
        tsv_url, gcs_output_path, cfg.gcs_bucket, description="hello_world_fw: Raw Data Download", return_job_url=True
    )
    write_provenance_json(
        gcs_output_path,
        cfg.gcs_bucket,
        metadata={"dataset": "hello_world_fw", "version": cfg.revision, "links": hf_urls},
    )

    # Finalize
    print(
        f"Transfer Job Launched & `provenance.json` written to `{gcs_output_path}`; check Transfer Job status at:\n"
        f"\t=> {job_url}"
    )


if __name__ == "__main__":
    download()
