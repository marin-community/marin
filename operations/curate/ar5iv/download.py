"""
ar5iv/download.py

Download script for the ar5iv raw HTML data, provided by SIGMathLing (Forum/Resource Cooperative for the Linguistics of
Mathematical/Technical Documents).

Home Page: https://sigmathling.kwarc.info/resources/ar5iv-dataset-2024/

Run with:
    - [Local] python operations/curate/ar5iv/download.py --gcs_output_path="scratch/raw/ar5iv"
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path

import draccus

from marin.utilities.storage_transfer_utils import create_gcs_transfer_job_from_tsv, create_url_list_tsv_on_gcs
from marin.utilities.validation_utils import write_provenance_json

# TODO (siddk, dlwh, percyliang) :: This is basically the README... but if it lives in the "download.py" anyway, not
#  sure if we should write a separate file to GCS?
DOWNLOAD_INSTRUCTIONS = (
    "\n===\n"
    "Invalid `personalized_url_json` (see `operations/curate/ar5iv/ar5iv-04-2024.json` for format).\n\n"
    "Downloading the ar5iv requires agreeing to a *per-user* License Agreement before getting access to the data.\n"
    "See https://sigmathling.kwarc.info/resources/ar5iv-dataset-2024/ for instructions.\n\n"
    "Once approved, add URLs for each zip file to `personalized_url_json` (and bump 'version' if appropriate)."
    "\n===\n"
)


@dataclass
class DownloadConfig:
    # fmt: off
    gcs_output_path: Path = Path("scratch/raw/ar5iv")   # Path to store (versioned) raw data on GCS
    gcs_bucket: str | None = None                       # Default GCS Bucket (`None` defaults to `os.environ["MARIN"]`)

    # Dataset-Specific Parameters
    personalized_url_json: Path | None = Path(          # Path to JSON File defining (personalized) links to ar5iv data
        "operations/curate/ar5iv/ar5iv-v04-2024.json"
    )

    def __post_init__(self) -> None:
        self.gcs_bucket = os.environ["MARIN"] if self.gcs_bucket is None else self.gcs_bucket

    # fmt: on


@draccus.wrap()
def download(cfg: DownloadConfig) -> None:
    print(f"[*] Downloading ar5iv Dataset to `gs://{cfg.gcs_bucket}/{cfg.gcs_output_path}`")
    if cfg.personalized_url_json is None or not cfg.personalized_url_json.exists():
        print(DOWNLOAD_INSTRUCTIONS)
        raise ValueError("Missing `personalized_url_json`")

    # Load JSON =>> Gently Validate URLs
    with open(cfg.personalized_url_json, "r") as f:
        ar5iv_url_cfg = json.load(f)
        if any(dl_file["url"] == "" for dl_file in ar5iv_url_cfg["links"]):
            print(DOWNLOAD_INSTRUCTIONS)
            raise ValueError("Missing `personalized_url_json`")

    # Parse Version =>> update `gcs_output_path`
    gcs_output_path = cfg.gcs_output_path / ar5iv_url_cfg["version"]

    # Create URL Transfer Job =>> Launch Transfer Job =>> Write Provenance
    tsv_url = create_url_list_tsv_on_gcs(
        [dl_file["url"] for dl_file in ar5iv_url_cfg["links"]], gcs_output_path, return_url=True
    )
    job_url = create_gcs_transfer_job_from_tsv(
        tsv_url, gcs_output_path, cfg.gcs_bucket, description="ar5iv: Raw Data Download", return_job_url=True
    )
    write_provenance_json(gcs_output_path, cfg.gcs_bucket, metadata=ar5iv_url_cfg)

    # TODO (siddk) =>> Figure out if we want to block on job completion then checksum here (vs. part of "verify")?

    # Finalize
    print(
        f"Transfer Job Launched & `provenance.json` written to `{gcs_output_path}`; check Transfer Job status at:\n"
        f"\t=> {job_url}"
    )


if __name__ == "__main__":
    download()
