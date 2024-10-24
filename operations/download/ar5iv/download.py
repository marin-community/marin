"""
ar5iv/download.py

Download script for the ar5iv raw HTML data, provided by SIGMathLing (Forum/Resource Cooperative for the Linguistics of
Mathematical/Technical Documents).

Home Page: https://sigmathling.kwarc.info/resources/ar5iv-dataset-2024/

Run with:
    - [Local] python operations/download/ar5iv/download.py --gcs_output_path="gs://marin-us-central2/raw/ar5iv"
"""

import json
from dataclasses import dataclass
from pathlib import Path

import draccus

from marin.utilities.gcs_utils import split_gcs_path
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
    gcs_output_path: str = (                            # Path to store raw data on GCS (including gs://$BUCKET)
        "gs://marin-us-central2/raw/ar5iv"
    )

    # Dataset-Specific Parameters
    personalized_url_json: Path | None = Path(          # Path to JSON File defining (personalized) links to ar5iv data
        "operations/download/ar5iv/ar5iv-v04-2024.json"
    )

    # Additional GCS Parameters
    public_gcs_path: str = (                            # Path to Publicly Readable Bucket (for Storage Transfer)
        "gs://hf_dataset_transfer_bucket"
    )

    def __post_init__(self) -> None:
        if not self.gcs_output_path.startswith("gs://"):
            raise ValueError(
                f"Invalid `{self.gcs_output_path = }`; expected URI of form `gs://BUCKET/path/to/resource`"
            )

        if not self.public_gcs_path.startswith("gs://"):
            raise ValueError(
                f"Invalid `{self.public_gcs_path = }`; expected URI of form `gs://BUCKET`"
            )

    # fmt: on


@draccus.wrap()
def download(cfg: DownloadConfig) -> None:
    print(f"[*] Downloading ar5iv Dataset to `{cfg.gcs_output_path}`")
    if cfg.personalized_url_json is None or not cfg.personalized_url_json.exists():
        print(DOWNLOAD_INSTRUCTIONS)
        raise ValueError("Missing `personalized_url_json`")

    # Load JSON =>> Gently Validate URLs
    with open(cfg.personalized_url_json, "r") as f:
        ar5iv_url_cfg = json.load(f)
        if any(dl_file["url"] == "" for dl_file in ar5iv_url_cfg["links"]):
            print(DOWNLOAD_INSTRUCTIONS)
            raise ValueError("Missing `personalized_url_json`")

    # Parse GCS Bucket, Relative Path from `gcs_output_path`
    gcs_bucket, gcs_relative_path = split_gcs_path(cfg.gcs_output_path)

    # Parse Version =>> update `gcs_output_path`
    gcs_versioned_relative_path = gcs_relative_path / ar5iv_url_cfg["version"]

    # Parse Public GCS Bucket from `public_gcs_path`
    public_gcs_bucket, _ = split_gcs_path(cfg.public_gcs_path)

    # Create a TSV File Manifest (publicly accessible URL)
    tsv_url = create_url_list_tsv_on_gcs(
        [dl_file["url"] for dl_file in ar5iv_url_cfg["links"]],
        gcs_versioned_relative_path,
        public_gcs_bucket,
        return_url=True,
    )

    # Initialize and Launch STS Job (using GCloud API)
    _, job_url = create_gcs_transfer_job_from_tsv(
        tsv_url,
        gcs_versioned_relative_path,
        gcs_bucket,
        description="Raw Custom Data Download: `ar5iv`",
        return_job_url=True,
    )

    # Write Provenance JSON
    # TODO: convert this script to fsspec
    gcs_path = f"gs://{gcs_bucket}/{gcs_versioned_relative_path}"
    write_provenance_json(gcs_path, metadata=ar5iv_url_cfg)

    # Finalize
    print(f"[*] Launched Transfer Job & wrote `provenance.json`; check Transfer Job status at:\n\t=> {job_url}")


if __name__ == "__main__":
    download()
