"""
huggingface_hub_utils.py

Helpful functions for facilitating downloads/verification of datasets/artifacts hosted on the HuggingFace Hub.
"""

from pathlib import Path

import fsspec
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import GatedRepoError

from marin.utilities.storage_transfer_utils import create_gcs_transfer_job_from_tsv, create_url_list_tsv_on_gcs
from marin.utilities.validation_utils import write_provenance_json


def get_hf_dataset_urls(hf_dataset_id: str, revision: str) -> list[str]:
    """Walk through Dataset Repo using the `hf://` fsspec built-ins."""
    fs = fsspec.filesystem("hf")

    # Check if Dataset is Public or Gated
    try:
        fs.info(f"hf://datasets/{hf_dataset_id}", revision=revision)
    except GatedRepoError as err:
        raise NotImplementedError(f"Unable to automatically download gated dataset `{hf_dataset_id}`") from err

    url_list = []
    for fpath in fs.find(f"hf://datasets/{hf_dataset_id}", revision=revision):
        if ".git" in fpath:
            continue

        # Resolve to HF Path =>> grab URL
        resolved_fpath = fs.resolve_path(fpath)
        url_list.append(
            hf_hub_url(
                resolved_fpath.repo_id,
                resolved_fpath.path_in_repo,
                revision=resolved_fpath.revision,
                repo_type=resolved_fpath.repo_type,
            )
        )

    return url_list


def download_hf_dataset(hf_dataset_id: str, revision: str, gcs_output_path: Path, gcs_bucket: str) -> str:
    """Create & Launch a Google Cloud Storage Transfer Job to Download a (Public) HuggingFace Dataset."""
    hf_urls = get_hf_dataset_urls(hf_dataset_id, revision)

    # Use `revision` as "version" for writing to GCS
    gcs_versioned_output_path = gcs_output_path / revision

    # Create a TSV File Manifest (publicly accessible URL)
    tsv_url = create_url_list_tsv_on_gcs(hf_urls, gcs_versioned_output_path, return_url=True)

    # Initialize and Launch STS Job (using GCloud API)
    job_url = create_gcs_transfer_job_from_tsv(
        tsv_url,
        gcs_versioned_output_path,
        gcs_bucket,
        description=f"Raw HF Dataset Download: `{hf_dataset_id}`",
        return_job_url=True,
    )

    # Write Provenance JSON
    write_provenance_json(
        gcs_versioned_output_path, gcs_bucket, metadata={"dataset": hf_dataset_id, "version": revision, "links": hf_urls}
    )

    return job_url
