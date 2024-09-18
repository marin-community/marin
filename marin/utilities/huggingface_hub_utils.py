"""
huggingface_hub_utils.py

Helpful functions for facilitating downloads/verification of datasets/artifacts hosted on the HuggingFace Hub.
"""

import fsspec
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import GatedRepoError

from marin.utilities.storage_transfer_utils import create_gcs_transfer_job_from_tsv, create_url_list_tsv_on_gcs
from marin.utilities.gcs_utils import split_gcs_path
from marin.utilities.validation_utils import write_provenance_json


def get_hf_dataset_urls(hf_dataset_id: str, revision: str, hf_url_glob: str) -> list[str]:
    """Walk through Dataset Repo using the `hf://` fsspec built-ins."""
    fs = fsspec.filesystem("hf")

    # Check if Dataset is Public or Gated
    try:
        fs.info(f"hf://datasets/{hf_dataset_id}", revision=revision)
    except GatedRepoError as err:
        raise NotImplementedError(f"Unable to automatically download gated dataset `{hf_dataset_id}`") from err

    url_list = []
    base_dir = f"hf://datasets/{hf_dataset_id}"
    for fpath in fs.find(base_dir, revision=revision):
        if ".git" in fpath:
            continue

        # Continue if the file does not match the hf_url_glob pattern
        if not fs.glob(f"{base_dir}/{hf_url_glob}"):
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


def download_hf_dataset(hf_dataset_id: str, revision: str, hf_url_glob: str, gcs_output_path: str, public_gcs_path: str) -> str:
    """Create & Launch a Google Cloud Storage Transfer Job to Download a (Public) HuggingFace Dataset."""
    hf_urls = get_hf_dataset_urls(hf_dataset_id, revision, hf_url_glob)

    # Parse GCS Bucket, Relative Path from `gcs_output_path`
    gcs_bucket, gcs_relative_path = split_gcs_path(gcs_output_path)

    # Use `revision` as "version" for writing to GCS
    gcs_versioned_relative_path = gcs_relative_path / revision

    # Parse Public GCS Bucket from `public_gcs_path`
    public_gcs_bucket, _ = split_gcs_path(public_gcs_path)

    # Create a TSV File Manifest (publicly accessible URL)
    tsv_url = create_url_list_tsv_on_gcs(hf_urls, gcs_versioned_relative_path, public_gcs_bucket, return_url=True)

    # Initialize and Launch STS Job (using GCloud API)
    job_name, job_url = create_gcs_transfer_job_from_tsv(
        tsv_url,
        gcs_versioned_relative_path,
        gcs_bucket,
        description=f"Raw HF Dataset Download: `{hf_dataset_id}`",
        return_job_url=True,
    )

    # Write Provenance JSON
    write_provenance_json(
        gcs_versioned_relative_path,
        gcs_bucket,
        metadata={"dataset": hf_dataset_id, "version": revision, "links": hf_urls},
    )

    return job_name, job_url
