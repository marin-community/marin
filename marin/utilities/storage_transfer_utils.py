"""
storage_transfer_utils.py

Helpful functions for programmatically creating and verifying GCS Storage Transfer jobs (from external URLs,
HuggingFace, external blob-stores like S3, as well as other GCS buckets).
"""

import urllib.parse
from pathlib import Path

import fsspec
from google.cloud import storage_transfer
from huggingface_hub import hf_hub_url


def create_url_list_tsv_on_gcs(
    url_list: list[str],
    gcs_output_path: Path,
    public_gcs_bucket: str = "hf_dataset_transfer_bucket",
    return_url: bool = False,
) -> str:
    """
    Creates a TSV file specifying the URLs to download using the Google Cloud Storage Transfer Service. This TSV file
    must be publicly readable, which is why we write to `public_gcs_bucket` instead of $MARIN.
    """
    gcs_tsv_path = f"{public_gcs_bucket}/{gcs_output_path!s}/download-urls.tsv"
    with fsspec.open(f"gs://{gcs_tsv_path}", "wt") as f:
        f.write("TsvHttpData-1.0\n")
        for url in url_list:
            f.write(f"{url}\n")

    return gcs_tsv_path if not return_url else f"https://storage.googleapis.com/{gcs_tsv_path}"


def create_gcs_transfer_job_from_tsv(
    tsv_url: str,
    gcs_output_path: Path,
    gcs_bucket: str,
    description: str,
    gcp_project_id: str = "hai-gcp-models",
    return_job_url: bool = False,
) -> str:
    """
    Creates and runs a one-time storage transfer job that downloads all files from a URL list specified in `tsv_url`
    to the specified GCS output path (in `gcs_bucket`).

    Note: The Google Storage Transfer Service always downloads files from URLs to the (messy, possibly non-homogeneous)
    path: `gcs_output_path/[URL_HOSTNAME]/[PATH]/[TO]/[FILENAME]/[IN]/URL`.

    As an example, with:
     - GCS Output Path = "raw/ar5iv/v04.2024"`
     - File URL: "https://data.fau.de/share/zEkvNxgWQ6W/ar5iv-04-2024-no-problem.zip"

    Writes a file to "raw/ar5iv/v04.2024/data.fau.de/share/zEkvNxgWQ6W/ar5iv-04-2024-no-problem.zip" on GCS.
    """
    print(f"[*] Creating Storage Transfer Job :: Download `{tsv_url}` to `gs://{gcs_bucket}/{gcs_output_path}`")
    client = storage_transfer.StorageTransferServiceClient()

    # Create a Transfer Job Specification with an HTTP Data "Source", and GCS Bucket "Sink"
    transfer_job_spec = storage_transfer.CreateTransferJobRequest(
        {
            "transfer_job": {
                "project_id": gcp_project_id,
                "description": description,
                "status": storage_transfer.TransferJob.Status.ENABLED,
                "transfer_spec": {
                    "http_data_source": {"list_url": tsv_url},
                    "gcs_data_sink": {"bucket_name": gcs_bucket, "path": f"{gcs_output_path!s}/"},
                },
            }
        }
    )

    # Create Job and Run
    #   => `creation_request` =>> Specifies Job, creates new "entry" in `console.cloud.google.com/jobs/transferJobs`
    #   => `run_transfer_job` =>> Actually invokes the transfer operation
    creation_request = client.create_transfer_job(transfer_job_spec)
    client.run_transfer_job({"job_name": creation_request.name, "project_id": gcp_project_id})

    if not return_job_url:
        return creation_request.name

    else:
        return (
            f"https://console.cloud.google.com/transfer/jobs/"
            f"{urllib.parse.quote_plus(creation_request.name)}/"
            f"monitoring?hl=en&project={gcp_project_id}"
        )


def get_hf_dataset_urls(hf_dataset_id: str, revision: str) -> list[str]:
    """Walk through Dataset Repo using the `hf://` fsspec built-ins."""
    fs = fsspec.filesystem("hf")

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
