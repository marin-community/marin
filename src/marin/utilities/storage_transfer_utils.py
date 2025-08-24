"""
storage_transfer_utils.py

Helpful functions for programmatically creating and verifying GCS Storage Transfer jobs (from external URLs,
HuggingFace, external blob-stores like S3, as well as other GCS buckets).
"""

import logging
import urllib.parse
from pathlib import Path
from time import time

import fsspec
from google.api_core import operations_v1
from google.cloud import storage_transfer, storage_transfer_v1

logger = logging.getLogger(__name__)


def create_url_list_tsv_on_gcs(
    url_list: list[str],
    gcs_output_path: Path,
    public_gcs_bucket: str,
    return_url: bool = False,
) -> str:
    """
    Creates a TSV file specifying the URLs to download using the Google Cloud Storage Transfer Service. This TSV file
    must be publicly readable, which is why we write to `public_gcs_bucket`.
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
    logger.info(f"[*] Creating Storage Transfer Job :: Download `{tsv_url}` to `gs://{gcs_bucket}/{gcs_output_path}`")
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
        job_url = (
            "https://console.cloud.google.com/transfer/jobs/"
            f"{urllib.parse.quote_plus(creation_request.name)}/"
            f"monitoring?hl=en&project={gcp_project_id}"
        )
        return creation_request.name, job_url


def wait_for_transfer_job(job_name: str, timeout: int, poll_interval: int, gcp_project_id: str = "hai-gcp-models"):
    """
    Waits for a Transfer Job to complete by polling the job status every 10 seconds. Raises a `TimeoutError` if the
    job does not complete within the specified `timeout` (default: 30 minutes).

    Parameters:
        job_name (str): The name of the Transfer Job to wait for.
        timeout (int): The maximum number of seconds to wait for the job to complete.
        poll_interval (int): The number of seconds to wait between polling the job status.

    Raises:
        TimeoutError: If the Transfer Job does not complete within the specified `timeout`.
    """
    logger.info(f"[*] Waiting for Transfer Job :: {job_name}")

    transfer_client = storage_transfer_v1.StorageTransferServiceClient()
    channel = transfer_client.transport.grpc_channel
    operations_client = operations_v1.OperationsClient(channel)
    start_time = time()

    while time() - start_time < timeout:
        if (time() - start_time) % poll_interval == 0:
            # Prepare the filter string to get the operations for the job
            filter_string = f'{{"project_id": "{gcp_project_id}", "job_names": ["{job_name}"]}}'
            # List transfer operations for the job
            # Use operations_client to list operations related to this transfer job
            transfer_operations = operations_client.list_operations("transferOperations", filter_string)
            # Check the status of all operations
            operation_statuses = [operation.done for operation in transfer_operations]
            complete_operations = operation_statuses.count(True)
            total_operations = len(operation_statuses)
            if total_operations > 0:
                percent_complete = (complete_operations / total_operations) * 100
            else:
                percent_complete = 0
            logger.info(f"{complete_operations}/{total_operations} ({percent_complete}%) complete")
            if percent_complete == 100:
                logger.info(f"[*] Transfer Job Completed :: {job_name}")
                return

    raise TimeoutError(f"Transfer Job did not complete within {timeout} seconds; check status for {job_name}")
