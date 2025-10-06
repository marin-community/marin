# Copyright 2025 The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

import re
import sys
import time

import google.auth
from google.api_core import operations_v1
from google.cloud import storage_transfer_v1


EMPTY_BUCKET = "levanter-empty"


def schedule_gcs_deletion_job(project_id, gcs_bucket_name, path_to_delete):
    """
    Schedules an STS job to delete all files in a GCS path and waits for completion.

    This function uses a "trick" in STS to delete all files in a GCS path by transferring files from an empty bucket to
    the target path with the `delete_objects_unique_in_sink` option enabled. This will delete all objects in the target
    path that do not exist in the source path (which is empty).

    """

    client = storage_transfer_v1.StorageTransferServiceClient()

    # Define the transfer job
    transfer_job = storage_transfer_v1.types.TransferJob(
        project_id=project_id,
        transfer_spec=storage_transfer_v1.types.TransferSpec(
            gcs_data_sink=storage_transfer_v1.types.GcsData(bucket_name=gcs_bucket_name, path=path_to_delete),
            gcs_data_source=storage_transfer_v1.types.GcsData(bucket_name=EMPTY_BUCKET),
            transfer_options=storage_transfer_v1.types.TransferOptions(delete_objects_unique_in_sink=True),
        ),
        status=storage_transfer_v1.types.TransferJob.Status.ENABLED,
        description=f"Delete all files in {gcs_bucket_name}/{path_to_delete}",
    )

    # Create the transfer job
    response = client.create_transfer_job(request={"transfer_job": transfer_job})
    print(f"Created transfer job: {response.name}")
    client.run_transfer_job({"job_name": response.name, "project_id": project_id})

    # Wait for job completion
    wait_for_transfer_job(response.name, timeout=3600, poll_interval=2, project_id=project_id)


def wait_for_transfer_job(job_name: str, timeout: int, poll_interval: int, project_id: str):
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
    print(f"[*] Waiting for Transfer Job :: {job_name}")

    transfer_client = storage_transfer_v1.StorageTransferServiceClient()
    channel = transfer_client.transport.grpc_channel
    operations_client = operations_v1.OperationsClient(channel)
    start_time = time.time()

    from tqdm import tqdm

    pbar = tqdm(desc=f"Waiting for Transfer Job :: {job_name}", unit="B", unit_scale=True)
    while time.time() - start_time < timeout:
        if (time.time() - start_time) % poll_interval == 0:
            # Prepare the filter string to get the operations for the job
            filter_string = f'{{"project_id": "{project_id}", "job_names": ["{job_name}"]}}'

            # List transfer operations for the job
            transfer_operations = operations_client.list_operations("transferOperations", filter_string)

            # Get the latest operation
            latest_operation = None
            for operation in transfer_operations:
                if operation.metadata is not None:
                    latest_operation = operation

            if latest_operation:
                # Extract relevant counters
                # Unpack the Any type to get TransferOperation
                metadata = storage_transfer_v1.types.TransferOperation()
                # Access the descriptor from the _pb2 module
                if latest_operation.metadata.Is(metadata._pb.DESCRIPTOR):
                    latest_operation.metadata.Unpack(metadata._pb)

                objects_deleted = metadata.counters.objects_deleted_from_sink
                objects_found = metadata.counters.objects_found_only_from_sink
                bytes_found_only_from_sink = metadata.counters.bytes_found_only_from_sink
                bytes_deleted_from_sink = metadata.counters.bytes_deleted_from_sink

                # Update the progress bar
                pbar.total = bytes_found_only_from_sink
                pbar.n = bytes_deleted_from_sink
                pbar.set_postfix(
                    objects_deleted=objects_deleted,
                    objects_found=objects_found,
                )
                pbar.update(0)

                if latest_operation.done:
                    print(f"[*] Transfer Job Completed :: {job_name}")
                    pbar.close()
                    return

    raise TimeoutError(f"Transfer Job did not complete within {timeout} seconds; check status for {job_name}")


def parse_gcs_url(gcs_url):
    """Parse the GCS URL and return the bucket name and prefix path."""
    match = re.match(r"gs://([^/]+)/(.+)", gcs_url)
    if match:
        bucket_name, path_prefix = match.groups()
        return bucket_name, path_prefix
    else:
        raise ValueError(f"Invalid GCS URL format: {gcs_url}")


if __name__ == "__main__":
    # Check for correct usage
    if len(sys.argv) != 2:
        print("Usage: python gcs_bulk_delete.py gs://bucket_name/path/to/delete")
        sys.exit(1)

    # Parse the GCS URL
    gcs_url = sys.argv[1]
    try:
        bucket_name, path_prefix = parse_gcs_url(gcs_url)
    except ValueError as e:
        print(e)
        sys.exit(1)

    # Get the project ID
    credentials, project_id = google.auth.default()

    # Schedule the deletion job
    schedule_gcs_deletion_job(project_id, bucket_name, path_prefix)
