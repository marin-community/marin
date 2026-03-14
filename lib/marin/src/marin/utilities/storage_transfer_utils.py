# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
storage_transfer_utils.py

Helpful functions for programmatically creating and verifying GCS Storage Transfer jobs (from external URLs,
HuggingFace, external blob-stores like S3, as well as other GCS buckets).
"""

import logging
from time import time

from google.api_core import operations_v1
from google.cloud import storage_transfer_v1

logger = logging.getLogger(__name__)


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
