# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RemoteClusterClient retry logic."""

import time
from unittest.mock import MagicMock, patch

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.types import JobName
from iris.rpc import cluster_pb2


@pytest.fixture
def mock_client():
    """Create a RemoteClusterClient with a mocked RPC client."""
    with patch("iris.cluster.client.remote_client.ControllerServiceClientSync") as mock_cls:
        client = RemoteClusterClient(controller_address="http://test:8080")
        client._client = mock_cls.return_value
        yield client


def test_get_job_status_retries_on_unavailable(mock_client):
    """Verify get_job_status retries on UNAVAILABLE errors."""
    job_id = JobName.root("test-job")

    # Mock: fail twice with UNAVAILABLE, then succeed
    mock_response = MagicMock()
    mock_response.job = cluster_pb2.JobStatus(job_id=job_id.to_wire(), state=cluster_pb2.JOB_STATE_RUNNING)

    call_count = 0

    def side_effect(_request):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            raise ConnectError(Code.UNAVAILABLE, "Controller temporarily unavailable")
        return mock_response

    mock_client._client.get_job_status.side_effect = side_effect

    # Should eventually succeed after retries
    start = time.monotonic()
    result = mock_client.get_job_status(job_id)
    elapsed = time.monotonic() - start

    assert result.state == cluster_pb2.JOB_STATE_RUNNING
    assert call_count == 3  # 2 failures + 1 success
    # Verify some delay occurred (initial backoff is 0.1s, so at least 0.1s)
    assert elapsed >= 0.1


def test_get_job_status_retries_on_internal_error(mock_client):
    """Verify get_job_status retries on INTERNAL errors (network errors)."""
    job_id = JobName.root("test-job")

    # Mock: fail once with INTERNAL, then succeed
    mock_response = MagicMock()
    mock_response.job = cluster_pb2.JobStatus(job_id=job_id.to_wire(), state=cluster_pb2.JOB_STATE_SUCCEEDED)

    call_count = 0

    def side_effect(_request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectError(Code.INTERNAL, "[Errno 104] Connection reset by peer")
        return mock_response

    mock_client._client.get_job_status.side_effect = side_effect

    result = mock_client.get_job_status(job_id)
    assert result.state == cluster_pb2.JOB_STATE_SUCCEEDED
    assert call_count == 2


def test_get_job_status_fails_after_max_retries(mock_client):
    """Verify get_job_status gives up after max_attempts."""
    job_id = JobName.root("test-job")

    # Mock: always fail with UNAVAILABLE
    mock_client._client.get_job_status.side_effect = ConnectError(Code.UNAVAILABLE, "Controller down")

    # Should raise after max_attempts (default 5)
    with pytest.raises(ConnectError) as exc_info:
        mock_client.get_job_status(job_id)

    assert exc_info.value.code == Code.UNAVAILABLE
    assert mock_client._client.get_job_status.call_count == 5


def test_get_job_status_no_retry_on_not_found(mock_client):
    """Verify get_job_status does not retry on NOT_FOUND."""
    job_id = JobName.root("test-job")

    # Mock: return NOT_FOUND (not retryable)
    mock_client._client.get_job_status.side_effect = ConnectError(Code.NOT_FOUND, "Job not found")

    # Should fail immediately without retry
    with pytest.raises(ConnectError) as exc_info:
        mock_client.get_job_status(job_id)

    assert exc_info.value.code == Code.NOT_FOUND
    # Only one call - no retries
    assert mock_client._client.get_job_status.call_count == 1


def test_fetch_task_logs_retries_on_connection_error(mock_client):
    """Verify fetch_task_logs retries the GetTaskLogs RPC on connection errors."""
    job_id = JobName.root("test-job")

    logs_response = cluster_pb2.Controller.GetTaskLogsResponse(
        task_logs=[],
        last_timestamp_ms=0,
        truncated=False,
    )

    call_count = 0

    def side_effect(_request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectError(Code.UNAVAILABLE, "Connection lost")
        return logs_response

    mock_client._client.get_task_logs.side_effect = side_effect

    result = mock_client.fetch_task_logs(job_id)
    assert result.task_logs == []
    assert result.truncated is False
    assert result.last_timestamp_ms == 0
    assert call_count == 2


def test_list_jobs_retries_on_unavailable(mock_client):
    """Verify list_jobs retries on UNAVAILABLE."""
    mock_response = cluster_pb2.Controller.ListJobsResponse()

    call_count = 0

    def side_effect(_request):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise ConnectError(Code.UNAVAILABLE, "Temporary failure")
        return mock_response

    mock_client._client.list_jobs.side_effect = side_effect

    result = mock_client.list_jobs()
    assert result == []
    assert call_count == 2
