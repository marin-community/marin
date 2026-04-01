# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Test that wait_for_job_with_streaming passes since_ms to fetch_logs."""

from unittest.mock import MagicMock, patch

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.types import JobName
from iris.rpc import cluster_pb2


def test_streaming_passes_since_ms_to_fetch_logs():
    """since_ms must reach the fetch_logs call inside wait_for_job_with_streaming."""
    client = object.__new__(RemoteClusterClient)
    client._client = MagicMock()

    job_id = JobName.from_wire("/user/test-job")
    since_ms_value = 1711900000000

    # _poll_job_state returns a terminal state immediately so the loop exits.
    with (
        patch.object(client, "_poll_job_state", return_value=cluster_pb2.JOB_STATE_SUCCEEDED),
        patch.object(
            client,
            "get_job_status",
            return_value=cluster_pb2.JobStatus(state=cluster_pb2.JOB_STATE_SUCCEEDED),
        ),
        patch.object(
            client,
            "fetch_logs",
            return_value=cluster_pb2.FetchLogsResponse(entries=[], cursor=0),
        ) as mock_fetch,
    ):
        client.wait_for_job_with_streaming(
            job_id,
            timeout=5.0,
            poll_interval=0.1,
            since_ms=since_ms_value,
        )

    assert mock_fetch.call_count >= 1, "fetch_logs should have been called at least once"
    for call in mock_fetch.call_args_list:
        assert call.kwargs.get("since_ms") == since_ms_value, f"Expected since_ms={since_ms_value}, got {call.kwargs}"
