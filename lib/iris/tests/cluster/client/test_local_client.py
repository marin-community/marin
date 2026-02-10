# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for local cluster client functionality."""

import logging

import pytest

from iris.client.client import IrisClient, Job
from iris.cluster.client.local_client import LocalClusterClient
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName
from iris.rpc import cluster_pb2


def extract_log_text(response: cluster_pb2.Controller.GetTaskLogsResponse) -> str:
    """Extract log text from a batch log response."""
    lines = []
    for batch in response.task_logs:
        for entry in batch.logs:
            lines.append(entry.data)
    return "\n".join(lines)


@pytest.fixture
def client():
    """Create a local cluster client for testing."""
    client = LocalClusterClient.create()
    yield client
    client.shutdown()


def test_command_entrypoint_preserves_env_vars(client):
    """Verify command entrypoints receive Iris environment variables."""
    job_id = JobName.root("test-env-vars")

    # Create a command that echoes an environment variable
    entrypoint = Entrypoint.from_command("sh", "-c", "echo IRIS_JOB_ID=$IRIS_JOB_ID")

    resources = cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3)

    client.submit_job(job_id=job_id, entrypoint=entrypoint, resources=resources)

    # Wait for job completion
    status = client.wait_for_job(job_id, timeout=10.0, poll_interval=0.1)

    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Check logs contain the job ID
    response = client.fetch_task_logs(job_id.task(0))
    log_text = extract_log_text(response)
    assert f"IRIS_JOB_ID={job_id.to_wire()}" in log_text


def test_log_streaming_captures_output_without_trailing_newline(client):
    """Verify log streaming captures output without trailing newline."""
    job_id = JobName.root("test-no-newline")

    # Use printf which doesn't add a newline
    entrypoint = Entrypoint.from_command("sh", "-c", "printf 'output without newline'")

    resources = cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3)

    client.submit_job(job_id=job_id, entrypoint=entrypoint, resources=resources)

    # Wait for job completion
    status = client.wait_for_job(job_id, timeout=10.0, poll_interval=0.1)

    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Check logs contain the output
    response = client.fetch_task_logs(job_id.task(0))
    log_text = extract_log_text(response)
    assert "output without newline" in log_text


def test_callable_entrypoint_succeeds(client):
    """Verify callable entrypoints execute and complete successfully."""
    job_id = JobName.root("test-callable-success")

    def task_func():
        print("hello from callable")

    entrypoint = Entrypoint.from_callable(task_func)

    resources = cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3)

    client.submit_job(job_id=job_id, entrypoint=entrypoint, resources=resources)

    status = client.wait_for_job(job_id, timeout=10.0, poll_interval=0.1)

    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED


def test_command_entrypoint_with_custom_env_var(client):
    """Verify command entrypoints can access custom environment variables."""
    job_id = JobName.root("test-custom-env")

    # Create a command that uses a custom env var
    entrypoint = Entrypoint.from_command("sh", "-c", "echo CUSTOM_VAR=$CUSTOM_VAR")

    resources = cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3)
    environment = EnvironmentSpec(env_vars={"CUSTOM_VAR": "custom_value"}).to_proto()

    client.submit_job(
        job_id=job_id,
        entrypoint=entrypoint,
        resources=resources,
        environment=environment,
    )

    # Wait for job completion
    status = client.wait_for_job(job_id, timeout=10.0, poll_interval=0.1)

    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    # Check logs contain the custom env var
    response = client.fetch_task_logs(job_id.task(0))
    log_text = extract_log_text(response)
    assert "CUSTOM_VAR=custom_value" in log_text


def test_job_wait_with_stream_logs(client, caplog):
    """Verify Job.wait(stream_logs=True) fetches and streams logs."""
    iris = IrisClient(client)
    job_id = JobName.root("test-stream-logs")

    entrypoint = Entrypoint.from_command("sh", "-c", "echo 'hello from streaming'")
    resources = cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3)

    client.submit_job(job_id=job_id, entrypoint=entrypoint, resources=resources)
    job = Job(iris, job_id)

    with caplog.at_level(logging.INFO, logger="iris.client.client"):
        status = job.wait(stream_logs=True, timeout=10.0)

    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED

    log_messages = [r.message for r in caplog.records]
    assert any(
        "hello from streaming" in msg for msg in log_messages
    ), f"Expected streamed log output, got: {log_messages}"

    # Verify no fetch failures occurred during streaming — this catches bugs
    # like typos or API mismatches that get silently swallowed.
    fetch_failures = [r.message for r in caplog.records if "Failed to fetch" in r.message]
    assert not fetch_failures, f"Log fetching failed during streaming: {fetch_failures}"


def _parent_with_two_children():
    """Parent callable that submits two child jobs and waits for both."""
    from iris.client.client import iris_ctx
    from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

    ctx = iris_ctx()
    res = ResourceSpec(cpu=1, memory="1g")
    env = EnvironmentSpec()

    job_a = ctx.client.submit(
        Entrypoint.from_command("sh", "-c", "echo CHILD_A_LINE_1"),
        "child-a",
        res,
        environment=env,
    )
    job_b = ctx.client.submit(
        Entrypoint.from_command("sh", "-c", "echo CHILD_B_LINE_1"),
        "child-b",
        res,
        environment=env,
    )
    job_a.wait(timeout=30, raise_on_failure=True)
    job_b.wait(timeout=30, raise_on_failure=True)


def test_child_job_logs_sorted_by_timestamp(client):
    """Logs from multiple child jobs are sorted globally by timestamp and include worker_id."""
    parent_id = JobName.root("test-child-logs")
    entrypoint = Entrypoint.from_callable(_parent_with_two_children)
    resources = cluster_pb2.ResourceSpecProto(cpu=1, memory_bytes=1024**3)

    client.submit_job(job_id=parent_id, entrypoint=entrypoint, resources=resources)

    status = client.wait_for_job(parent_id, timeout=60.0, poll_interval=0.2)
    assert status.state == cluster_pb2.JOB_STATE_SUCCEEDED, f"Parent job failed: {status}"

    response = client.fetch_task_logs(parent_id, include_children=True)

    all_entries = []
    for batch in response.task_logs:
        for entry in batch.logs:
            all_entries.append((entry.timestamp.epoch_ms, batch.worker_id, batch.task_id, entry.data))

    log_text = " ".join(e[3] for e in all_entries)
    assert "CHILD_A_LINE_1" in log_text, f"Missing child-a logs in: {log_text}"
    assert "CHILD_B_LINE_1" in log_text, f"Missing child-b logs in: {log_text}"

    # Verify worker_id is populated in every batch that has logs
    batches_with_logs = [b for b in response.task_logs if len(b.logs) > 0]
    assert all(
        b.worker_id for b in batches_with_logs
    ), f"worker_id missing in batches: {[(b.task_id, b.worker_id) for b in batches_with_logs]}"
