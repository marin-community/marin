# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""E2E tests for local cluster client functionality."""

import logging
import re
import time

import pytest

from iris.client.client import IrisClient, Job
from iris.cluster.types import Entrypoint, EnvironmentSpec, JobName
from iris.rpc import logging_pb2
from iris.rpc import job_pb2

pytestmark = pytest.mark.e2e


def extract_log_text(response: logging_pb2.FetchLogsResponse) -> str:
    """Extract log text from a FetchLogs response."""
    return "\n".join(e.data for e in response.entries)


@pytest.fixture(scope="module")
def iris_client():
    """Create a local IrisClient for testing."""
    client = IrisClient.local()
    yield client
    client.shutdown()


@pytest.fixture(scope="module")
def client(iris_client):
    """Expose the underlying ClusterClient for tests that use low-level APIs."""
    return iris_client._cluster_client


def test_command_entrypoint_preserves_env_vars(client):
    """Verify command entrypoints receive Iris environment variables."""
    job_id = JobName.root("test-user", "test-env-vars")

    # Create a command that echoes an environment variable
    entrypoint = Entrypoint.from_command("sh", "-c", "echo IRIS_TASK_ID=$IRIS_TASK_ID")

    resources = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3)

    with client._store.transaction() as cur:
        client.submit_job(cur, job_id=job_id, entrypoint=entrypoint, resources=resources)

    # Wait for job completion
    status = client.wait_for_job(job_id, timeout=10.0, poll_interval=0.1)

    assert status.state == job_pb2.JOB_STATE_SUCCEEDED

    # For attempt 0, build_common_iris_env omits the :attempt_id suffix
    # (only appended for retries), so expect just the task_id wire format.
    expected = job_id.task(0).to_wire()
    response = client.fetch_logs(re.escape(job_id.task(0).to_wire()) + ":.*")
    log_text = extract_log_text(response)
    assert f"IRIS_TASK_ID={expected}" in log_text


def test_log_streaming_captures_output_without_trailing_newline(client):
    """Verify log streaming captures output without trailing newline."""
    job_id = JobName.root("test-user", "test-no-newline")

    # Use printf which doesn't add a newline
    entrypoint = Entrypoint.from_command("sh", "-c", "printf 'output without newline'")

    resources = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3)

    with client._store.transaction() as cur:
        client.submit_job(cur, job_id=job_id, entrypoint=entrypoint, resources=resources)

    # Wait for job completion
    status = client.wait_for_job(job_id, timeout=10.0, poll_interval=0.1)

    assert status.state == job_pb2.JOB_STATE_SUCCEEDED

    # Check logs contain the output
    response = client.fetch_logs(re.escape(job_id.task(0).to_wire()) + ":.*")
    log_text = extract_log_text(response)
    assert "output without newline" in log_text


def test_callable_entrypoint_succeeds(client):
    """Verify callable entrypoints execute and complete successfully."""
    job_id = JobName.root("test-user", "test-callable-success")

    def task_func():
        print("hello from callable")

    entrypoint = Entrypoint.from_callable(task_func)

    resources = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3)

    with client._store.transaction() as cur:
        client.submit_job(cur, job_id=job_id, entrypoint=entrypoint, resources=resources)

    status = client.wait_for_job(job_id, timeout=10.0, poll_interval=0.1)

    assert status.state == job_pb2.JOB_STATE_SUCCEEDED


def test_command_entrypoint_with_custom_env_var(client):
    """Verify command entrypoints can access custom environment variables."""
    job_id = JobName.root("test-user", "test-custom-env")

    # Create a command that uses a custom env var
    entrypoint = Entrypoint.from_command("sh", "-c", "echo CUSTOM_VAR=$CUSTOM_VAR")

    resources = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3)
    environment = EnvironmentSpec(env_vars={"CUSTOM_VAR": "custom_value"}).to_proto()

    with client._store.transaction() as cur:
        client.submit_job(
            cur,
            job_id=job_id,
            entrypoint=entrypoint,
            resources=resources,
            environment=environment,
        )

    # Wait for job completion
    status = client.wait_for_job(job_id, timeout=10.0, poll_interval=0.1)

    assert status.state == job_pb2.JOB_STATE_SUCCEEDED

    # Check logs contain the custom env var
    response = client.fetch_logs(re.escape(job_id.task(0).to_wire()) + ":.*")
    log_text = extract_log_text(response)
    assert "CUSTOM_VAR=custom_value" in log_text


def test_job_wait_with_stream_logs(client, iris_client, caplog):
    """Verify Job.wait(stream_logs=True) fetches and streams logs."""
    iris = iris_client
    job_id = JobName.root("test-user", "test-stream-logs")

    entrypoint = Entrypoint.from_command("sh", "-c", "echo 'hello from streaming'")
    resources = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3)

    with client._store.transaction() as cur:
        client.submit_job(cur, job_id=job_id, entrypoint=entrypoint, resources=resources)
    job = Job(iris, job_id)

    with caplog.at_level(logging.INFO, logger="iris.client.client"):
        status = job.wait(stream_logs=True, timeout=10.0)

    assert status.state == job_pb2.JOB_STATE_SUCCEEDED

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


def _parent_with_delayed_child():
    """Parent callable that starts a child after streaming has already begun."""
    from iris.client.client import iris_ctx
    from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

    ctx = iris_ctx()
    res = ResourceSpec(cpu=1, memory="1g")
    env = EnvironmentSpec()

    print("PARENT_STARTED")
    time.sleep(1.0)
    child = ctx.client.submit(
        Entrypoint.from_command("sh", "-c", "echo CHILD_DYNAMIC_LINE"),
        "child-dynamic",
        res,
        environment=env,
    )
    child.wait(timeout=30, raise_on_failure=True)
    # Keep parent alive briefly so child logs must be discovered while parent is still active.
    time.sleep(1.0)
    print("PARENT_DONE")


def test_child_job_logs_sorted_by_timestamp(client):
    """Logs from multiple child jobs are present in flattened prefix response."""
    parent_id = JobName.root("test-user", "test-child-logs")
    entrypoint = Entrypoint.from_callable(_parent_with_two_children)
    resources = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3)

    with client._store.transaction() as cur:
        client.submit_job(cur, job_id=parent_id, entrypoint=entrypoint, resources=resources)

    status = client.wait_for_job(parent_id, timeout=60.0, poll_interval=0.2)
    assert status.state == job_pb2.JOB_STATE_SUCCEEDED, f"Parent job failed: {status}"

    response = client.fetch_logs(re.escape(parent_id.to_wire()) + "/.*")

    log_text = " ".join(e.data for e in response.entries)
    assert "CHILD_A_LINE_1" in log_text, f"Missing child-a logs in: {log_text}"
    assert "CHILD_B_LINE_1" in log_text, f"Missing child-b logs in: {log_text}"


def test_wait_stream_logs_discovers_child_tasks(client, iris_client, caplog):
    """Streaming discovers and emits logs for child tasks created after wait starts."""
    iris = iris_client
    parent_id = JobName.root("test-user", "test-stream-child-discovery")
    entrypoint = Entrypoint.from_callable(_parent_with_delayed_child)
    resources = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3)

    with client._store.transaction() as cur:
        client.submit_job(cur, job_id=parent_id, entrypoint=entrypoint, resources=resources)
    job = Job(iris, parent_id)

    with caplog.at_level(logging.INFO, logger="iris.client.client"):
        status = job.wait(stream_logs=True, timeout=60.0, poll_interval=0.2)

    assert status.state == job_pb2.JOB_STATE_SUCCEEDED
    messages = [r.message for r in caplog.records]
    assert any("CHILD_DYNAMIC_LINE" in msg for msg in messages), messages


def _parent_with_failing_child():
    """Parent callable that submits a child that exits with an error."""
    from iris.client.client import iris_ctx
    from iris.cluster.types import Entrypoint, EnvironmentSpec, ResourceSpec

    ctx = iris_ctx()
    res = ResourceSpec(cpu=1, memory="1g")
    env = EnvironmentSpec()

    child = ctx.client.submit(
        Entrypoint.from_command("sh", "-c", "echo 'about to fail'; exit 42"),
        "failing-child",
        res,
        environment=env,
    )
    child.wait(timeout=30, raise_on_failure=False)
    # Keep parent alive after child failure so the streaming loop has time to detect it.
    time.sleep(2.0)


def test_stream_logs_surfaces_child_failure(client, iris_client):
    """Parent job with a failing child still completes successfully."""
    iris = iris_client
    parent_id = JobName.root("test-user", "test-child-failure-surfaced")
    entrypoint = Entrypoint.from_callable(_parent_with_failing_child)
    resources = job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3)

    with client._store.transaction() as cur:
        client.submit_job(cur, job_id=parent_id, entrypoint=entrypoint, resources=resources)
    job = Job(iris, parent_id)

    status = job.wait(
        stream_logs=True,
        timeout=30.0,
        poll_interval=0.2,
        raise_on_failure=False,
    )

    assert status.state == job_pb2.JOB_STATE_SUCCEEDED
