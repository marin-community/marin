# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tests for RemoteClusterClient parameter propagation."""

from unittest.mock import MagicMock

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.types import Entrypoint, JobName
from iris.rpc import cluster_pb2


def _make_client() -> tuple[RemoteClusterClient, MagicMock]:
    """Create a RemoteClusterClient with a mocked RPC stub."""
    client = RemoteClusterClient.__new__(RemoteClusterClient)
    mock_rpc = MagicMock()
    client._client = mock_rpc
    client._bundle_id = "test-bundle"
    client._bundle_blob = None
    return client, mock_rpc


def _submit_with_policies(
    preemption_policy: cluster_pb2.JobPreemptionPolicy = cluster_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED,
    existing_job_policy: cluster_pb2.ExistingJobPolicy = cluster_pb2.EXISTING_JOB_POLICY_UNSPECIFIED,
) -> cluster_pb2.Controller.LaunchJobRequest:
    """Submit a job with given policies and return the captured LaunchJobRequest."""
    client, mock_rpc = _make_client()
    job_id = JobName.root("test-user", "test-job")
    entrypoint = Entrypoint.from_command("echo", "hello")
    resources = cluster_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024**3)

    client.submit_job(
        job_id=job_id,
        entrypoint=entrypoint,
        resources=resources,
        preemption_policy=preemption_policy,
        existing_job_policy=existing_job_policy,
    )

    request = mock_rpc.launch_job.call_args[0][0]
    return request


def test_submit_job_default_policies():
    request = _submit_with_policies()
    assert request.preemption_policy == cluster_pb2.JOB_PREEMPTION_POLICY_UNSPECIFIED
    assert request.existing_job_policy == cluster_pb2.EXISTING_JOB_POLICY_UNSPECIFIED


def test_submit_job_preserve_children_policy():
    request = _submit_with_policies(
        preemption_policy=cluster_pb2.JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN,
    )
    assert request.preemption_policy == cluster_pb2.JOB_PREEMPTION_POLICY_PRESERVE_CHILDREN


def test_submit_job_recreate_existing_policy():
    request = _submit_with_policies(
        existing_job_policy=cluster_pb2.EXISTING_JOB_POLICY_RECREATE,
    )
    assert request.existing_job_policy == cluster_pb2.EXISTING_JOB_POLICY_RECREATE


def test_submit_job_both_policies():
    request = _submit_with_policies(
        preemption_policy=cluster_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN,
        existing_job_policy=cluster_pb2.EXISTING_JOB_POLICY_KEEP,
    )
    assert request.preemption_policy == cluster_pb2.JOB_PREEMPTION_POLICY_TERMINATE_CHILDREN
    assert request.existing_job_policy == cluster_pb2.EXISTING_JOB_POLICY_KEEP
