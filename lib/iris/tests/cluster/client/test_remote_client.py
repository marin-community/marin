# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.types import Entrypoint, JobName
from iris.rpc import controller_pb2, job_pb2
from rigging.timing import Deadline


def test_external_endpoint_resolution_uses_controller_proxy_path():
    client = RemoteClusterClient("http://controller.example:8080/")
    try:
        address = client.resolve_endpoint(LOG_SERVER_ENDPOINT_NAME)
    finally:
        client.shutdown()

    assert address == "http://controller.example:8080/proxy/system.log-server"


def test_task_status_rpc_uses_the_wait_deadline():
    class ControllerStub:
        timeout_ms = 0

        def get_task_status(self, _request, *, timeout_ms=None):
            self.timeout_ms = timeout_ms or 0
            return controller_pb2.Controller.GetTaskStatusResponse()

    stub = ControllerStub()
    client = object.__new__(RemoteClusterClient)
    client._client = stub
    client._timeout_ms = 30_000

    client.get_task_status(JobName.from_wire("/alice/train/0"), deadline=Deadline.from_seconds(1))

    assert 0 < stub.timeout_ms <= 1_000


def test_sandbox_submission_carries_no_bundle_or_submitter_env(tmp_path, monkeypatch):
    class ControllerStub:
        request: controller_pb2.Controller.LaunchJobRequest | None = None

        def launch_job(self, request, *, timeout_ms=None):
            self.request = request
            return controller_pb2.Controller.LaunchJobResponse(job_id=request.name)

        def close(self):
            pass

    monkeypatch.setenv("HF_TOKEN", "submitter-token")
    (tmp_path / "secrets.env").write_text("KEY=value\n")
    stub = ControllerStub()
    client = RemoteClusterClient("http://controller.example:8080/", bundle_id="parent-bundle", workspace=tmp_path)
    client._client = stub
    try:
        client.submit_job(
            job_id=JobName.root("alice", "sandbox"),
            entrypoint=Entrypoint.from_command("sleep", "infinity"),
            resources=job_pb2.ResourceSpecProto(cpu_millicores=1000),
            container_profile=job_pb2.CONTAINER_PROFILE_SANDBOX,
        )
    finally:
        client.shutdown()

    assert stub.request is not None
    assert stub.request.bundle_id == ""
    assert stub.request.bundle_blob == b""
    assert dict(stub.request.environment.env_vars) == {}
    assert list(stub.request.entrypoint.setup_commands) == []
