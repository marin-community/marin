# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import traceback

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.client.endpoint_client import EndpointClient
from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.types import Entrypoint, JobName
from iris.preemption import TERMINATION_GRACE_ENV
from iris.rpc import controller_pb2, job_pb2
from iris.rpc.errors import connect_error_sanitized
from rigging.timing import Deadline, Duration


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


def test_submission_serializes_termination_grace_as_duration():
    class ControllerStub:
        request: controller_pb2.Controller.LaunchJobRequest | None = None

        def launch_job(self, request, *, timeout_ms=None):
            self.request = request
            return controller_pb2.Controller.LaunchJobResponse(job_id=request.name)

        def close(self):
            pass

    stub = ControllerStub()
    client = RemoteClusterClient("http://controller.example:8080/", bundle_id="test-bundle")
    client._client = stub
    try:
        client.submit_job(
            job_id=JobName.root("alice", "training"),
            entrypoint=Entrypoint.from_command("python", "train.py"),
            resources=job_pb2.ResourceSpecProto(cpu_millicores=1000),
            termination_grace=Duration.from_seconds(60),
        )
    finally:
        client.shutdown()

    assert stub.request is not None
    assert stub.request.termination_grace.milliseconds == 60_000
    assert TERMINATION_GRACE_ENV not in stub.request.environment.env_vars


@pytest.mark.parametrize("rpc_code", [None, Code.NOT_FOUND, Code.PERMISSION_DENIED])
def test_endpoint_resolution_reports_absence_without_hiding_rpc_failures(rpc_code):
    class EndpointStub:
        def register_endpoint(self, request):
            raise NotImplementedError

        def unregister_endpoint(self, request, *, timeout_ms=None):
            raise NotImplementedError

        def list_endpoints(self, request, *, timeout_ms=None):
            if rpc_code is not None:
                raise ConnectError(rpc_code, "registry response")
            return controller_pb2.Controller.ListEndpointsResponse()

        def close(self):
            pass

    client = RemoteClusterClient("http://controller.example:8080/", use_controller_proxy=False)
    client._endpoint_client = EndpointClient(EndpointStub())
    try:
        with pytest.raises(ConnectError) as raised:
            client.resolve_endpoint("/alice/train/inference")
        assert raised.value.code == (rpc_code or Code.NOT_FOUND)
        if rpc_code is not None:
            assert raised.value.message == "registry response"
    finally:
        client.shutdown()


def test_submission_failure_keeps_rpc_details_and_prints_code_and_reason():
    error = connect_error_sanitized(Code.PERMISSION_DENIED, "task cannot submit through the hub")

    class ControllerStub:
        def launch_job(self, request, *, timeout_ms=None):
            raise error

        def close(self):
            pass

    client = RemoteClusterClient("http://controller.example:8080/")
    client._client = ControllerStub()
    try:
        with pytest.raises(ConnectError) as raised:
            client.submit_job(
                job_id=JobName.from_wire("/alice/train/eval"),
                entrypoint=Entrypoint.from_command("true"),
                resources=job_pb2.ResourceSpecProto(cpu_millicores=1000),
            )
        assert raised.value is error
        rendered = "".join(traceback.format_exception(raised.value))
        assert "launch_job(/alice/train/eval)" in rendered
        assert error.code.name in rendered
        assert error.message in rendered
    finally:
        client.shutdown()
