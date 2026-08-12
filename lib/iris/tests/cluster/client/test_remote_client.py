# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib

import iris.cluster.client.remote_client as remote_client_module
import pytest
from iris.cluster.client.remote_client import ExactWorkspaceBundle, RemoteClusterClient
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.types import Entrypoint, JobName
from iris.rpc import controller_pb2, job_pb2


def test_external_endpoint_resolution_uses_controller_proxy_path():
    client = RemoteClusterClient("http://controller.example:8080/")
    try:
        address = client.resolve_endpoint(LOG_SERVER_ENDPOINT_NAME)
    finally:
        client.shutdown()

    assert address == "http://controller.example:8080/proxy/system.log-server"


def test_exact_workspace_bundle_rejects_declared_content_mismatch():
    with pytest.raises(ValueError, match="content ID does not match"):
        ExactWorkspaceBundle(blob=b"reviewed bytes", bundle_id="0" * 64)


def test_exact_workspace_bundle_is_sent_verbatim_with_declared_id(monkeypatch):
    requests = []

    class RecordingControllerClient:
        def launch_job(self, request, *, timeout_ms):
            del timeout_ms
            copy = controller_pb2.Controller.LaunchJobRequest()
            copy.CopyFrom(request)
            requests.append(copy)
            return controller_pb2.Controller.LaunchJobResponse(job_id=request.name)

        def close(self):
            pass

    monkeypatch.setattr(
        remote_client_module,
        "ControllerServiceClientSync",
        lambda **_kwargs: RecordingControllerClient(),
    )
    blob = b"reviewed bytes"
    bundle_id = hashlib.sha256(blob).hexdigest()
    client = RemoteClusterClient(
        "http://controller.example:8080/",
        exact_bundle=ExactWorkspaceBundle(blob=blob, bundle_id=bundle_id),
    )
    try:
        client.submit_job(
            JobName.root("test-user", "exact-bundle"),
            Entrypoint.from_command("true"),
            job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024),
            bundle_init_image="registry.example/iris-init@sha256:" + "a" * 64,
        )
    finally:
        client.shutdown()

    assert requests[0].exact_bundle_upload.blob == blob
    assert requests[0].exact_bundle_upload.bundle_id == bundle_id
    assert requests[0].bundle_init_image == "registry.example/iris-init@sha256:" + "a" * 64
