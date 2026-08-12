# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import hashlib
from pathlib import Path

import iris.cluster.client.remote_client as remote_client_module
import pytest
from iris.cluster.client.remote_client import ExactWorkspaceBundle, RemoteClusterClient
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.types import Entrypoint, JobName
from iris.rpc import controller_pb2, job_pb2


def _recording_remote_client(
    monkeypatch, **kwargs
) -> tuple[RemoteClusterClient, list[controller_pb2.Controller.LaunchJobRequest]]:
    requests: list[controller_pb2.Controller.LaunchJobRequest] = []

    class RecordingControllerClient:
        def launch_job(self, request, *, timeout_ms):
            del timeout_ms
            copy = controller_pb2.Controller.LaunchJobRequest()
            copy.CopyFrom(request)
            requests.append(copy)
            return controller_pb2.Controller.LaunchJobResponse(job_id=request.name)

        def close(self):
            pass

    class RecordingEndpointClient:
        def register_endpoint(self, request, *, timeout_ms=None):
            raise AssertionError(f"unexpected endpoint registration: {request}, {timeout_ms}")

        def close(self):
            pass

    monkeypatch.setattr(
        remote_client_module,
        "ControllerServiceClientSync",
        lambda **_kwargs: RecordingControllerClient(),
    )
    monkeypatch.setattr(
        remote_client_module,
        "EndpointServiceClientSync",
        lambda **_kwargs: RecordingEndpointClient(),
    )
    return RemoteClusterClient("http://controller.example:8080/", **kwargs), requests


def _submit_one(client: RemoteClusterClient) -> None:
    client.submit_job(
        JobName.root("test-user", "bundle-source"),
        Entrypoint.from_command("true"),
        job_pb2.ResourceSpecProto(cpu_millicores=1000, memory_bytes=1024),
    )


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
    blob = b"reviewed bytes"
    bundle_id = hashlib.sha256(blob).hexdigest()
    client, requests = _recording_remote_client(
        monkeypatch,
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


def test_nonempty_bundle_id_still_wins_over_workspace(monkeypatch, tmp_path):
    client, requests = _recording_remote_client(monkeypatch, bundle_id="parent-bundle", workspace=tmp_path)
    try:
        _submit_one(client)
    finally:
        client.shutdown()

    assert requests[0].bundle_id == "parent-bundle"
    assert requests[0].bundle_blob == b""


def test_empty_bundle_id_still_falls_back_to_workspace_zip(monkeypatch, tmp_path):
    zipped_paths: list[Path] = []

    def create_workspace_zip(workspace: Path, *, extra_includes=()) -> bytes:
        del extra_includes
        zipped_paths.append(workspace)
        return b"workspace zip"

    monkeypatch.setattr(remote_client_module, "create_workspace_zip", create_workspace_zip)
    client, requests = _recording_remote_client(monkeypatch, bundle_id="", workspace=tmp_path)
    try:
        _submit_one(client)
    finally:
        client.shutdown()

    assert zipped_paths == [tmp_path.resolve()]
    assert requests[0].bundle_id == ""
    assert requests[0].bundle_blob == b"workspace zip"


@pytest.mark.parametrize(
    "legacy_sources",
    [
        {"bundle_id": "parent-bundle"},
        {"bundle_id": ""},
        {"workspace": Path("workspace")},
        {"bundle_id": "parent-bundle", "workspace": Path("workspace")},
    ],
)
def test_exact_workspace_bundle_rejects_every_legacy_source(legacy_sources):
    blob = b"reviewed bytes"
    exact_bundle = ExactWorkspaceBundle(blob=blob, bundle_id=hashlib.sha256(blob).hexdigest())

    with pytest.raises(ValueError, match="exact_bundle is mutually exclusive"):
        RemoteClusterClient("http://controller.example:8080/", exact_bundle=exact_bundle, **legacy_sources)
