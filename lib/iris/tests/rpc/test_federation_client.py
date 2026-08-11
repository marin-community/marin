# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import pytest
from connectrpc.code import Code
from connectrpc.errors import ConnectError
from iris.cluster.federation.protocol import PeerCallError, PeerErrorCode
from iris.resources.endpoint import ExecRequest
from iris.resources.identity import AttemptIdentity, ResourceKey, ResourceKind
from iris.resources.names import JobName
from iris.rpc import controller_pb2, resource_pb2
from iris.rpc import federation_client as federation_transport
from iris.rpc.worker_client import EXEC_IN_CONTAINER_MAX_TIMEOUT
from rigging.timing import Duration


class _ControllerStub:
    def __init__(self, *, error: ConnectError | None = None):
        self.error = error

    def list_backends(self, request):
        backend = controller_pb2.Controller.BackendSummary(
            backend_id="gpu",
            kind="kubernetes",
            worker_count=3,
            advertised_attributes={"device-type": controller_pb2.StringList(values=["gpu"])},
        )
        backend.availability.version = 2
        backend.availability.observation_epoch_ms = 123
        backend.availability.amounts["h100"] = 8
        backend.availability.held_by_band.add(band=2, amounts={"h100": 4})
        return controller_pb2.Controller.ListBackendsResponse(backends=[backend])

    def terminate_job(self, request):
        if self.error is not None:
            raise self.error

    def close(self):
        pass


class _ResourceStub:
    def __init__(self):
        self.exec_timeout_ms = 0
        self.exec_request = None

    def create_resource(self, request, timeout_ms):
        self.exec_timeout_ms = timeout_ms
        self.exec_request = resource_pb2.ExecAttemptRequest()
        assert request.body.Unpack(self.exec_request)
        result = resource_pb2.Operation()
        result.result.Pack(resource_pb2.ExecAttemptResponse())
        return result

    def close(self):
        pass


def _connection(monkeypatch, controller: _ControllerStub | None = None):
    controller = controller or _ControllerStub()
    resources = _ResourceStub()
    monkeypatch.setattr(federation_transport, "ControllerServiceClientSync", lambda **kwargs: controller)
    monkeypatch.setattr(federation_transport, "ResourceServiceClientSync", lambda **kwargs: resources)
    return federation_transport.ConnectPeerConnection("http://peer:10000", []), resources


def test_connect_peer_decodes_backend_heartbeat_to_native_records(monkeypatch) -> None:
    connection, _ = _connection(monkeypatch)

    (backend,) = connection.list_backends()

    assert (backend.backend_id, backend.kind, backend.worker_count) == ("gpu", "kubernetes", 3)
    assert backend.advertised_attributes == {"device-type": ("gpu",)}
    assert backend.availability is not None
    assert (
        backend.availability.version,
        backend.availability.observation_epoch_ms,
        backend.availability.amounts,
        backend.availability.held_by_band,
    ) == (2, 123, {"h100": 8}, {2: {"h100": 4}})


def test_connect_peer_translates_transport_errors_to_native_errors(monkeypatch) -> None:
    connection, _ = _connection(monkeypatch, _ControllerStub(error=ConnectError(Code.NOT_FOUND, "gone")))

    with pytest.raises(PeerCallError) as error:
        connection.terminate_job(JobName.from_wire("/u/j"))

    assert (error.value.code, error.value.message) == (PeerErrorCode.NOT_FOUND, "gone")


def test_exec_proxy_deadline_outlasts_the_peer(monkeypatch) -> None:
    connection, resources = _connection(monkeypatch)
    attempt = AttemptIdentity(ResourceKey("parent", ResourceKind.TASK, "/u/j/0"), 0, "attempt-uid")

    connection.exec_in_container(ExecRequest(attempt, (), Duration.from_seconds(-1)))
    assert resources.exec_timeout_ms >= EXEC_IN_CONTAINER_MAX_TIMEOUT.to_ms()
    assert resources.exec_request.attempt.attempt_uid == "attempt-uid"

    connection.exec_in_container(ExecRequest(attempt, (), Duration.from_seconds(30)))
    assert resources.exec_timeout_ms > 30 * 1000
