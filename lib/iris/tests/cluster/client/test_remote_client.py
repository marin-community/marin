# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.types import JobName
from iris.rpc import controller_pb2
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
