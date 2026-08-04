# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME
from iris.cluster.types import JobName


def test_external_endpoint_resolution_uses_controller_proxy_path():
    client = RemoteClusterClient("http://controller.example:8080/")
    try:
        address = client.resolve_endpoint(LOG_SERVER_ENDPOINT_NAME)
    finally:
        client.shutdown()

    assert address == "http://controller.example:8080/proxy/system.log-server"


def test_readiness_queries_cap_each_rpc_to_retry_deadline():
    rpc_calls: list[tuple[str, int | None]] = []

    class ControllerStub:
        def get_job_state(self, _request, *, timeout_ms=None):
            rpc_calls.append(("get_job_state", timeout_ms))
            return SimpleNamespace(states={})

        def list_tasks(self, _request, *, timeout_ms=None):
            rpc_calls.append(("list_tasks", timeout_ms))
            return SimpleNamespace(tasks=[])

    client = object.__new__(RemoteClusterClient)
    client.__dict__["_client"] = ControllerStub()
    client.__dict__["_timeout_ms"] = 30_000
    job_id = JobName.from_string("/tester/inference")

    client.get_job_states([job_id], retry_max_elapsed=10)
    client.list_tasks(job_id, retry_max_elapsed=10)

    assert rpc_calls == [("get_job_state", 10_000), ("list_tasks", 10_000)]
