# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""ActorServer mounts telltale alongside its RPC service, on the same port."""

import httpx
import pytest
from iris.actor import ActorServer
from rigging import telltale


@pytest.fixture
def server():
    server = ActorServer(host="127.0.0.1")
    server.serve_background()
    yield server
    server.stop()


def _base_url(server: ActorServer) -> str:
    return f"http://127.0.0.1:{server._actual_port}"


def test_metrics_served_on_the_actor_port(server):
    telltale.counter("actor_telltale_probe", "d").inc()

    response = httpx.get(f"{_base_url(server)}/metrics")

    assert response.status_code == 200
    assert "actor_telltale_probe_total 1.0" in response.text


def test_health_and_index_served_on_the_actor_port(server):
    assert httpx.get(f"{_base_url(server)}/health").json() == {"status": "healthy"}
    assert httpx.get(f"{_base_url(server)}/").status_code == 200
