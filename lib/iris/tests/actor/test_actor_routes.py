# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest
from iris.actor.server import ActorServer


@pytest.fixture
def server():
    server = ActorServer(host="127.0.0.1")
    server.serve_background()
    yield server
    server.stop()


def _base_url(server: ActorServer) -> str:
    return f"http://{server.address}"


def test_actor_port_does_not_expose_legacy_status_or_metrics_routes(server):
    assert httpx.get(f"{_base_url(server)}/health").status_code == 404
    assert httpx.get(f"{_base_url(server)}/").status_code == 404
    assert httpx.get(f"{_base_url(server)}/metrics").status_code == 404
