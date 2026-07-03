# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

from iris.cluster.client.remote_client import RemoteClusterClient
from iris.cluster.endpoints import LOG_SERVER_ENDPOINT_NAME


def test_external_endpoint_resolution_uses_controller_proxy_path():
    client = RemoteClusterClient("http://controller.example:8080/")
    try:
        address = client.resolve_endpoint(LOG_SERVER_ENDPOINT_NAME)
    finally:
        client.shutdown()

    assert address == "http://controller.example:8080/proxy/system.log-server"
