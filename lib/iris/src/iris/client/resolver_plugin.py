# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registers ``iris://<cluster>?endpoint=<name>`` with ``rigging.resolver``."""

from rigging.resolver import ServiceURL, gcp_vm_address, register_scheme

from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.controller_pb2 import Controller as _Controller


def _resolve_iris(url: ServiceURL) -> tuple[str, int]:
    if url.port is not None:
        raise ValueError(f"iris:// URLs cannot have a port (use ?endpoint=<name>): {url!r}")
    cluster = url.host
    name = url.query.get("endpoint")
    if not name:
        raise ValueError(f"iris:// URL requires ?endpoint=<name>: {url!r}")
    controller_host, controller_port = gcp_vm_address(f"iris-controller-{cluster}")
    with ControllerServiceClientSync(address=f"http://{controller_host}:{controller_port}") as client:
        response = client.list_endpoints(_Controller.ListEndpointsRequest(prefix=name, exact=True))
    if not response.endpoints:
        raise KeyError(f"iris endpoint not found: {name!r} on cluster {cluster!r}")
    host, port = response.endpoints[0].address.rsplit(":", 1)
    return host, int(port)


register_scheme("iris", _resolve_iris)
