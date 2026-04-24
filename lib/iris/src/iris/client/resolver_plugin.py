# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris resolver plugin: registers ``iris://`` with :mod:`rigging.resolver`.

Importing this module installs an ``iris://`` handler that goes through the
existing :class:`ControllerService.ListEndpoints` RPC. Off-cluster code that
does not import ``iris.client`` therefore cannot resolve ``iris://`` URLs —
which is the right failure mode (the handler needs the iris RPC stack).

URL shape::

    iris://<cluster>?endpoint=<name>

The cluster name maps to a controller VM ``iris-controller-<cluster>`` (looked
up via ``rigging.resolver.providers.vm_address``). The ``endpoint`` query
parameter is the system-service name registered on that controller (e.g.
``/system/log-server``); the handler returns its registered ``host:port``.
"""

from rigging.resolver import ServiceURL, register_scheme
from rigging.resolver.providers import vm_address

from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.controller_pb2 import Controller as _Controller


def _resolve_iris(url: ServiceURL) -> tuple[str, int]:
    cluster = url.host
    name = url.query["endpoint"]
    controller_host, controller_port = vm_address(f"iris-controller-{cluster}", provider="gcp")
    client = ControllerServiceClientSync(address=f"http://{controller_host}:{controller_port}")
    try:
        response = client.list_endpoints(_Controller.ListEndpointsRequest(prefix=name, exact=True))
    finally:
        client.close()
    if not response.endpoints:
        raise KeyError(f"iris endpoint not found: {name!r} on cluster {cluster!r}")
    address = response.endpoints[0].address
    host, port = address.rsplit(":", 1)
    return host, int(port)


register_scheme("iris", _resolve_iris)
