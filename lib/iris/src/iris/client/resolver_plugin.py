# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registers ``iris://<cluster>?endpoint=<name>`` with ``rigging.resolver``.

The handler resolves a cluster name by loading its YAML config and asking the
platform's ``ControllerProvider`` where the controller is. This makes the
scheme work uniformly across GCP-VM clusters (``GcpControllerProvider``
returns a labeled VM's internal IP), CoreWeave/K8s clusters
(``K8sControllerProvider`` returns a Kubernetes Service DNS name), and
Manual/Local clusters (returns the static address from config).

Off-cluster callers (a laptop without direct access to the controller's
internal address) wrap their use of the resolver in
``bundle.controller.tunnel(addr)`` — the existing pattern at
``iris.cluster.config:1095``. The architecture doc's ``maybe_proxy()`` is
the planned ergonomic wrapper for that case.
"""

from rigging.resolver import ServiceURL, register_scheme

from iris.cluster.config import load_cluster_config
from iris.rpc.controller_connect import ControllerServiceClientSync
from iris.rpc.controller_pb2 import Controller as _Controller


def _resolve_iris(url: ServiceURL) -> tuple[str, int]:
    if url.port is not None:
        raise ValueError(f"iris:// URLs cannot have a port (use ?endpoint=<name>): {url!r}")
    cluster = url.host
    name = url.query.get("endpoint")
    if not name:
        raise ValueError(f"iris:// URL requires ?endpoint=<name>: {url!r}")

    cluster_config = load_cluster_config(cluster)
    bundle = cluster_config.provider_bundle()
    controller_addr = bundle.controller.discover_controller(cluster_config.proto.controller)

    with ControllerServiceClientSync(address=f"http://{controller_addr}") as client:
        response = client.list_endpoints(_Controller.ListEndpointsRequest(prefix=name, exact=True))
    if not response.endpoints:
        raise KeyError(f"iris endpoint not found: {name!r} on cluster {cluster!r}")
    host, port = response.endpoints[0].address.rsplit(":", 1)
    return host, int(port)


register_scheme("iris", _resolve_iris)
