# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Registers ``iris://<cluster>?endpoint=<name>`` with ``rigging.resolver``.

The handler resolves a cluster name by loading its YAML config and asking the
platform's ``ControllerProvider`` where the controller is. This makes the
scheme work uniformly across GCP-VM clusters (``GcpControllerProvider``
returns a labeled VM's internal IP), CoreWeave/K8s clusters
(``K8sControllerProvider`` returns a Kubernetes Service DNS name), and
Manual/Local clusters (returns the static address from config).

Off-cluster callers wrap their use of :func:`rigging.resolver.resolve`
in :func:`rigging.proxy.proxy_stack`. When that scope is active and the
controller or returned endpoint is not directly reachable, the handler
opens an ssh tunnel via ``bundle.controller.tunnel_to(...)`` and caches
it on the active stack for the duration of the block.
"""

from urllib.parse import urlsplit

from rigging.proxy import ProxyStack, active_stack, is_reachable
from rigging.resolver import ServiceURL, register_scheme

from iris.cluster.config import load_cluster_config
from iris.cluster.providers.factory import ProviderBundle
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
    stack = active_stack()

    controller_addr = _split_hostport(bundle.controller.discover_controller(cluster_config.proto.controller))
    controller_addr = _maybe_tunnel(bundle, controller_addr, stack)

    grpc_target = f"http://{controller_addr[0]}:{controller_addr[1]}"
    with ControllerServiceClientSync(address=grpc_target) as client:
        response = client.list_endpoints(_Controller.ListEndpointsRequest(prefix=name, exact=True))
    if not response.endpoints:
        raise KeyError(f"iris endpoint not found: {name!r} on cluster {cluster!r}")

    log_addr = _split_hostport(response.endpoints[0].address)
    return _maybe_tunnel(bundle, log_addr, stack)


def _split_hostport(addr: str) -> tuple[str, int]:
    """Split ``host:port`` or ``scheme://host:port`` into ``(host, port)``."""
    if "://" in addr:
        parts = urlsplit(addr)
        if not parts.hostname or parts.port is None:
            raise ValueError(f"address {addr!r} missing host or port")
        return parts.hostname, parts.port
    host, port = addr.rsplit(":", 1)
    return host, int(port)


def _maybe_tunnel(
    bundle: ProviderBundle,
    addr: tuple[str, int],
    stack: ProxyStack | None,
) -> tuple[str, int]:
    if stack is None or is_reachable(*addr):
        return addr
    return stack.proxy(addr, lambda: bundle.controller.tunnel_to(*addr))


register_scheme("iris", _resolve_iris)
