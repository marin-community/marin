# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Construct an :class:`IrisClient` backed by an in-process :class:`LocalCluster`.

This module is the single place where the lightweight ``iris.client`` layer
meets the heavy ``iris.cluster`` provider/controller code. Importing it
transitively pulls in the cluster admin tree (autoscaler, providers, etc.),
which is why it lives in its own module — ``import iris.client`` stays cheap
for remote clients.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from iris.client.client import IrisClient, LocalClientConfig
from iris.cluster.client import RemoteClusterClient
from iris.cluster.providers.local.cluster import LocalCluster, make_local_cluster_config


def make_local_client(config: LocalClientConfig | None = None) -> IrisClient:
    """Start an in-process LocalCluster and return an IrisClient connected to it.

    The returned client owns the cluster lifecycle: :meth:`IrisClient.shutdown`
    (or context-manager exit) tears the cluster down.
    """
    cfg = config or LocalClientConfig()
    cluster = LocalCluster(make_local_cluster_config(cfg.max_workers))
    address = cluster.start()
    return IrisClient(
        RemoteClusterClient(controller_address=address, timeout_ms=30000),
        controller=cluster,
    )


@contextmanager
def local_client(config: LocalClientConfig | None = None) -> Iterator[IrisClient]:
    """Context-manager wrapper around :func:`make_local_client`.

    Use directly in tests or scripts; the cluster is closed on exit even if
    the body raises.
    """
    client = make_local_client(config)
    try:
        yield client
    finally:
        client.shutdown()
