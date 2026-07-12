# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Iris cluster lifecycle orchestration.

Provides high-level helpers that construct concrete provider implementations and
drive cluster startup/shutdown.
"""

from collections.abc import Iterator
from contextlib import contextmanager

from iris.cluster.composer import provider_bundle
from iris.cluster.config import IrisClusterConfig, validate_config
from iris.cluster.local_cluster import LocalCluster


@contextmanager
def connect_cluster(config: IrisClusterConfig) -> Iterator[str]:
    """Start controller, open tunnel, yield address, stop on exit.

    Local mode uses LocalCluster directly (in-process controller + workers).
    Remote modes delegate controller lifecycle to the platform (GCP, CoreWeave, etc.).
    """
    validate_config(config)
    is_local = config.controller.controller_kind() == "local"

    if is_local:
        cluster = LocalCluster(config)
        address = cluster.start()
        try:
            yield address
        finally:
            cluster.close()
    else:
        bundle = provider_bundle(config)
        address = bundle.controller.start_controller(config)
        try:
            with bundle.controller.tunnel(address) as tunnel_url:
                yield tunnel_url
        finally:
            bundle.controller.stop_controller(config)
            bundle.controller.shutdown()
