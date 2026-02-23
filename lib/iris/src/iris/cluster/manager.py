# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Cluster lifecycle management.

Provides free functions for cluster lifecycle operations:
- connect_cluster(): context manager that starts controller, opens tunnel, yields address
- stop_all(): stops controller and all worker slices
"""

from __future__ import annotations

import logging
from collections.abc import Iterator
from contextlib import contextmanager

from iris.cluster.config import IrisConfig, validate_config
from iris.rpc import config_pb2

logger = logging.getLogger(__name__)


@contextmanager
def connect_cluster(config: config_pb2.IrisClusterConfig) -> Iterator[str]:
    """Start controller, open tunnel, yield address, stop on exit.

    Delegates controller lifecycle to the platform (GCP, CoreWeave, Local, etc.).
    """
    validate_config(config)
    iris_config = IrisConfig(config)
    platform = iris_config.platform()

    address = platform.start_controller(config)
    try:
        with platform.tunnel(address) as tunnel_url:
            yield tunnel_url
    finally:
        platform.stop_controller(config)
        platform.shutdown()


def stop_all(
    config: config_pb2.IrisClusterConfig,
    dry_run: bool = False,
    label_prefix: str | None = None,
) -> list[str]:
    """Stop controller and all worker slices.

    Delegates to the platform's stop_all() for platform-specific teardown,
    then calls platform.shutdown() to release platform resources.
    """
    iris_config = IrisConfig(config)
    platform = iris_config.platform()
    try:
        return platform.stop_all(config, dry_run=dry_run, label_prefix=label_prefix)
    finally:
        platform.shutdown()
