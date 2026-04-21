# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Weight transfer management.

This module provides abstractions for communicating weights between training and inference workers.

Currently GCS checkpoint and Arrow Flight methods are supported.
"""

import logging

from haliax.partitioning import ResourceMapping
from jax.sharding import Mesh

from .arrow_flight import ArrowFlightClient, ArrowFlightCoordinator, ArrowFlightServer
from .base import (
    WeightTransferClient,
    WeightTransferClientMetrics,
    WeightTransferConfig,
    WeightTransferMode,
    WeightTransferServer,
    WeightTransferServerMetrics,
    WeightUpdate,
)
from .checkpoint import GCSCheckpointClient, GCSCheckpointServer

logger = logging.getLogger(__name__)


def create_weight_transfer_server(
    config: WeightTransferConfig,
    mesh: Mesh | None = None,
    axis_mapping: ResourceMapping | None = None,
    coordinator_handle=None,
) -> WeightTransferServer:
    """Factory function to create appropriate transfer server for Levanter models.

    Args:
        config: Weight transfer configuration
        mesh: JAX mesh for distributed computation (optional)
        axis_mapping: Levanter axis mapping for sharding (optional)
        coordinator_handle: Pre-created actor handle for the coordinator.
            If provided, the server uses it directly instead of discovering via fray v1.
    """
    if config.mode == WeightTransferMode.ARROW_FLIGHT:
        return ArrowFlightServer(config, mesh, axis_mapping, coordinator_handle=coordinator_handle)

    # Default to GCS checkpoint mode
    return GCSCheckpointServer(
        config,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )


def create_weight_transfer_client(
    config: WeightTransferConfig,
    mesh: Mesh | None = None,
    axis_mapping: ResourceMapping | None = None,
    coordinator_handle=None,
) -> WeightTransferClient:
    """Factory function to create appropriate transfer client for Levanter models.

    Args:
        config: Weight transfer configuration
        mesh: JAX mesh for distributed computation (optional)
        axis_mapping: Levanter axis mapping for sharding (optional)
        coordinator_handle: Pre-created actor handle for the coordinator.
            If provided, the client uses it directly instead of discovering via fray v1.
    """
    if config.mode == WeightTransferMode.ARROW_FLIGHT:
        return ArrowFlightClient(config, mesh, axis_mapping, coordinator_handle=coordinator_handle)

    # Default to GCS checkpoint mode
    return GCSCheckpointClient(
        config,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )


__all__ = [
    "ArrowFlightClient",
    "ArrowFlightCoordinator",
    "ArrowFlightServer",
    "GCSCheckpointClient",
    "GCSCheckpointServer",
    "WeightTransferClient",
    "WeightTransferClientMetrics",
    "WeightTransferConfig",
    "WeightTransferMode",
    "WeightTransferServer",
    "WeightTransferServerMetrics",
    "WeightUpdate",
    "create_weight_transfer_client",
    "create_weight_transfer_server",
]
