# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Weight transfer management.

This module provides abstractions for communicating weights between training and inference workers.

Currently GCS, Ray remoting, and JAX transfer server methods are supported.
"""

import logging
from typing import TYPE_CHECKING

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

try:
    from .jax import JAXTransferClient, JAXTransferServer, WeightTransferCoordinator
except (ImportError, AttributeError):
    JAXTransferClient = None
    JAXTransferServer = None
    WeightTransferCoordinator = None

if TYPE_CHECKING:
    from fray.v2.actor import ActorHandle

logger = logging.getLogger(__name__)

# Check if JAX transfer is available
JAX_TRANSFER_AVAILABLE = JAXTransferClient is not None and JAXTransferServer is not None


def create_weight_transfer_server(
    config: WeightTransferConfig,
    coordinator: "ActorHandle | None" = None,
    mesh: Mesh | None = None,
    axis_mapping: ResourceMapping | None = None,
) -> WeightTransferServer:
    """Factory function to create appropriate transfer server for Levanter models.

    Args:
        config: Weight transfer configuration
        coordinator: Actor handle for the weight transfer coordinator (required for JAX/Arrow Flight modes)
        mesh: JAX mesh for distributed computation (optional)
        axis_mapping: Levanter axis mapping for sharding (optional)

    Returns:
        WeightTransferServer instance
    """
    if config.mode == WeightTransferMode.JAX_TRANSFER_SERVER:
        if coordinator is None:
            raise ValueError("coordinator handle is required for JAX transfer server mode")
        return JAXTransferServer(config, coordinator, mesh, axis_mapping)

    elif config.mode == WeightTransferMode.ARROW_FLIGHT:
        if coordinator is None:
            raise ValueError("coordinator handle is required for Arrow Flight mode")
        return ArrowFlightServer(config, coordinator, mesh, axis_mapping)

    # Default to GCS checkpoint mode (no coordinator needed)
    return GCSCheckpointServer(
        config,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )


def create_weight_transfer_client(
    config: WeightTransferConfig,
    coordinator: "ActorHandle | None" = None,
    mesh: Mesh | None = None,
    axis_mapping: ResourceMapping | None = None,
) -> WeightTransferClient:
    """Factory function to create appropriate transfer client for Levanter models.

    Args:
        config: Weight transfer configuration
        coordinator: Actor handle for the weight transfer coordinator (required for JAX/Arrow Flight modes)
        mesh: JAX mesh for distributed computation (optional)
        axis_mapping: Levanter axis mapping for sharding (optional)

    Returns:
        WeightTransferClient instance
    """
    if config.mode == WeightTransferMode.JAX_TRANSFER_SERVER:
        if coordinator is None:
            raise ValueError("coordinator handle is required for JAX transfer server mode")
        return JAXTransferClient(config, coordinator, mesh, axis_mapping)

    elif config.mode == WeightTransferMode.ARROW_FLIGHT:
        if coordinator is None:
            raise ValueError("coordinator handle is required for Arrow Flight mode")
        return ArrowFlightClient(config, coordinator, mesh, axis_mapping)

    # Default to GCS checkpoint mode (no coordinator needed)
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
    "JAXTransferClient",
    "JAXTransferServer",
    "WeightTransferClient",
    "WeightTransferClientMetrics",
    "WeightTransferConfig",
    "WeightTransferCoordinator",
    "WeightTransferMode",
    "WeightTransferServer",
    "WeightTransferServerMetrics",
    "WeightUpdate",
    "create_weight_transfer_client",
    "create_weight_transfer_server",
]
