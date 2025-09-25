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

from haliax.partitioning import ResourceMapping
from jax.sharding import Mesh

from .base import (
    WeightTransferClient,
    WeightTransferClientMetrics,
    WeightTransferConfig,
    WeightTransferMode,
    WeightTransferServer,
    WeightTransferServerMetrics,
)
from .checkpoint import GCSCheckpointClient, GCSCheckpointServer
from .ray import RayRemotingClient, RayRemotingServer, RayWeightCoordinator

try:
    from .jax import JAXTransferClient, JAXTransferServer
except (ImportError, AttributeError):
    JAXTransferClient = None
    JAXTransferServer = None

logger = logging.getLogger(__name__)

# Check if JAX transfer is available
try:
    from ..jax_weight_transfer import (
        WeightTransferCoordinator,
        instantiate_coordinator,
        start_transfer_server,
    )

    JAX_TRANSFER_AVAILABLE = True
except (ImportError, AttributeError):
    JAX_TRANSFER_AVAILABLE = False
    WeightTransferCoordinator = None
    instantiate_coordinator = None
    start_transfer_server = None


def create_coordinator(mode: WeightTransferMode, name: str):
    """Create coordinator based on transfer mode.

    Args:
        mode: The weight transfer mode
        name: Unique name for the coordinator

    Returns:
        Ray actor handle for coordinator, or None if not needed
    """
    if mode == WeightTransferMode.RAY_REMOTING:
        return RayWeightCoordinator.options(name=name).remote()
    elif mode == WeightTransferMode.JAX_TRANSFER_SERVER:
        if not JAX_TRANSFER_AVAILABLE:
            raise RuntimeError("JAX transfer server not available")
        transfer_server = start_transfer_server()
        return instantiate_coordinator(transfer_server, name=name)
    else:
        return None  # GCS_CHECKPOINT doesn't need coordinator


def create_weight_transfer_server(
    config: WeightTransferConfig,
    mesh: Mesh | None = None,
    axis_mapping: ResourceMapping | None = None,
    coordinator=None,
) -> WeightTransferServer:
    """Factory function to create appropriate transfer server for Levanter models.

    Args:
        config: Weight transfer configuration
        mesh: JAX mesh for distributed computation (optional)
        axis_mapping: Levanter axis mapping for sharding (optional)
        coordinator: Ray coordinator for distributed modes (required for RAY_REMOTING)

    Returns:
        WeightTransferServer instance
    """
    if config.mode == WeightTransferMode.JAX_TRANSFER_SERVER:
        raise RuntimeError("JAX transfer server not supported for Levanter models")

    elif config.mode == WeightTransferMode.RAY_REMOTING:
        if coordinator is None:
            raise ValueError("Coordinator required for Ray remoting")
        return RayRemotingServer(config, coordinator)

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
    coordinator=None,
) -> WeightTransferClient:
    """Factory function to create appropriate transfer client for Levanter models.

    Args:
        config: Weight transfer configuration
        mesh: JAX mesh for distributed computation (optional)
        axis_mapping: Levanter axis mapping for sharding (optional)
        target_model: Target parameter structure for reconstructing NamedArrays (optional)
        coordinator: Ray coordinator for distributed modes (required for RAY_REMOTING)

    Returns:
        WeightTransferClient instance
    """
    if config.mode == WeightTransferMode.JAX_TRANSFER_SERVER:
        # JAX transfer server is currently not supported with Levanter models
        logger.warning("JAX transfer server not supported, falling back to GCS checkpoints")
        config.mode = WeightTransferMode.GCS_CHECKPOINT

    elif config.mode == WeightTransferMode.RAY_REMOTING:
        if coordinator is None:
            raise ValueError("Coordinator required for Ray remoting")
        return RayRemotingClient(
            config,
            coordinator,
        )

    # Default to GCS checkpoint mode
    return GCSCheckpointClient(
        config,
        axis_mapping=axis_mapping,
        mesh=mesh,
    )


__all__ = [
    "GCSCheckpointClient",
    "GCSCheckpointServer",
    "JAXTransferClient",
    "JAXTransferServer",
    "RayRemotingClient",
    "RayRemotingServer",
    "RayWeightCoordinator",
    "WeightTransferClient",
    "WeightTransferClientMetrics",
    "WeightTransferConfig",
    "WeightTransferMode",
    "WeightTransferServer",
    "WeightTransferServerMetrics",
    "create_coordinator",
    "create_weight_transfer_client",
    "create_weight_transfer_server",
]
