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
Base classes and configurations for weight transfer management.

This module provides the core abstractions and configurations used by all
weight transfer implementations.
"""

import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from jaxtyping import PyTree

logger = logging.getLogger(__name__)


class WeightTransferMode(Enum):
    GCS_CHECKPOINT = "gcs_checkpoint"
    JAX_TRANSFER_SERVER = "jax_transfer_server"
    ARROW_FLIGHT = "arrow_flight"


@dataclass
class WeightTransferServerMetrics:
    """Metrics for weight transfer servers."""

    total_transfers: int = 0
    successful_transfers: int = 0
    failed_transfers: int = 0


@dataclass
class WeightTransferClientMetrics:
    """Metrics for weight transfer clients."""

    total_polls: int = 0
    successful_receives: int = 0
    failed_receives: int = 0
    fetch_time: float = 0
    decode_time: float = 0
    poll_time: float = 0


@dataclass
class WeightUpdate:
    """Result of receiving weights from a weight transfer server."""

    model: PyTree
    weight_id: int


@dataclass
class WeightTransferConfig:
    mode: WeightTransferMode = WeightTransferMode.GCS_CHECKPOINT
    # Common settings
    sync_interval_steps: int = 100
    poll_interval_seconds: float = 30.0
    coordinator_name: str = "weight_transfer_coordinator"

    transfer_timeout: float = 5.0

    # GCS Checkpoint specific
    checkpoint_dir: str = ""
    max_checkpoints: int | None = 5

    # Arrow Flight specific
    flight_host: str = "0.0.0.0"
    flight_port: int = 0  # 0 = auto-assign
    flight_batch_size: int = 1024 * 1024 * 100  # 100MB chunks
    flight_use_tls: bool = False


class WeightTransferServer(ABC):
    """Abstract base class for weight transfer servers (training worker side)."""

    @abstractmethod
    def serve_weights(self, weight_id: int, model) -> None:
        """Serve weights to clients.

        Args:
            weight_id: Unique identifier for this weight update
            model: Levanter model parameters (PyTree of NamedArrays)
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        pass


class WeightTransferClient(ABC):
    """Abstract base class for weight transfer clients (inference worker side)."""

    @abstractmethod
    def receive_weights(self, old_model: PyTree) -> WeightUpdate | None:
        """Receive weights from server.

        Args:
            old_model: Previous model for memory optimization (optional)

        Returns:
            WeightUpdate containing the new model and weight_id if update available, None otherwise.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        pass


def get_or_create_actor(actor_class, name: str, *args, **kwargs):
    """Fetch an existing actor reference or create it if it doesn't exist.

    Args:
        actor_class: Ray remote class (e.g., RayWeightCoordinator, WeightTransferCoordinator)
        name: Actor name for registration
        *args: Arguments to pass to actor constructor
        max_retries: Number of retry attempts
        **kwargs: Keyword arguments to pass to actor constructor

    Returns:
        Ray actor handle
    """
    max_retries = 5

    for attempt in range(max_retries):
        logger.info("Retrieving or creating actor '%s' (attempt %d)", name, attempt + 1)
        try:
            return actor_class.options(name=name, get_if_exists=True, max_restarts=-1).remote(*args, **kwargs)
        except ValueError:
            # Another process might have created it, wait and retry
            if attempt < max_retries - 1:
                retry_timeout = 0.1 * (attempt**2)
                logger.info(
                    "Actor '%s' not found, retrying in %.2f seconds (attempt %d)", name, retry_timeout, attempt + 1
                )
                time.sleep(retry_timeout)
                continue
            raise

    raise RuntimeError(f"Failed to get or create actor '{name}' after {max_retries} attempts")
