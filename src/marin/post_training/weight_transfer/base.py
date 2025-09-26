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
from typing import Any

from jaxtyping import PyTree

logger = logging.getLogger(__name__)


class WeightTransferMode(Enum):
    GCS_CHECKPOINT = "gcs_checkpoint"
    JAX_TRANSFER_SERVER = "jax_transfer_server"
    RAY_REMOTING = "ray_remoting"
    ARROW_FLIGHT = "arrow_flight"


@dataclass
class WeightTransferServerMetrics:
    """Metrics for weight transfer servers."""

    total_transfers: int = 0
    successful_transfers: int = 0
    failed_transfers: int = 0
    start_time: float = 0.0

    @property
    def transfer_rate(self) -> float:
        """Transfers per second."""
        elapsed = time.time() - self.start_time
        return self.successful_transfers / max(elapsed, 1.0)


@dataclass
class WeightTransferClientMetrics:
    """Metrics for weight transfer clients."""

    total_polls: int = 0
    successful_receives: int = 0
    failed_receives: int = 0
    start_time: float = 0.0

    @property
    def poll_rate(self) -> float:
        """Polls per second."""
        elapsed = time.time() - self.start_time
        return self.total_polls / max(elapsed, 1.0)

    @property
    def success_rate(self) -> float:
        """Ratio of successful receives to total polls."""
        return self.successful_receives / max(self.total_polls, 1)


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
    def get_metrics(self) -> WeightTransferServerMetrics:
        """Get transfer metrics."""
        pass


class WeightTransferClient(ABC):
    """Abstract base class for weight transfer clients (inference worker side)."""

    @abstractmethod
    def receive_weights(self, old_model: PyTree) -> Any:
        """Receive weights from server.

        Args:
            old_model: Previous model for memory optimization (optional)

        Returns:
            new_model or None if no update available.
            new_model will be Levanter model parameters (PyTree of NamedArrays).
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    def get_metrics(self) -> WeightTransferClientMetrics:
        """Get transfer metrics."""
        return WeightTransferClientMetrics(start_time=time.time())


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
            return actor_class.options(name=name, get_if_exists=True).remote(*args, **kwargs)
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
