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
    mode: WeightTransferMode = WeightTransferMode.ARROW_FLIGHT
    # Common settings
    sync_interval_steps: int = 1
    coordinator_name: str = "weight_transfer_coordinator"

    transfer_timeout: float = 30.0
    max_weight_transfer_wait_time: float = 0.0
    """Maximum time (in seconds) to wait for new weights before proceeding. 0 means run ahead without waiting."""

    # GCS Checkpoint specific
    checkpoint_dir: str = ""
    max_checkpoints: int | None = 5

    # Arrow Flight specific
    flight_host: str = "0.0.0.0"
    flight_port: int = 0  # 0 = auto-assign


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
