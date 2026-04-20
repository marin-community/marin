# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
    ARROW_FLIGHT = "arrow_flight"


@dataclass
class WeightTransferServerMetrics:
    """Metrics for weight transfer servers."""

    total_transfers: int = 0
    successful_transfers: int = 0
    failed_transfers: int = 0
    total_transfer_bytes: int = 0
    transfer_bytes: int = 0
    param_count: int = 0
    largest_param_bytes: int = 0
    state_dict_time: float = 0.0
    materialize_time: float = 0.0
    serialize_time: float = 0.0
    store_time: float = 0.0
    update_time: float = 0.0
    serve_time: float = 0.0
    materialize_mib_per_second: float = 0.0
    serialize_mib_per_second: float = 0.0
    store_mib_per_second: float = 0.0


@dataclass
class WeightTransferClientMetrics:
    """Metrics for weight transfer clients."""

    total_polls: int = 0
    successful_receives: int = 0
    failed_receives: int = 0
    total_receive_bytes: int = 0
    receive_bytes: int = 0
    param_count: int = 0
    largest_param_bytes: int = 0
    fetch_time: float = 0
    decode_time: float = 0
    poll_time: float = 0
    fetch_mib_per_second: float = 0.0
    decode_mib_per_second: float = 0.0


@dataclass
class WeightUpdate:
    """Result of receiving weights from a weight transfer server."""

    model: PyTree | None
    state_dict: dict
    weight_id: int


@dataclass
class WeightTransferConfig:
    mode: WeightTransferMode = WeightTransferMode.ARROW_FLIGHT
    # Common settings
    sync_interval_steps: int = 1
    coordinator_name: str = "weight_transfer_coordinator"

    transfer_timeout: float = 600.0
    max_weight_transfer_wait_time: float = 0.0
    """Maximum time (in seconds) to wait for new weights before proceeding. 0 means run ahead without waiting."""

    # GCS Checkpoint specific
    checkpoint_dir: str = ""
    max_checkpoints: int | None = 5

    # Arrow Flight specific
    flight_host: str = "0.0.0.0"
    flight_port: int = 0  # 0 = auto-assign
    convert_to_bfloat16: bool = True
    """Whether to convert weights to bfloat16 during transfer. Reduces transfer size by 50% for float32 weights."""
    debug_weight_transfer: bool = False


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

    def get_debug_snapshot(self) -> dict[str, object]:
        """Return lightweight server-local debug state for checkpoint diagnostics."""
        return {}


class WeightTransferClient(ABC):
    """Abstract base class for weight transfer clients (inference worker side)."""

    @abstractmethod
    def receive_weights(self, old_model: PyTree | None) -> WeightUpdate | None:
        """Receive weights from server.

        Args:
            old_model: Previous model for memory optimization (optional)
            apply_weight_update: Whether to apply the weight update to the model

        Returns:
            WeightUpdate containing the new model or state_dict and weight_id if update available,
            None otherwise. If old_model is None, the weight update will only contain the new state
            dict but will not apply the weight update. If old model is not None, the weight update
            will be applied to the model.
        """
        pass

    @abstractmethod
    def cleanup(self) -> None:
        """Cleanup resources."""
        pass

    @abstractmethod
    def get_metrics(self) -> dict:
        pass
