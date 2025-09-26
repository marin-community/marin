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
Apache Arrow Flight-based weight transfer implementation.

This module provides weight transfer using Apache Arrow Flight RPC for
high-performance binary communication between training and inference workers.
"""

import logging
import socket
import threading
import time

import haliax.state_dict as hsd
import jax
import numpy as np
import pyarrow as pa
import pyarrow.flight as flight
import ray
from haliax.partitioning import ResourceMapping
from jax.sharding import Mesh
from jaxtyping import PyTree
from levanter.utils.jax_utils import barrier_sync

from .base import (
    WeightTransferClient,
    WeightTransferClientMetrics,
    WeightTransferConfig,
    WeightTransferServer,
    WeightTransferServerMetrics,
    get_or_create_actor,
)

logger = logging.getLogger(__name__)


def serialize_pytree_to_arrow(model: PyTree, weight_id: int) -> pa.RecordBatch:
    """Convert model to Arrow RecordBatch using Haliax state_dict for efficient transfer.

    Args:
        model: Haliax/Equinox model or PyTree
        weight_id: Unique identifier for this weight version

    Returns:
        Arrow RecordBatch containing serialized state dict data
    """
    # Convert model to state dict using Haliax's torch-compatible format
    state_dict = hsd.to_torch_compatible_state_dict(model)

    # Prepare data for Arrow RecordBatch
    param_names = []
    param_data = []
    param_shapes = []
    param_dtypes = []

    for name, array in state_dict.items():
        param_names.append(name)
        # Convert to numpy if needed and serialize to bytes
        array_np = np.asarray(array)
        param_data.append(array_np.tobytes())
        param_shapes.append(str(array_np.shape))
        param_dtypes.append(str(array_np.dtype))

    # Create Arrow arrays
    names_array = pa.array(param_names, type=pa.string())
    data_array = pa.array(param_data, type=pa.binary())
    shapes_array = pa.array(param_shapes, type=pa.string())
    dtypes_array = pa.array(param_dtypes, type=pa.string())

    # Create schema with metadata
    schema = pa.schema(
        [
            pa.field("param_names", pa.string()),
            pa.field("param_data", pa.binary()),
            pa.field("param_shapes", pa.string()),
            pa.field("param_dtypes", pa.string()),
        ],
        metadata={"weight_id": str(weight_id), "timestamp": str(time.time()), "num_params": str(len(state_dict))},
    )

    # Create RecordBatch
    return pa.RecordBatch.from_arrays([names_array, data_array, shapes_array, dtypes_array], schema=schema)


def deserialize_arrow_to_pytree(record_batch: pa.RecordBatch, target_model: PyTree) -> PyTree:
    """Convert Arrow RecordBatch back to model using Haliax state_dict.

    Args:
        record_batch: Arrow RecordBatch containing serialized state dict data
        target_model: Target model template to preserve structure and NamedArray axes

    Returns:
        Reconstructed model with updated weights
    """
    # Extract parameter data from Arrow RecordBatch
    param_names = record_batch.column("param_names").to_pylist()
    param_data = record_batch.column("param_data").to_pylist()
    param_shapes = record_batch.column("param_shapes").to_pylist()
    param_dtypes = record_batch.column("param_dtypes").to_pylist()

    # Reconstruct state dict
    state_dict = {}
    for name, data, shape_str, dtype_str in zip(param_names, param_data, param_shapes, param_dtypes, strict=True):
        # Parse shape and dtype
        shape = eval(shape_str)
        dtype = np.dtype(dtype_str)

        # Reconstruct array from bytes
        array_np = np.frombuffer(data, dtype=dtype).reshape(shape)
        state_dict[name] = array_np

    # Use Haliax to reconstruct the model with proper structure
    return hsd.from_torch_compatible_state_dict(target_model, state_dict)


@ray.remote(num_cpus=0)
class ArrowFlightCoordinator:
    """Ray actor for coordinating Arrow Flight weight transfers."""

    def __init__(self):
        self.server_address = None
        self.latest_weight_id = None
        self.server_location = None

    def register_server(self, server_address: str, server_port: int) -> None:
        """Register the Arrow Flight server address."""
        self.server_address = server_address
        self.server_port = server_port
        self.server_location = f"grpc://{server_address}:{server_port}"
        logger.info(f"Registered Arrow Flight server at {self.server_location}")

    def get_server_info(self) -> tuple[str | None, int | None]:
        """Get the current server location and latest weight ID."""
        return self.server_location, self.latest_weight_id

    def update_weight_id(self, weight_id: int) -> None:
        """Update the latest weight ID."""
        if self.latest_weight_id is None or weight_id > self.latest_weight_id:
            self.latest_weight_id = weight_id
            logger.info(f"Updated latest weight ID to {weight_id}")


class MarinFlightServer(flight.FlightServerBase):
    """Arrow Flight server for serving model weights."""

    def __init__(self, location: str, config: WeightTransferConfig):
        super().__init__(location)
        self.config = config
        self.weights_store = {}  # weight_id -> RecordBatch
        self.latest_weight_id = None
        self._lock = threading.Lock()
        self._location = location

    def do_put(self, context, descriptor, reader, writer):
        """Handle incoming weight data from training worker."""
        # This would be used if we want to support pushing weights via Flight
        # For now, we use direct calls to store_weights
        pass

    def do_get(self, context, ticket):
        """Serve weight data to inference workers."""
        try:
            # Parse ticket to get weight ID
            ticket_data = ticket.ticket.decode("utf-8")
            weight_id = int(ticket_data)

            with self._lock:
                if weight_id not in self.weights_store:
                    raise flight.FlightNotFoundError(f"Weight ID {weight_id} not found")

                record_batch = self.weights_store[weight_id]

            # Create a generator that yields the record batch
            def generate_batches():
                yield record_batch

            return flight.GeneratorStream(record_batch.schema, generate_batches())

        except Exception as e:
            logger.error(f"Error in do_get: {e}")
            raise flight.FlightInternalError(f"Failed to get weights: {e}") from e

    def list_flights(self, context, criteria):
        """List available weight transfers."""
        with self._lock:
            for weight_id in self.weights_store.keys():
                descriptor = flight.FlightDescriptor.for_command(str(weight_id))

                # Create flight info
                info = flight.FlightInfo(
                    schema=self.weights_store[weight_id].schema,
                    descriptor=descriptor,
                    endpoints=[flight.FlightEndpoint(str(weight_id), [self.location])],
                    total_records=1,
                    total_bytes=self.weights_store[weight_id].nbytes,
                )
                yield info

    def store_weights(self, weight_id: int, record_batch: pa.RecordBatch) -> None:
        """Store weights in the server."""
        with self._lock:
            self.weights_store[weight_id] = record_batch
            self.latest_weight_id = weight_id
            logger.info(f"Stored weights for weight_id {weight_id}")

    def get_latest_weight_id(self) -> int | None:
        """Get the latest weight ID."""
        with self._lock:
            return self.latest_weight_id


class ArrowFlightServer(WeightTransferServer):
    """Arrow Flight-based weight transfer server for Haliax/Equinox models.

    Uses Haliax state_dict for proper serialization of model parameters.
    """

    def __init__(
        self, config: WeightTransferConfig, mesh: Mesh | None = None, axis_mapping: ResourceMapping | None = None
    ):
        self.config = config
        self.mesh = mesh
        self.axis_mapping = axis_mapping

        # Start Flight server
        location = f"grpc://{config.flight_host}:{config.flight_port}"
        self.flight_server = MarinFlightServer(location, config)

        # Server starts immediately when created, get the actual port
        actual_port = self.flight_server.port
        actual_host = config.flight_host if config.flight_host != "0.0.0.0" else socket.gethostname()

        self.server_location = f"grpc://{actual_host}:{actual_port}"
        logger.info(f"Arrow Flight server started at {self.server_location}")

        # Start the server in a background thread
        self._server_thread = threading.Thread(target=self.flight_server.serve, daemon=True)
        self._server_thread.start()

        # Register with coordinator
        self.coordinator = get_or_create_actor(ArrowFlightCoordinator, config.coordinator_name)
        ray.get(self.coordinator.register_server.remote(actual_host, actual_port))

        self.metrics = WeightTransferServerMetrics(start_time=time.time())

    def serve_weights(self, weight_id: int, model: PyTree) -> None:
        """Serve weights via Arrow Flight using Haliax state_dict serialization."""
        self.metrics.total_transfers += 1

        try:
            logger.info(f"Serving weights for weight_id {weight_id} via Arrow Flight...")
            barrier_sync()

            if jax.process_index() == 0:
                # Convert model to Arrow RecordBatch
                record_batch = serialize_pytree_to_arrow(model, weight_id)

                # Store in Flight server
                self.flight_server.store_weights(weight_id, record_batch)

                # Update coordinator
                ray.get(self.coordinator.update_weight_id.remote(weight_id))

                self.metrics.successful_transfers += 1
                logger.info(f"Served weights for weight_id {weight_id} via Arrow Flight")

            barrier_sync()

        except Exception as e:
            self.metrics.failed_transfers += 1
            logger.error(f"Failed to serve weights {weight_id} via Arrow Flight: {e}")
            raise

    def cleanup(self) -> None:
        """Cleanup Flight server resources."""
        try:
            self.flight_server.shutdown()
        except Exception as e:
            logger.warning(f"Error during Arrow Flight server cleanup: {e}")

    def get_metrics(self) -> WeightTransferServerMetrics:
        """Get transfer metrics."""
        return self.metrics


class ArrowFlightClient(WeightTransferClient):
    """Arrow Flight-based weight transfer client for Haliax/Equinox models.

    Uses Haliax state_dict for proper deserialization of model parameters.
    """

    def __init__(
        self, config: WeightTransferConfig, mesh: Mesh | None = None, axis_mapping: ResourceMapping | None = None
    ):
        self.config = config
        self.mesh = mesh
        self.axis_mapping = axis_mapping

        # Get coordinator
        self.coordinator = get_or_create_actor(ArrowFlightCoordinator, config.coordinator_name)

        self.last_weight_id = None
        self.flight_client = None
        self.server_location = None

        self.metrics = WeightTransferClientMetrics(start_time=time.time())

    def _connect_to_server(self) -> bool:
        """Connect to the Arrow Flight server."""
        try:
            server_location, _latest_weight_id = ray.get(self.coordinator.get_server_info.remote())

            if server_location is None:
                return False

            if self.server_location != server_location:
                # New server or first connection
                self.server_location = server_location
                if self.flight_client:
                    self.flight_client.close()

                self.flight_client = flight.FlightClient(server_location)
                logger.info(f"Connected to Arrow Flight server at {server_location}")

            return True

        except Exception as e:
            logger.warning(f"Failed to connect to Arrow Flight server: {e}")
            return False

    def receive_weights(self, old_model: PyTree = None) -> PyTree | None:
        """Receive weights from Arrow Flight server.

        Args:
            old_model: Template model to preserve structure. Required for proper deserialization.
        """
        self.metrics.total_polls += 1

        if old_model is None:
            raise ValueError("old_model is required for Arrow Flight weight transfer to preserve model structure")

        try:
            # Connect to server if needed
            if not self._connect_to_server():
                return None

            # Get latest weight info
            _server_location, latest_weight_id = ray.get(self.coordinator.get_server_info.remote())

            if latest_weight_id is None or latest_weight_id == self.last_weight_id:
                return None

            # Request weights from Flight server
            ticket = flight.Ticket(str(latest_weight_id).encode("utf-8"))
            flight_reader = self.flight_client.do_get(ticket)

            # Read the record batch
            record_batch = flight_reader.read_all()

            # Convert back to model using state_dict
            model = deserialize_arrow_to_pytree(record_batch, old_model)

            self.metrics.successful_receives += 1
            self.last_weight_id = latest_weight_id

            logger.info(f"Received weights for weight_id {latest_weight_id} via Arrow Flight")

            return model

        except Exception as e:
            self.metrics.failed_receives += 1
            logger.error(f"Failed to receive weights via Arrow Flight: {e}")
            raise

    def cleanup(self) -> None:
        """Cleanup Flight client resources."""
        if self.flight_client:
            try:
                self.flight_client.close()
            except Exception as e:
                logger.warning(f"Error during Arrow Flight client cleanup: {e}")

    def get_metrics(self) -> WeightTransferClientMetrics:
        """Get transfer metrics."""
        return self.metrics
