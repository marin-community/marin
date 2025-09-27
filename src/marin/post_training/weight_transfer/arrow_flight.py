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

import dataclasses
import logging
import socket
import threading
import time
from collections.abc import Sequence

import haliax as hax
import haliax.state_dict as hsd
import jax
import jax.numpy as jnp
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

# The maximum number of array elements in a single Arrow RecordBatch.
# Larger arrays are chunked into multiple RecordBatches to avoid hitting Arrow limits.
MAX_ELEMENTS_PER_RECORD = 32 * 1000 * 1000


def _create_binary_array(buffer_data: bytes) -> pa.Array:
    """Create a PyArrow binary array from buffer data."""
    block = pa.py_buffer(buffer_data)
    return pa.Array.from_buffers(
        pa.large_binary(),
        1,  # length
        [None, pa.array([0, len(block)], type=pa.int64()).buffers()[1], block],
    )


def _create_record_batch(
    schema: pa.Schema, name: str, binary_array: pa.Array, shape: tuple, dtype: str, part_idx: int, total_parts: int
) -> pa.RecordBatch:
    """Create a RecordBatch with the standard schema."""
    return pa.RecordBatch.from_arrays(
        [
            pa.array([name], type=pa.string()),
            pa.array(binary_array, type=pa.large_binary()),
            pa.array([list(shape)], type=pa.list_(pa.int64())),
            pa.array([str(dtype)], type=pa.string()),
            pa.array([part_idx], type=pa.int64()),
            pa.array([total_parts], type=pa.int64()),
        ],
        schema=schema,
    )


def serialize_pytree_to_arrow(model: PyTree, weight_id: int) -> tuple[pa.Schema, Sequence[pa.RecordBatch]]:
    """Convert model to Arrow RecordBatch using Haliax state_dict for efficient transfer.

    Args:
        model: Haliax/Equinox model or PyTree
        weight_id: Unique identifier for this weight version

    Large arrays are split into multiple RecordBatches if needed to avoid hitting the Arrow
    2GB limit.

    Returns:
        pa.Schema, List of Arrow RecordBatch containing serialized state dict data
    """
    # Convert model to state dict using Haliax's JAX array format
    state_dict = hsd.to_state_dict(model)
    batches = []
    schema = pa.schema(
        [
            pa.field("name", pa.string()),
            pa.field("data", pa.large_binary()),
            pa.field("shape", pa.list_(pa.int64())),
            pa.field("dtype", pa.string()),
            pa.field("idx", pa.int64()),
            pa.field("count", pa.int64()),
        ],
        metadata={"weight_id": str(weight_id), "timestamp": str(time.time()), "num_params": str(len(state_dict))},
    )

    # make a thread pool for big arrays
    import concurrent.futures

    # serialize directly from JAX array buffers
    pool = concurrent.futures.ThreadPoolExecutor()
    with pool:
        for name, value in state_dict.items():
            value = jax.device_get(value)
            shape = value.shape
            is_scalar = len(shape) == 0
            dtype = value.dtype

            # Handle splitting for large arrays
            if is_scalar:
                splits = [value]
                total_parts = 1
            else:
                flat_data = value.ravel()
                splits = list(jnp.array_split(flat_data, max(1, len(flat_data) // MAX_ELEMENTS_PER_RECORD)))
                total_parts = len(splits)

            # Create batches for each split
            for i, split in enumerate(splits):
                buffer_data = split.tobytes()
                binary_array = _create_binary_array(buffer_data)

                def _task(
                    schema=schema,
                    name=name,
                    binary_array=binary_array,
                    shape=shape,
                    dtype=dtype,
                    i=i,
                    total_parts=total_parts,
                ):
                    return _create_record_batch(schema, name, binary_array, shape, dtype, i, total_parts)

                batches.append(pool.submit(_task))
    # Wait for all tasks to complete and collect results
    batches = [batch.result() for batch in batches]
    return schema, batches


def deserialize_arrow_to_pytree(reader: pa.RecordBatchReader, target_model: PyTree) -> PyTree:
    """Convert Arrow RecordBatch back to model using Haliax state_dict."""
    state_dict = {}
    shapes = {}
    dtypes = {}

    for batch in reader:
        name = batch.column("name")[0].as_py()
        data = batch.column("data")[0]

        if name not in state_dict:
            state_dict[name] = []
        state_dict[name].append(data)

        shape = tuple(batch.column("shape")[0].as_py())
        dtype = batch.column("dtype")[0].as_py()
        shapes[name] = shape
        dtypes[name] = dtype

    # Coerce arrays to correct shapes and dtypes, construct JAX arrays directly
    for name in state_dict.keys():
        shape = shapes[name]
        dtype = dtypes[name]

        if len(shape) == 0:
            # scalar - get buffer directly
            buffer = state_dict[name][0].as_buffer()
            array_np = np.frombuffer(buffer, dtype=dtype)
            state_dict[name] = jax.numpy.asarray(array_np.item())
        else:
            # Get buffers directly without converting to Python lists
            buffers = [part.as_buffer() for part in state_dict[name]]
            parts = [np.frombuffer(buf, dtype=np.uint8) for buf in buffers]
            array_np = np.concatenate(parts)
            array_np = array_np.view(dtype).reshape(shape)
            # Convert to JAX array directly
            state_dict[name] = jax.numpy.asarray(array_np)

    return hsd.from_state_dict(target_model, state_dict)


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

    weights_store: dict[int, tuple[pa.Schema, Sequence[pa.RecordBatch]]]

    def __init__(self, location: str, config: WeightTransferConfig):
        super().__init__(location)
        self.config = config
        self.weights_store = {}
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
                    logger.warning(f"Requested weight_id {weight_id} not found in store, returning latest.")
                    weight_id = self.latest_weight_id

                (schema, batches) = self.weights_store[weight_id]

            return flight.GeneratorStream(schema, batches)

        except Exception as e:
            logger.error(f"Error in do_get: {e}")
            raise flight.FlightInternalError(f"Failed to get weights: {e}") from e

    def list_flights(self, context, criteria):
        """List available weight transfers."""
        with self._lock:
            for weight_id in self.weights_store.keys():
                descriptor = flight.FlightDescriptor.for_command(str(weight_id))
                schema = self.weights_store[weight_id].schema
                batches = self.weights_store[weight_id].batches

                # Create flight info
                info = flight.FlightInfo(
                    schema=schema,
                    descriptor=descriptor,
                    endpoints=[flight.FlightEndpoint(str(weight_id), [self.location])],
                    total_records=len(batches),
                    total_bytes=sum(batch.nbytes for batch in batches),
                )
                yield info

    def store_weights(self, weight_id: int, schema: pa.Schema, batches: pa.RecordBatch) -> None:
        """Store weights in the server."""
        with self._lock:
            # remove all other weights
            self.weights_store.clear()
            self.weights_store[weight_id] = (schema, batches)
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

        self.metrics = WeightTransferServerMetrics()

    def serve_weights(self, weight_id: int, model: PyTree) -> None:
        """Serve weights via Arrow Flight using Haliax state_dict serialization."""
        self.metrics.total_transfers += 1

        try:
            logger.info(f"Serving weights for weight_id {weight_id} via Arrow Flight...")
            barrier_sync()

            if jax.process_index() == 0:
                # Convert model to Arrow RecordBatch
                schema, batches = serialize_pytree_to_arrow(model, weight_id)

                # Store in Flight server
                self.flight_server.store_weights(weight_id, schema, batches)

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

        self.metrics = WeightTransferClientMetrics()

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

        except Exception:
            logger.warning("Failed to connect to Arrow Flight server.", exc_info=True)
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

            start_time = time.time()
            # Get latest weight info
            _server_location, latest_weight_id = ray.get(self.coordinator.get_server_info.remote())

            if latest_weight_id is None or latest_weight_id == self.last_weight_id:
                return None

            poll_time = time.time()

            # Request weights from Flight server
            ticket = flight.Ticket(str(latest_weight_id).encode("utf-8"))
            read_options = pa.ipc.IpcReadOptions(
                ensure_alignment=pa.ipc.Alignment.DataTypeSpecific, use_threads=True, ensure_native_endian=True
            )

            call_options = pa.flight.FlightCallOptions(read_options=read_options)

            flight_reader = self.flight_client.do_get(ticket, options=call_options)

            reader = flight_reader.to_reader()

            fetch_time = time.time()
            print(reader.schema.metadata)

            # Use the actual weight ID from the server, not the requested one
            received_weight_id = int(reader.schema.metadata[b"weight_id"].decode("utf-8"))

            # Convert back to model using state_dict and move to target device
            with self.mesh, hax.axis_mapping(self.axis_mapping):
                model = deserialize_arrow_to_pytree(reader, old_model)

            decode_time = time.time()

            self.metrics.successful_receives += 1
            self.metrics.poll_time = poll_time - start_time
            self.metrics.fetch_time = fetch_time - poll_time
            self.metrics.decode_time = decode_time - fetch_time
            self.last_weight_id = received_weight_id

            logger.info(f"Received weights for weight_id {received_weight_id} via Arrow Flight")

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

    def get_metrics(self) -> dict:
        return dataclasses.asdict(self.metrics)
