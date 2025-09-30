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
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial

import haliax as hax
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

# The maximum number of array elements in a single Arrow RecordBatch.
# Larger arrays are chunked into multiple RecordBatches to avoid hitting Arrow limits.
# We assume our largest dtype is 4 bytes (e.g., float32/int32).
MAX_ELEMENTS_PER_RECORD = (2000 * 1000 * 1000) / 4


def _create_binary_array(buffer_data: jax.Array) -> pa.Array:
    """Create a PyArrow binary array from buffer data."""
    # dlpack_capsule = jax.dlpack.to_dlpack(buffer_data)
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


def serialize_pytree_to_arrow(model: PyTree, weight_id: int) -> dict[str, tuple[pa.Schema, Sequence[pa.RecordBatch]]]:
    """Convert model to Arrow RecordBatch per parameter using Haliax state_dict for efficient transfer.

    Args:
        model: Haliax/Equinox model or PyTree
        weight_id: Unique identifier for this weight version

    Large arrays are split into multiple RecordBatches if needed to avoid hitting the Arrow
    2GB limit.

    Returns:
        Dict mapping param_name -> (schema, batches) for per-parameter flights
    """
    # Convert model to state dict using Haliax's JAX array format
    state_dict = hsd.to_state_dict(model)
    state_dict = jax.device_get(state_dict)

    result = {}
    sz = 0

    for name, value in state_dict.items():
        shape = value.shape
        is_scalar = len(shape) == 0
        dtype = value.dtype
        sz += value.nbytes

        # Create schema for this parameter
        schema = pa.schema(
            [
                pa.field("data", pa.large_binary()),
                pa.field("shape", pa.list_(pa.int64())),
                pa.field("dtype", pa.string()),
                pa.field("idx", pa.int64()),
                pa.field("count", pa.int64()),
            ],
            metadata={
                "weight_id": str(weight_id),
                "timestamp": str(time.time()),
                "param_name": name,
            },
        )

        # Handle splitting for large arrays
        if is_scalar:
            splits = [value]
            total_parts = 1
        else:
            splits = np.array_split(value.flatten(), max(1, value.size // MAX_ELEMENTS_PER_RECORD))
            total_parts = len(splits)

        # Create batches for each split
        batches = []
        for i, split in enumerate(splits):
            binary_array = _create_binary_array(split)
            batch = pa.RecordBatch.from_arrays(
                [
                    pa.array(binary_array, type=pa.large_binary()),
                    pa.array([list(shape)], type=pa.list_(pa.int64())),
                    pa.array([str(dtype)], type=pa.string()),
                    pa.array([i], type=pa.int64()),
                    pa.array([total_parts], type=pa.int64()),
                ],
                schema=schema,
            )
            batches.append(batch)

        result[name] = (schema, batches)

    logger.info(f"Serialized model to Arrow with {len(state_dict)} parameters, total size {sz / (1024 * 1024):.2f} MB")
    return result


@partial(jax.jit, donate_argnums=0)
def update_model(old_model, new_state_dict):
    return hsd.from_state_dict(old_model, new_state_dict)


def deserialize_arrow_to_pytree(param_name: str, reader: pa.RecordBatchReader, target_model: PyTree) -> jax.Array:
    """Convert Arrow RecordBatch back to a single parameter array.

    Args:
        param_name: Name of the parameter being deserialized
        reader: Arrow RecordBatch reader containing the parameter data
        target_model: Template model to preserve structure

    Returns:
        JAX array for the parameter
    """
    parts = []
    shape = None
    dtype = None

    for batch in reader:
        data = batch.column("data")[0]
        parts.append(data)

        if shape is None:
            shape = tuple(batch.column("shape")[0].as_py())
            dtype = batch.column("dtype")[0].as_py()

    # Coerce arrays to correct shapes and dtypes, construct JAX arrays directly
    if len(shape) == 0:
        # scalar - get buffer directly
        buffer = parts[0].as_buffer()
        array_np = np.frombuffer(buffer, dtype=dtype)
        return jax.numpy.asarray(array_np.item())
    else:
        # Get buffers directly without converting to Python lists
        st = time.time()
        buffers = [part.as_buffer() for part in parts]
        buffer_parts = [np.frombuffer(buf, dtype=np.uint8) for buf in buffers]
        array_np = np.concatenate(buffer_parts)
        array_np = array_np.view(dtype).reshape(shape)
        # Convert to JAX array directly
        res = jax.numpy.asarray(array_np)
        ed = time.time()
        if ed - st > 0.1:
            logger.info(f"Deserialized param {param_name} of shape {shape} and dtype {dtype} in {ed - st:.2f}s")
        return res


@ray.remote(num_cpus=0)
class ArrowFlightCoordinator:
    """Ray actor for coordinating Arrow Flight weight transfers."""

    def __init__(self):
        self.server_address = None
        self.latest_weight_id = None
        self.server_location = None
        self.available_params: dict[int, list[str]] = {}
        self.all_server_locations: list[tuple[str, int]] = []

    def register_server(self, server_address: str, server_port: int) -> None:
        """Register the Arrow Flight server address."""
        self.server_address = server_address
        self.server_port = server_port
        self.server_location = f"grpc://{server_address}:{server_port}"
        logger.info(f"Registered Arrow Flight server at {self.server_location}")

    def register_all_servers(self, server_locations: list[tuple[str, int]]) -> None:
        """Register all server locations for load balancing."""
        self.all_server_locations = server_locations
        logger.info(f"Registered {len(server_locations)} Arrow Flight servers")

    def get_server_info(self) -> tuple[str | None, int | None]:
        """Get the current server location and latest weight ID."""
        return self.server_location, self.latest_weight_id

    def get_all_server_locations(self) -> list[str]:
        """Get all server locations for parallel fetching."""
        return [f"grpc://{host}:{port}" for host, port in self.all_server_locations]

    def update_weight_id(self, weight_id: int, param_names: list[str]) -> None:
        """Update the latest weight ID and register available params atomically."""
        if self.latest_weight_id is None or weight_id > self.latest_weight_id:
            self.available_params[weight_id] = param_names
            self.latest_weight_id = weight_id
            logger.info(f"Updated latest weight ID to {weight_id} with {len(param_names)} params")

    def get_available_params(self, weight_id: int) -> list[str] | None:
        """Get the list of available param names for a weight_id."""
        return self.available_params.get(weight_id)


class MarinFlightServer(flight.FlightServerBase):
    """Arrow Flight server for serving model weights."""

    weights_store: dict[int, dict[str, tuple[pa.Schema, Sequence[pa.RecordBatch]]]]

    def __init__(self, location: str, config: WeightTransferConfig):
        super().__init__(location)
        self.config = config
        self.weights_store = {}
        self.latest_weight_id = None
        self._lock = threading.Lock()
        self._location = location

    def do_put(self, context, descriptor, reader, writer):
        pass

    def do_get(self, context, ticket):
        """Serve weight data to inference workers."""
        try:
            ticket_data = ticket.ticket.decode("utf-8")

            # Parse ticket as "weight_id/param_name"
            if "/" not in ticket_data:
                raise ValueError(f"Invalid ticket format: {ticket_data}. Expected 'weight_id/param_name'")

            weight_id_str, param_name = ticket_data.split("/", 1)
            weight_id = int(weight_id_str)

            with self._lock:
                if weight_id not in self.weights_store:
                    logger.warning(f"Requested weight_id {weight_id} not found in store, using latest.")
                    weight_id = self.latest_weight_id

                (schema, batches) = self.weights_store[weight_id][param_name]

            return flight.RecordBatchStream(pa.RecordBatchReader.from_batches(schema, batches))

        except Exception as e:
            logger.error(f"Error in do_get: {e}")
            raise flight.FlightInternalError(f"Failed to get weights: {e}") from e

    def list_flights(self, context, criteria):
        """List available weight transfers."""
        with self._lock:
            for weight_id, params_dict in self.weights_store.items():
                for param_name, (schema, batches) in params_dict.items():
                    ticket_str = f"{weight_id}/{param_name}"
                    descriptor = flight.FlightDescriptor.for_command(ticket_str)

                    # Create flight info for this param
                    info = flight.FlightInfo(
                        schema=schema,
                        descriptor=descriptor,
                        endpoints=[flight.FlightEndpoint(ticket_str, [self._location])],
                        total_records=len(batches),
                        total_bytes=sum(batch.nbytes for batch in batches),
                    )
                    yield info

    def store_weights(self, weight_id: int, params_dict: dict[str, tuple[pa.Schema, Sequence[pa.RecordBatch]]]) -> None:
        """Store weights in the server atomically."""
        with self._lock:
            # remove all other weights
            self.weights_store.clear()
            self.weights_store[weight_id] = params_dict
            self.latest_weight_id = weight_id
            logger.info(f"Stored {len(params_dict)} params for weight_id {weight_id}")

    def get_latest_weight_id(self) -> int | None:
        """Get the latest weight ID."""
        with self._lock:
            return self.latest_weight_id


class ArrowFlightServer(WeightTransferServer):
    """Arrow Flight-based weight transfer server for Haliax/Equinox models.

    Uses Haliax state_dict for proper serialization of model parameters.
    Spawns multiple flight server instances for parallel serving.
    """

    def __init__(
        self,
        config: WeightTransferConfig,
        mesh: Mesh | None = None,
        axis_mapping: ResourceMapping | None = None,
        num_servers: int = 16,
    ):
        self.config = config
        self.mesh = mesh
        self.axis_mapping = axis_mapping
        self.num_servers = num_servers

        # Start multiple Flight servers
        self.flight_servers = []
        self._server_threads = []
        self.server_locations = []

        actual_host = config.flight_host if config.flight_host != "0.0.0.0" else socket.gethostname()

        for i in range(num_servers):
            # Use port 0 to auto-assign for all servers
            location = f"grpc://{config.flight_host}:0"
            flight_server = MarinFlightServer(location, config)

            # Server starts immediately when created, get the actual port
            actual_port = flight_server.port
            server_location = f"grpc://{actual_host}:{actual_port}"

            self.flight_servers.append(flight_server)
            self.server_locations.append(server_location)

            # Start the server in a background thread
            server_thread = threading.Thread(target=flight_server.serve, daemon=True)
            server_thread.start()
            self._server_threads.append(server_thread)

            logger.info(f"Arrow Flight server {i} started at {server_location}")

        # Register with coordinator (use first server's location for compatibility)
        self.coordinator = get_or_create_actor(ArrowFlightCoordinator, config.coordinator_name)
        first_port = self.flight_servers[0].port
        ray.get(self.coordinator.register_server.remote(actual_host, first_port))

        # Also register all server locations
        ray.get(
            self.coordinator.register_all_servers.remote([(actual_host, server.port) for server in self.flight_servers])
        )

        self.metrics = WeightTransferServerMetrics()

    def serve_weights(self, weight_id: int, model: PyTree) -> None:
        """Serve weights via Arrow Flight using Haliax state_dict serialization.

        Distributes parameters across multiple flight servers for parallel serving.
        """
        self.metrics.total_transfers += 1

        logger.info(f"Serving weights for weight_id {weight_id} via {self.num_servers} Arrow Flight servers...")
        start_time = time.time()
        try:
            barrier_sync()

            if jax.process_index() == 0:
                # Convert model to Arrow RecordBatch per parameter
                params_dict = serialize_pytree_to_arrow(model, weight_id)
                serialize_time = time.time()

                for flight_server in self.flight_servers:
                    flight_server.store_weights(weight_id, params_dict)

                store_time = time.time()

                # Update coordinator with param names
                param_names = list(params_dict.keys())
                ray.get(self.coordinator.update_weight_id.remote(weight_id, param_names))
                update_time = time.time()

                self.metrics.successful_transfers += 1

            barrier_sync()
            logger.info(
                f"Served weights for weight_id {weight_id}. serialize={serialize_time - start_time:.2f}s, "
                f"store={store_time - serialize_time:.2f}s, "
                f"update={update_time - store_time:.2f}s"
            )

        except Exception as e:
            self.metrics.failed_transfers += 1
            logger.error(f"Failed to serve weights {weight_id} via Arrow Flight: {e}")
            raise

    def cleanup(self) -> None:
        """Cleanup Flight server resources."""
        for i, flight_server in enumerate(self.flight_servers):
            try:
                flight_server.shutdown()
            except Exception as e:
                logger.warning(f"Error during Arrow Flight server {i} cleanup: {e}")

    def get_metrics(self) -> WeightTransferServerMetrics:
        """Get transfer metrics."""
        return self.metrics


class ArrowFlightClient(WeightTransferClient):
    """Arrow Flight-based weight transfer client for Haliax/Equinox models.

    Uses Haliax state_dict for proper deserialization of model parameters.
    """

    _coordinator: ray.actor.ActorHandle
    _receive_pool: ThreadPoolExecutor

    def __init__(
        self, config: WeightTransferConfig, mesh: Mesh | None = None, axis_mapping: ResourceMapping | None = None
    ):
        self.config = config
        self.mesh = mesh
        self.axis_mapping = axis_mapping

        # Get coordinator
        self._coordinator = get_or_create_actor(ArrowFlightCoordinator, config.coordinator_name)

        self.last_weight_id = None
        self.flight_clients: list[flight.FlightClient] = []
        self.server_locations: list[str] = []

        self.metrics = WeightTransferClientMetrics()
        self._receive_pool = ThreadPoolExecutor(max_workers=16)

    def _connect_to_servers(self) -> bool:
        """Connect to all Arrow Flight servers."""
        try:
            # Get all server locations
            new_locations = ray.get(self._coordinator.get_all_server_locations.remote())

            if not new_locations:
                # Fall back to single server
                server_location, _ = ray.get(self._coordinator.get_server_info.remote())
                if server_location is None:
                    return False
                new_locations = [server_location]

            # Connect to new servers
            if set(new_locations) != set(self.server_locations):
                # Close old clients
                for client in self.flight_clients:
                    client.close()
                self.flight_clients.clear()

                # Create new clients
                for loc in new_locations:
                    self.flight_clients.append(
                        flight.FlightClient(loc, generic_options=[("grpc.per_message_compression", "0")])
                    )
                    logger.info(f"Connected to Arrow Flight server at {loc}")

                self.server_locations = new_locations

            return True

        except Exception:
            logger.warning("Failed to connect to Arrow Flight servers.", exc_info=True)
            return False

    def _fetch_param(self, weight_id: int, param_name: str) -> tuple[str, jax.Array]:
        """Fetch a single parameter from any available server."""
        ticket_str = f"{weight_id}/{param_name}"
        ticket = flight.Ticket(ticket_str.encode("utf-8"))

        read_options = pa.ipc.IpcReadOptions(
            ensure_alignment=pa.ipc.Alignment.DataTypeSpecific, use_threads=False, ensure_native_endian=False
        )
        call_options = pa.flight.FlightCallOptions(read_options=read_options)

        server_id = hash(param_name) % len(self.flight_clients)
        reader = self.flight_clients[server_id].do_get(ticket, options=call_options).to_reader()
        param_array = deserialize_arrow_to_pytree(param_name, reader, None)
        return param_name, param_array

    def receive_weights(self, old_model: PyTree = None) -> PyTree | None:
        """Receive weights from Arrow Flight servers in parallel.

        Args:
            old_model: Template model to preserve structure. Required for proper deserialization.
        """
        self.metrics.total_polls += 1

        if old_model is None:
            raise ValueError("old_model is required for Arrow Flight weight transfer to preserve model structure")

        try:
            # Connect to servers if needed
            if not self._connect_to_servers():
                return None

            start_time = time.time()
            # Get latest weight info
            _server_location, latest_weight_id = ray.get(self._coordinator.get_server_info.remote())

            if latest_weight_id is None or latest_weight_id == self.last_weight_id:
                return None

            # Get available params for this weight_id
            param_names = ray.get(self._coordinator.get_available_params.remote(latest_weight_id))
            if param_names is None:
                logger.warning(f"No params available for weight_id {latest_weight_id}")
                return None

            poll_time = time.time()

            # Fetch all params in parallel
            state_dict = {}
            futures = {
                self._receive_pool.submit(self._fetch_param, latest_weight_id, param_name): param_name
                for param_name in param_names
            }

            for future in as_completed(futures):
                param_name, param_array = future.result()
                state_dict[param_name] = param_array

            fetch_time = time.time()

            # Convert back to model using state_dict and move to target device
            logger.info(f"Mesh {self.mesh} Axis mapping {self.axis_mapping}")
            with self.mesh, hax.axis_mapping(self.axis_mapping):
                model = update_model(old_model, state_dict)

            decode_time = time.time()

            self.metrics.successful_receives += 1
            self.metrics.poll_time = poll_time - start_time
            self.metrics.fetch_time = fetch_time - poll_time
            self.metrics.decode_time = decode_time - fetch_time
            self.last_weight_id = latest_weight_id

            logger.info(
                f"Received {len(param_names)} params for weight_id {latest_weight_id} via Arrow Flight "
                f"(poll={poll_time - start_time:.2f}s, fetch={fetch_time - poll_time:.2f}s, "
                f"decode={decode_time - fetch_time:.2f}s)"
            )

            return model

        except Exception:
            self.metrics.failed_receives += 1
            logger.error("Failed to receive weights via Arrow Flight", exc_info=True)
            return None

    def cleanup(self) -> None:
        """Cleanup Flight client resources."""
        self._receive_pool.shutdown(wait=False)
        for client in self.flight_clients:
            try:
                client.close()
            except Exception as e:
                logger.warning(f"Error during Arrow Flight client cleanup: {e}")

    def get_metrics(self) -> dict:
        return dataclasses.asdict(self.metrics)
