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

Currently all parameters are replicated to worker 0 and served from their for simplicity.
For performance, we start multiple Flight servers on the worker and load balance client requests
based on the parameter name hash.

This gets us to about ~7GB/s transfer on a TPUv5-4 VM (running the server & client on the same VM).
We can likely extract a bit more performance by:

* Batching smaller parameters to avoid tiny requests
* Tweaking gRPC settings for larger message sizes, compression, etc.
"""

import dataclasses
import logging
import socket
import threading
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
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
    WeightUpdate,
    get_or_create_actor,
)

logger = logging.getLogger(__name__)


@dataclass
class ServerInfo:
    """Information about available weights and servers."""

    weight_id: int | None
    server_addresses: list[str]
    param_names: list[str]


# The maximum number of array elements in a single Arrow RecordBatch.
# Larger arrays are chunked into multiple RecordBatches to avoid hitting Arrow limits.
# We assume our largest dtype is 4 bytes (e.g., float32/int32).
MAX_ELEMENTS_PER_RECORD = (2000 * 1000 * 1000) // 4

# Thread pool configuration for parallel serving and fetching
NUM_PARALLEL_SERVERS = 16


def _create_binary_array(buffer_data: np.ndarray) -> pa.Array:
    """Construct a single element Arrow LargeBinary array from numpy buffer data without copies"""
    block = pa.py_buffer(buffer_data)
    return pa.Array.from_buffers(
        pa.large_binary(),
        1,  # length
        [None, pa.array([0, len(block)], type=pa.int64()).buffers()[1], block],
    )


def state_dict_to_batches(
    state_dict: dict[str, np.ndarray], weight_id: int
) -> dict[str, tuple[pa.Schema, Sequence[pa.RecordBatch]]]:
    """Convert state_dict to Arrow RecordBatch per parameter using Haliax state_dict for efficient transfer.

    Large arrays are split into multiple RecordBatches if needed to avoid hitting the Arrow
    2GB limit.

    Returns:
        Dict mapping param_name -> (schema, batches) for per-parameter flights
    """

    result = {}
    sz = 0

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
        },
    )

    for name, value in state_dict.items():
        shape = value.shape
        is_scalar = len(shape) == 0
        dtype = value.dtype
        sz += value.nbytes

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


def deserialize_arrow_to_pytree(param_name: str, reader: pa.RecordBatchReader) -> jax.Array:
    """Convert Arrow RecordBatch back to a single parameter array.

    Args:
        param_name: Name of the parameter being deserialized
        reader: Arrow RecordBatch reader containing the parameter data

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
            logger.debug(f"Deserialized param {param_name} of shape {shape} and dtype {dtype} in {ed - st:.2f}s")
        return res


@ray.remote(num_cpus=0)
class ArrowFlightCoordinator:
    """Ray actor for coordinating Arrow Flight weight transfers."""

    _server_info = ServerInfo | None

    def __init__(self):
        self._server_info = None

    def update_server(self, weight_id: int, param_names: list[str], server_locations: list[tuple[str, int]]) -> None:
        self._server_info = ServerInfo(
            weight_id=weight_id,
            server_addresses=[f"grpc://{host}:{port}" for host, port in server_locations],
            param_names=param_names,
        )
        logger.info(f"Updated server: weight_id={weight_id}, params={len(param_names)}, servers={len(server_locations)}")
        return 123

    def fetch_server(self) -> ServerInfo:
        return self._server_info


class MarinFlightServer(flight.FlightServerBase):
    """Arrow Flight server for serving model weights."""

    config: WeightTransferConfig
    _weights_store: dict[int, dict[str, tuple[pa.Schema, Sequence[pa.RecordBatch]]]]
    _latest_weight_id: int | None
    _lock: threading.Lock
    _location: str

    def __init__(self, location: str, config: WeightTransferConfig):
        super().__init__(location)
        self.config = config
        self._weights_store = {}
        self._latest_weight_id = None
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
                if weight_id != self._latest_weight_id:
                    logger.debug(f"Requested weight_id {weight_id} stale, returning {self._latest_weight_id}")
                    weight_id = self._latest_weight_id

                (schema, batches) = self._weights_store[weight_id][param_name]

            return flight.RecordBatchStream(pa.RecordBatchReader.from_batches(schema, batches))

        except Exception as e:
            logger.error(f"Error in do_get: {e}")
            raise flight.FlightInternalError(f"Failed to get weights: {e}") from e

    def list_flights(self, context, criteria):
        """List available weight transfers."""
        with self._lock:
            for weight_id, params_dict in self._weights_store.items():
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
        with self._lock:
            # remove all other weights
            self._weights_store.clear()
            self._weights_store[weight_id] = params_dict
            self._latest_weight_id = weight_id

    def get_latest_weight_id(self) -> int | None:
        """Get the latest weight ID."""
        with self._lock:
            return self._latest_weight_id


class ArrowFlightServer(WeightTransferServer):
    """Arrow Flight-based weight transfer server for Haliax/Equinox models.

    Uses Haliax state_dict for proper serialization of model parameters.
    Spawns multiple flight server instances for parallel serving.

    Threading model: Each flight server runs in its own daemon thread via serve().
    """

    config: WeightTransferConfig
    mesh: Mesh | None
    axis_mapping: ResourceMapping | None
    num_servers: int
    _flight_servers: list[MarinFlightServer]
    _server_threads: list[threading.Thread]
    _server_locations: list[str]
    coordinator: ray.actor.ActorHandle
    metrics: WeightTransferServerMetrics

    def __init__(
        self,
        config: WeightTransferConfig,
        mesh: Mesh | None = None,
        axis_mapping: ResourceMapping | None = None,
        num_servers: int = NUM_PARALLEL_SERVERS,
    ):
        self.config = config
        self.mesh = mesh
        self.axis_mapping = axis_mapping
        self.num_servers = num_servers

        # Start multiple Flight servers
        self._flight_servers = []
        self._server_threads = []
        self._server_locations = []

        actual_host = config.flight_host if config.flight_host != "0.0.0.0" else socket.gethostname()

        for i in range(num_servers):
            # Use port 0 to auto-assign for all servers
            location = f"grpc://{config.flight_host}:0"
            flight_server = MarinFlightServer(location, config)

            # Server starts immediately when created, get the actual port
            actual_port = flight_server.port
            server_location = f"grpc://{actual_host}:{actual_port}"

            self._flight_servers.append(flight_server)
            self._server_locations.append(server_location)

            # Start the server in a background thread
            server_thread = threading.Thread(target=flight_server.serve, daemon=True)
            server_thread.start()
            self._server_threads.append(server_thread)

            logger.info(f"Arrow Flight server {i} started at {server_location}")

        # Get coordinator
        self.coordinator = get_or_create_actor(ArrowFlightCoordinator, config.coordinator_name)

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
                # Fetching the entire state dict to CPU allows JAX to parallelize the individual transfers
                state_dict = hsd.to_state_dict(model)
                state_dict = jax.device_get(state_dict)
                copy_time = time.time()

                # Convert to Arrow RecordBatch per parameter
                params_dict = state_dict_to_batches(state_dict, weight_id)
                serialize_time = time.time()

                for flight_server in self._flight_servers:
                    flight_server.store_weights(weight_id, params_dict)

                store_time = time.time()

                # Update coordinator with weight info and server locations
                param_names = list(params_dict.keys())
                actual_host = self.config.flight_host if self.config.flight_host != "0.0.0.0" else socket.gethostname()
                server_locations = [(actual_host, server.port) for server in self._flight_servers]
                ray.get(self.coordinator.update_server.remote(weight_id, param_names, server_locations))
                update_time = time.time()

                self.metrics.successful_transfers += 1

                logger.info(
                    "Served weights for weight_id %s, timings: copy=%.2fs, serialize=%.2fs, store=%.2fs, update=%.2fs",
                    weight_id,
                    copy_time - start_time,
                    serialize_time - copy_time,
                    store_time - serialize_time,
                    update_time - store_time,
                )

            barrier_sync()

        except Exception as e:
            self.metrics.failed_transfers += 1
            logger.error(f"Failed to serve weights {weight_id} via Arrow Flight: {e}")
            raise

    def cleanup(self) -> None:
        """Cleanup Flight server resources."""
        # shutdown servers in parallel in threads to avoid blocking on shutdown
        for flight_server in self._flight_servers:
            logger.info(f"Shutting down Arrow Flight server at {flight_server._location}...")
            threading.Thread(target=flight_server.shutdown, daemon=True).start()

    def get_metrics(self) -> WeightTransferServerMetrics:
        """Get transfer metrics."""
        return self.metrics


class ArrowFlightClient(WeightTransferClient):
    """Arrow Flight-based weight transfer client for Haliax/Equinox models."""

    config: WeightTransferConfig
    mesh: Mesh | None
    axis_mapping: ResourceMapping | None
    _last_weight_id: int | None
    _flight_clients: list[flight.FlightClient]
    _server_locations: list[str]
    metrics: WeightTransferClientMetrics
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

        self._last_weight_id = None
        self._flight_clients = []
        self._server_locations = []

        self.metrics = WeightTransferClientMetrics()
        self._receive_pool = ThreadPoolExecutor(max_workers=NUM_PARALLEL_SERVERS)

    def _connect_to_servers(self, new_locations) -> bool:
        """Connect to all Arrow Flight servers."""
        try:
            # Connect to new servers
            if set(new_locations) != set(self._server_locations):
                # Close old clients
                for client in self._flight_clients:
                    client.close()
                self._flight_clients.clear()

                # Create new clients
                for loc in new_locations:
                    self._flight_clients.append(
                        flight.FlightClient(loc, generic_options=[("grpc.per_message_compression", 0)])
                    )
                    logger.debug(f"Connected to Arrow Flight server at {loc}")

                self._server_locations = new_locations

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

        server_id = hash(param_name) % len(self._flight_clients)
        reader = self._flight_clients[server_id].do_get(ticket, options=call_options).to_reader()
        param_array = deserialize_arrow_to_pytree(param_name, reader)
        return param_name, param_array

    def receive_weights(self, old_model: PyTree = None) -> WeightUpdate | None:
        """Receive weights from Arrow Flight servers in parallel.

        Args:
            old_model: Template model to preserve structure. Required for proper deserialization.
        """
        self.metrics.total_polls += 1

        if old_model is None:
            raise ValueError("old_model is required for Arrow Flight weight transfer to preserve model structure")

        try:
            start_time = time.time()

            # Fetch server info from coordinator
            server_info = ray.get(self._coordinator.fetch_server.remote())

            if not server_info:
                logger.info("No Arrow Flight server info available from coordinator.")
                return None

            if server_info.weight_id is None or server_info.weight_id == self._last_weight_id:
                logger.info("No new weights available from Arrow Flight server.")
                return None

            # Connect to servers if needed
            if not self._connect_to_servers(server_info.server_addresses):
                logger.info("Failed to connect to Arrow Flight servers.")
                return None

            poll_time = time.time()

            state_dict = {}
            futures = {
                self._receive_pool.submit(self._fetch_param, server_info.weight_id, param_name): param_name
                for param_name in server_info.param_names
            }

            for future in as_completed(futures):
                param_name, param_array = future.result()
                state_dict[param_name] = param_array

            fetch_time = time.time()

            # Convert back to model using state_dict and move to target device
            with self.mesh, hax.axis_mapping(self.axis_mapping):
                model = update_model(old_model, state_dict)

            decode_time = time.time()

            self.metrics.successful_receives += 1
            self.metrics.poll_time = poll_time - start_time
            self.metrics.fetch_time = fetch_time - poll_time
            self.metrics.decode_time = decode_time - fetch_time
            self._last_weight_id = server_info.weight_id

            logger.info(
                f"Received {len(server_info.param_names)} params for weight_id {server_info.weight_id} via Arrow Flight "
                f"(poll={poll_time - start_time:.2f}s, fetch={fetch_time - poll_time:.2f}s, "
                f"decode={decode_time - fetch_time:.2f}s)"
            )

            return WeightUpdate(model=model, weight_id=server_info.weight_id)

        except Exception:
            self.metrics.failed_receives += 1
            logger.error("Failed to receive weights via Arrow Flight", exc_info=True)
            return None

    def cleanup(self) -> None:
        """Cleanup Flight client resources."""
        try:
            logger.info("Shutting down Arrow Flight client thread pool...")
            self._receive_pool.shutdown(wait=False, cancel_futures=False)
            logger.info("Thread pool shutdown completed")
        except Exception as e:
            logger.warning(f"Error shutting down thread pool: {e}")

        logger.info("Closing Arrow Flight client connection...")
        for client in self._flight_clients:
            try:
                client.close()
            except Exception as e:
                logger.warning(f"Error during Arrow Flight client cleanup: {e}")

    def get_metrics(self) -> dict:
        return dataclasses.asdict(self.metrics)
