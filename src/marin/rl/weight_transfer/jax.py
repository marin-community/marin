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
JAX transfer server-based weight transfer implementation.

This module provides weight transfer using JAX's native transfer server for
high-performance communication between training and inference workers.
"""

import asyncio
import dataclasses
import logging
import queue
import socket
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import jax
import jax.experimental.transfer as jax_transfer
import numpy as np
import ray
import ray.runtime_context
from haliax.jax_utils import is_jax_array_like
from jax.sharding import Mesh
from jaxtyping import PyTree
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

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


@dataclass(frozen=True)
class WeightTransferSpec:
    """
    Specifies to a client how to pull a weight from a server.
    """

    address: str  # address of the jax transfer server
    transfer_uuid: int
    weight_id: int
    time_start: float  # in epoch seconds


@dataclass(frozen=True)
class WeightTransferMetadata:
    """
    Metadata about a weight transfer, including the weight ID, size in bytes, and start/end times.
    """

    weight_id: int
    weight_bytes: int
    time_start: float  # in epoch seconds
    time_end: float  # in epoch seconds

    @property
    def time_elapsed(self) -> float:
        return self.time_end - self.time_start


@dataclass(frozen=True)
class _WeightTransferRequest:
    uuid: int
    time_start: float  # in epoch seconds
    transfer_ready_future: asyncio.Future[WeightTransferSpec]


@dataclass(frozen=True)
class _EnqueuedWeightTransferRequest:
    uuid: int
    time_start: float  # in epoch seconds


# -- Actual Coordinator --


@ray.remote
class WeightTransferCoordinator:
    """
    WeightTransferCoordinator is a Ray actor that coordinates weight transfers between a server and clients.

    Note we have to be careful here. We can't actually start the transfer server inside this class because
    the JAX transfer server needs to be started in the actual JAX training process, not in the Ray actor.
    So this class just exists to coordinate weight transfers.

    The actual weight transfers are handled by JAX's transfer server.
    """

    _requested_transfers: list[_WeightTransferRequest]
    _latest_weight_id: int | None
    _pending_completion: dict[int, asyncio.Event]  # transfer_uuid -> event

    def __init__(self):
        self.transfer_server_address = None  # Actual JAX transfer server address
        self._requested_transfers = []
        self._lock = asyncio.Lock()
        self._latest_weight_id = None
        self._pending_completion = {}
        self._transfer_id = 0

    def latest_weight_id(self) -> int | None:
        """
        Returns the latest weight ID that has been transferred.
        """
        return self._latest_weight_id

    def get_transfer_info(self) -> tuple[int | None, str | None]:
        """
        Returns the latest weight ID and transfer server address without blocking.
        Returns (None, None) if no weights are available or server not registered.
        """
        return self._latest_weight_id, self.transfer_server_address

    def register_transfer_server(self, transfer_server_address: str):
        """
        Register the actual JAX transfer server address with the coordinator.
        Called by the server when it starts up.
        """
        self.transfer_server_address = transfer_server_address

    async def schedule_weight_transfer(self) -> WeightTransferSpec:
        """
        Requests a weight transfer from the coordinator.
        Blocks until the underlying weight transfer has picked up the request.
        """
        transfer_id = self._transfer_id
        self._transfer_id += 1

        request = _WeightTransferRequest(
            # can't actually be a full uuid because they want a 32-bit int. Could take 32 bits of uuid, but
            # there won't be collisions with a central coordinator...
            uuid=transfer_id,
            time_start=time.time(),
            transfer_ready_future=asyncio.Future(),
        )
        async with self._lock:
            self._requested_transfers.append(request)

        return await request.transfer_ready_future

    async def poll_transfers(self, latest_weight_id: int) -> list[_EnqueuedWeightTransferRequest]:
        """Called by the training process to poll for weight transfers."""
        self._latest_weight_id = latest_weight_id
        async with self._lock:
            requests = self._requested_transfers
            self._requested_transfers = []

            out: list[_EnqueuedWeightTransferRequest] = []
            for request in requests:
                if self.transfer_server_address is None:
                    raise RuntimeError(
                        "Transfer server address not registered. Server must call register_transfer_server() first."
                    )

                transfer = WeightTransferSpec(
                    address=self.transfer_server_address,
                    transfer_uuid=request.uuid,
                    weight_id=latest_weight_id,
                    time_start=request.time_start,
                )
                request.transfer_ready_future.set_result(transfer)
                out.append(
                    _EnqueuedWeightTransferRequest(
                        uuid=request.uuid,
                        time_start=request.time_start,
                    )
                )
                event = asyncio.Event()
                self._pending_completion[request.uuid] = event

        return out

    async def report_transfer_finished(self, transfer_uuid: int):
        """Called by clients to report that they have finished a transfer."""
        async with self._lock:
            if transfer_uuid in self._pending_completion:
                self._pending_completion[transfer_uuid].set()
                logger.info("Transfer %d finished", transfer_uuid)
            else:
                raise ValueError(f"Transfer {transfer_uuid} not found")

    async def await_transfers(self, transfer_uuids: list[int]):
        """Blocks until all specified transfers are complete. Called by the server."""
        if not transfer_uuids:
            return

        async with self._lock:
            pending_events = [self._pending_completion[uuid] for uuid in transfer_uuids]

        logger.info("Awaiting %d transfers", len(pending_events))

        out = await asyncio.gather(*(event.wait() for event in pending_events))
        logger.info("Awaited %d transfers", len(out))

        async with self._lock:
            for uuid in transfer_uuids:
                self._pending_completion.pop(uuid)

        return out


# -- Various Public Functions --


async def process_weight_transfers(
    transfer_server: jax_transfer.TransferServer,
    coordinator: ActorHandle,
    latest_weight_id: int,
    latest_weights: PyTree,
):
    """
    For the *server* to call. Polls for weight transfers and blocks until they are complete.
    Returns the number of transfers that were enqueued.

    This is an async function, so you can run it in the background.

    Returns the number of transfers that were enqueued.
    """

    # TODO: JAX doesn't expose any kind of timeout mechanism so we'll just block forever??!?
    enqueued_requests = await coordinator.poll_transfers.remote(latest_weight_id)  # type: ignore

    e: Exception | None = None
    failed_uuids: list[int] = []

    weight_bytes = num_bytes(latest_weights)
    # 4GBps is the best we get on v4, add a comfortable margin on top of that
    timeout = 30 + (weight_bytes / (4 * 1024 * 1024 * 1024 / 2))  # in seconds

    if enqueued_requests:
        uuids_to_wait_for = [req.uuid for req in enqueued_requests]
        for request in enqueued_requests:
            try:
                transfer_server.await_pull(request.uuid, latest_weights)
            except Exception as exc:
                logger.exception("Error waiting for transfer %d", request.uuid)
                failed_uuids.append(request.uuid)
                e = exc

        # TODO: need to handle failed transfers.
        # JAX doesn't provide any mechanism to handling cancellation :(
        transfers_finished = coordinator.await_transfers.remote(uuids_to_wait_for)  # type: ignore
        try:
            await asyncio.wait_for(transfers_finished, timeout=timeout)
        except asyncio.TimeoutError:
            logger.warning(
                "Timed out waiting for transfers %s after %.2f seconds. "
                "This may indicate a slow network or large weights.",
                uuids_to_wait_for,
                timeout,
            )
            e = asyncio.TimeoutError(f"Timed out waiting for transfers {uuids_to_wait_for} after {timeout:.2f} seconds")
        except Exception as exc:
            logger.exception("Error waiting for transfers %s", uuids_to_wait_for)
            e = exc

        if e:
            raise Exception(f"Failed to wait for transfers {failed_uuids}") from e

    return len(enqueued_requests)


async def receive_weight_transfers(
    coordinator: ActorHandle,
    client_server: jax_transfer.TransferServer,
    placeholder: PyTree,
) -> tuple[PyTree, WeightTransferMetadata]:
    """
    Asks the coordinator to schedule a weight transfer for this client, and blocks until the transfer is complete.
    """
    if placeholder is None:
        raise ValueError("Placeholder model cannot be None.")

    transfer_info: WeightTransferSpec = await coordinator.schedule_weight_transfer.remote()  # type: ignore
    total_bytes = num_bytes(placeholder)

    try:
        connection = client_server.connect(transfer_info.address)
        out = connection.pull(transfer_info.transfer_uuid, placeholder)
        if out is None:
            raise RuntimeError(f"Transfer failed: received None from server at {transfer_info.address}")

        # TODO: this should be pushed into a thread to avoid blocking the event loop
        out = jax.block_until_ready(out)
    except Exception as e:
        await coordinator.report_transfer_finished.remote(transfer_info.transfer_uuid)  # type: ignore
        raise RuntimeError(f"JAX transfer failed from {transfer_info.address}: {e}") from e

    await coordinator.report_transfer_finished.remote(transfer_info.transfer_uuid)  # type: ignore

    return out, WeightTransferMetadata(
        weight_id=transfer_info.weight_id,
        weight_bytes=total_bytes,
        time_start=transfer_info.time_start,
        time_end=time.time(),
    )


def start_transfer_server() -> jax_transfer.TransferServer:
    """Start JAX transfer server."""
    ip = get_local_ip_from_hostname()
    backend_client = jax.devices()[0].client

    # Use random port binding for proper network resource management
    server = jax_transfer.start_transfer_server(
        backend_client,
        f"{ip}:0",  # Random port binding
        [f"{ip}:0"] * jax.device_count(),
    )
    return server


# -- Helpers --


def get_local_ip_from_hostname():
    hostname = socket.gethostname()
    ip_address = socket.gethostbyname(hostname)
    return ip_address


def this_node_affinity_strategy(soft: bool = False) -> NodeAffinitySchedulingStrategy:
    """
    Returns a NodeAffinitySchedulingStrategy that will only schedule weight transfers to the current node.
    """
    return NodeAffinitySchedulingStrategy(node_id=ray.runtime_context.get_runtime_context().get_node_id(), soft=soft)


def num_bytes(model: PyTree):
    # especially with jax.vjp, we get duplicate arrays and want to uniq them
    # NB we need to use object identity here, mostly because of ShapedDtypeStruct
    leaves = {id(x): x for x in jax.tree_util.tree_leaves(model) if is_jax_array_like(x)}
    return sum(x.nbytes for x in leaves.values())


class JAXTransferServer(WeightTransferServer):
    """JAX transfer server-based weight transfer server."""

    coordinator: WeightTransferCoordinator

    def __init__(self, config: WeightTransferConfig, mesh, params_sharding_rules=None):
        self.config = config
        self.mesh = mesh
        self.params_sharding_rules = params_sharding_rules

        self.coordinator = get_or_create_actor(WeightTransferCoordinator, config.coordinator_name)

        # Start transfer server and register its address with coordinator
        self.transfer_server = start_transfer_server()
        self.coordinator.register_transfer_server.remote(self.transfer_server.address())
        self._setup_cpu_transfer()

        # Single-item queue for polling
        self.poll_queue = queue.Queue(maxsize=1)
        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="weight_transfer")

        self.metrics = WeightTransferServerMetrics()

    def _setup_cpu_transfer(self):
        """Setup CPU mesh for transfers."""
        try:
            cpu_devices = jax.devices("cpu")
            if cpu_devices:
                self.cpu_mesh = Mesh(np.array(cpu_devices[:1]), axis_names=("cpu",))
                logger.info(f"Setup CPU mesh with {len(cpu_devices)} devices")
            else:
                raise RuntimeError("No CPU devices found, cannot perform weight transfer")
        except Exception as e:
            raise RuntimeError(f"Failed to setup CPU mesh: {e}") from e

    def _transfer_to_cpu(self, model) -> PyTree:
        """Transfer params to CPU devices."""
        try:
            with self.cpu_mesh:
                cpu_devices = jax.devices("cpu")
                return jax.device_put(model, cpu_devices[0])
        except Exception as e:
            logger.warning(f"Failed to transfer to CPU: {e}, using original params")
            return model

    def serve_weights(self, weight_id: int, model) -> None:
        """Serve weights with CPU transfer and threading."""
        self.metrics.total_transfers += 1

        def _serve_in_thread():
            try:
                cpu_params = self._transfer_to_cpu(model)
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    result = loop.run_until_complete(
                        process_weight_transfers(self.transfer_server, self.coordinator, weight_id, cpu_params)
                    )
                    logger.info(f"Processed {result} weight transfers for weight_id {weight_id}")

                    # Update metrics
                    self.metrics.successful_transfers += 1

                    return result
                finally:
                    loop.close()
            except Exception as e:
                logger.error(f"Error serving weights {weight_id}: {e}")
                self.metrics.failed_transfers += 1
                return 0

        # Use single-item queue - drop old requests if training runs ahead
        try:
            self.poll_queue.put_nowait((weight_id, model))
        except queue.Full:
            old_item = self.poll_queue.get_nowait()
            logger.info(f"Dropping old weight transfer {old_item[0]} for new {weight_id}")
            self.poll_queue.put_nowait((weight_id, model))

        _ = self.executor.submit(_serve_in_thread)

    def cleanup(self) -> None:
        """Cleanup transfer server and thread pool."""
        logger.info("Cleaning up JAX transfer server")
        self.executor.shutdown(wait=True)
        self.transfer_server = None

    def get_metrics(self) -> dict:
        return dataclasses.asdict(self.metrics)


class JAXTransferClient(WeightTransferClient):
    """JAX transfer server-based weight transfer client."""

    def __init__(self, config: WeightTransferConfig, mesh, params_sharding_rules=None):
        self.config = config
        self.mesh = mesh
        self.params_sharding_rules = params_sharding_rules

        self.coordinator = get_or_create_actor(WeightTransferCoordinator, config.coordinator_name)

        # Start transfer server for client (doesn't register address with coordinator)
        self.transfer_server = start_transfer_server()
        self._setup_cpu_transfer()

        self.executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="weight_transfer")

        self._last_received_weight_id: int = -1

        # Metrics tracking
        self.metrics = WeightTransferClientMetrics()

    def _setup_cpu_transfer(self):
        """Setup CPU mesh for transfers."""
        try:
            cpu_devices = jax.devices("cpu")
            if cpu_devices:
                self.cpu_mesh = Mesh(np.array(cpu_devices[:1]), axis_names=("cpu",))
                logger.info(f"Setup CPU mesh with {len(cpu_devices)} devices")
            else:
                raise RuntimeError("No CPU devices found, cannot perform weight transfer")
        except Exception as e:
            raise RuntimeError(f"Failed to setup CPU mesh: {e}") from e

    def _transfer_from_cpu(self, model) -> PyTree:
        """Transfer params from CPU back to target devices."""
        if self.params_sharding_rules is not None:
            return jax.device_put(model, self.params_sharding_rules)
        else:
            # Use default device placement
            return jax.device_put(model, jax.devices()[0])

    def receive_weights(self, old_model: PyTree) -> WeightUpdate | None:
        """Receive weights with CPU transfer."""
        self.metrics.total_polls += 1

        # First check if new weights are available without blocking
        try:
            latest_weight_id, server_address = ray.get(self.coordinator.get_transfer_info.remote())
            logger.info(
                "Current weight id %s, Latest weight ID: %s, Server address: %s",
                self._last_received_weight_id,
                latest_weight_id,
                server_address,
            )

            # Early exit if no weights available or no new weights
            if latest_weight_id is None or server_address is None:
                return None

            if latest_weight_id <= self._last_received_weight_id:
                return None
        except Exception as e:
            logger.error(f"Failed to check transfer info: {e}")
            self.metrics.failed_receives += 1
            return None

        def _receive_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                logger.info("Receiving weight transfers from server at %s", server_address)
                cpu_params, metadata = loop.run_until_complete(
                    receive_weight_transfers(self.coordinator, self.transfer_server, old_model)
                )

                tpu_params = self._transfer_from_cpu(cpu_params)
                return tpu_params, metadata
            finally:
                loop.close()

        try:
            logger.info("Fetching new weights, current=%s, latest=%s", self._last_received_weight_id, latest_weight_id)
            future = self.executor.submit(_receive_in_thread)
            params, metadata = future.result(timeout=self.config.transfer_timeout)

            if params is not None and metadata is not None:
                # Update metrics and track received weight ID
                self.metrics.successful_receives += 1
                self._last_received_weight_id = metadata.weight_id
                return WeightUpdate(model=params, weight_id=metadata.weight_id)

            return None

        except Exception as e:
            self.metrics.failed_receives += 1
            logger.error(f"Failed to receive weights: {e}")
            return None

    def cleanup(self) -> None:
        """Cleanup transfer server and thread pool."""
        logger.info("Cleaning up JAX transfer client")
        self.executor.shutdown(wait=True)
        self.transfer_server = None

    def get_metrics(self) -> dict:
        return dataclasses.asdict(self.metrics)
