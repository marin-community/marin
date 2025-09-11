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
Coordinator for managing weight transfers during RL. Actual transfers are handled by JAX's transfer server.
This module is responsible for:
- scheduling weight transfers
- polling for weight transfers
- reporting that weight transfers are complete
- waiting for weight transfers to complete
"""

import asyncio
import logging
import socket
import time
from dataclasses import dataclass

import jax
import jax.experimental.transfer as jax_transfer
import ray
import ray.runtime_context
from haliax.jax_utils import is_jax_array_like
from jaxtyping import PyTree
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

logger = logging.getLogger(__name__)

# -- Types --


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

    def __init__(self, address: str):
        self.address = address
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
                transfer = WeightTransferSpec(
                    address=self.address,
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
    transfer_server: jax_transfer.TransferServer, coordinator: ActorHandle, latest_weight_id: int, latest_weights: PyTree
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
    transfer_info: WeightTransferSpec = await coordinator.schedule_weight_transfer.remote()  # type: ignore
    total_bytes = num_bytes(placeholder)

    out = do_transfer(transfer_info, client_server, placeholder)
    # TODO: this should be pushed into a thread to avoid blocking the event loop
    out = jax.block_until_ready(out)

    await coordinator.report_transfer_finished.remote(transfer_info.transfer_uuid)  # type: ignore

    return out, WeightTransferMetadata(
        weight_id=transfer_info.weight_id,
        weight_bytes=total_bytes,
        time_start=transfer_info.time_start,
        time_end=time.time(),
    )


def instantiate_coordinator(server: jax_transfer.TransferServer, name: str | None = None):
    """
    Instantiates the WeightTransferCoordinator on the current node.
    If a name is provided, the actor will be named, allowing it to be looked up with ray.get_actor(name).
    """
    options = {
        "num_cpus": 0,
        "scheduling_strategy": this_node_affinity_strategy(),
    }

    if name:
        options["name"] = name

    return WeightTransferCoordinator.options(**options).remote(server.address())  # type: ignore


def start_transfer_server() -> jax_transfer.TransferServer:
    ip = get_local_ip_from_hostname()
    backend_client = jax.devices()[0].client
    return jax_transfer.start_transfer_server(
        backend_client,
        f"{ip}:0",  # bind to the local IP address
        [f"{ip}:0"] * jax.device_count(),
    )


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


def do_transfer(spec: WeightTransferSpec, client_transfer_server: jax_transfer.TransferServer, placeholder: PyTree):
    # TODO: JAX doesn't expose any kind of timeout mechanism
    connection = client_transfer_server.connect(spec.address)
    return connection.pull(spec.transfer_uuid, placeholder)
