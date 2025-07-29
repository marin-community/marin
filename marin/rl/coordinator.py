"""
Coordinator for managing the training process.
"""

import asyncio
import logging
import socket
import time
import uuid
from dataclasses import dataclass

import jax
import jax.experimental.transfer as jax_transfer
import ray
import ray.runtime_context
from jaxtyping import PyTree
from ray.actor import ActorHandle
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class WeightTransfer:
    address: str  # address of the jax transfer server
    transfer_uuid: int
    weight_id: int
    time_start: float  # in epoch seconds

    def do_transfer(self, client_transfer_server: jax_transfer.TransferServer, placeholder: PyTree):
        # TODO: JAX doesn't expose any kind of timeout mechanism
        connection = client_transfer_server.connect(self.address)
        return connection.pull(self.transfer_uuid, placeholder)


@dataclass(frozen=True)
class WeightTransferMetadata:
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
    transfer_ready_future: asyncio.Future[WeightTransfer]


@dataclass(frozen=True)
class _EnqueuedWeightTransferRequest:
    uuid: int
    time_start: float  # in epoch seconds


@ray.remote
class WeightTransferCoordinator:
    """
    A WeightTransferCoordinator that lives on a particular TPU training node.

    Note we have to be careful here. We can't actually start the transfer server inside this class because
    the JAX transfer server needs to be started in the actual JAX training process, not in the Ray actor.

    So this class just exists to coordinate weight transfers.
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

    def latest_weight_id(self) -> int | None:
        """
        Returns the latest weight ID that has been transferred.
        """
        return self._latest_weight_id

    async def schedule_weight_transfer(self) -> WeightTransfer:
        """
        Requests a weight transfer from the coordinator.
        Blocks until the underlying weight transfer has picked up the request.
        """
        request = _WeightTransferRequest(
            uuid=uuid.uuid4().int,
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
                transfer = WeightTransfer(
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

    async def await_transfers(self, transfer_uuids: list[int]):
        """Blocks until all specified transfers are complete."""
        if not transfer_uuids:
            return

        async with self._lock:
            this_pending_completions = [self._pending_completion.pop(uuid) for uuid in transfer_uuids]

        return await asyncio.gather(*this_pending_completions)


async def process_weight_transfers(
    transfer_server: jax_transfer.TransferServer, coordinator: ActorHandle, latest_weight_id: int, latest_weights: PyTree
):
    """
    Processes weight transfers for the given latest weight ID.
    This blocks until all transfers are complete.
    """
    enqueued_requests = await coordinator.poll_transfers.remote(latest_weight_id)

    if enqueued_requests:
        uuids_to_wait_for = [req.uuid for req in enqueued_requests]
        for request in enqueued_requests:
            transfer_server.await_pull(request.uuid, latest_weights)

        await coordinator.await_transfers.remote(uuids_to_wait_for)

    return latest_weight_id


async def receive_weight_transfers(
    coordinator: ActorHandle,
    client_server: jax_transfer.TransferServer,
    placeholder: PyTree,
) -> tuple[PyTree, WeightTransferMetadata]:
    """
    Asks the coordinator to schedule a weight transfer for this client, and blocks until the transfer is complete.
    """
    transfer_info: WeightTransfer = await coordinator.schedule_weight_transfer.remote()
    this_size = sum(jax.tree_util.tree_map(lambda x: x.nbytes if isinstance(x, jax.Array) else 0, placeholder))
    out = client_server.await_pull(transfer_info.transfer_uuid, placeholder)
    out = jax.block_until_ready(out)

    return out, WeightTransferMetadata(
        weight_id=transfer_info.weight_id,
        weight_bytes=this_size,
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

    return WeightTransferCoordinator.options(**options).remote(server.address())


def get_local_ip_from_hostname():
    try:
        hostname = socket.gethostname()
        ip_address = socket.gethostbyname(hostname)
        return ip_address
    except socket.gaierror:
        return "Could not resolve hostname to IP address."


def start_transfer_server():
    ip = get_local_ip_from_hostname()
    return jax_transfer.start_transfer_server(
        jax.devices()[0].client,
        f"{ip}:0",  # bind to the local IP address
        [f"{ip}:0"] * jax.device_count(),
    )


def this_node_affinity_strategy(soft: bool = False) -> NodeAffinitySchedulingStrategy:
    """
    Returns a NodeAffinitySchedulingStrategy that will only schedule weight transfers to the current node.
    """
    return NodeAffinitySchedulingStrategy(node_id=ray.runtime_context.get_runtime_context().get_node_id(), soft=soft)
