#!/usr/bin/env python
"""Benchmark cross-slice weight transfer using ``jax.experimental.transfer``.

A persistent large TPU slice hosts the weights while small slices
periodically connect to fetch them. The large slice runs a
:class:`~marin.rl.coordinator.WeightTransferCoordinator` to manage an all-gather cycle
and simulate parameter updates.
"""

import argparse
import asyncio
import logging
import time
from collections.abc import Iterable
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import ray
import wandb
from jax._src.mesh_utils import create_device_mesh
from jax.experimental import transfer as jax_transfer
from levanter.infra import ray_tpu
from ray.actor import ActorHandle

from marin.rl.coordinator import (
    instantiate_coordinator,
    process_weight_transfers,
    receive_weight_transfers,
    start_transfer_server,
)

logger = logging.getLogger("ray")


@dataclass
class TransferStats:
    """Simple container for timing results."""

    bytes_transferred: int
    transfer_time: float


@ray.remote
class TransferProgressTracker:
    """Tracks and logs transfer progress using wandb."""

    def __init__(self, num_transfers: int, **wandb_args) -> None:
        self.coordinator: ActorHandle | None = None
        wandb.init(**wandb_args)
        self._total_transfers = num_transfers
        self._num_finished = 0

    def set_coordinator(self, coordinator: ActorHandle) -> None:
        """Stores the coordinator actor handle."""
        self.coordinator = coordinator

    def get_coordinator(self) -> ActorHandle | None:
        """Returns the coordinator actor handle."""
        return self.coordinator

    def mark_transfer_finished(self, step: int) -> None:
        """Mark a transfer as finished for logging purposes."""
        if step >= self._total_transfers:
            raise ValueError(f"Step {step} exceeds total transfers {self._total_transfers}.")

        self._num_finished += 1
        wandb.log(
            {
                "manager/finished_transfers": self._num_finished,
            }
        )

    def log(self, metrics):
        wandb.log(metrics)

    def finish(self):
        wandb.finish()

    def is_finished(self) -> bool:
        """Check if all transfers are finished."""
        return self._num_finished >= self._total_transfers

    def num_finished(self) -> int:
        return self._num_finished


full_sharded = jax.sharding.PartitionSpec(
    ("p", "d"),
)
host_replicated = jax.sharding.PartitionSpec(
    ("d",),
)


def make_process_mesh():
    """Create a process mesh for the current TPU slice."""
    # num_devices = jax.local_device_count()
    # return jax.make_mesh((jax.process_count(), num_devices,), ("p", "d",))
    # from jax.experimental.mesh_utils import create_device_mesh

    # jax.experimental
    device_array = create_device_mesh(
        (jax.process_count(), jax.local_device_count()),
        jax.devices(),
    )

    return jax.sharding.Mesh(device_array, ("p", "d"))


@jax.jit
def deshard(arr):
    """Deshard the array to a single device."""
    return jax.lax.with_sharding_constraint(arr, host_replicated)


def large_slice_loop(
    size: int,
    rounds: int,
    tracker: ActorHandle,
    receiver_count: int,
) -> None:
    """Run on the large slice; serves weights and simulates updates."""
    mesh = make_process_mesh()
    with jax.sharding.use_mesh(mesh):
        logger.info("Large slice started with %d devices", jax.device_count())

        @jax.jit
        def make_arr(rng_key):
            """Create a random array of the given size on the mesh."""
            out = jax.random.normal(rng_key, (size,), dtype=jnp.float32)
            return jax.lax.with_sharding_constraint(out, full_sharded)

        rng = jax.random.PRNGKey(0)
        server: jax_transfer.TransferServer | None = None
        coordinator: ActorHandle | None = None

        if jax.process_index() == 0:
            server = start_transfer_server()
            coordinator = instantiate_coordinator(server)
            ray.get(tracker.set_coordinator.remote(coordinator))
            logger.info("Large slice started transfer server at %s", server.address())

        for step in range(rounds):
            rng, subkey = jax.random.split(rng)
            arr = make_arr(subkey)
            logger.info("Step %d", step)

            time_in = time.time()
            gathered = deshard(arr)
            gathered = jax.block_until_ready(gathered)
            gather_elapsed = time.time() - time_in

            if jax.process_index() == 0 and server is not None and coordinator is not None:
                logger.info("Gather took %.2f seconds for step %d", gather_elapsed, step)
                num_transfers_this_round = 0

                while num_transfers_this_round < receiver_count:
                    num_transfers = asyncio.run(process_weight_transfers(server, coordinator, step, gathered))
                    if num_transfers == 0:
                        time.sleep(0.5)
                        continue
                    num_transfers_this_round += num_transfers
                    logger.info("Processed %d transfers for step %d", num_transfers, step)

                    metrics = {
                        "host/gather_time": gather_elapsed,
                        "host/step": step,
                        "host/gather_bytes": gathered.nbytes,
                        "host/num_transfers": num_transfers,
                        "host/weight_norm": float(
                            jax.tree_util.tree_reduce(lambda x, y: x + jnp.linalg.norm(y), gathered, 0.0)
                        ),
                    }
                    ray.get(tracker.log.remote(metrics))

                ray.get(
                    tracker.log.remote(
                        {
                            "host/step_finished": step,
                        }
                    )
                )

            logger.info("Finished all transfers for step %d", step)

        logger.info("Finished all rounds.")
        while not ray.get(tracker.is_finished.remote()):
            time.sleep(5.0)


def small_slice_loop(
    tracker: ActorHandle,
    worker_id: int,
    shape: Iterable[int],
    rounds: int,
) -> list[TransferStats]:
    """Persistent worker that performs `rounds` pulls from the weight server."""
    client_server = start_transfer_server()

    coordinator: ActorHandle | None = None
    while coordinator is None:
        coordinator = ray.get(tracker.get_coordinator.remote())
        if coordinator is None:
            time.sleep(0.5)

    logger.info("Small slice %d connected to coordinator", worker_id)

    mesh = make_process_mesh()
    stats: list[TransferStats] = []

    with jax.sharding.use_mesh(mesh):

        @jax.jit
        def make_placeholder():
            """Create a placeholder array with the given shape."""
            out = jnp.zeros(shape, dtype=jnp.float32)
            return jax.lax.with_sharding_constraint(out, full_sharded)

        placeholder = make_placeholder()

        @jax.jit
        def reshard(arr):
            return jax.lax.with_sharding_constraint(arr, full_sharded)

        for step in range(rounds):
            logger.info("Small slice %d pulling weights for step %d", worker_id, step)
            result, info = asyncio.run(receive_weight_transfers(coordinator, client_server, placeholder))
            logger.info(
                "Small slice %d finished pulling weights for step %d. It took %.2f seconds",
                worker_id,
                step,
                info.time_elapsed,
            )

            time_in = time.time()
            result = reshard(result)
            result = jax.block_until_ready(result)
            elapsed = time.time() - time_in

            transfer_finished = tracker.mark_transfer_finished.remote(step)

            metrics = {
                "client/pull_time": info.time_end - info.time_start,
                "client/reshard_time": elapsed,
                "client/bytes_transferred": info.weight_bytes,
                "client/worker_id": worker_id,
                "client/round": step,
                "client/weight_norm": float(jax.tree_util.tree_reduce(lambda x, y: x + jnp.linalg.norm(y), result, 0.0)),
            }
            ray.get([tracker.log.remote(metrics), transfer_finished])
            stats.append(TransferStats(bytes_transferred=info.weight_bytes, transfer_time=info.time_elapsed + elapsed))

    logger.info("Small slice %d finished %d rounds", worker_id, rounds)
    return stats


def run_benchmark(large_type: str, small_type: str, size: int, num_small: int, rounds: int) -> list[TransferStats]:
    """Initializes and runs the weight transfer benchmark."""
    tracker: ActorHandle = TransferProgressTracker.options(num_cpus=0).remote(  # type: ignore
        rounds * num_small,
        project="levanter-tpu-transfer-benchmark",
    )

    def _large_slice_fn():
        return large_slice_loop(size, rounds, tracker, num_small)

    large_future = ray_tpu.run_on_pod_ray.remote(_large_slice_fn, large_type)

    shape = (size,)

    def _make_worker_fn(wid: int):
        def _worker_fn():
            return small_slice_loop(tracker, wid, shape, rounds)

        return _worker_fn

    worker_futures = [ray_tpu.run_on_pod_ray.remote(_make_worker_fn(wid), small_type) for wid in range(num_small)]

    results_nested = ray.get(worker_futures)
    results: list[TransferStats] = []
    for pod_stats in results_nested:
        # run_on_pod_ray returns a list of results, one for each process on the pod, so we have to flatten twice
        for process_stats in pod_stats:
            results.extend(process_stats)

    ray.get(large_future)
    ray.get(tracker.finish.remote())
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--large-type", default="v5p-128", help="Shape of the large slice")
    parser.add_argument("--small-type", default="v5p-8", help="Shape of the small slices")
    parser.add_argument("--num-small", type=int, default=1, help="Number of small slices")
    parser.add_argument("--size", type=int, default=int(2e9), help="Number of fp32 weights")
    parser.add_argument("--rounds", type=int, default=1, help="Number of transfer rounds")
    args = parser.parse_args()

    ray.init()

    stats = run_benchmark(
        args.large_type,
        args.small_type,
        args.size,
        args.num_small,
        args.rounds,
    )

    for i, s in enumerate(stats):
        print(f"Transfer {i}: {s.bytes_transferred / 1e6:.2f} MB in {s.transfer_time:.2f}s")


if __name__ == "__main__":
    main()
