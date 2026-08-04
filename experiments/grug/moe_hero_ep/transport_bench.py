# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compare MoonEP token transports at the real row shape on a GB200 gang.

MNEP-105 shows that a padded transport gives the same rack step time as the
exact ragged transport, but a four-GPU probe of the collective alone shows a
large rate difference. That probe moved a plain buffer, so it measured neither
the padding gather nor the compaction scatter.

This benchmark separates the three costs:

- ``ragged``: the direct device kernel, which is the current transport.
- ``padded_collective``: one standard all-to-all with no layout work.
- ``padded_full``: the same collective inside the real gather and scatter.

The difference between the last two is the layout cost. The difference between
the first two is the collective cost.
"""

import logging
import time
from dataclasses import dataclass
from functools import partial

import click
import jax
import jax.numpy as jnp
import numpy as np
from fray.cluster import ResourceConfig
from jax.experimental.shard_map import shard_map
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P
from levanter.distributed import DistributedConfig
from levanter.grug._moe.ep_moonep import _static_padded_token_all_to_all

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.jax_wheel_setup import MoonEPJaxWheelBuild, moonep_jax_setup_scripts
from experiments.grug.moe_hero_ep.train import MoonEPTransport, _apply_hero_ep_runtime_defaults

BENCH_GPUS_PER_NODE = 4
BENCH_WORKER_CPU = 8
BENCH_WORKER_RAM = "64g"
BENCH_WORKER_DISK = "64g"
# One EP64 rank holds tokens_per_rank * top_k assignment rows of hidden_dim.
BENCH_ROWS_PER_RANK = 524_288
BENCH_ROW_ELEMENTS = 5_120
BENCH_REPEATS = 5


@dataclass(frozen=True)
class TransportBenchConfig:
    """Shape and identity for one transport benchmark."""

    run_id: str
    device_count: int
    rows_per_rank: int
    row_elements: int
    capacity_factor: float


def _balanced_send_matrix(num_ranks: int, rows_per_rank: int) -> jax.Array:
    rows_per_peer = rows_per_rank // num_ranks
    return jnp.full((num_ranks, num_ranks), rows_per_peer, dtype=jnp.int32)


def _time_median(run, repeats: int) -> float:
    jax.block_until_ready(run())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(run())
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]


def _benchmark(config: TransportBenchConfig) -> dict[str, float]:
    devices = np.asarray(jax.devices())
    if devices.size != config.device_count:
        raise ValueError(f"benchmark requires {config.device_count} devices, got {devices.size}")
    mesh = Mesh(devices, axis_names=("expert",), axis_types=(AxisType.Explicit,))
    num_ranks = config.device_count
    rows = config.rows_per_rank
    elements = config.row_elements
    rows_per_peer = rows // num_ranks
    capacity = max(int(np.ceil(config.capacity_factor * rows / num_ranks)), 1)

    def _source(rank: jax.Array, total_rows: int) -> jax.Array:
        return jnp.broadcast_to(rank.astype(jnp.bfloat16), (total_rows, elements))

    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=P(), check_rep=False)
    def ragged() -> jax.Array:
        rank = jax.lax.axis_index("expert")
        source = _source(rank, rows)
        offsets = jnp.arange(num_ranks, dtype=jnp.int32) * rows_per_peer
        sizes = jnp.full((num_ranks,), rows_per_peer, dtype=jnp.int32)
        received = jax.lax.ragged_all_to_all(
            source, jnp.zeros_like(source), offsets, sizes, offsets, sizes, axis_name="expert"
        )
        return jnp.sum(received[::8192, 0], dtype=jnp.float32)

    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=P(), check_rep=False)
    def padded_collective() -> jax.Array:
        rank = jax.lax.axis_index("expert")
        source = _source(rank, num_ranks * capacity)
        received = jax.lax.all_to_all(source, "expert", split_axis=0, concat_axis=0, tiled=True)
        return jnp.sum(received[::8192, 0], dtype=jnp.float32)

    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=P(), check_rep=False)
    def padded_full() -> jax.Array:
        rank = jax.lax.axis_index("expert")
        source = _source(rank, rows)
        received = _static_padded_token_all_to_all(
            source,
            _balanced_send_matrix(num_ranks, rows),
            rank,
            capacity=capacity,
            num_rounds=1,
        )
        return jnp.sum(received[::8192, 0], dtype=jnp.float32)

    results = {}
    with jax.set_mesh(mesh):
        for label, fn in (
            ("ragged", ragged),
            ("padded_collective", padded_collective),
            ("padded_full", padded_full),
        ):
            compiled = jax.jit(fn)
            results[label] = _time_median(compiled, BENCH_REPEATS)
    return results


def _run_benchmark_local(config: TransportBenchConfig) -> None:
    DistributedConfig().initialize()
    results = _benchmark(config)
    if jax.process_index() != 0:
        return
    num_ranks = config.device_count
    exact_bytes = config.rows_per_rank * config.row_elements * 2 * (num_ranks - 1) / num_ranks
    padded_bytes = exact_bytes * config.capacity_factor
    # The worker runs this function through the callable runner, which does not
    # configure the root logger. Write the result to stdout so that the job log
    # keeps it.
    lines = [f"transport_bench ranks={num_ranks}"]
    for label, seconds in results.items():
        moved = exact_bytes if label == "ragged" else padded_bytes
        lines.append(
            f"transport_bench label={label} median_ms={seconds * 1e3:.3f} "
            f"gigabytes={moved / 1e9:.3f} gigabytes_per_second={moved / seconds / 1e9:.1f}"
        )
    layout = results["padded_full"] - results["padded_collective"]
    lines.append(f"transport_bench layout_cost_ms={layout * 1e3:.3f}")
    print("\n".join(lines), flush=True)


@click.command()
@click.option("--run-id", required=True, help="Iris job identifier.")
@click.option("--nodes", type=click.IntRange(min=1), required=True, help="GB200 nodes to reserve.")
@click.option("--rows-per-rank", type=click.IntRange(min=1), default=BENCH_ROWS_PER_RANK, show_default=True)
@click.option("--row-elements", type=click.IntRange(min=1), default=BENCH_ROW_ELEMENTS, show_default=True)
@click.option("--capacity-factor", type=click.FloatRange(min=0.1), default=1.0, show_default=True)
@click.option(
    "--moonep-jax-wheel-build",
    type=click.Choice([build.value for build in MoonEPJaxWheelBuild]),
    required=True,
)
def main(
    run_id: str,
    nodes: int,
    rows_per_rank: int,
    row_elements: int,
    capacity_factor: float,
    moonep_jax_wheel_build: str,
) -> None:
    """Dispatch a transport benchmark on a GB200 gang."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    device_count = nodes * BENCH_GPUS_PER_NODE
    if rows_per_rank % device_count != 0:
        raise ValueError(f"rows_per_rank={rows_per_rank} must be divisible by {device_count} ranks")

    resources = ResourceConfig.with_gpu(
        "GB200",
        count=BENCH_GPUS_PER_NODE,
        cpu=BENCH_WORKER_CPU,
        ram=BENCH_WORKER_RAM,
        disk=BENCH_WORKER_DISK,
        replicas=nodes,
    )
    build = MoonEPJaxWheelBuild(moonep_jax_wheel_build)
    config = TransportBenchConfig(
        run_id=run_id,
        device_count=device_count,
        rows_per_rank=rows_per_rank,
        row_elements=row_elements,
        capacity_factor=capacity_factor,
    )
    _apply_hero_ep_runtime_defaults("moonep_jax", MoonEPTransport.DIRECT_DEVICE)
    dispatch_grug_training_run(
        run_id=run_id,
        config=config,
        local_entrypoint=_run_benchmark_local,
        resources=resources,
        max_retries_failure=0,
        processes_per_task=1,
        setup_scripts=moonep_jax_setup_scripts(build, resources),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
