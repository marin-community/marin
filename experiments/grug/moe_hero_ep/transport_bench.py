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

import json
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
from rigging.filesystem import StoragePath

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
BENCH_RESULT_ROOT = "s3://marin-us-east-02a/tmp/ttl=30d/transport_bench"


@dataclass(frozen=True)
class TransportBenchConfig:
    """Shape and identity for one transport benchmark."""

    run_id: str
    device_count: int
    row_counts: tuple[int, ...]
    row_elements: int
    capacity_factor: float
    skew: float
    result_uri: str


def _send_matrix(num_ranks: int, rows_per_rank: int, skew: float) -> np.ndarray:
    """Build a send matrix whose largest cell is ``skew`` times the mean cell.

    Each sender gives one peer the large message and divides the rest equally.
    Every row sums to ``rows_per_rank``, so the total traffic does not change.
    """
    mean = rows_per_rank // num_ranks
    matrix = np.zeros((num_ranks, num_ranks), dtype=np.int32)
    large = round(mean * skew)
    for source in range(num_ranks):
        rest = (rows_per_rank - large) // (num_ranks - 1)
        matrix[source, :] = rest
        matrix[source, (source + 1) % num_ranks] = large
        matrix[source, source] += rows_per_rank - int(matrix[source].sum())
    return matrix


def _time_median(run, repeats: int) -> float:
    jax.block_until_ready(run())
    times = []
    for _ in range(repeats):
        start = time.perf_counter()
        jax.block_until_ready(run())
        times.append(time.perf_counter() - start)
    return sorted(times)[len(times) // 2]


def _benchmark(config: TransportBenchConfig, rows: int) -> dict[str, float]:
    devices = np.asarray(jax.devices())
    if devices.size != config.device_count:
        raise ValueError(f"benchmark requires {config.device_count} devices, got {devices.size}")
    mesh = Mesh(devices, axis_names=("expert",), axis_types=(AxisType.Explicit,))
    num_ranks = config.device_count
    elements = config.row_elements
    send_matrix = _send_matrix(num_ranks, rows, config.skew)
    capacity = max(int(np.ceil(config.capacity_factor * send_matrix.max())), 1)

    def _source(rank: jax.Array, total_rows: int) -> jax.Array:
        return jnp.broadcast_to(rank.astype(jnp.bfloat16), (total_rows, elements))

    @partial(shard_map, mesh=mesh, in_specs=(), out_specs=P(), check_rep=False)
    def ragged() -> jax.Array:
        rank = jax.lax.axis_index("expert")
        source = _source(rank, rows)
        matrix = jnp.asarray(send_matrix)
        local_sizes = matrix[rank]
        send_offsets = jnp.cumsum(local_sizes, dtype=jnp.int32) - local_sizes
        recv_sizes = matrix[:, rank]
        recv_offsets = jnp.cumsum(recv_sizes, dtype=jnp.int32) - recv_sizes
        received = jax.lax.ragged_all_to_all(
            source,
            jnp.zeros_like(source),
            send_offsets,
            local_sizes,
            recv_offsets,
            recv_sizes,
            axis_name="expert",
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
            jnp.asarray(send_matrix),
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
    num_ranks = config.device_count
    sweep = {rows: _benchmark(config, rows) for rows in config.row_counts}
    if jax.process_index() != 0:
        return
    # This dispatch path does not deliver worker stdout or worker logs to the
    # job log stream, so write the result to object storage instead.
    report = {
        "run_id": config.run_id,
        "ranks": num_ranks,
        "row_elements": config.row_elements,
        "capacity_factor": config.capacity_factor,
        "skew": config.skew,
        "sweep": [],
    }
    for rows, results in sweep.items():
        exact_bytes = rows * config.row_elements * 2 * (num_ranks - 1) / num_ranks
        padded_bytes = exact_bytes * config.capacity_factor
        entry = {
            "rows_per_rank": rows,
            "rows_per_peer": rows // num_ranks,
            "measurements": {},
        }
        for label, seconds in results.items():
            moved = exact_bytes if label == "ragged" else padded_bytes
            entry["measurements"][label] = {
                "median_ms": round(seconds * 1e3, 3),
                "gigabytes": round(moved / 1e9, 3),
                "gigabytes_per_second": round(moved / seconds / 1e9, 1),
            }
        entry["layout_cost_ms"] = round((results["padded_full"] - results["padded_collective"]) * 1e3, 3)
        report["sweep"].append(entry)
    StoragePath(config.result_uri).write_text(json.dumps(report, indent=2))


@click.command()
@click.option("--run-id", required=True, help="Iris job identifier.")
@click.option("--nodes", type=click.IntRange(min=1), required=True, help="GB200 nodes to reserve.")
@click.option(
    "--rows-per-rank",
    default=str(BENCH_ROWS_PER_RANK),
    show_default=True,
    help="Comma-separated row counts for each rank. Each value gives one sweep point.",
)
@click.option("--row-elements", type=click.IntRange(min=1), default=BENCH_ROW_ELEMENTS, show_default=True)
@click.option("--capacity-factor", type=click.FloatRange(min=0.1), default=1.0, show_default=True)
@click.option(
    "--skew",
    type=click.FloatRange(min=1.0),
    default=1.0,
    show_default=True,
    help="Ratio of the largest peer message to the mean peer message.",
)
@click.option(
    "--moonep-jax-wheel-build",
    type=click.Choice([build.value for build in MoonEPJaxWheelBuild]),
    required=True,
)
def main(
    run_id: str,
    nodes: int,
    rows_per_rank: str,
    row_elements: int,
    capacity_factor: float,
    skew: float,
    moonep_jax_wheel_build: str,
) -> None:
    """Dispatch a transport benchmark on a GB200 gang."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")
    device_count = nodes * BENCH_GPUS_PER_NODE
    row_counts = tuple(int(value) for value in rows_per_rank.split(","))
    for rows in row_counts:
        if rows % device_count != 0:
            raise ValueError(f"rows_per_rank={rows} must be divisible by {device_count} ranks")

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
        row_counts=row_counts,
        row_elements=row_elements,
        capacity_factor=capacity_factor,
        skew=skew,
        result_uri=f"{BENCH_RESULT_ROOT}/{run_id}.json",
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
    main()
