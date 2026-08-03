# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe XLA's direct multi-node ragged all-to-all kernel on one NVL72."""

import logging
import os
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

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe_hero_ep.jax_wheel_setup import MoonEPJaxWheelBuild, moonep_jax_setup_scripts
from experiments.grug.moe_hero_ep.train import MoonEPTransport, _apply_hero_ep_runtime_defaults

logger = logging.getLogger(__name__)

PROBE_NODES = 16
PROBE_GPUS_PER_NODE = 4
PROBE_DEVICE_COUNT = PROBE_NODES * PROBE_GPUS_PER_NODE
PROBE_WORKER_CPU = 8
PROBE_WORKER_RAM = "64g"
PROBE_WORKER_DISK = "64g"
PROBE_ROWS_PER_RANK = 524_288
PROBE_ROW_ELEMENTS = 5_120


@dataclass(frozen=True)
class RaggedDeviceProbeConfig:
    """Shape and identity for one direct-device transport probe."""

    run_id: str
    device_count: int
    rows_per_rank: int
    row_elements: int


def _balanced_ragged_probe(config: RaggedDeviceProbeConfig) -> tuple[jax.Array, jax.Array]:
    devices = np.asarray(jax.devices())
    if devices.size != config.device_count:
        raise ValueError(f"probe requires {config.device_count} devices, got {devices.size}")
    if config.rows_per_rank % config.device_count != 0:
        raise ValueError(f"rows_per_rank={config.rows_per_rank} must be divisible by device_count={config.device_count}")

    mesh = Mesh(devices, axis_names=("expert",), axis_types=(AxisType.Explicit,))
    rows_per_peer = config.rows_per_rank // config.device_count

    @partial(
        shard_map,
        mesh=mesh,
        in_specs=(),
        out_specs=(P(), P()),
        check_rep=False,
    )
    def transport() -> tuple[jax.Array, jax.Array]:
        rank = jax.lax.axis_index("expert")
        source = jnp.broadcast_to(
            rank.astype(jnp.bfloat16),
            (config.rows_per_rank, config.row_elements),
        )
        peer_ids = jnp.arange(config.device_count, dtype=jnp.int32)
        input_offsets = peer_ids * rows_per_peer
        send_sizes = jnp.full((config.device_count,), rows_per_peer, dtype=jnp.int32)
        output_offsets = jnp.full((config.device_count,), rank * rows_per_peer, dtype=jnp.int32)
        destination = jnp.zeros_like(source)
        received = jax.lax.ragged_all_to_all(
            source,
            destination,
            input_offsets,
            send_sizes,
            output_offsets,
            send_sizes,
            axis_name="expert",
        )

        expected_sources = jnp.arange(config.rows_per_rank, dtype=jnp.int32) // rows_per_peer
        sample_columns = jnp.asarray((0, config.row_elements // 2, config.row_elements - 1), dtype=jnp.int32)
        samples = received[:, sample_columns].astype(jnp.int32)
        mismatches = jnp.sum(samples != expected_sources[:, None], dtype=jnp.int32)
        checksum = jnp.sum(samples, dtype=jnp.int32)
        return checksum, mismatches

    with jax.set_mesh(mesh):
        return jax.jit(transport)()


def _run_probe_local(config: RaggedDeviceProbeConfig) -> None:
    DistributedConfig().initialize()
    logger.info(
        "Starting %s with %d rows per rank and %d row elements",
        config.run_id,
        config.rows_per_rank,
        config.row_elements,
    )
    start = time.perf_counter()
    checksum, mismatches = jax.block_until_ready(_balanced_ragged_probe(config))
    duration = time.perf_counter() - start
    mismatch_count = int(mismatches)
    if mismatch_count != 0:
        raise ValueError(f"ragged transport returned {mismatch_count} sampled value mismatches")
    if jax.process_index() == 0:
        logger.info(
            "Ragged device probe passed: checksum=%d mismatches=%d duration=%.6f",
            int(checksum),
            mismatch_count,
            duration,
        )


@click.command()
@click.option("--run-id", required=True, help="Iris job identifier.")
@click.option("--rows-per-rank", type=click.IntRange(min=PROBE_DEVICE_COUNT), required=True)
@click.option("--row-elements", type=click.IntRange(min=1), required=True)
@click.option(
    "--moonep-jax-wheel-build",
    type=click.Choice([build.value for build in MoonEPJaxWheelBuild]),
    required=True,
)
def main(
    run_id: str,
    rows_per_rank: int,
    row_elements: int,
    moonep_jax_wheel_build: str,
) -> None:
    """Dispatch a direct-device ragged all-to-all probe."""
    if not run_id.strip():
        raise ValueError("run_id must not be empty")

    resources = ResourceConfig.with_gpu(
        "GB200",
        count=PROBE_GPUS_PER_NODE,
        cpu=PROBE_WORKER_CPU,
        ram=PROBE_WORKER_RAM,
        disk=PROBE_WORKER_DISK,
        replicas=PROBE_NODES,
    )
    build = MoonEPJaxWheelBuild(moonep_jax_wheel_build)
    config = RaggedDeviceProbeConfig(
        run_id=run_id,
        device_count=PROBE_DEVICE_COUNT,
        rows_per_rank=rows_per_rank,
        row_elements=row_elements,
    )
    _apply_hero_ep_runtime_defaults("moonep_jax", MoonEPTransport.DIRECT_DEVICE)
    os.environ.setdefault("NCCL_DEBUG", "INFO")
    dispatch_grug_training_run(
        run_id=run_id,
        config=config,
        local_entrypoint=_run_probe_local,
        resources=resources,
        max_retries_failure=0,
        processes_per_task=1,
        setup_scripts=moonep_jax_setup_scripts(build, resources),
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
