# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Launch the staged GB200 symmetric-memory gate for MoK EP64."""

import click
from fray.current_client import current_client
from fray.types import Entrypoint, JobRequest, ResourceConfig, create_environment
from iris.rpc.proto_display import priority_band_value
from iris.runtime.jax_init import XLA_AUTOTUNE_CACHE_MODE_ENV
from marin.training.run_environment import extras_for_resources

from experiments.grug.moe_hero_ep.mok_ep64_symmetric_memory_probe import ProbeConfig, probe_entrypoint

GPUS_PER_TASK = 4
PROCESSES_PER_TASK = 4
CPUS_PER_TASK = 16
RAM_PER_TASK = "64g"
DISK_PER_TASK = "64g"
SUPPORTED_NUM_NODES = (2, 16)


def build_probe_request(
    *,
    run_id: str,
    num_nodes: int,
    arena_bytes: int = 4096,
    iterations: int = 3,
) -> JobRequest:
    """Build an EP8 or EP64 gang with one supervised process per GPU."""
    if num_nodes not in SUPPORTED_NUM_NODES:
        raise ValueError(f"num_nodes must be one of {SUPPORTED_NUM_NODES}, got {num_nodes}")
    resources = ResourceConfig.with_gpu(
        "GB200",
        count=GPUS_PER_TASK,
        cpu=CPUS_PER_TASK,
        ram=RAM_PER_TASK,
        disk=DISK_PER_TASK,
        replicas=num_nodes,
    )
    config = ProbeConfig(
        expected_world_size=num_nodes * PROCESSES_PER_TASK,
        arena_bytes=arena_bytes,
        iterations=iterations,
    )
    return JobRequest(
        name=f"mok-ep{config.expected_world_size}-symm-mem-{run_id}",
        entrypoint=Entrypoint.from_callable(probe_entrypoint, args=[config]),
        resources=resources,
        environment=create_environment(
            env_vars={
                XLA_AUTOTUNE_CACHE_MODE_ENV: "local_only",
                "TORCH_SYMMMEM_IMPLICIT_POOL": "0",
            },
            extras=extras_for_resources(resources),
        ),
        processes_per_task=PROCESSES_PER_TASK,
        max_retries_failure=0,
        max_retries_preemption=0,
        max_task_failures=0,
        priority=priority_band_value("interactive"),
    )


@click.command()
@click.option("--run-id", required=True)
@click.option("--num-nodes", type=click.Choice([str(value) for value in SUPPORTED_NUM_NODES]), required=True)
@click.option("--arena-bytes", type=click.IntRange(min=64), default=4096, show_default=True)
@click.option("--iterations", type=click.IntRange(min=1), default=3, show_default=True)
def main(run_id: str, num_nodes: str, arena_bytes: int, iterations: int) -> None:
    """Submit the probe and wait for every task and child process."""
    request = build_probe_request(
        run_id=run_id,
        num_nodes=int(num_nodes),
        arena_bytes=arena_bytes,
        iterations=iterations,
    )
    current_client().submit(request).wait(raise_on_failure=True)


if __name__ == "__main__":
    main()
