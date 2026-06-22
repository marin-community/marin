# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave launcher for a one-GPU-per-process JAX/DeepEP topology probe."""

from __future__ import annotations

import datetime
import os
from dataclasses import dataclass

from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.execution.types import ExecutorStep

from experiments.grug.dispatch import dispatch_grug_training_run
from experiments.grug.moe.cw_jax_clique_probe import main as run_clique_probe_local
from experiments.grug.moe.cw_storage import set_default_cw_grug_moe_prefix

set_default_cw_grug_moe_prefix()

DEFAULT_OUTPUT_SUBDIR = "experiments/grug-moe-cw/deepep-topology-probe"


def env_int(key: str, default: int) -> int:
    raw = os.environ.get(key, "")
    return int(raw) if raw else default


@dataclass(frozen=True)
class DeepEPTopologyProbeConfig:
    run_id: str
    resources: ResourceConfig


def _run_deepep_topology_probe_local(config: DeepEPTopologyProbeConfig) -> None:
    del config
    run_clique_probe_local()


def run_deepep_topology_probe(config: DeepEPTopologyProbeConfig) -> None:
    dispatch_grug_training_run(
        run_id=config.run_id,
        config=config,
        local_entrypoint=_run_deepep_topology_probe_local,
        resources=config.resources,
        max_retries_failure=0,
    )


def build_step() -> ExecutorStep:
    run_id = os.environ.get("RUN_ID") or datetime.datetime.now(datetime.timezone.utc).strftime(
        "DEEPEP-TOPOLOGY-PROBE-N2-GPU1-%Y%m%d-%H%M%S"
    )
    resources = ResourceConfig.with_gpu(
        "H100",
        count=env_int("DEEPEP_TOPOLOGY_GPU_COUNT", 1),
        cpu=env_int("DEEPEP_TOPOLOGY_WORKER_CPU", 8),
        ram=os.environ.get("DEEPEP_TOPOLOGY_WORKER_RAM", "64g"),
        disk=os.environ.get("DEEPEP_TOPOLOGY_WORKER_DISK", "64g"),
        replicas=env_int("DEEPEP_TOPOLOGY_REPLICAS", 16),
        image=os.environ.get("DEEPEP_TOPOLOGY_TASK_IMAGE") or None,
    )
    return ExecutorStep(
        name=f"{DEFAULT_OUTPUT_SUBDIR}/{run_id}",
        fn=run_deepep_topology_probe,
        config=DeepEPTopologyProbeConfig(run_id=run_id, resources=resources),
    )


def main() -> None:
    executor_main(steps=[build_step()])


if __name__ == "__main__":
    main()
