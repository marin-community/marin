#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Reproduce a collective-initialization failure after a fast process restart.

Run this program as a fresh Iris-supervised process at least twice on the same
GPU allocation and with the same persistent compilation cache. The workload
contains ten distinct fusions and one ``psum``, matching the smallest
previously observed trigger.
"""

from __future__ import annotations

import importlib.metadata
import os
import platform
import time

import jax
import jax.numpy as jnp
import numpy as np
from iris.runtime.jax_init import initialize_jax
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

_IRIS_PROCESS_COUNT_ENV = "IRIS_MULTIGPU_PROCESS_COUNT"
_IRIS_PROCESS_INDEX_ENV = "IRIS_MULTIGPU_PROCESS_INDEX"
_REPEAT_ENV = "MARIN_REPRO_REPEAT"


def log(message: str) -> None:
    """Emit a rank-stamped progress marker immediately."""
    process_index = (
        jax.process_index() if jax.distributed.is_initialized() else int(os.environ.get(_IRIS_PROCESS_INDEX_ENV, "0"))
    )
    print(f"REPRO[rank={process_index}] {message}", flush=True)


def initialize_distributed() -> None:
    """Join the process group described by the Iris multigpu supervisor."""
    if _IRIS_PROCESS_COUNT_ENV not in os.environ:
        return

    repeat = int(os.environ.get(_REPEAT_ENV, "0"))
    initialize_jax(
        endpoint_name=f"b200_nccl_fast_restart_{repeat}",
        poll_timeout=180,
    )


def package_version(name: str) -> str:
    """Return an installed package version for the environment record."""
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def main() -> None:
    initialize_distributed()
    devices = jax.devices()
    mesh = Mesh(np.asarray(devices), ("data",))
    sharding = NamedSharding(mesh, P("data"))
    log(
        "environment "
        f"platform={platform.machine()} "
        f"jax={jax.__version__} jaxlib={jax.lib.__version__} "
        f"cuda_runtime={package_version('nvidia-cuda-runtime')} "
        f"cuda_nvcc={package_version('nvidia-cuda-nvcc')} "
        f"nccl={package_version('nvidia-nccl-cu13')} "
        f"processes={jax.process_count()} devices={len(devices)} "
        f"device_kind={devices[0].device_kind} "
        f"xla_flags={os.environ.get('XLA_FLAGS', '')!r} "
        f"nccl_debug={os.environ.get('NCCL_DEBUG', '')!r}"
    )

    def collective_fusions(local_values: jax.Array) -> jax.Array:
        values = local_values
        for index in range(10):
            values = jax.lax.optimization_barrier(jnp.sin(values) * np.float32(1.0 + index * 1e-6) + np.float32(index))
            if index == 0:
                values = values + jax.lax.psum(jnp.sum(values) * np.float32(1e-30), "data")
        return values

    execute = jax.jit(
        jax.shard_map(
            collective_fusions,
            mesh=mesh,
            in_specs=P("data"),
            out_specs=P("data"),
            check_vma=False,
        )
    )
    values = jax.jit(
        lambda: jnp.ones((len(devices) * 1024,), dtype=jnp.float32),
        out_shardings=sharding,
    )()
    log("compile-start")
    compile_start = time.monotonic()
    compiled = execute.lower(values).compile()
    log(f"compile-complete seconds={time.monotonic() - compile_start:.3f}")
    log("first-execution-start")
    execution_start = time.monotonic()
    result = compiled(values)
    result.block_until_ready()
    checksum = float(jnp.sum(result))
    log(f"first-execution-complete seconds={time.monotonic() - execution_start:.3f} checksum={checksum:.6e}")
    log("REPRO_OK")


if __name__ == "__main__":
    main()
