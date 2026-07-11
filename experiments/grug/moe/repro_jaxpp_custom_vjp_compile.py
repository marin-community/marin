# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Bounded JaxPP custom-VJP compile/load isolation for large MoE gradients.

This script compares three backward implementations at the per-pipeline-stage
shape used by the Grug SonicMoE experiment:

* plain: JAX batched matmuls and autodiff
* opaque: a custom VJP returning large gradients through Pallas custom calls
* quack: Marin's QuACK grouped expert MLP custom VJP

Run either directly on one GPU or through the smallest JaxPP MPMD topology
(two local processes). The watchdog makes every invocation bounded.
"""

from __future__ import annotations

import argparse
import faulthandler
import importlib.metadata
import json
import multiprocessing as mp
import os
import platform
import sys
import threading
import time
from dataclasses import dataclass
from functools import partial
from typing import Any, Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental import multihost_utils
from jax.experimental import pallas as pl
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxpp.experimental import mpmd as jaxpp_mpmd
from levanter.grug._moe.sonic_quack import quack_mlp_varlen

JAXPP_REVISION = "7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
SONIC_IMPLEMENTATION_REVISION = "7952c5e5fd"
Mode = Literal["plain", "opaque", "quack"]
Runtime = Literal["direct", "distributed_direct", "jaxpp"]


def event(name: str, **fields: Any) -> None:
    print(json.dumps({"time": time.time(), "event": name, **fields}, default=str), flush=True)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def environment() -> dict[str, Any]:
    backend = jax.extend.backend.get_backend()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": package_version("jaxlib"),
        "jaxpp": package_version("jaxpp"),
        "jaxpp_revision": JAXPP_REVISION,
        "sonic_implementation_revision": SONIC_IMPLEMENTATION_REVISION,
        "quack_kernels": package_version("quack-kernels"),
        "jax_tvm_ffi": package_version("jax-tvm-ffi"),
        "backend_platform": backend.platform,
        "backend_platform_version": backend.platform_version,
        "devices": [str(device) for device in jax.devices()],
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "xla_python_client_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
    }


@dataclass(frozen=True)
class Config:
    mode: Mode
    runtime: Runtime
    layers: int
    experts: int
    tokens_per_expert: int
    hidden: int
    intermediate: int
    fsdp: int
    timeout: int
    stack_after: int

    @property
    def assignments(self) -> int:
        return self.experts * self.tokens_per_expert


def start_watchdog(config: Config) -> None:
    faulthandler.enable()
    faulthandler.dump_traceback_later(config.stack_after, repeat=True)

    def hard_stop() -> None:
        event("watchdog_timeout", timeout=config.timeout)
        os._exit(124)

    timer = threading.Timer(config.timeout, hard_stop)
    timer.daemon = True
    timer.start()


def swiglu(x: jax.Array) -> jax.Array:
    gate, up = jnp.split(x, 2, axis=-1)
    return jax.nn.silu(gate) * up


def plain_layer(x: jax.Array, up: jax.Array, down: jax.Array, config: Config) -> jax.Array:
    grouped = x.reshape(config.experts, config.tokens_per_expert, config.hidden)
    hidden = swiglu(jnp.einsum("emh,ehi->emi", grouped, up))
    return jnp.einsum("emi,eih->emh", hidden, down).reshape(x.shape)


def _pallas_zero_like(x: jax.Array) -> jax.Array:
    flat = x.reshape((-1,))
    block = 256
    if flat.size % block:
        raise ValueError(f"opaque mode requires a multiple of {block} elements, got {flat.size}")
    blocks = flat.size // block

    def kernel(x_ref, out_ref):
        out_ref[...] = x_ref[...] * jnp.asarray(0, x_ref.dtype)

    out = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(flat.shape, flat.dtype),
        grid=(blocks,),
        in_specs=(pl.BlockSpec((block,), lambda index: (index,)),),
        out_specs=pl.BlockSpec((block,), lambda index: (index,)),
        name="jaxpp_repro_zero_like",
    )(flat)
    return out.reshape(x.shape)


@jax.custom_vjp
def opaque_layer(x: jax.Array, up: jax.Array, down: jax.Array) -> jax.Array:
    return x


def _opaque_layer_fwd(x: jax.Array, up: jax.Array, down: jax.Array):
    return x, (up, down)


def _opaque_layer_bwd(residuals: tuple[jax.Array, jax.Array], dout: jax.Array):
    up, down = residuals
    return dout, _pallas_zero_like(up), _pallas_zero_like(down)


opaque_layer.defvjp(_opaque_layer_fwd, _opaque_layer_bwd)


def quack_layer(
    x: jax.Array,
    up: jax.Array,
    down: jax.Array,
    config: Config,
) -> jax.Array:
    group_sizes = jnp.full((config.experts,), config.tokens_per_expert, dtype=jnp.int32)
    return quack_mlp_varlen(x, up, down, group_sizes)


def loss_and_grads(
    up_weights: tuple[jax.Array, ...],
    down_weights: tuple[jax.Array, ...],
    x: jax.Array,
    dependency: jax.Array,
    config: Config,
):
    def loss_fn(up_tree, down_tree, activation):
        activation = activation + dependency.astype(activation.dtype)
        for up, down in zip(up_tree, down_tree, strict=True):
            if config.mode == "plain":
                activation = plain_layer(activation, up, down, config)
            elif config.mode == "opaque":
                activation = opaque_layer(activation, up, down)
            else:
                activation = quack_layer(activation, up, down, config)
        return jnp.mean(activation.astype(jnp.float32))

    return jax.value_and_grad(loss_fn, argnums=(0, 1, 2))(up_weights, down_weights, x)


def shapes(config: Config, sharding: NamedSharding):
    up_shape = (config.experts, config.hidden, 2 * config.intermediate)
    down_shape = (config.experts, config.intermediate, config.hidden)
    x_shape = (config.assignments, config.hidden)
    ups = tuple(jax.ShapeDtypeStruct(up_shape, jnp.bfloat16, sharding=sharding) for _ in range(config.layers))
    downs = tuple(jax.ShapeDtypeStruct(down_shape, jnp.bfloat16, sharding=sharding) for _ in range(config.layers))
    return ups, downs, x_shape


def initialize_array(shape: tuple[int, ...], sharding: NamedSharding) -> jax.Array:
    if not any(device.process_index == jax.process_index() for device in sharding.mesh.devices.flat):
        return jax.make_array_from_single_device_arrays(shape, sharding, [], dtype=jnp.bfloat16)
    with jax.set_mesh(sharding.mesh):
        return jax.jit(lambda: jnp.zeros(shape, jnp.bfloat16), out_shardings=sharding)()


def run_direct(config: Config, devices: list[jax.Device] | None = None, event_prefix: str = "direct") -> None:
    if devices is None:
        devices = jax.devices()[: config.fsdp]
    mesh = Mesh(np.asarray(devices, dtype=object), ("fsdp",), axis_types=(AxisType.Explicit,))
    replicated = NamedSharding(mesh, P())
    ups_shape, downs_shape, x_shape = shapes(config, replicated)
    ups = tuple(initialize_array(shape.shape, replicated) for shape in ups_shape)
    downs = tuple(initialize_array(shape.shape, replicated) for shape in downs_shape)
    x = initialize_array(x_shape, replicated)
    dependency = jax.device_put(np.asarray(0, np.float32), replicated)
    step = jax.jit(partial(loss_and_grads, config=config))
    event(f"{event_prefix}_compile_execute_entered", process_id=jax.process_index())
    started = time.perf_counter()
    result = step(ups, downs, x, dependency)
    jax.block_until_ready(result)
    event(
        f"{event_prefix}_compile_execute_returned",
        process_id=jax.process_index(),
        elapsed=time.perf_counter() - started,
    )


def run_distributed_direct_worker(config: Config, process_id: int, coordinator: str, devices: list[int]) -> None:
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=2,
        process_id=process_id,
        local_device_ids=devices,
        cluster_detection_method="deactivate",
    )
    start_watchdog(config)
    try:
        if process_id == 1:
            run_direct(config, devices=jax.local_devices(), event_prefix="distributed_direct")
        multihost_utils.sync_global_devices("distributed_direct_complete")
        event("distributed_direct_barrier_returned", process_id=process_id)
    finally:
        jax.distributed.shutdown()


def run_jaxpp_worker(config: Config, process_id: int, coordinator: str, devices: list[int]) -> None:
    jax.distributed.initialize(
        coordinator_address=coordinator,
        num_processes=2,
        process_id=process_id,
        local_device_ids=devices,
        cluster_detection_method="deactivate",
    )
    start_watchdog(config)
    try:
        all_devices = np.asarray(jax.devices(), dtype=object).reshape(1, 2, config.fsdp)
        mpmd_mesh = jaxpp_mpmd.MpmdMesh(
            Mesh(
                all_devices,
                ("replica", "pp", "fsdp"),
                axis_types=(AxisType.Explicit,) * 3,
            ),
            "pp",
        )
        stage0_scalar = NamedSharding(mpmd_mesh.unstack[0], P())
        stage1_scalar = NamedSharding(mpmd_mesh.unstack[1], P())
        stage1_weight = NamedSharding(mpmd_mesh.unstack[1], P("fsdp", None, None))
        stage1_replicated = NamedSharding(mpmd_mesh.unstack[1], P())
        up_shapes, down_shapes, x_shape = shapes(config, stage1_weight)
        up_out = tuple(stage1_weight for _ in up_shapes)
        down_out = tuple(stage1_weight for _ in down_shapes)

        def stage1(up_weights, down_weights, x, dependency):
            replicated_ups = jax.tree.map(lambda value: jax.reshard(value, P()), up_weights)
            replicated_downs = jax.tree.map(lambda value: jax.reshard(value, P()), down_weights)
            return loss_and_grads(replicated_ups, replicated_downs, x, dependency, config)

        @jaxpp_mpmd.mpmd(
            mpmd_mesh,
            in_shardings=(stage0_scalar, up_out, down_out, stage1_replicated),
            infer_donation=False,
        )
        def program(seed, up_weights, down_weights, x):
            seed = jaxpp_mpmd.task(lambda value: value + 1, out_shardings=stage0_scalar)(seed)
            dependency = jaxpp_mpmd.transfer(seed, out_shardings=stage1_scalar).done()
            return jaxpp_mpmd.task(
                stage1,
                name=f"repro_{config.mode}_stage1_backward",
                out_shardings=(stage1_scalar, (up_out, down_out, stage1_replicated)),
            )(up_weights, down_weights, x, dependency)

        seed_shape = jax.ShapeDtypeStruct((), jnp.float32, sharding=stage0_scalar)
        x_struct = jax.ShapeDtypeStruct(x_shape, jnp.bfloat16, sharding=stage1_replicated)
        event("jaxpp_lower_entered", process_id=process_id)
        lower_started = time.perf_counter()
        lowered = program.lower(seed_shape, up_shapes, down_shapes, x_struct)
        event("jaxpp_lower_returned", process_id=process_id, elapsed=time.perf_counter() - lower_started)

        seed = (
            jax.device_put(np.asarray(0, np.float32), stage0_scalar)
            if mpmd_mesh.my_mpmd_axis_index == 0
            else jax.make_array_from_single_device_arrays((), stage0_scalar, [], dtype=jnp.float32)
        )
        ups = tuple(initialize_array(shape.shape, stage1_weight) for shape in up_shapes)
        downs = tuple(initialize_array(shape.shape, stage1_weight) for shape in down_shapes)
        x = initialize_array(x_shape, stage1_replicated)
        flat_args, _ = jax.tree_util.tree_flatten((seed, ups, downs, x))
        local = lowered._local_jaxpr
        local_args = [flat_args[index] for index in local.global_invar_indices]
        event("jaxpp_eval_local_entered", process_id=process_id, local_inputs=len(local_args))
        started = time.perf_counter()
        result = lowered.eval_local(*local_args)
        jax.block_until_ready(result)
        event("jaxpp_eval_local_returned", process_id=process_id, elapsed=time.perf_counter() - started)
        multihost_utils.sync_global_devices("jaxpp_compile_repro_complete")
        event("jaxpp_completion_barrier_returned", process_id=process_id)
    finally:
        jax.distributed.shutdown()


def run_jaxpp(config: Config) -> None:
    if len(jax.devices()) != 2 * config.fsdp:
        raise ValueError(f"jaxpp runtime requires {2 * config.fsdp} visible GPUs, got {len(jax.devices())}")
    context = mp.get_context("spawn")
    coordinator = "127.0.0.1:5789"
    processes = []
    try:
        for process_id in range(2):
            local_devices = list(range(process_id * config.fsdp, (process_id + 1) * config.fsdp))
            process = context.Process(target=run_jaxpp_worker, args=(config, process_id, coordinator, local_devices))
            process.start()
            processes.append(process)
        deadline = time.monotonic() + config.timeout + 60
        while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
            for process in processes:
                if process.exitcode not in (None, 0):
                    raise SystemExit(process.exitcode)
            time.sleep(1)
        if any(process.is_alive() for process in processes):
            raise TimeoutError("worker cleanup deadline exceeded")
        bad = next((process.exitcode for process in processes if process.exitcode), None)
        if bad is not None:
            raise SystemExit(bad)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


def run_distributed_direct(config: Config) -> None:
    if len(jax.devices()) != 2 * config.fsdp:
        raise ValueError(f"distributed_direct runtime requires {2 * config.fsdp} visible GPUs, got {len(jax.devices())}")
    context = mp.get_context("spawn")
    coordinator = "127.0.0.1:5790"
    processes = []
    try:
        for process_id in range(2):
            local_devices = list(range(process_id * config.fsdp, (process_id + 1) * config.fsdp))
            process = context.Process(
                target=run_distributed_direct_worker,
                args=(config, process_id, coordinator, local_devices),
            )
            process.start()
            processes.append(process)
        deadline = time.monotonic() + config.timeout + 60
        while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
            for process in processes:
                if process.exitcode not in (None, 0):
                    raise SystemExit(process.exitcode)
            time.sleep(1)
        if any(process.is_alive() for process in processes):
            raise TimeoutError("worker cleanup deadline exceeded")
        bad = next((process.exitcode for process in processes if process.exitcode), None)
        if bad is not None:
            raise SystemExit(bad)
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
        for process in processes:
            process.join(timeout=10)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("plain", "opaque", "quack"), required=True)
    parser.add_argument("--runtime", choices=("direct", "distributed_direct", "jaxpp"), required=True)
    parser.add_argument("--layers", type=int, default=6)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--tokens-per-expert", type=int, default=1024)
    parser.add_argument("--hidden", type=int, default=2560)
    parser.add_argument("--intermediate", type=int, default=1280)
    parser.add_argument("--fsdp", type=int, default=1)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--stack-after", type=int, default=300)
    args = parser.parse_args()
    return Config(**vars(args))


def main() -> None:
    config = parse_args()
    event("environment", config=config.__dict__, environment=environment())
    if config.runtime == "direct":
        start_watchdog(config)
        run_direct(config)
    elif config.runtime == "distributed_direct":
        run_distributed_direct(config)
    else:
        run_jaxpp(config)


if __name__ == "__main__":
    main()
