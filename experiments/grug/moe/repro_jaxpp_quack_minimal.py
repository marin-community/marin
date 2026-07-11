# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone two-rank JaxPP/QuACK non-return diagnostic.

This intentionally avoids Marin and Levanter imports. It uses one grouped
QuACK GEMM so forward and custom-VJP-backward placement can be tested
independently. Install the pinned runtime before invoking the script::

    uv pip install cupy-cuda13x quack-kernels==0.5.0 jax-tvm-ffi==0.1.3
    uv pip install --no-deps \
      'jaxpp @ git+https://github.com/NVIDIA/jaxpp.git@7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9'

The default dimensions are the minimum failing shape. The two-rank JaxPP case
needs six visible GPUs::

    CUDA_VISIBLE_DEVICES=0,1,2,3,4,5 python -u repro_jaxpp_quack_minimal.py \
      --runtime jaxpp --operation quack --transform forward --transfer scalar

The nearest passing topology uses four visible GPUs plus ``--fsdp 2
--experts 2``. A single-process direct run has the same fsdp=2/3 boundary.
Each invocation is bounded by per-rank watchdogs and emits JSON-line events.
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
from collections.abc import Callable
from dataclasses import asdict, dataclass
from typing import Any

import cutlass.cute as cute  # pyrefly: ignore[missing-import]  # GPU-only dependency
import jax
import jax.numpy as jnp
import numpy as np
from cuda.bindings import driver as cuda  # pyrefly: ignore[missing-import]  # GPU-only dependency
from cutlass import BFloat16, Float32, Int32  # pyrefly: ignore[missing-import]  # GPU-only dependency
from jax.experimental import multihost_utils
from jax.experimental import pallas as pl
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from jaxpp.experimental import mpmd as jaxpp_mpmd
from quack.compile_utils import make_fake_tensor  # pyrefly: ignore[missing-import]  # GPU-only dependency
from quack.cute_dsl_utils import (  # pyrefly: ignore[missing-import]  # GPU-only dependency
    get_max_active_clusters,
)
from quack.gemm_default_epi import (  # pyrefly: ignore[missing-import]  # GPU-only dependency
    GemmDefaultEpiMixin,
    GemmDefaultSm90,
)
from quack.jax_utils import TvmFfiKernel  # pyrefly: ignore[missing-import]  # GPU-only dependency
from quack.tile_scheduler import TileSchedulerOptions  # pyrefly: ignore[missing-import]  # GPU-only dependency
from quack.varlen_utils import VarlenArguments  # pyrefly: ignore[missing-import]  # GPU-only dependency

JAXPP_REVISION = "7091a9b5ce02cd1a6bdc905f6a36e89370a5fba9"
EXPECTED_PACKAGES = {"jaxpp": "0.10.2", "jax-tvm-ffi": "0.1.3", "quack-kernels": "0.5.0"}
TILE_SHAPE = (128, 192)
CLUSTER_SHAPE = (2, 1, 1)
ALIGNMENT = 8
ALLOW_CUDA_GRAPH = os.environ.get("JAXPP_QUACK_ALLOW_CUDA_GRAPH", "true").lower() in ("1", "true", "yes", "on")


def event(name: str, **fields: Any) -> None:
    """Emit one structured diagnostic event."""
    print(json.dumps({"time": time.time(), "event": name, **fields}, default=str), flush=True)


def package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def jaxpp_revision() -> str:
    direct_url = importlib.metadata.distribution("jaxpp").read_text("direct_url.json")
    if direct_url is None:
        return "unknown"
    return json.loads(direct_url).get("vcs_info", {}).get("commit_id", "unknown")


def check_versions() -> None:
    actual = {name: package_version(name) for name in EXPECTED_PACKAGES}
    mismatches = {
        name: {"expected": expected, "actual": actual[name]}
        for name, expected in EXPECTED_PACKAGES.items()
        if actual[name] != expected
    }
    if mismatches:
        raise RuntimeError(f"package version mismatch: {mismatches}")
    actual_revision = jaxpp_revision()
    if actual_revision != JAXPP_REVISION:
        raise RuntimeError(f"JaxPP revision mismatch: expected {JAXPP_REVISION}, got {actual_revision}")


def environment() -> dict[str, Any]:
    backend = jax.extend.backend.get_backend()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "jax": jax.__version__,
        "jaxlib": package_version("jaxlib"),
        "jaxpp": package_version("jaxpp"),
        "jaxpp_revision": jaxpp_revision(),
        "quack_kernels": package_version("quack-kernels"),
        "jax_tvm_ffi": package_version("jax-tvm-ffi"),
        "backend_platform": backend.platform,
        "backend_platform_version": backend.platform_version,
        "devices": [str(device) for device in jax.devices()],
        "allow_cuda_graph": ALLOW_CUDA_GRAPH,
        "xla_python_client_mem_fraction": os.environ.get("XLA_PYTHON_CLIENT_MEM_FRACTION", ""),
    }


@dataclass(frozen=True)
class Config:
    runtime: str
    operation: str
    transform: str
    transfer: str
    input_sharding: str
    compute_rank: int
    experts: int
    tokens_per_expert: int
    input_dim: int
    output_dim: int
    fsdp: int
    timeout: int
    stack_after: int
    coordinator_port: int

    @property
    def tokens(self) -> int:
        return self.experts * self.tokens_per_expert


def start_watchdog(config: Config, process_id: int) -> None:
    faulthandler.enable()
    faulthandler.dump_traceback_later(config.stack_after, repeat=True)

    def hard_stop() -> None:
        event("watchdog_timeout", process_id=process_id, timeout=config.timeout)
        os._exit(124)

    timer = threading.Timer(config.timeout, hard_stop)
    timer.daemon = True
    timer.start()


def _weight_view(storage: Any) -> Any:
    return cute.make_tensor(
        storage.iterator,
        cute.make_layout(
            (storage.shape[1], storage.shape[2], storage.shape[0]),
            stride=(storage.shape[2], 1, storage.shape[1] * storage.shape[2]),
        ),
    )


class _GroupedVarlenFfi:
    def __init__(self) -> None:
        self.gemm = GemmDefaultSm90(
            Float32,
            BFloat16,
            TILE_SHAPE,
            CLUSTER_SHAPE,
            pingpong=True,
            is_persistent=True,
        )
        self.max_active_clusters = get_max_active_clusters(CLUSTER_SHAPE[0] * CLUSTER_SHAPE[1])

    @cute.jit
    def __call__(
        self,
        x: cute.Tensor,
        weight_storage: cute.Tensor,
        offsets: cute.Tensor,
        output: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        epilogue = GemmDefaultEpiMixin.EpilogueArguments()
        scheduler = TileSchedulerOptions(
            max_active_clusters=Int32(self.max_active_clusters),
            max_swizzle_size=Int32(8),
        )
        self.gemm(
            x,
            _weight_view(weight_storage),
            output,
            None,
            epilogue,
            scheduler,
            VarlenArguments(mCuSeqlensM=offsets),
            stream,
        )


def _compile_grouped_varlen() -> Any:
    tokens = cute.sym_int()
    input_dim = cute.sym_int()
    output_dim = cute.sym_int()
    experts = cute.sym_int()
    offsets_size = cute.sym_int()
    x = make_fake_tensor(BFloat16, (tokens, input_dim), leading_dim=1, divisibility=ALIGNMENT)
    weights = make_fake_tensor(
        BFloat16,
        (experts, output_dim, input_dim),
        leading_dim=2,
        divisibility=ALIGNMENT,
    )
    offsets = make_fake_tensor(Int32, (offsets_size,), leading_dim=0, divisibility=4)
    output = make_fake_tensor(BFloat16, (tokens, output_dim), leading_dim=1, divisibility=ALIGNMENT)
    return cute.compile(
        _GroupedVarlenFfi(),
        x,
        weights,
        offsets,
        output,
        cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
        options="--enable-tvm-ffi",
    )


GROUPED_VARLEN = TvmFfiKernel(
    "jaxpp_quack_minimal_grouped_varlen",
    _compile_grouped_varlen,
    allow_cuda_graph=ALLOW_CUDA_GRAPH,
)


def grouped_varlen(x: jax.Array, weights: jax.Array, offsets: jax.Array) -> jax.Array:
    weight_storage = jnp.swapaxes(weights, 1, 2)
    output_shape = jax.ShapeDtypeStruct((x.shape[0], weights.shape[2]), x.dtype)
    return GROUPED_VARLEN(x, weight_storage, offsets, key=(), output_shape_dtype=output_shape)


def opaque_zero(x: jax.Array) -> jax.Array:
    flat = x.reshape((-1,))
    block = 128
    padded_size = ((flat.size + block - 1) // block) * block
    padded = jnp.pad(flat, (0, padded_size - flat.size))

    def kernel(x_ref: Any, output_ref: Any) -> None:
        output_ref[...] = x_ref[...] * jnp.asarray(0, x_ref.dtype)

    output = pl.pallas_call(
        kernel,
        out_shape=jax.ShapeDtypeStruct(padded.shape, padded.dtype),
        grid=(padded_size // block,),
        in_specs=(pl.BlockSpec((block,), lambda index: (index,)),),
        out_specs=pl.BlockSpec((block,), lambda index: (index,)),
        name="jaxpp_quack_minimal_opaque_zero",
    )(padded)
    return output[: flat.size].reshape(x.shape)


def operation_output(x: jax.Array, weights: jax.Array, offsets: jax.Array, operation: str) -> jax.Array:
    if operation == "plain":
        grouped_x = x.reshape(weights.shape[0], -1, weights.shape[1])
        return jnp.einsum("eti,eio->eto", grouped_x, weights).reshape(x.shape[0], weights.shape[2])
    if operation == "opaque":
        del offsets
        if x.shape[1] < weights.shape[2]:
            x = jnp.pad(x, ((0, 0), (0, weights.shape[2] - x.shape[1])))
        return opaque_zero(x[:, : weights.shape[2]])
    return grouped_varlen(x, weights, offsets)


def run_probe(
    x: jax.Array,
    weights: jax.Array,
    offsets: jax.Array,
    sink: jax.Array,
    dependency: jax.Array,
    config: Config,
) -> Any:
    x = x + dependency.astype(x.dtype)
    if config.transform == "forward":
        output = operation_output(x, weights, offsets, config.operation)
        return jnp.mean(output.astype(jnp.float32))

    @jax.custom_vjp
    def backward_only(
        primal_x: jax.Array,
        primal_weights: jax.Array,
        primal_offsets: jax.Array,
        primal_sink: jax.Array,
    ) -> jax.Array:
        del primal_x, primal_weights, primal_offsets
        return jnp.mean(primal_sink.astype(jnp.float32))

    def probe_fwd(primal_x: jax.Array, primal_weights: jax.Array, primal_offsets: jax.Array, primal_sink: jax.Array):
        return jnp.mean(primal_sink.astype(jnp.float32)), (primal_x, primal_weights, primal_offsets)

    def probe_bwd(residuals: tuple[jax.Array, jax.Array, jax.Array], cotangent: jax.Array):
        primal_x, primal_weights, primal_offsets = residuals
        output = operation_output(primal_x, primal_weights, primal_offsets, config.operation)
        sink_gradient = output * cotangent.astype(output.dtype)
        return jnp.zeros_like(primal_x), jnp.zeros_like(primal_weights), None, sink_gradient

    backward_only.defvjp(probe_fwd, probe_bwd)
    return jax.value_and_grad(backward_only, argnums=3)(x, weights, offsets, sink)


def array_shapes(config: Config) -> tuple[tuple[int, ...], ...]:
    return (
        (config.tokens, config.input_dim),
        (config.experts, config.input_dim, config.output_dim),
        (config.experts + 1,),
        (config.tokens, config.output_dim),
    )


def initialize_array(shape: tuple[int, ...], dtype: jnp.dtype, sharding: NamedSharding) -> jax.Array:
    owns_devices = any(device.process_index == jax.process_index() for device in sharding.mesh.devices.flat)
    if not owns_devices:
        return jax.make_array_from_single_device_arrays(shape, sharding, [], dtype=dtype)
    with jax.set_mesh(sharding.mesh):
        return jax.jit(lambda: jnp.zeros(shape, dtype), out_shardings=sharding)()


def initialize_inputs(
    config: Config,
    replicated: NamedSharding,
    weight_sharding: NamedSharding,
) -> tuple[jax.Array, ...]:
    x_shape, weight_shape, offsets_shape, sink_shape = array_shapes(config)
    x = initialize_array(x_shape, jnp.bfloat16, replicated)
    weights = initialize_array(weight_shape, jnp.bfloat16, weight_sharding)
    owns_devices = any(device.process_index == jax.process_index() for device in replicated.mesh.devices.flat)
    if owns_devices:
        with jax.set_mesh(replicated.mesh):
            offsets = jax.jit(
                lambda: jnp.arange(config.experts + 1, dtype=jnp.int32) * config.tokens_per_expert,
                out_shardings=replicated,
            )()
    else:
        offsets = jax.make_array_from_single_device_arrays(offsets_shape, replicated, [], dtype=jnp.int32)
    sink = initialize_array(sink_shape, jnp.bfloat16, replicated)
    return x, weights, offsets, sink


def compute_on_mesh(config: Config, devices: list[jax.Device], event_prefix: str) -> None:
    mesh = Mesh(np.asarray(devices, dtype=object), ("fsdp",), axis_types=(AxisType.Explicit,))
    replicated = NamedSharding(mesh, P())
    weight_sharding = NamedSharding(
        mesh,
        P("fsdp", None, None) if config.input_sharding == "sharded" else P(),
    )
    x, weights, offsets, sink = initialize_inputs(config, replicated, weight_sharding)

    def local_probe(local_x: jax.Array, local_weights: jax.Array, local_offsets: jax.Array, local_sink: jax.Array):
        local_weights = jax.reshard(local_weights, replicated)
        return run_probe(local_x, local_weights, local_offsets, local_sink, jnp.asarray(0, jnp.float32), config)

    step = jax.jit(local_probe)
    event(f"{event_prefix}_entered", process_id=jax.process_index())
    started = time.perf_counter()
    result = step(x, weights, offsets, sink)
    jax.block_until_ready(result)
    event(f"{event_prefix}_returned", process_id=jax.process_index(), elapsed=time.perf_counter() - started)


def initialize_distributed(config: Config, process_id: int, local_device_ids: list[int]) -> None:
    jax.distributed.initialize(
        coordinator_address=f"127.0.0.1:{config.coordinator_port}",
        num_processes=2,
        process_id=process_id,
        local_device_ids=local_device_ids,
        cluster_detection_method="deactivate",
    )


def distributed_direct_worker(config: Config, process_id: int, local_device_ids: list[int]) -> None:
    initialize_distributed(config, process_id, local_device_ids)
    start_watchdog(config, process_id)
    try:
        compute_on_mesh(config, jax.local_devices(), "distributed_direct_eval")
        multihost_utils.sync_global_devices("jaxpp_quack_minimal_direct_complete")
        event("distributed_direct_barrier_returned", process_id=process_id)
    finally:
        event("distributed_shutdown_entered", process_id=process_id)
        jax.distributed.shutdown()
        event("distributed_shutdown_returned", process_id=process_id)


def jaxpp_worker(config: Config, process_id: int, local_device_ids: list[int]) -> None:
    initialize_distributed(config, process_id, local_device_ids)
    start_watchdog(config, process_id)
    try:
        devices = np.asarray(jax.devices(), dtype=object).reshape(2, config.fsdp)
        mpmd_mesh = jaxpp_mpmd.MpmdMesh(
            Mesh(devices, ("pp", "fsdp"), axis_types=(AxisType.Explicit, AxisType.Explicit)),
            "pp",
        )
        compute_mesh = mpmd_mesh.unstack[config.compute_rank]
        source_mesh = mpmd_mesh.unstack[1 - config.compute_rank]
        compute_scalar = NamedSharding(compute_mesh, P())
        source_scalar = NamedSharding(source_mesh, P())
        replicated = NamedSharding(compute_mesh, P())
        weight_sharding = NamedSharding(
            compute_mesh,
            P("fsdp", None, None) if config.input_sharding == "sharded" else P(),
        )
        x_shape, weight_shape, offsets_shape, sink_shape = array_shapes(config)
        x_struct = jax.ShapeDtypeStruct(x_shape, jnp.bfloat16, sharding=replicated)
        weight_struct = jax.ShapeDtypeStruct(weight_shape, jnp.bfloat16, sharding=weight_sharding)
        offsets_struct = jax.ShapeDtypeStruct(offsets_shape, jnp.int32, sharding=replicated)
        sink_struct = jax.ShapeDtypeStruct(sink_shape, jnp.bfloat16, sharding=replicated)
        result_sharding = compute_scalar if config.transform == "forward" else (compute_scalar, replicated)

        def task(
            local_x: jax.Array,
            local_weights: jax.Array,
            local_offsets: jax.Array,
            local_sink: jax.Array,
            dependency: jax.Array,
        ) -> Any:
            local_weights = jax.reshard(local_weights, replicated)
            return run_probe(local_x, local_weights, local_offsets, local_sink, dependency, config)

        def task_without_transfer(
            local_x: jax.Array,
            local_weights: jax.Array,
            local_offsets: jax.Array,
            local_sink: jax.Array,
        ) -> Any:
            local_weights = jax.reshard(local_weights, replicated)
            dependency = jnp.asarray(0, jnp.float32)
            return run_probe(local_x, local_weights, local_offsets, local_sink, dependency, config)

        if config.transfer == "scalar":

            @jaxpp_mpmd.mpmd(
                mpmd_mesh,
                in_shardings=(source_scalar, replicated, weight_sharding, replicated, replicated),
                infer_donation=False,
            )
            def program(seed: jax.Array, x: jax.Array, weights: jax.Array, offsets: jax.Array, sink: jax.Array):
                seed = jaxpp_mpmd.task(lambda value: value + 1, out_shardings=source_scalar)(seed)
                dependency = jaxpp_mpmd.transfer(seed, out_shardings=compute_scalar).done()
                return jaxpp_mpmd.task(task, out_shardings=result_sharding)(x, weights, offsets, sink, dependency)

            seed_struct = jax.ShapeDtypeStruct((), jnp.float32, sharding=source_scalar)
            lower_args = (seed_struct, x_struct, weight_struct, offsets_struct, sink_struct)
        else:

            @jaxpp_mpmd.mpmd(
                mpmd_mesh,
                in_shardings=(replicated, weight_sharding, replicated, replicated),
                infer_donation=False,
            )
            def program(x: jax.Array, weights: jax.Array, offsets: jax.Array, sink: jax.Array):
                return jaxpp_mpmd.task(task_without_transfer, out_shardings=result_sharding)(x, weights, offsets, sink)

            lower_args = (x_struct, weight_struct, offsets_struct, sink_struct)

        event("jaxpp_lower_entered", process_id=process_id)
        started = time.perf_counter()
        lowered = program.lower(*lower_args)
        event("jaxpp_lower_returned", process_id=process_id, elapsed=time.perf_counter() - started)

        values: tuple[jax.Array, ...] = initialize_inputs(config, replicated, weight_sharding)
        if config.transfer == "scalar":
            seed = initialize_array((), jnp.float32, source_scalar)
            values = (seed, *values)
        flat_args, _ = jax.tree_util.tree_flatten(values)
        local_args = [flat_args[index] for index in lowered._local_jaxpr.global_invar_indices]
        event("jaxpp_eval_local_entered", process_id=process_id, local_inputs=len(local_args))
        started = time.perf_counter()
        result = lowered.eval_local(*local_args)
        jax.block_until_ready(result)
        event("jaxpp_eval_local_returned", process_id=process_id, elapsed=time.perf_counter() - started)
        multihost_utils.sync_global_devices("jaxpp_quack_minimal_complete")
        event("jaxpp_barrier_returned", process_id=process_id)
    finally:
        event("distributed_shutdown_entered", process_id=process_id)
        jax.distributed.shutdown()
        event("distributed_shutdown_returned", process_id=process_id)


def run_workers(config: Config, worker: Callable[[Config, int, list[int]], None]) -> None:
    if len(jax.devices()) != 2 * config.fsdp:
        raise ValueError(f"{config.runtime} requires {2 * config.fsdp} visible GPUs, got {len(jax.devices())}")
    context = mp.get_context("spawn")
    processes: list[mp.Process] = []
    try:
        for process_id in range(2):
            local_device_ids = list(range(process_id * config.fsdp, (process_id + 1) * config.fsdp))
            process = context.Process(target=worker, args=(config, process_id, local_device_ids))
            process.start()
            processes.append(process)
        deadline = time.monotonic() + config.timeout + 30
        while any(process.is_alive() for process in processes) and time.monotonic() < deadline:
            bad = next((process.exitcode for process in processes if process.exitcode not in (None, 0)), None)
            if bad is not None:
                raise SystemExit(bad)
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
            event("worker_exit", pid=process.pid, exitcode=process.exitcode)
        for process in processes:
            if process.is_alive():
                process.kill()
                process.join()
                event("worker_killed", pid=process.pid, exitcode=process.exitcode)


def parse_args() -> Config:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime", choices=("direct", "distributed_direct", "jaxpp"), required=True)
    parser.add_argument("--operation", choices=("plain", "opaque", "quack"), required=True)
    parser.add_argument("--transform", choices=("forward", "gradient"), required=True)
    parser.add_argument("--transfer", choices=("none", "scalar"), default="none")
    parser.add_argument("--input-sharding", choices=("replicated", "sharded"), default="replicated")
    parser.add_argument("--compute-rank", type=int, choices=(0, 1), default=1)
    parser.add_argument("--experts", type=int, default=3)
    parser.add_argument("--tokens-per-expert", type=int, default=1)
    parser.add_argument("--input-dim", type=int, default=8)
    parser.add_argument("--output-dim", type=int, default=8)
    parser.add_argument("--fsdp", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--stack-after", type=int, default=60)
    parser.add_argument("--coordinator-port", type=int, default=5789)
    args = parser.parse_args()
    positive = ("experts", "tokens_per_expert", "input_dim", "output_dim", "fsdp", "timeout", "stack_after")
    for name in positive:
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.input_sharding == "sharded" and args.experts % args.fsdp:
        parser.error("--experts must be divisible by --fsdp for sharded weights")
    if args.runtime != "jaxpp" and args.transfer != "none":
        parser.error("--transfer applies only to --runtime jaxpp")
    return Config(**vars(args))


def main() -> None:
    config = parse_args()
    check_versions()
    event("start", config=asdict(config), environment=environment())
    if config.runtime == "direct":
        start_watchdog(config, 0)
        compute_on_mesh(config, jax.devices()[: config.fsdp], "direct_eval")
    elif config.runtime == "distributed_direct":
        run_workers(config, distributed_direct_worker)
    else:
        run_workers(config, jaxpp_worker)
    event("complete")


if __name__ == "__main__":
    main()
