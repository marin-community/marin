# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark XLA ragged all-to-all at the target EP8 microbatch geometry.

Run separate processes for the private-memory baseline and the device kernel
because XLA flags are fixed before JAX initializes. The device path requires:

    --xla_gpu_experimental_ragged_all_to_all_use_device_kernel=true
    --xla_gpu_ragged_all_to_all_mode=symmetric
    --xla_enable_nccl_symmetric_buffers_for_collectives=RaggedAllToAll
"""

import argparse
import importlib.metadata
import json
import os
import statistics
import time
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh
from jax.sharding import PartitionSpec as P

_EP_SIZE = 8


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--assignments-per-rank",
        type=int,
        default=65_536,
        help="Local routed assignments. The target 32x4096 top-k4 microbatch has 65,536 per EP rank.",
    )
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=30)
    return parser


def _validate_args(args: argparse.Namespace) -> None:
    if args.assignments_per_rank <= 0:
        raise ValueError(f"assignments_per_rank must be positive, got {args.assignments_per_rank}")
    if args.assignments_per_rank % _EP_SIZE:
        raise ValueError(f"assignments_per_rank={args.assignments_per_rank} must be divisible by EP size {_EP_SIZE}")
    if args.hidden_dim <= 0:
        raise ValueError(f"hidden_dim must be positive, got {args.hidden_dim}")
    if args.warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {args.warmup}")
    if args.iterations <= 0:
        raise ValueError(f"iterations must be positive, got {args.iterations}")


def _mesh() -> Mesh:
    local_devices = jax.local_devices()
    if len(local_devices) < _EP_SIZE:
        raise RuntimeError(f"ragged all-to-all benchmark requires 8 local devices, found {len(local_devices)}")
    return Mesh(
        np.asarray(local_devices[:_EP_SIZE]),
        axis_names=("expert",),
        axis_types=(AxisType.Explicit,),
    )


def _payload_bits(
    source: jax.Array,
    destination: jax.Array,
    source_rows: jax.Array,
    hidden_dim: int,
) -> jax.Array:
    columns = jnp.arange(hidden_dim, dtype=jnp.uint16)
    return (
        (source.astype(jnp.uint16) << jnp.uint16(13))
        ^ (destination.astype(jnp.uint16) << jnp.uint16(10))
        ^ ((source_rows.astype(jnp.uint16) & jnp.uint16(31)) << jnp.uint16(5))
        ^ (columns & jnp.uint16(31))
    )


def _local_input(assignments_per_rank: int, hidden_dim: int) -> jax.Array:
    source = jax.lax.axis_index("expert")
    rows = jnp.arange(assignments_per_rank, dtype=jnp.int32)[:, None]
    destination = rows // (assignments_per_rank // _EP_SIZE)
    bits = _payload_bits(source, destination, rows, hidden_dim)
    return jax.lax.bitcast_convert_type(bits, jnp.bfloat16)


def _local_transfer(x: jax.Array) -> jax.Array:
    assignments_per_rank = x.shape[0]
    per_peer = assignments_per_rank // _EP_SIZE
    source = jax.lax.axis_index("expert")
    input_offsets = jnp.arange(_EP_SIZE, dtype=jnp.int32) * per_peer
    sizes = jnp.full((_EP_SIZE,), per_peer, dtype=jnp.int32)
    output_offsets = jnp.full((_EP_SIZE,), source * per_peer, dtype=jnp.int32)
    return jax.lax.ragged_all_to_all(
        x,
        jnp.zeros_like(x),
        input_offsets,
        sizes,
        output_offsets,
        sizes,
        axis_name="expert",
    )


def _local_validation(output: jax.Array) -> tuple[jax.Array, jax.Array]:
    assignments_per_rank, hidden_dim = output.shape
    per_peer = assignments_per_rank // _EP_SIZE
    destination = jax.lax.axis_index("expert")
    output_rows = jnp.arange(assignments_per_rank, dtype=jnp.int32)[:, None]
    source = output_rows // per_peer
    source_rows = destination * per_peer + output_rows % per_peer
    expected_bits = _payload_bits(source, destination, source_rows, hidden_dim)
    output_bits = jax.lax.bitcast_convert_type(output, jnp.uint16)
    local_mismatches = jnp.sum(output_bits != expected_bits, dtype=jnp.int32)
    local_checksum = jnp.sum(output_bits.astype(jnp.uint32), dtype=jnp.uint32)
    return (
        jax.lax.psum(local_mismatches, "expert"),
        jax.lax.psum(local_checksum, "expert"),
    )


def _timings(compiled, x: jax.Array, *, warmup: int, iterations: int) -> dict[str, float]:
    for _ in range(warmup):
        jax.block_until_ready(compiled(x))
    durations = []
    for _ in range(iterations):
        started = time.perf_counter()
        jax.block_until_ready(compiled(x))
        durations.append(time.perf_counter() - started)
    return {
        "mean_ms": 1000.0 * statistics.fmean(durations),
        "median_ms": 1000.0 * statistics.median(durations),
        "min_ms": 1000.0 * min(durations),
        "max_ms": 1000.0 * max(durations),
    }


def main() -> None:
    args = _parser().parse_args()
    _validate_args(args)
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"ragged all-to-all timing requires GPUs, got {jax.default_backend()}")

    mesh = _mesh()
    sharding = P("expert", None)
    make_input = jax.jit(
        jax.shard_map(
            lambda _: _local_input(args.assignments_per_rank, args.hidden_dim),
            mesh=mesh,
            in_specs=P(),
            out_specs=sharding,
            check_vma=False,
        )
    )
    transfer = jax.jit(
        jax.shard_map(
            _local_transfer,
            mesh=mesh,
            in_specs=sharding,
            out_specs=sharding,
            check_vma=False,
        )
    )
    validate = jax.jit(
        jax.shard_map(
            _local_validation,
            mesh=mesh,
            in_specs=sharding,
            out_specs=(P(), P()),
            check_vma=False,
        )
    )

    with jax.set_mesh(mesh):
        x = jax.block_until_ready(make_input(jnp.array(0, dtype=jnp.int32)))
        compiled_transfer = transfer.lower(x).compile()
        output = jax.block_until_ready(compiled_transfer(x))
        mismatches, checksum = jax.device_get(validate(output))
        timing = _timings(compiled_transfer, x, warmup=args.warmup, iterations=args.iterations)

    mismatch_count = int(mismatches)
    if mismatch_count:
        raise AssertionError(f"ragged all-to-all payload mismatch: {mismatch_count} elements differ")

    result: dict[str, Any] = {
        "backend": jax.default_backend(),
        "device_kind": jax.devices()[0].device_kind,
        "devices": _EP_SIZE,
        "assignments_per_rank": args.assignments_per_rank,
        "assignments_per_peer": args.assignments_per_rank // _EP_SIZE,
        "hidden_dim": args.hidden_dim,
        "dtype": "bfloat16",
        "payload_bytes_per_rank": args.assignments_per_rank * args.hidden_dim * 2,
        "mismatch_count": mismatch_count,
        "checksum_uint32": int(checksum),
        "warmup": args.warmup,
        "iterations": args.iterations,
        "timing": timing,
        "jax_version": jax.__version__,
        "jaxlib_version": importlib.metadata.version("jaxlib"),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "tf_cpp_vmodule": os.environ.get("TF_CPP_VMODULE", ""),
    }
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
