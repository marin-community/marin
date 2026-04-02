# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Any, cast

import jax
from jax import core as jax_core
from jax._src import mesh as mesh_lib
from jax.sharding import NamedSharding


_AUTOTUNE_THREAD_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pallas_autotune")


def sharding_of(value: jax.Array):
    """Return array sharding metadata when available."""
    sharding = None
    try:
        sharding = value.sharding  # type: ignore[attr-defined]
    except Exception:
        sharding = None
    if sharding is not None:
        return sharding

    aval = getattr(value, "aval", None)
    if aval is None:
        return None
    return getattr(aval, "sharding", None)


def named_sharding_of(value: jax.Array) -> NamedSharding | None:
    """Return a NamedSharding for the value when one is attached."""
    sharding = sharding_of(value)
    if isinstance(sharding, NamedSharding):
        return sharding
    return None


def hlo_sharding_of(value: jax.Array):
    """Return XLA HLO sharding metadata when it can be derived."""
    sharding = sharding_of(value)
    if sharding is None:
        return None
    to_hlo = getattr(sharding, "_to_xla_hlo_sharding", None)
    if to_hlo is None:
        return None
    try:
        return to_hlo(value.ndim)
    except Exception:
        return None


def value_uses_manual_sharding(value: jax.Array) -> bool:
    """Detect shard_map-local tracer values that carry manual sharding."""
    hlo_sharding = hlo_sharding_of(value)
    return hlo_sharding is not None and hlo_sharding.is_manual()


def shape_dtype_struct_for_benchmark(value: jax.Array) -> jax.ShapeDtypeStruct:
    """Build a lowering spec while preserving compatible global sharding."""
    sharding = sharding_of(value)
    if sharding is None or value_uses_manual_sharding(value):
        return jax.ShapeDtypeStruct(value.shape, value.dtype)
    return jax.ShapeDtypeStruct(value.shape, value.dtype, sharding=sharding)


def contains_tracer(*values: jax.Array) -> bool:
    """Whether any lowering input is already a tracer."""
    return any(isinstance(value, jax_core.Tracer) for value in values)


def benchmark_lowering_args(*values: jax.Array) -> tuple[jax.Array | jax.ShapeDtypeStruct, ...]:
    """Choose tracer-aware lowering inputs for autotune benchmarks."""
    if contains_tracer(*values):
        return values
    return tuple(shape_dtype_struct_for_benchmark(value) for value in values)


def should_offload_compile(*values: jax.Array) -> bool:
    """Whether benchmark lowering should run on the shared autotune thread."""
    return (
        contains_tracer(*values)
        or any(value_uses_manual_sharding(value) for value in values)
        or jax_core.unsafe_am_i_under_a_jit_DO_NOT_USE()
        or not mesh_lib.thread_resources.env.physical_mesh.empty
    )


def compile_benchmark_fn_current_thread(
    benchmark_fn: Callable[..., jax.Array],
    lowering_args: tuple[jax.Array | jax.ShapeDtypeStruct, ...],
) -> float:
    """Compile a benchmark function on the current thread and return compile time."""
    jitted = jax.jit(benchmark_fn)
    start = time.perf_counter()
    lowered = jitted.lower(*lowering_args)
    lowered.compile()
    return time.perf_counter() - start


def compile_benchmark_fn(
    *,
    benchmark_fn: Callable[..., jax.Array],
    lowering_args: tuple[jax.Array | jax.ShapeDtypeStruct, ...],
    args: Sequence[jax.Array],
) -> float:
    """Compile a benchmark function, offloading when JAX thread-local state is unsafe."""
    if not should_offload_compile(*args):
        return compile_benchmark_fn_current_thread(benchmark_fn, lowering_args)
    return _AUTOTUNE_THREAD_POOL.submit(
        compile_benchmark_fn_current_thread,
        benchmark_fn,
        lowering_args,
    ).result()


def maybe_wrap_in_shard_map(
    fn: Callable[..., jax.Array],
    *,
    args: Sequence[jax.Array],
    out_specs: Any,
    check_vma: bool = False,
) -> Callable[..., jax.Array]:
    """Wrap a benchmark function in shard_map when inputs are globally NamedSharded."""
    if not args or any(value_uses_manual_sharding(value) for value in args):
        return fn

    shardings = tuple(named_sharding_of(value) for value in args)
    if any(sharding is None for sharding in shardings):
        return fn

    named_shardings = cast(tuple[NamedSharding, ...], shardings)
    mesh = named_shardings[0].mesh
    if any(sharding.mesh != mesh for sharding in named_shardings[1:]):
        return fn

    return jax.shard_map(
        fn,
        mesh=mesh,
        in_specs=tuple(sharding.spec for sharding in named_shardings),
        out_specs=out_specs,
        check_vma=check_vma,
    )


__all__ = [
    "benchmark_lowering_args",
    "compile_benchmark_fn",
    "compile_benchmark_fn_current_thread",
    "contains_tracer",
    "hlo_sharding_of",
    "maybe_wrap_in_shard_map",
    "named_sharding_of",
    "shape_dtype_struct_for_benchmark",
    "sharding_of",
    "should_offload_compile",
    "value_uses_manual_sharding",
]
