# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure tiny GPU launch overhead for Triton, JAX/Pallas, and XLA.

The benchmark intentionally uses a one-element increment/store kernel. It
reports two regimes:

- ``batch``: enqueue N dependent launches and synchronize once.
- ``sync_each``: synchronize after every launch.
- ``jax_pallas_compiled``: put N chained Pallas calls in one compiled JAX
  executable and synchronize once.

The batch numbers are the useful lower bound for training-style launch overhead;
the sync_each numbers include host synchronization cost and are intentionally
more pessimistic.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from collections.abc import Callable
from typing import Any, Literal

Backend = Literal["triton", "torch", "jax_xla", "jax_pallas", "jax_pallas_compiled"]
Mode = Literal["batch", "sync_each"]


def _parse_csv_ints(raw: str) -> list[int]:
    values = [int(part) for part in raw.split(",") if part.strip()]
    if not values:
        raise ValueError("expected at least one integer")
    if any(value <= 0 for value in values):
        raise ValueError(f"values must be positive, got {values}")
    return values


def _stats(values: list[float]) -> dict[str, float]:
    return {
        "steady_s": statistics.fmean(values),
        "median_s": statistics.median(values),
        "min_s": min(values),
        "max_s": max(values),
    }


def _emit(record: dict[str, Any]) -> None:
    print(json.dumps(record, sort_keys=True), flush=True)


def _time_python_loop(launches: int, steps: int, warmup: int) -> dict[str, float]:
    for _ in range(warmup):
        for _ in range(launches):
            pass

    timings = []
    for _ in range(steps):
        start = time.perf_counter()
        for _ in range(launches):
            pass
        timings.append(time.perf_counter() - start)
    return _stats(timings)


def _torch_timing(
    call: Callable[[], None],
    synchronize: Callable[[], None],
    *,
    launches: int,
    steps: int,
    warmup: int,
    mode: Mode,
    torch_module: Any,
) -> dict[str, float]:
    for _ in range(warmup):
        for _ in range(launches):
            call()
            if mode == "sync_each":
                synchronize()
        synchronize()

    wall_timings: list[float] = []
    event_timings: list[float] = []
    for _ in range(steps):
        start_event = torch_module.cuda.Event(enable_timing=True)
        end_event = torch_module.cuda.Event(enable_timing=True)
        synchronize()
        wall_start = time.perf_counter()
        if mode == "batch":
            start_event.record()
        for _ in range(launches):
            if mode == "sync_each":
                call()
                synchronize()
            else:
                call()
        if mode == "batch":
            end_event.record()
        synchronize()
        wall_timings.append(time.perf_counter() - wall_start)
        if mode == "batch":
            event_timings.append(start_event.elapsed_time(end_event) / 1000.0)

    record = {f"wall_{key}": value for key, value in _stats(wall_timings).items()}
    if event_timings:
        record.update({f"cuda_event_{key}": value for key, value in _stats(event_timings).items()})
    return record


def _run_triton_case(launches: int, steps: int, warmup: int, mode: Mode) -> dict[str, float]:
    import torch
    import triton
    import triton.language as tl

    @triton.jit
    def inc_kernel(x_ptr, y_ptr):
        value = tl.load(x_ptr)
        tl.store(y_ptr, value + 1.0)

    x = torch.ones((1,), device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)

    def call() -> None:
        inc_kernel[(1,)](x, y)

    return _torch_timing(
        call,
        torch.cuda.synchronize,
        launches=launches,
        steps=steps,
        warmup=warmup,
        mode=mode,
        torch_module=torch,
    )


def _run_torch_case(launches: int, steps: int, warmup: int, mode: Mode) -> dict[str, float]:
    import torch

    x = torch.ones((1,), device="cuda", dtype=torch.float32)
    y = torch.empty_like(x)

    def call() -> None:
        torch.add(x, 1.0, out=y)

    return _torch_timing(
        call,
        torch.cuda.synchronize,
        launches=launches,
        steps=steps,
        warmup=warmup,
        mode=mode,
        torch_module=torch,
    )


def _jax_timing(
    call: Callable[[Any], Any],
    state: Any,
    *,
    launches: int,
    steps: int,
    warmup: int,
    mode: Mode,
) -> dict[str, float]:
    for _ in range(warmup):
        for _ in range(launches):
            state = call(state)
            if mode == "sync_each":
                state.block_until_ready()
        state.block_until_ready()

    timings = []
    for _ in range(steps):
        start = time.perf_counter()
        for _ in range(launches):
            state = call(state)
            if mode == "sync_each":
                state.block_until_ready()
        state.block_until_ready()
        timings.append(time.perf_counter() - start)
    return {f"wall_{key}": value for key, value in _stats(timings).items()}


def _run_jax_xla_case(launches: int, steps: int, warmup: int, mode: Mode) -> dict[str, float]:
    import jax
    import jax.numpy as jnp

    state = jnp.ones((1,), dtype=jnp.float32)
    call = jax.jit(lambda x: x + jnp.float32(1.0))
    state = call(state)
    state.block_until_ready()
    return _jax_timing(call, state, launches=launches, steps=steps, warmup=warmup, mode=mode)


def _run_jax_pallas_case(launches: int, steps: int, warmup: int, mode: Mode) -> dict[str, float]:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    def inc_kernel(x_ref, y_ref):
        value = pltriton.load(x_ref.at[0])
        pltriton.store(y_ref.at[0], value + jnp.float32(1.0))

    def inc(x):
        return pl.pallas_call(
            inc_kernel,
            out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
            grid=(1,),
            compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=3),
            cost_estimate=pl.CostEstimate(flops=1, transcendentals=0, bytes_accessed=8),
            name="launch_overhead_pallas_inc",
        )(x)

    state = jnp.ones((1,), dtype=jnp.float32)
    call = jax.jit(inc)
    state = call(state)
    state.block_until_ready()
    return _jax_timing(call, state, launches=launches, steps=steps, warmup=warmup, mode=mode)


def _run_jax_pallas_compiled_case(launches: int, steps: int, warmup: int) -> dict[str, float]:
    import jax
    import jax.numpy as jnp
    from jax.experimental import pallas as pl
    from jax.experimental.pallas import triton as pltriton

    def inc_kernel(x_ref, y_ref):
        value = pltriton.load(x_ref.at[0])
        pltriton.store(y_ref.at[0], value + jnp.float32(1.0))

    def inc(x):
        return pl.pallas_call(
            inc_kernel,
            out_shape=jax.ShapeDtypeStruct((1,), jnp.float32),
            grid=(1,),
            compiler_params=pltriton.CompilerParams(num_warps=1, num_stages=3),
            cost_estimate=pl.CostEstimate(flops=1, transcendentals=0, bytes_accessed=8),
            name="launch_overhead_pallas_compiled_inc",
        )(x)

    def run_many(x):
        for _ in range(launches):
            x = inc(x)
            x = jax.lax.optimization_barrier(x)
        return x

    state = jnp.ones((1,), dtype=jnp.float32)
    call = jax.jit(run_many)
    state = call(state)
    state.block_until_ready()

    for _ in range(warmup):
        state = call(state)
        state.block_until_ready()

    timings = []
    for _ in range(steps):
        start = time.perf_counter()
        state = call(state)
        state.block_until_ready()
        timings.append(time.perf_counter() - start)
    return {f"wall_{key}": value for key, value in _stats(timings).items()}


def _run_backend(backend: Backend, launches: int, steps: int, warmup: int, mode: Mode) -> dict[str, float]:
    if backend == "triton":
        return _run_triton_case(launches, steps, warmup, mode)
    if backend == "torch":
        return _run_torch_case(launches, steps, warmup, mode)
    if backend == "jax_xla":
        return _run_jax_xla_case(launches, steps, warmup, mode)
    if backend == "jax_pallas":
        return _run_jax_pallas_case(launches, steps, warmup, mode)
    if backend == "jax_pallas_compiled":
        if mode != "batch":
            raise ValueError("jax_pallas_compiled only supports batch mode")
        return _run_jax_pallas_compiled_case(launches, steps, warmup)
    raise ValueError(f"unknown backend: {backend}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backends", default="triton,torch,jax_xla,jax_pallas")
    parser.add_argument("--launches", default="1,10,100,1000")
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--modes", default="batch,sync_each")
    args = parser.parse_args()

    launches_values = _parse_csv_ints(args.launches)
    backends = [part for part in args.backends.split(",") if part]
    modes = [part for part in args.modes.split(",") if part]

    for launches in launches_values:
        loop_stats = _time_python_loop(launches, args.steps, args.warmup)
        _emit(
            {
                "backend": "python_loop",
                "launches": launches,
                **loop_stats,
                "steady_per_launch_us": loop_stats["steady_s"] / launches * 1e6,
                "median_per_launch_us": loop_stats["median_s"] / launches * 1e6,
            }
        )

    for backend in backends:
        if backend not in {"triton", "torch", "jax_xla", "jax_pallas", "jax_pallas_compiled"}:
            raise ValueError(f"unknown backend: {backend}")
        for mode in modes:
            if mode not in {"batch", "sync_each"}:
                raise ValueError(f"unknown mode: {mode}")
            if backend == "jax_pallas_compiled" and mode != "batch":
                continue
            for launches in launches_values:
                timing = _run_backend(backend, launches, args.steps, args.warmup, mode)
                record = {
                    "backend": backend,
                    "mode": "compiled_batch" if backend == "jax_pallas_compiled" else mode,
                    "launches": launches,
                    **timing,
                    "wall_steady_per_launch_us": timing["wall_steady_s"] / launches * 1e6,
                    "wall_median_per_launch_us": timing["wall_median_s"] / launches * 1e6,
                }
                if "cuda_event_steady_s" in timing:
                    record["cuda_event_steady_per_launch_us"] = timing["cuda_event_steady_s"] / launches * 1e6
                    record["cuda_event_median_per_launch_us"] = timing["cuda_event_median_s"] / launches * 1e6
                _emit(record)


if __name__ == "__main__":
    main()
