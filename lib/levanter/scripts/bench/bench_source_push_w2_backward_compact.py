# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Compact expert-block benchmark for source-push W2 backward.

This intentionally skips SourcePushPlan construction, dy routing, W13 backward,
and dx return/combine. It measures the destination-local W2 backward problem
that remains after dy has already been routed into expert-major rows:

    H, route_weight, dy, W2 -> dH, d_route_weight, dW2

Use this harness to compare replacement grouped-GEMM W2 kernels against the
current staged implementation without paying unrelated MLP bookkeeping costs.
"""

from __future__ import annotations

import argparse
import json
import os
import time
import traceback
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from statistics import median
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P
from jaxtyping import Array, Bool, Float

from levanter.grug._moe.source_push_backward_w2 import (
    SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_FUSED,
    SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE,
    SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE,
    SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
    SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE,
    MIN_SOURCE_PUSH_W2_MATMUL_ROW_BLOCK,
    _SourcePushW2BackwardOutput,
    _SourcePushW2MatmulBackwardOutput,
    _pad_w2_matmul_rows_for_pallas,
    _source_push_w2_activation_and_weighted_activation_reference,
    _source_push_w2_backward_expert_blocks,
    _source_push_w2_d_weighted_activation_pallas_call,
    _source_push_w2_dw2_pallas_call,
    _source_push_w2_matmul_backward,
    _source_push_w2_matmul_backward_inferred_block_sizes,
    _source_push_w2_swiglu_backward,
)
from levanter.grug._moe.source_push_inbox import AXIS, _block_until_ready
from levanter.grug._moe.source_push_plan import SOURCE_PUSH_MESH_AXIS

KERNEL_NAME = "source_push_w2_backward_compact"

MODE_FULL = "full"
MODE_MATMUL = "matmul"
MODE_D_WEIGHTED_ACTIVATION = "d_weighted_activation"
MODE_DW2 = "dw2"
MODE_SWIGLU_FROM_D_WEIGHTED = "swiglu_from_d_weighted"
MODES = (
    MODE_FULL,
    MODE_MATMUL,
    MODE_D_WEIGHTED_ACTIVATION,
    MODE_DW2,
    MODE_SWIGLU_FROM_D_WEIGHTED,
)

IMPLEMENTATION_REFERENCE = "reference"
IMPLEMENTATION_PALLAS_MATMUL_REFERENCE_SWIGLU = "pallas_matmul_reference_swiglu"
IMPLEMENTATION_PALLAS_MATMUL_PALLAS_SWIGLU = "pallas_matmul_pallas_swiglu"
IMPLEMENTATION_PALLAS_FUSED = "pallas_fused"
IMPLEMENTATIONS = (
    IMPLEMENTATION_REFERENCE,
    IMPLEMENTATION_PALLAS_MATMUL_REFERENCE_SWIGLU,
    IMPLEMENTATION_PALLAS_MATMUL_PALLAS_SWIGLU,
    IMPLEMENTATION_PALLAS_FUSED,
)


@dataclass(frozen=True, slots=True)
class W2CompactConfig:
    ep_size: int
    experts_per_rank: int
    rows_per_expert: int
    live_rows_per_expert: int
    hidden_dim: int
    intermediate_dim: int
    dtype: str


class W2CompactInputs(NamedTuple):
    h: Float[Array, "Dst E C twoI"]
    route_weight: Float[Array, "Dst E C"]
    dy: Float[Array, "Dst E C D"]
    w2: Float[Array, "Dst E I D"]
    valid: Bool[Array, "Dst E C"]
    d_weighted_activation: Float[Array, "Dst E C I"]


class Timing(NamedTuple):
    first_call_time: float
    steady_state_times: list[float]
    output: Any


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--experts-per-rank", type=int, default=32)
    parser.add_argument("--rows-per-expert", type=int, default=5120)
    parser.add_argument("--live-rows-per-expert", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument("--intermediate-dim", type=int, default=1280)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--implementations", type=_parse_csv, default=(IMPLEMENTATION_REFERENCE,))
    parser.add_argument("--modes", type=_parse_csv, default=(MODE_FULL,))
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--repeat-runs", type=int, default=1)
    parser.add_argument("--debug-exceptions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--git-sha", type=str, default=None)
    parser.add_argument("--jsonl", type=str, default=None)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    config = W2CompactConfig(
        ep_size=args.ep_size,
        experts_per_rank=args.experts_per_rank,
        rows_per_expert=args.rows_per_expert,
        live_rows_per_expert=args.live_rows_per_expert,
        hidden_dim=args.hidden_dim,
        intermediate_dim=args.intermediate_dim,
        dtype=args.dtype,
    )
    _validate_config(config)
    mesh = _make_mesh(config.ep_size)
    inputs = _make_inputs(config, mesh)
    if args.jsonl:
        jsonl_dir = os.path.dirname(args.jsonl)
        if jsonl_dir:
            os.makedirs(jsonl_dir, exist_ok=True)

    for implementation in args.implementations:
        if implementation not in IMPLEMENTATIONS:
            raise ValueError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")
        for mode in args.modes:
            if mode not in MODES:
                raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
            try:
                timing = _time_mode(
                    config,
                    mesh,
                    inputs,
                    mode=mode,
                    implementation=implementation,
                    warmup=args.warmup,
                    steps=args.steps,
                    repeat_runs=args.repeat_runs,
                )
                rows = _timing_rows(
                    config,
                    mode=mode,
                    implementation=implementation,
                    timing=timing,
                    repeat_runs=args.repeat_runs,
                )
            except Exception as exc:
                if args.debug_exceptions:
                    traceback.print_exc()
                rows = [_error_row(config, mode=mode, implementation=implementation, exc=exc)]

            for row in rows:
                if args.git_sha is not None:
                    row["git_sha"] = args.git_sha
                line = json.dumps(row, sort_keys=True)
                print(line, flush=True)
                if args.jsonl:
                    with open(args.jsonl, "a", encoding="utf-8") as f:
                        print(line, file=f, flush=True)


def _parse_csv(value: str) -> tuple[str, ...]:
    parsed = tuple(part for part in value.split(",") if part)
    if not parsed:
        raise argparse.ArgumentTypeError("expected a comma-separated list")
    return parsed


def _validate_config(config: W2CompactConfig) -> None:
    if config.live_rows_per_expert > config.rows_per_expert:
        raise ValueError("live_rows_per_expert must be <= rows_per_expert")
    if config.ep_size < 1:
        raise ValueError("ep_size must be positive")
    if config.hidden_dim % 128 != 0:
        raise ValueError("hidden_dim must be a multiple of 128 for current W2 Pallas kernels")
    if config.intermediate_dim % 64 != 0:
        raise ValueError("intermediate_dim must be a multiple of 64 for current W2 Pallas kernels")


def _make_mesh(ep_size: int) -> Mesh:
    devices = np.asarray(jax.devices()[:ep_size])
    if devices.size < ep_size:
        raise RuntimeError(f"Need {ep_size} visible JAX devices, got {devices.size}")
    return Mesh(devices, (AXIS,), axis_types=(AxisType.Explicit,))


def _make_inputs(config: W2CompactConfig, mesh: Mesh) -> W2CompactInputs:
    dtype = jnp.bfloat16 if config.dtype == "bf16" else jnp.float32
    h_shape = (
        config.ep_size,
        config.experts_per_rank,
        config.rows_per_expert,
        2 * config.intermediate_dim,
    )
    route_shape = (config.ep_size, config.experts_per_rank, config.rows_per_expert)
    dy_shape = (
        config.ep_size,
        config.experts_per_rank,
        config.rows_per_expert,
        config.hidden_dim,
    )
    w2_shape = (
        config.ep_size,
        config.experts_per_rank,
        config.intermediate_dim,
        config.hidden_dim,
    )
    sharding_4d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None, None))
    sharding_3d = NamedSharding(mesh, P(SOURCE_PUSH_MESH_AXIS, None, None))

    @jax.jit(
        out_shardings=W2CompactInputs(
            h=sharding_4d,
            route_weight=sharding_3d,
            dy=sharding_4d,
            w2=sharding_4d,
            valid=sharding_3d,
            d_weighted_activation=sharding_4d,
        )
    )
    def init_inputs() -> W2CompactInputs:
        row_ids = jnp.arange(config.rows_per_expert)[None, None, :]
        valid = jnp.broadcast_to(row_ids < config.live_rows_per_expert, route_shape)
        valid_f = valid.astype(dtype)
        h = jnp.ones(h_shape, dtype=dtype) * valid_f[..., None] * jnp.asarray(0.125, dtype)
        route_weight = jnp.ones(route_shape, dtype=dtype) * valid_f * jnp.asarray(0.25, dtype)
        dy = jnp.ones(dy_shape, dtype=dtype) * valid_f[..., None] * jnp.asarray(0.0625, dtype)
        w2 = jnp.ones(w2_shape, dtype=dtype) * jnp.asarray(0.03125, dtype)
        d_weighted_activation = jnp.ones(w2_shape[:2] + (config.rows_per_expert, config.intermediate_dim), dtype=dtype)
        d_weighted_activation = d_weighted_activation * valid_f[..., None] * jnp.asarray(0.015625, dtype)
        return W2CompactInputs(
            h=h,
            route_weight=route_weight,
            dy=dy,
            w2=w2,
            valid=valid,
            d_weighted_activation=d_weighted_activation,
        )

    with jax.set_mesh(mesh):
        return _block_inputs(init_inputs())


def _time_mode(
    config: W2CompactConfig,
    mesh: Mesh,
    inputs: W2CompactInputs,
    *,
    mode: str,
    implementation: str,
    warmup: int,
    steps: int,
    repeat_runs: int,
) -> Timing:
    call = jax.jit(_mode_fn(mode, implementation, mesh))

    start = time.perf_counter()
    with jax.set_mesh(mesh):
        output = call(inputs)
    _block_until_ready(output)
    first_call_time = time.perf_counter() - start

    for _ in range(warmup):
        with jax.set_mesh(mesh):
            output = call(inputs)
        _block_until_ready(output)

    steady_state_times = []
    for _ in range(repeat_runs):
        start = time.perf_counter()
        for _ in range(steps):
            with jax.set_mesh(mesh):
                output = call(inputs)
            _block_until_ready(output)
        steady_state_times.append((time.perf_counter() - start) / steps)

    return Timing(first_call_time=first_call_time, steady_state_times=steady_state_times, output=output)


def _mode_fn(mode: str, implementation: str, mesh: Mesh) -> Callable[[W2CompactInputs], Any]:
    if mode == MODE_FULL:
        return lambda inputs: _full_w2_backward(inputs, implementation, mesh)
    if mode == MODE_MATMUL:
        return lambda inputs: _matmul_w2_backward(inputs, implementation, mesh)
    if mode == MODE_D_WEIGHTED_ACTIVATION:
        return lambda inputs: _d_weighted_activation(inputs, implementation, mesh)
    if mode == MODE_DW2:
        return lambda inputs: _dw2(inputs, implementation, mesh)
    if mode == MODE_SWIGLU_FROM_D_WEIGHTED:
        return lambda inputs: _swiglu_from_d_weighted(inputs, implementation, mesh)
    raise ValueError(f"mode must be one of {MODES}, got {mode!r}")


def _full_w2_backward(inputs: W2CompactInputs, implementation: str, mesh: Mesh) -> _SourcePushW2BackwardOutput:
    if implementation == IMPLEMENTATION_REFERENCE:
        return _source_push_w2_backward_expert_blocks(
            inputs.h,
            inputs.route_weight,
            inputs.dy,
            inputs.w2,
            inputs.valid,
            implementation=SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE,
        )
    if implementation == IMPLEMENTATION_PALLAS_MATMUL_REFERENCE_SWIGLU:
        return _source_push_w2_backward_expert_blocks(
            inputs.h,
            inputs.route_weight,
            inputs.dy,
            inputs.w2,
            inputs.valid,
            implementation=SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE,
            matmul_implementation=SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
            swiglu_implementation=SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE,
            mesh=mesh,
        )
    if implementation == IMPLEMENTATION_PALLAS_MATMUL_PALLAS_SWIGLU:
        return _source_push_w2_backward_expert_blocks(
            inputs.h,
            inputs.route_weight,
            inputs.dy,
            inputs.w2,
            inputs.valid,
            implementation=SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_REFERENCE,
            matmul_implementation=SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
            swiglu_implementation=SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_PALLAS_MGPU,
            mesh=mesh,
        )
    if implementation == IMPLEMENTATION_PALLAS_FUSED:
        return _source_push_w2_backward_expert_blocks(
            inputs.h,
            inputs.route_weight,
            inputs.dy,
            inputs.w2,
            inputs.valid,
            implementation=SOURCE_PUSH_W2_BACKWARD_IMPLEMENTATION_PALLAS_MGPU_FUSED,
            mesh=mesh,
        )
    raise ValueError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")


def _matmul_w2_backward(inputs: W2CompactInputs, implementation: str, mesh: Mesh) -> _SourcePushW2MatmulBackwardOutput:
    _activation, weighted_activation = _source_push_w2_activation_and_weighted_activation_reference(
        inputs.h,
        inputs.route_weight,
        inputs.valid,
    )
    matmul_implementation = _matmul_implementation(implementation)
    return _source_push_w2_matmul_backward(
        weighted_activation,
        inputs.dy,
        inputs.w2,
        inputs.valid,
        implementation=matmul_implementation,
        mesh=mesh if matmul_implementation == SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU else None,
    )


def _d_weighted_activation(inputs: W2CompactInputs, implementation: str, mesh: Mesh) -> Float[Array, "Dst E C I"]:
    if _matmul_implementation(implementation) == SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE:
        valid_f = inputs.valid.astype(jnp.float32)
        dy = inputs.dy.astype(jnp.float32) * valid_f[..., None]
        w2 = inputs.w2.astype(jnp.float32)
        return jnp.einsum("dech,deih->deci", dy, w2)

    dy, _weighted_activation, valid, block_sizes = _padded_matmul_inputs(
        inputs.dy,
        inputs.dy[..., : inputs.w2.shape[-2]],
        inputs.valid,
        inputs.w2,
    )
    d_weighted_activation = _source_push_w2_d_weighted_activation_pallas_call(
        dy.astype(inputs.w2.dtype),
        inputs.w2,
        valid,
        row_block=block_sizes.row_block,
        intermediate_block=block_sizes.intermediate_block,
        hidden_block=block_sizes.hidden_block,
        interpret=False,
        mesh=mesh,
    )
    return d_weighted_activation[:, :, : inputs.dy.shape[2], :]


def _dw2(inputs: W2CompactInputs, implementation: str, mesh: Mesh) -> Float[Array, "Dst E I D"]:
    _activation, weighted_activation = _source_push_w2_activation_and_weighted_activation_reference(
        inputs.h,
        inputs.route_weight,
        inputs.valid,
    )
    if _matmul_implementation(implementation) == SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE:
        valid_f = inputs.valid.astype(jnp.float32)
        weighted_activation = weighted_activation.astype(jnp.float32) * valid_f[..., None]
        dy = inputs.dy.astype(jnp.float32) * valid_f[..., None]
        return jnp.einsum("deci,dech->deih", weighted_activation, dy)

    dy, weighted_activation, valid, block_sizes = _padded_matmul_inputs(
        inputs.dy,
        weighted_activation,
        inputs.valid,
        inputs.w2,
    )
    return _source_push_w2_dw2_pallas_call(
        weighted_activation.astype(inputs.w2.dtype),
        dy.astype(inputs.w2.dtype),
        valid,
        row_block=block_sizes.row_block,
        intermediate_block=block_sizes.intermediate_block,
        hidden_block=block_sizes.hidden_block,
        interpret=False,
        mesh=mesh,
    )


def _swiglu_from_d_weighted(inputs: W2CompactInputs, implementation: str, mesh: Mesh) -> Any:
    swiglu_implementation = (
        SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_PALLAS_MGPU
        if implementation == IMPLEMENTATION_PALLAS_MATMUL_PALLAS_SWIGLU
        else SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_REFERENCE
    )
    return _source_push_w2_swiglu_backward(
        inputs.h,
        inputs.route_weight,
        inputs.d_weighted_activation,
        inputs.valid,
        implementation=swiglu_implementation,
        mesh=mesh if swiglu_implementation == SOURCE_PUSH_W2_SWIGLU_BACKWARD_IMPLEMENTATION_PALLAS_MGPU else None,
    )


def _padded_matmul_inputs(
    dy: Float[Array, "Dst E C D"],
    weighted_activation: Float[Array, "Dst E C I"],
    valid: Bool[Array, "Dst E C"],
    w2: Float[Array, "Dst E I D"],
):
    weighted_activation, dy, valid = _pad_w2_matmul_rows_for_pallas(
        weighted_activation,
        dy,
        valid,
        row_multiple=MIN_SOURCE_PUSH_W2_MATMUL_ROW_BLOCK,
    )
    block_sizes = _source_push_w2_matmul_backward_inferred_block_sizes(weighted_activation, dy, w2)
    return dy, weighted_activation, valid, block_sizes


def _matmul_implementation(implementation: str) -> str:
    if implementation == IMPLEMENTATION_REFERENCE:
        return SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_REFERENCE
    if implementation in (
        IMPLEMENTATION_PALLAS_MATMUL_REFERENCE_SWIGLU,
        IMPLEMENTATION_PALLAS_MATMUL_PALLAS_SWIGLU,
        IMPLEMENTATION_PALLAS_FUSED,
    ):
        return SOURCE_PUSH_W2_MATMUL_BACKWARD_IMPLEMENTATION_PALLAS_MGPU
    raise ValueError(f"implementation must be one of {IMPLEMENTATIONS}, got {implementation!r}")


def _timing_rows(
    config: W2CompactConfig,
    *,
    mode: str,
    implementation: str,
    timing: Timing,
    repeat_runs: int,
) -> list[dict[str, Any]]:
    rows = [
        _timing_row(
            config,
            mode=mode,
            implementation=implementation,
            row_type="repeat",
            repeat_run=repeat_run,
            repeat_runs=repeat_runs,
            first_call_time=timing.first_call_time,
            steady_state_time=steady_state_time,
        )
        for repeat_run, steady_state_time in enumerate(timing.steady_state_times)
    ]
    return [*rows, _summary_row(rows)]


def _timing_row(
    config: W2CompactConfig,
    *,
    mode: str,
    implementation: str,
    row_type: str,
    repeat_run: int | None,
    repeat_runs: int,
    first_call_time: float | None,
    steady_state_time: float,
) -> dict[str, Any]:
    useful_flops, rounded_flops = _flops_per_rank(config, mode)
    return {
        "kernel": KERNEL_NAME,
        "implementation": implementation,
        "mode": mode,
        "row_type": row_type,
        "config": asdict(config),
        "device_type": jax.devices()[0].device_kind,
        "device_count": config.ep_size,
        "backend": jax.default_backend(),
        "dtype": config.dtype,
        "repeat_run": repeat_run,
        "repeat_runs": repeat_runs,
        "first_call_time": first_call_time,
        "steady_state_time": steady_state_time,
        "useful_flops_per_rank": useful_flops,
        "rounded_flops_per_rank": rounded_flops,
        "useful_tflops_per_rank": useful_flops / steady_state_time / 1e12,
        "rounded_tflops_per_rank": rounded_flops / steady_state_time / 1e12,
        "live_rows_per_rank": config.experts_per_rank * config.live_rows_per_expert,
        "rounded_rows_per_rank": config.experts_per_rank * config.rows_per_expert,
        "error": None,
        "error_type": None,
        "error_message": None,
    }


def _summary_row(rows: list[dict[str, Any]]) -> dict[str, Any]:
    first = rows[0]
    summary = dict(first)
    summary.update(
        row_type="summary",
        repeat_run=None,
        repeat_rows=len(rows),
        min_steady_state_time=min(row["steady_state_time"] for row in rows),
        max_steady_state_time=max(row["steady_state_time"] for row in rows),
        median_steady_state_time=median(row["steady_state_time"] for row in rows),
        median_useful_tflops_per_rank=median(row["useful_tflops_per_rank"] for row in rows),
        median_rounded_tflops_per_rank=median(row["rounded_tflops_per_rank"] for row in rows),
    )
    return summary


def _error_row(config: W2CompactConfig, *, mode: str, implementation: str, exc: Exception) -> dict[str, Any]:
    return {
        "kernel": KERNEL_NAME,
        "implementation": implementation,
        "mode": mode,
        "row_type": "error",
        "config": asdict(config),
        "device_type": jax.devices()[0].device_kind,
        "device_count": config.ep_size,
        "backend": jax.default_backend(),
        "dtype": config.dtype,
        "steady_state_time": None,
        "useful_tflops_per_rank": None,
        "rounded_tflops_per_rank": None,
        "error": True,
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def _flops_per_rank(config: W2CompactConfig, mode: str) -> tuple[float, float]:
    useful_rows = config.experts_per_rank * config.live_rows_per_expert
    rounded_rows = config.experts_per_rank * config.rows_per_expert
    if mode in (MODE_D_WEIGHTED_ACTIVATION, MODE_DW2):
        multiplier = 2
    elif mode == MODE_SWIGLU_FROM_D_WEIGHTED:
        multiplier = 0
    else:
        multiplier = 4
    useful = useful_rows * config.hidden_dim * config.intermediate_dim * multiplier
    rounded = rounded_rows * config.hidden_dim * config.intermediate_dim * multiplier
    return float(useful), float(rounded)


def _block_inputs(inputs: W2CompactInputs) -> W2CompactInputs:
    _block_until_ready(inputs)
    return inputs


if __name__ == "__main__":
    main()
