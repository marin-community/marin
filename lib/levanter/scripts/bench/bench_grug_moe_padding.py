# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure the training-step effect of skipping padding in Grug MoE dispatch.

The benchmark runs two fresh processes with identical padded shapes and weights:

- ``padding-counted`` treats every position as a routed token, matching the old behavior.
- ``padding-skipped`` passes a token-validity mask to dispatch.

Fresh processes keep allocator peaks independent. The JSON report includes average step time,
capacity and padding counts, and peak allocator memory when the backend exposes it. Defaults are
small enough for a workstation; pass production-local shapes and an expert-parallel backend for a
representative accelerator measurement.

Both cases run the same padded shapes, so peak memory is expected to match: every backend sizes
its buffers from the padded token count. Runtime shrinks where transport rows and grouped-GEMM
segments follow the valid rows (``scatter``, ``sonic``, ``ragged_all_to_all``) and stays flat for
the dense fixed-buffer collectives (``fixed_all_to_all``, ``fixed_pooled_wave_all_to_all``).
Shrinking the buffers themselves needs a static valid-token budget, which packing provides.

Examples::

    python lib/levanter/scripts/bench/bench_grug_moe_padding.py
    python lib/levanter/scripts/bench/bench_grug_moe_padding.py \
        --implementation ragged_all_to_all --expert-axis-size 8 \
        --tokens 65536 --hidden 3072 --intermediate 3072 --experts 384 --top-k 8
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from enum import StrEnum

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug.grug_moe import moe_mlp


class BenchmarkCase(StrEnum):
    PADDING_COUNTED = "padding-counted"
    PADDING_SKIPPED = "padding-skipped"


@dataclass(frozen=True)
class BenchmarkConfig:
    implementation: str
    expert_axis_size: int
    tokens: int
    hidden: int
    intermediate: int
    experts: int
    top_k: int
    padding_fraction: float
    capacity_factor: float
    transport_capacity_factor: float
    num_expert_waves: int
    warmup: int
    iterations: int
    forward_only: bool


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    implementation: str
    device_kind: str
    tokens: int
    valid_tokens: int
    top_k: int
    forward_only: bool
    mean_step_ms: float
    peak_bytes_in_use: int | None
    dropped_assignments: int
    skipped_padding_assignments: int


def _validate(config: BenchmarkConfig) -> None:
    positive = {
        "expert_axis_size": config.expert_axis_size,
        "tokens": config.tokens,
        "hidden": config.hidden,
        "intermediate": config.intermediate,
        "experts": config.experts,
        "top_k": config.top_k,
        "num_expert_waves": config.num_expert_waves,
        "iterations": config.iterations,
    }
    for name, value in positive.items():
        if value <= 0:
            raise ValueError(f"{name} must be positive, got {value}")
    if config.warmup < 0:
        raise ValueError(f"warmup must be non-negative, got {config.warmup}")
    if not 0 <= config.padding_fraction < 1:
        raise ValueError(f"padding_fraction must be in [0, 1), got {config.padding_fraction}")
    if config.capacity_factor <= 0 or config.transport_capacity_factor <= 0:
        raise ValueError("capacity factors must be positive")
    if config.top_k > config.experts:
        raise ValueError(f"top_k={config.top_k} must not exceed experts={config.experts}")
    if config.experts % config.expert_axis_size != 0:
        raise ValueError(f"experts={config.experts} must be divisible by expert_axis_size={config.expert_axis_size}")


def _run_case(config: BenchmarkConfig, case: BenchmarkCase) -> BenchmarkResult:
    if config.expert_axis_size > len(jax.devices()):
        raise ValueError(f"expert_axis_size={config.expert_axis_size} exceeds available devices={len(jax.devices())}")
    devices = np.asarray(jax.devices()[: config.expert_axis_size])
    mesh = Mesh(devices, ("expert",), axis_types=(AxisType.Explicit,))
    token_sharding = NamedSharding(mesh, P("expert"))
    token_matrix_sharding = NamedSharding(mesh, P("expert", None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))

    key_x, key_experts, key_logits, key_up_gate, key_down = jax.random.split(jax.random.key(0), 5)
    x = jax.random.normal(key_x, (config.tokens, config.hidden), dtype=jnp.bfloat16)
    selected_experts = jax.random.randint(
        key_experts,
        (config.tokens, config.top_k),
        minval=0,
        maxval=config.experts,
        dtype=jnp.int32,
    )
    combine_weights = jax.nn.softmax(
        jax.random.normal(key_logits, (config.tokens, config.top_k), dtype=jnp.float32),
        axis=-1,
    ).astype(jnp.bfloat16)
    w_up_gate = jax.random.normal(
        key_up_gate,
        (config.experts, config.hidden, 2 * config.intermediate),
        dtype=jnp.bfloat16,
    )
    w_down = jax.random.normal(
        key_down,
        (config.experts, config.intermediate, config.hidden),
        dtype=jnp.bfloat16,
    )
    valid_tokens = max(1, round(config.tokens * (1 - config.padding_fraction)))
    token_valid = jnp.arange(config.tokens) < valid_tokens
    if case == BenchmarkCase.PADDING_COUNTED:
        token_valid = jnp.ones_like(token_valid)

    x = jax.device_put(x, token_matrix_sharding)
    selected_experts = jax.device_put(selected_experts, token_matrix_sharding)
    combine_weights = jax.device_put(combine_weights, token_matrix_sharding)
    token_valid = jax.device_put(token_valid, token_sharding)
    w_up_gate = jax.device_put(w_up_gate, expert_sharding)
    w_down = jax.device_put(w_down, expert_sharding)

    pooled_capacity = (
        config.transport_capacity_factor if config.implementation == "fixed_pooled_wave_all_to_all" else None
    )

    def objective(x, selected_experts, combine_weights, token_valid, w_up_gate, w_down):
        out, overflow = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            token_valid=token_valid,
            implementation=config.implementation,
            mesh=mesh,
            capacity_factor=config.capacity_factor,
            pooled_transport_capacity_factor=pooled_capacity,
            num_expert_waves=config.num_expert_waves,
            report_capacity_overflow=True,
        )
        return jnp.mean(out.astype(jnp.float32) ** 2), overflow

    if config.forward_only:
        step = jax.jit(objective)
    else:
        step = jax.jit(jax.value_and_grad(objective, argnums=(0, 4, 5), has_aux=True))
    args = (x, selected_experts, combine_weights, token_valid, w_up_gate, w_down)

    with jax.set_mesh(mesh):
        result = jax.block_until_ready(step(*args))
        for _ in range(config.warmup - 1):
            result = jax.block_until_ready(step(*args))
        start = time.perf_counter()
        for _ in range(config.iterations):
            result = step(*args)
        result = jax.block_until_ready(result)
        elapsed = time.perf_counter() - start

    overflow = result[1] if config.forward_only else result[0][1]
    memory_stats = jax.local_devices()[0].memory_stats() or {}
    peak_bytes = memory_stats.get("peak_bytes_in_use")
    return BenchmarkResult(
        case=case,
        implementation=config.implementation,
        device_kind=jax.devices()[0].device_kind,
        tokens=config.tokens,
        valid_tokens=(config.tokens if case == BenchmarkCase.PADDING_COUNTED else valid_tokens),
        top_k=config.top_k,
        forward_only=config.forward_only,
        mean_step_ms=elapsed * 1000 / config.iterations,
        peak_bytes_in_use=None if peak_bytes is None else int(peak_bytes),
        dropped_assignments=int(overflow.dropped),
        skipped_padding_assignments=int(overflow.padding_skipped),
    )


def _child_command(config: BenchmarkConfig, case: BenchmarkCase) -> list[str]:
    command = [
        sys.executable,
        __file__,
        "--case",
        case,
        "--implementation",
        config.implementation,
        "--expert-axis-size",
        str(config.expert_axis_size),
        "--tokens",
        str(config.tokens),
        "--hidden",
        str(config.hidden),
        "--intermediate",
        str(config.intermediate),
        "--experts",
        str(config.experts),
        "--top-k",
        str(config.top_k),
        "--padding-fraction",
        str(config.padding_fraction),
        "--capacity-factor",
        str(config.capacity_factor),
        "--transport-capacity-factor",
        str(config.transport_capacity_factor),
        "--num-expert-waves",
        str(config.num_expert_waves),
        "--warmup",
        str(config.warmup),
        "--iterations",
        str(config.iterations),
    ]
    if config.forward_only:
        command.append("--forward-only")
    return command


def _compare(config: BenchmarkConfig) -> dict[str, object]:
    environment = os.environ.copy()
    environment["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    results = []
    for case in BenchmarkCase:
        completed = subprocess.run(
            _child_command(config, case),
            env=environment,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"{case} benchmark failed:\n{completed.stderr}")
        results.append(json.loads(completed.stdout))

    counted, skipped = results
    runtime_reduction = 100 * (counted["mean_step_ms"] - skipped["mean_step_ms"]) / counted["mean_step_ms"]
    counted_peak = counted["peak_bytes_in_use"]
    skipped_peak = skipped["peak_bytes_in_use"]
    memory_reduction = None
    if counted_peak is not None and skipped_peak is not None:
        memory_reduction = 100 * (counted_peak - skipped_peak) / counted_peak
    return {
        "results": results,
        "padding_skipped_runtime_reduction_percent": runtime_reduction,
        "padding_skipped_peak_memory_reduction_percent": memory_reduction,
    }


def _parse_args() -> tuple[BenchmarkConfig, BenchmarkCase | None]:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--implementation", default="scatter")
    parser.add_argument("--expert-axis-size", type=int, default=1)
    parser.add_argument("--tokens", type=int, default=4096)
    parser.add_argument("--hidden", type=int, default=512)
    parser.add_argument("--intermediate", type=int, default=256)
    parser.add_argument("--experts", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--padding-fraction", type=float, default=0.5)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--transport-capacity-factor", type=float, default=1.25)
    parser.add_argument("--num-expert-waves", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--forward-only", action="store_true")
    parser.add_argument("--case", choices=tuple(BenchmarkCase), default=None, help=argparse.SUPPRESS)
    args = parser.parse_args()
    config = BenchmarkConfig(
        implementation=args.implementation,
        expert_axis_size=args.expert_axis_size,
        tokens=args.tokens,
        hidden=args.hidden,
        intermediate=args.intermediate,
        experts=args.experts,
        top_k=args.top_k,
        padding_fraction=args.padding_fraction,
        capacity_factor=args.capacity_factor,
        transport_capacity_factor=args.transport_capacity_factor,
        num_expert_waves=args.num_expert_waves,
        warmup=args.warmup,
        iterations=args.iterations,
        forward_only=args.forward_only,
    )
    return config, None if args.case is None else BenchmarkCase(args.case)


def main() -> None:
    config, case = _parse_args()
    _validate(config)
    report = asdict(_run_case(config, case)) if case is not None else _compare(config)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
