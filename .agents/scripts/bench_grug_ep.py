#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Benchmark Grug MoE expert-parallel backends on local JAX devices."""

import argparse
import json
import time
from collections.abc import Callable
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P
from levanter.grug._moe.common import _DEFAULT_EP_CAPACITY_FACTOR
from levanter.grug.grug_moe import moe_mlp
from levanter.utils.activation import ActivationFunctionEnum


def _json_print(payload: dict) -> None:
    if jax.process_index() == 0:
        print(json.dumps(payload, sort_keys=True), flush=True)


def _block_until_ready(value):
    return jax.block_until_ready(jax.tree_util.tree_map(lambda x: x, value))


def _time_call(fn: Callable, args: tuple, *, warmup: int, iters: int) -> tuple[float, float]:
    start_compile = time.perf_counter()
    _block_until_ready(fn(*args))
    compile_s = time.perf_counter() - start_compile

    for _ in range(warmup):
        _block_until_ready(fn(*args))

    start = time.perf_counter()
    for _ in range(iters):
        _block_until_ready(fn(*args))
    steady_s = (time.perf_counter() - start) / iters
    return compile_s, steady_s


def _scalar_int(value) -> int:
    return int(np.asarray(jax.device_get(value)))


def _benchmark_dropped_assignments(value, *, pass_mode: str) -> int:
    if pass_mode == "forward":
        _, dropped = value
        return _scalar_int(dropped)

    (loss, dropped), _ = value
    del loss
    return _scalar_int(dropped)


def _make_mesh(num_devices: int, *, data_axis: int, model_axis: int) -> Mesh:
    devices = np.asarray(jax.devices()[:num_devices], dtype=object)
    if devices.size != num_devices:
        raise ValueError(f"Requested {num_devices} devices, but JAX only sees {devices.size}")
    if num_devices % (data_axis * model_axis) != 0:
        raise ValueError(
            f"num_devices={num_devices} must be divisible by data_axis * model_axis={data_axis * model_axis}"
        )
    expert_axis = num_devices // (data_axis * model_axis)
    return Mesh(
        devices.reshape(data_axis, expert_axis, model_axis),
        ("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _sharded_inputs(
    *,
    mesh: Mesh,
    tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    local_experts: int,
    topk: int,
    dtype: jnp.dtype,
    seed: int,
):
    expert_axis = mesh.shape["expert"]
    num_experts = local_experts * expert_axis
    keys = jax.random.split(jax.random.key(seed), 6)
    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))

    x = jax.random.normal(keys[0], (tokens, hidden_dim), dtype=dtype)
    selected_experts = jax.random.randint(keys[1], (tokens, topk), 0, num_experts, dtype=jnp.int32)
    combine_logits = jax.random.normal(keys[2], (tokens, topk), dtype=jnp.float32)
    combine_weights = jax.nn.softmax(combine_logits, axis=-1).astype(dtype)
    w_gate = jax.random.normal(keys[3], (num_experts, hidden_dim, intermediate_dim), dtype=dtype)
    w_up = jax.random.normal(keys[4], (num_experts, hidden_dim, intermediate_dim), dtype=dtype)
    w_down = jax.random.normal(keys[5], (num_experts, intermediate_dim, hidden_dim), dtype=dtype)
    w_gate_up = jnp.concatenate([w_gate, w_up], axis=-1)

    return (
        jax.device_put(x, batch_sharding),
        jax.device_put(selected_experts, batch_sharding),
        jax.device_put(combine_weights, batch_sharding),
        jax.device_put(w_gate_up, expert_sharding),
        jax.device_put(w_down, expert_sharding),
    )


def _forward_fn(
    x,
    selected_experts,
    combine_weights,
    w_gate_up,
    w_down,
    *,
    mesh: Mesh,
    implementation: str,
    capacity_factor: float,
):
    return moe_mlp(
        x,
        selected_experts,
        combine_weights,
        w_gate_up,
        w_down,
        activation=ActivationFunctionEnum.silu,
        implementation=implementation,
        mesh=mesh,
        capacity_factor=capacity_factor,
        report_capacity_overflow=True,
    )


def _loss_fn(
    x,
    selected_experts,
    combine_weights,
    w_gate_up,
    w_down,
    *,
    mesh: Mesh,
    implementation: str,
    capacity_factor: float,
):
    out = _forward_fn(
        x,
        selected_experts,
        combine_weights,
        w_gate_up,
        w_down,
        mesh=mesh,
        implementation=implementation,
        capacity_factor=capacity_factor,
    )
    if isinstance(out, tuple):
        out, dropped = out
    else:
        dropped = jnp.array(0, dtype=jnp.int32)
    return jnp.mean(out.astype(jnp.float32) * out.astype(jnp.float32)), dropped


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=131072)
    parser.add_argument("--hidden-dim", type=int, default=5120)
    parser.add_argument("--intermediate-dim", type=int, default=2560)
    parser.add_argument("--local-experts", type=int, default=8)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--capacity-factor", type=float, default=_DEFAULT_EP_CAPACITY_FACTOR)
    parser.add_argument("--num-devices", type=int, default=0)
    parser.add_argument("--data-axis", type=int, default=1)
    parser.add_argument("--model-axis", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--pass-mode", choices=("forward", "forward_backward"), default="forward")
    parser.add_argument("--implementations", nargs="+", default=("ring", "ragged_all_to_all"))
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    num_devices = args.num_devices or jax.local_device_count()
    mesh = _make_mesh(num_devices, data_axis=args.data_axis, model_axis=args.model_axis)

    with jax.set_mesh(mesh):
        inputs = _sharded_inputs(
            mesh=mesh,
            tokens=args.tokens,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            local_experts=args.local_experts,
            topk=args.topk,
            dtype=dtype,
            seed=args.seed,
        )

        for implementation in args.implementations:
            if args.pass_mode == "forward":
                bench_fn = jax.jit(
                    partial(
                        _forward_fn,
                        mesh=mesh,
                        implementation=implementation,
                        capacity_factor=args.capacity_factor,
                    )
                )
            else:
                bench_fn = jax.jit(
                    jax.value_and_grad(
                        partial(
                            _loss_fn,
                            mesh=mesh,
                            implementation=implementation,
                            capacity_factor=args.capacity_factor,
                        ),
                        has_aux=True,
                        argnums=(0, 3, 4),
                    )
                )

            compile_s, steady_s = _time_call(bench_fn, inputs, warmup=args.warmup, iters=args.iters)
            dropped_assignments = _benchmark_dropped_assignments(bench_fn(*inputs), pass_mode=args.pass_mode)
            _json_print(
                {
                    "implementation": implementation,
                    "pass_mode": args.pass_mode,
                    "compile_s": compile_s,
                    "steady_s": steady_s,
                    "tokens_per_s": args.tokens / steady_s,
                    "tokens": args.tokens,
                    "topk": args.topk,
                    "hidden_dim": args.hidden_dim,
                    "intermediate_dim": args.intermediate_dim,
                    "local_experts": args.local_experts,
                    "expert_axis": mesh.shape["expert"],
                    "num_devices": num_devices,
                    "dtype": args.dtype,
                    "capacity_factor": args.capacity_factor,
                    "dropped_assignments": dropped_assignments,
                }
            )


if __name__ == "__main__":
    main()
