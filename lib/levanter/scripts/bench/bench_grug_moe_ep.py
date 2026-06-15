# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Perf smoke for Grug MoE expert-parallel transport backends."""

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
from haliax.nn.ragged_dot import ragged_dot
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.grug._moe.common import split_moe_w13_output
from levanter.grug._moe.ep_common import _prefix_cap_counts
from levanter.grug.grug_moe import moe_mlp
from levanter.kernels.deepep import (
    deepep_collapse_local_assignments,
    deepep_combine_intranode,
    deepep_dispatch_intranode_with_assignments,
    deepep_get_dispatch_layout,
    transport_ffi,
)
from levanter.utils.activation import ActivationFunctionEnum


@dataclass(frozen=True)
class BenchResult:
    implementation: str
    compile_seconds: float
    median_seconds: float
    mean_seconds: float
    tokens_per_second: float


@dataclass(frozen=True)
class DeepEPConfigOverride:
    dispatch: transport_ffi.IntranodeConfig
    combine: transport_ffi.IntranodeConfig


@dataclass(frozen=True)
class BenchmarkContext:
    tokens: int
    mesh: Mesh
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]


_DEEPEP_COMPONENT_STAGES = (
    "deepep_dispatch",
    "deepep_dispatch_w13",
    "deepep_dispatch_w13_w2",
    "deepep_dispatch_w13_w2_collapse",
    "deepep_full",
)


def _ep_mesh(expert_axis_size: int) -> Mesh:
    devices = np.asarray(jax.devices())
    if devices.size % expert_axis_size != 0:
        raise ValueError(f"device count {devices.size} must be divisible by expert axis size {expert_axis_size}")
    data_axis_size = devices.size // expert_axis_size
    return Mesh(
        devices.reshape(data_axis_size, expert_axis_size, 1),
        axis_names=("data", "expert", "model"),
        axis_types=(AxisType.Explicit, AxisType.Explicit, AxisType.Explicit),
    )


def _balanced_topk_assignments(tokens: int, *, topk: int, num_experts: int) -> jax.Array:
    token_ids = jnp.arange(tokens, dtype=jnp.int32)[:, None]
    topk_offsets = jnp.arange(topk, dtype=jnp.int32)[None, :]
    return (token_ids * topk + topk_offsets) % num_experts


def _make_inputs(
    *,
    tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    num_experts: int,
    topk: int,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    k_x, k_weights, k_w13, k_w2 = jax.random.split(jax.random.key(6215), 4)
    x = jax.random.normal(k_x, (tokens, hidden_dim), dtype=dtype)
    selected_experts = _balanced_topk_assignments(tokens, topk=topk, num_experts=num_experts)
    combine_weights = jax.nn.sigmoid(jax.random.normal(k_weights, (tokens, topk), dtype=jnp.float32)).astype(dtype)
    w_up_gate = jax.random.normal(k_w13, (num_experts, hidden_dim, 2 * intermediate_dim), dtype=dtype)
    w_down = jax.random.normal(k_w2, (num_experts, intermediate_dim, hidden_dim), dtype=dtype)
    return x, selected_experts, combine_weights, w_up_gate, w_down


def _shard_inputs(
    mesh: Mesh,
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    x, selected_experts, combine_weights, w_up_gate, w_down = inputs
    batch_sharding = NamedSharding(mesh, P(("data", "expert"), None))
    expert_sharding = NamedSharding(mesh, P("expert", None, None))
    return (
        jax.sharding.reshard(x, batch_sharding),
        jax.sharding.reshard(selected_experts, batch_sharding),
        jax.sharding.reshard(combine_weights, batch_sharding),
        jax.sharding.reshard(w_up_gate, expert_sharding),
        jax.sharding.reshard(w_down, expert_sharding),
    )


def _forward_fn(implementation: str, mesh: Mesh, *, capacity_factor: float):
    def run(x, selected_experts, combine_weights, w_up_gate, w_down):
        return moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation=implementation,
            mesh=mesh,
            capacity_factor=capacity_factor,
            report_capacity_overflow=True,
        )

    return jax.jit(run)


def _forward_backward_fn(implementation: str, mesh: Mesh, *, capacity_factor: float):
    def loss_fn(x, selected_experts, combine_weights, w_up_gate, w_down):
        out, _dropped = moe_mlp(
            x,
            selected_experts,
            combine_weights,
            w_up_gate,
            w_down,
            activation=ActivationFunctionEnum.silu,
            implementation=implementation,
            mesh=mesh,
            capacity_factor=capacity_factor,
            report_capacity_overflow=True,
        )
        return jnp.sum(out.astype(jnp.float32))

    return jax.jit(jax.value_and_grad(loss_fn, argnums=(0, 2, 3, 4)))


def _deepep_component_local(
    x_local,
    selected_experts_local,
    combine_weights_local,
    moe_w13_local,
    moe_w2_local,
    *,
    num_experts: int,
    capacity_factor: float,
    stage: str,
    dispatch_config: transport_ffi.IntranodeConfig | None,
    combine_config: transport_ffi.IntranodeConfig | None,
):
    local_experts = moe_w13_local.shape[0]
    ep_size = num_experts // local_experts
    topk = selected_experts_local.shape[1]
    local_capacity = int(np.ceil(capacity_factor * x_local.shape[0] * topk))
    local_capacity = max(local_experts, local_capacity)
    max_recv_tokens = x_local.shape[0] * ep_size

    num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank = deepep_get_dispatch_layout(
        selected_experts_local,
        num_ranks=ep_size,
        num_experts=num_experts,
    )
    (
        recv_x,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        local_group_sizes,
        num_recv_tokens,
        x_dispatch,
        assignment_weights,
        recv_token_indices,
        assignment_destinations,
    ) = deepep_dispatch_intranode_with_assignments(
        x_local,
        selected_experts_local,
        combine_weights_local,
        num_tokens_per_rank,
        num_tokens_per_expert,
        is_token_in_rank,
        num_experts=num_experts,
        dispatch_config=dispatch_config,
        combine_config=combine_config,
        max_recv_tokens=max_recv_tokens,
    )
    accepted_group_sizes = _prefix_cap_counts(local_group_sizes, capacity=local_capacity)
    x_dispatch = x_dispatch[:local_capacity]
    assignment_weights = assignment_weights[:local_capacity]
    recv_token_indices = recv_token_indices[:local_capacity]
    if stage == "deepep_dispatch":
        local_value = jnp.sum(x_dispatch.astype(jnp.float32)) + jnp.sum(assignment_weights.astype(jnp.float32))
        return jax.lax.psum(local_value, ("data", "expert"))

    w13_out = ragged_dot(x_dispatch, moe_w13_local, accepted_group_sizes)
    if stage == "deepep_dispatch_w13":
        return jax.lax.psum(jnp.sum(w13_out.astype(jnp.float32)), ("data", "expert"))

    moe_dim = moe_w2_local.shape[1]
    gate, up = split_moe_w13_output(w13_out, intermediate_dim=moe_dim, interleaved=False)
    out_dispatch = ragged_dot(jax.nn.silu(gate) * up, moe_w2_local, accepted_group_sizes)
    if stage == "deepep_dispatch_w13_w2":
        return jax.lax.psum(jnp.sum(out_dispatch.astype(jnp.float32)), ("data", "expert"))

    recv_out = deepep_collapse_local_assignments(
        out_dispatch,
        assignment_weights,
        recv_token_indices,
        assignment_destinations,
        accepted_group_sizes,
        num_recv_tokens,
        recv_capacity=recv_x.shape[0],
    )
    if stage == "deepep_dispatch_w13_w2_collapse":
        return jax.lax.psum(jnp.sum(recv_out.astype(jnp.float32)), ("data", "expert"))

    if stage != "deepep_full":
        raise AssertionError(f"Unhandled DeepEP component stage {stage!r}")
    out_local, _ = deepep_combine_intranode(
        recv_out,
        recv_topk_weights,
        recv_src_idx,
        rank_prefix_matrix,
        channel_prefix_matrix,
        recv_channel_prefix_matrix,
        send_head,
        num_recv_tokens,
        is_token_in_rank,
    )
    return jax.lax.psum(jnp.sum(out_local.astype(jnp.float32)), ("data", "expert"))


def _deepep_component_fn(
    stage: str,
    mesh: Mesh,
    *,
    num_experts: int,
    capacity_factor: float,
    deepep_config: DeepEPConfigOverride | None,
):
    batch_spec = P(("data", "expert"), None)
    w_spec = P("expert", None, None)
    shard_fn = shard_map(
        partial(
            _deepep_component_local,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
            stage=stage,
            dispatch_config=deepep_config.dispatch if deepep_config is not None else None,
            combine_config=deepep_config.combine if deepep_config is not None else None,
        ),
        mesh=mesh,
        in_specs=(batch_spec, batch_spec, batch_spec, w_spec, w_spec),
        out_specs=P(),
        check_vma=False,
    )
    return jax.jit(shard_fn)


def _block_until_ready(value):
    jax.block_until_ready(value)
    return value


def _time(
    fn, args: tuple[jax.Array, ...], *, tokens: int, warmup: int, steps: int, implementation: str
) -> BenchResult:
    start = time.perf_counter()
    _block_until_ready(fn(*args))
    compile_seconds = time.perf_counter() - start

    for _ in range(warmup):
        _block_until_ready(fn(*args))

    samples: list[float] = []
    for _ in range(steps):
        start = time.perf_counter()
        _block_until_ready(fn(*args))
        samples.append(time.perf_counter() - start)

    median_seconds = statistics.median(samples)
    mean_seconds = statistics.fmean(samples)
    return BenchResult(
        implementation=implementation,
        compile_seconds=compile_seconds,
        median_seconds=median_seconds,
        mean_seconds=mean_seconds,
        tokens_per_second=tokens / median_seconds,
    )


def _as_json(result: BenchResult) -> dict[str, float | str]:
    return {
        "implementation": result.implementation,
        "compile_seconds": result.compile_seconds,
        "median_seconds": result.median_seconds,
        "mean_seconds": result.mean_seconds,
        "tokens_per_second": result.tokens_per_second,
    }


def _max_tree_abs_diff(reference, candidate) -> float:
    max_diff = 0.0
    for reference_leaf, candidate_leaf in zip(
        jax.tree_util.tree_leaves(reference),
        jax.tree_util.tree_leaves(candidate),
        strict=True,
    ):
        diff = jnp.max(jnp.abs(reference_leaf.astype(jnp.float32) - candidate_leaf.astype(jnp.float32)))
        max_diff = max(max_diff, float(diff))
    return max_diff


def _deepep_config_override(args: argparse.Namespace) -> DeepEPConfigOverride | None:
    dispatch_values = (
        args.deepep_dispatch_sms,
        args.deepep_dispatch_max_send_tokens,
        args.deepep_dispatch_max_recv_tokens,
    )
    combine_values = (
        args.deepep_combine_sms,
        args.deepep_combine_max_send_tokens,
        args.deepep_combine_max_recv_tokens,
    )
    if all(value is None for value in (*dispatch_values, *combine_values)):
        return None
    if any(value is None for value in (*dispatch_values, *combine_values)):
        raise ValueError("DeepEP config override requires all dispatch and combine config fields")

    dispatch = transport_ffi.IntranodeConfig(
        num_sms=args.deepep_dispatch_sms,
        num_max_send_tokens=args.deepep_dispatch_max_send_tokens,
        num_max_recv_tokens=args.deepep_dispatch_max_recv_tokens,
    )
    combine = transport_ffi.IntranodeConfig(
        num_sms=args.deepep_combine_sms,
        num_max_send_tokens=args.deepep_combine_max_send_tokens,
        num_max_recv_tokens=args.deepep_combine_max_recv_tokens,
    )
    return DeepEPConfigOverride(dispatch=dispatch, combine=combine)


def _deepep_config_payload(config: DeepEPConfigOverride | None) -> dict[str, dict[str, int]] | None:
    if config is None:
        return None
    return {
        "dispatch": {
            "num_sms": config.dispatch.num_sms,
            "num_max_send_tokens": config.dispatch.num_max_send_tokens,
            "num_max_recv_tokens": config.dispatch.num_max_recv_tokens,
        },
        "combine": {
            "num_sms": config.combine.num_sms,
            "num_max_send_tokens": config.combine.num_max_send_tokens,
            "num_max_recv_tokens": config.combine.num_max_recv_tokens,
        },
    }


def _benchmark_context(args: argparse.Namespace, expert_axis_size: int) -> BenchmarkContext:
    tokens = args.batch_size * args.seq_len
    dtype = jnp.bfloat16 if args.dtype == "bfloat16" else jnp.float32
    mesh = _ep_mesh(expert_axis_size)
    inputs = _shard_inputs(
        mesh,
        _make_inputs(
            tokens=tokens,
            hidden_dim=args.hidden_dim,
            intermediate_dim=args.intermediate_dim,
            num_experts=args.num_experts,
            topk=args.topk,
            dtype=dtype,
        ),
    )
    return BenchmarkContext(tokens=tokens, mesh=mesh, inputs=inputs)


def _shape_payload(args: argparse.Namespace, tokens: int) -> dict[str, float | int | str]:
    return {
        "tokens": tokens,
        "batch_size": args.batch_size,
        "seq_len": args.seq_len,
        "hidden_dim": args.hidden_dim,
        "intermediate_dim": args.intermediate_dim,
        "num_experts": args.num_experts,
        "topk": args.topk,
        "capacity_factor": args.capacity_factor,
        "dtype": args.dtype,
        "mode": args.mode,
    }


def _mesh_payload(expert_axis_size: int) -> dict[str, int]:
    return {
        "devices": len(jax.devices()),
        "expert_axis_size": expert_axis_size,
    }


def _emit_payload(args: argparse.Namespace, payload: dict[str, object]) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    print(rendered)
    if args.json_output is None:
        return
    with open(args.json_output, "w", encoding="utf-8") as handle:
        handle.write(rendered)
        handle.write("\n")


def _component_results(
    args: argparse.Namespace,
    context: BenchmarkContext,
    deepep_config: DeepEPConfigOverride | None,
) -> list[BenchResult]:
    return [
        _time(
            _deepep_component_fn(
                stage,
                context.mesh,
                num_experts=args.num_experts,
                capacity_factor=args.capacity_factor,
                deepep_config=deepep_config,
            ),
            context.inputs,
            tokens=context.tokens,
            warmup=args.warmup,
            steps=args.steps,
            implementation=stage,
        )
        for stage in _DEEPEP_COMPONENT_STAGES
    ]


def _run_deepep_component_benchmark(
    args: argparse.Namespace,
    context: BenchmarkContext,
    expert_axis_size: int,
    deepep_config: DeepEPConfigOverride | None,
) -> None:
    with jax.set_mesh(context.mesh):
        results = _component_results(args, context, deepep_config)
    _emit_payload(
        args,
        {
            "shape": _shape_payload(args, context.tokens),
            "mesh": _mesh_payload(expert_axis_size),
            "deepep_config": _deepep_config_payload(deepep_config),
            "results": [_as_json(result) for result in results],
        },
    )


def _correctness_vs_ring(
    args: argparse.Namespace,
    context: BenchmarkContext,
    implementations: tuple[str, ...],
) -> tuple[float, dict[str, dict[str, float]]]:
    ring_out, ring_dropped = _block_until_ready(
        _forward_fn("ring", context.mesh, capacity_factor=args.capacity_factor)(*context.inputs)
    )
    ring_forward_backward = None
    if args.mode == "forward_backward":
        ring_forward_backward = _block_until_ready(
            _forward_backward_fn("ring", context.mesh, capacity_factor=args.capacity_factor)(*context.inputs)
        )
    max_reference_abs = float(jnp.max(jnp.abs(ring_out.astype(jnp.float32))))
    correctness: dict[str, dict[str, float]] = {}
    for implementation in implementations:
        candidate_out, candidate_dropped = _block_until_ready(
            _forward_fn(implementation, context.mesh, capacity_factor=args.capacity_factor)(*context.inputs)
        )
        diff = jnp.abs(ring_out.astype(jnp.float32) - candidate_out.astype(jnp.float32))
        max_abs_diff = float(jnp.max(diff))
        correctness[implementation] = {
            "max_abs_diff": max_abs_diff,
            "mean_abs_diff": float(jnp.mean(diff)),
            "max_relative_diff": max_abs_diff / max(max_reference_abs, 1.0),
            "ring_dropped": int(ring_dropped),
            "candidate_dropped": int(candidate_dropped),
            "dropped_abs_diff": int(jnp.abs(ring_dropped - candidate_dropped)),
        }
        if ring_forward_backward is not None:
            candidate_loss, candidate_grads = _block_until_ready(
                _forward_backward_fn(implementation, context.mesh, capacity_factor=args.capacity_factor)(
                    *context.inputs
                )
            )
            ring_loss, ring_grads = ring_forward_backward
            correctness[implementation]["loss_abs_diff"] = float(
                jnp.abs(ring_loss.astype(jnp.float32) - candidate_loss.astype(jnp.float32))
            )
            correctness[implementation]["grad_max_abs_diff"] = _max_tree_abs_diff(ring_grads, candidate_grads)
    return max_reference_abs, correctness


def _benchmark_results(
    args: argparse.Namespace,
    context: BenchmarkContext,
    implementations: tuple[str, ...],
) -> list[BenchResult]:
    fn_factory = _forward_fn if args.mode == "forward" else _forward_backward_fn
    return [
        _time(
            fn_factory(implementation, context.mesh, capacity_factor=args.capacity_factor),
            context.inputs,
            tokens=context.tokens,
            warmup=args.warmup,
            steps=args.steps,
            implementation=implementation,
        )
        for implementation in implementations
    ]


def _run_moe_benchmark(
    args: argparse.Namespace,
    context: BenchmarkContext,
    expert_axis_size: int,
    deepep_config: DeepEPConfigOverride | None,
) -> None:
    implementations = tuple(args.implementations)
    with jax.set_mesh(context.mesh):
        max_reference_abs, correctness = _correctness_vs_ring(args, context, implementations)
        results = _benchmark_results(args, context, implementations)
    _emit_payload(
        args,
        {
            "shape": _shape_payload(args, context.tokens),
            "mesh": _mesh_payload(expert_axis_size),
            "deepep_config": _deepep_config_payload(deepep_config),
            "max_reference_abs": max_reference_abs,
            "correctness_vs_ring": correctness,
            "results": [_as_json(result) for result in results],
        },
    )


def _run_benchmark(
    args: argparse.Namespace,
    expert_axis_size: int,
    deepep_config: DeepEPConfigOverride | None,
) -> None:
    if deepep_config is not None and args.mode != "deepep_components":
        raise ValueError("DeepEP config override is only supported with --mode deepep_components")

    context = _benchmark_context(args, expert_axis_size)
    if args.mode == "deepep_components":
        _run_deepep_component_benchmark(args, context, expert_axis_size, deepep_config)
        return
    _run_moe_benchmark(args, context, expert_axis_size, deepep_config)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=2560)
    parser.add_argument("--intermediate-dim", type=int, default=1280)
    parser.add_argument("--num-experts", type=int, default=64)
    parser.add_argument("--topk", type=int, default=4)
    parser.add_argument("--capacity-factor", type=float, default=1.25)
    parser.add_argument("--expert-axis-size", type=int, default=None)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--mode", choices=("forward", "forward_backward", "deepep_components"), default="forward")
    parser.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    parser.add_argument("--implementations", nargs="+", default=("ring", "deepep", "assigned_token"))
    parser.add_argument("--json-output", type=str, default=None)
    parser.add_argument("--deepep-dispatch-sms", type=int, default=None)
    parser.add_argument("--deepep-dispatch-max-send-tokens", type=int, default=None)
    parser.add_argument("--deepep-dispatch-max-recv-tokens", type=int, default=None)
    parser.add_argument("--deepep-combine-sms", type=int, default=None)
    parser.add_argument("--deepep-combine-max-send-tokens", type=int, default=None)
    parser.add_argument("--deepep-combine-max-recv-tokens", type=int, default=None)
    args = parser.parse_args()

    expert_axis_size = args.expert_axis_size or len(jax.devices())
    if "deepep" in args.implementations and len(jax.devices()) != expert_axis_size:
        raise ValueError(
            "DeepEP intranode transport currently requires the expert group to span all visible local GPUs; "
            f"got visible_devices={len(jax.devices())}, expert_axis_size={expert_axis_size}"
        )
    _run_benchmark(args, expert_axis_size, _deepep_config_override(args))


if __name__ == "__main__":
    main()
