# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the experimental MoE dispatch-up Mosaic GPU subkernel."""

import argparse
import math
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax, shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from haliax.nn.ragged_dot import ragged_dot
from levanter.grug._moe.ep_common import (
    _clip_receiver_group_sizes,
    _compact_by_keep_mask,
    _expert_prefix_keep_mask,
    _local_permute_from_counts,
    _permute_by_global_expert,
    _shard_a2a_params,
)
from levanter.grug._moe.ep_padded_all_to_all import _dispatch_fixed_buckets
from levanter.kernels.pallas.moe_dispatch_up.mosaic_gpu import (
    dispatch_prepacked_moe_dispatch_up_mosaic_gpu_direct_ready_local,
    dispatch_prepacked_moe_dispatch_up_mosaic_gpu_local,
    dispatch_prepacked_moe_dispatch_up_mosaic_gpu_ready_local,
    compute_moe_up_mosaic_gpu_local,
    compute_moe_up_mosaic_gpu_block_ready_local,
    compute_moe_up_mosaic_gpu_ready_local,
)
from levanter.kernels.pallas.moe_dispatch_up.reference import (
    MoeDispatchUpLayout,
    dispatch_prepacked_moe_dispatch_up_reference,
    compute_moe_up_from_layout_reference,
    prepack_moe_dispatch_up_reference,
)


def _time_block(label: str, fn: Callable[[], jax.Array | MoeDispatchUpLayout]) -> tuple[object, float]:
    start = time.perf_counter()
    result = fn()
    jax.block_until_ready(result)
    elapsed = time.perf_counter() - start
    print(f"{label}: {elapsed * 1e3:.3f} ms")
    return result, elapsed


def _measure_steady_state(
    label: str,
    fn: Callable[[], jax.Array | MoeDispatchUpLayout],
    *,
    warmup_steps: int,
    bench_iters: int,
) -> float:
    for _ in range(warmup_steps):
        jax.block_until_ready(fn())

    times = []
    for _ in range(bench_iters):
        start = time.perf_counter()
        jax.block_until_ready(fn())
        times.append(time.perf_counter() - start)
    times_ms = np.asarray(times) * 1e3
    print(
        f"{label}/steady: "
        f"mean={float(np.mean(times_ms)):.3f} ms "
        f"min={float(np.min(times_ms)):.3f} ms "
        f"max={float(np.max(times_ms)):.3f} ms "
        f"iters={bench_iters}"
    )
    return float(np.mean(times_ms))


def _print_speedup(label: str, baseline_ms: float, candidate_ms: float) -> None:
    print(
        f"{label}_speedup: "
        f"{baseline_ms / candidate_ms:.3f}x baseline={baseline_ms:.3f} ms candidate={candidate_ms:.3f} ms"
    )


def _print_roofline(
    *,
    ep_size: int,
    tokens_per_rank: int,
    experts_per_rank: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    dtype_bytes: int,
    dispatch_ms: float | None,
    w13_ms: float,
) -> None:
    routed_rows = ep_size * tokens_per_rank * top_k
    dispatch_payload_bytes = routed_rows * hidden * dtype_bytes
    w13_flops = 2 * routed_rows * hidden * (2 * intermediate)
    w13_bytes = (
        routed_rows * hidden * dtype_bytes
        + ep_size * experts_per_rank * hidden * (2 * intermediate) * dtype_bytes
        + routed_rows * intermediate * dtype_bytes
    )
    dispatch_payload = f"{dispatch_payload_bytes / 1024:.1f} KiB"
    if dispatch_ms is None:
        dispatch_payload_bw = "not-run"
    else:
        dispatch_payload_bw = f"{dispatch_payload_bytes / (dispatch_ms / 1e3) / 1e9:.6f} GB/s"
    w13_tflops = w13_flops / (w13_ms / 1e3) / 1e12
    w13_intensity = w13_flops / w13_bytes
    w13_hbm_bound_tflops = 3.35e12 * w13_intensity / 1e12
    h100_bf16_peak_tflops = 989.0
    w13_roofline_tflops = min(h100_bf16_peak_tflops * ep_size, w13_hbm_bound_tflops * ep_size)
    print(
        "roofline: "
        f"routed_rows={routed_rows} "
        f"dispatch_payload={dispatch_payload} "
        f"dispatch_payload_bw={dispatch_payload_bw} "
        f"w13_flops={w13_flops / 1e6:.3f} MFLOP "
        f"w13_bytes={w13_bytes / 1024:.1f} KiB "
        f"w13_intensity={w13_intensity:.3f} flop/byte "
        f"w13_measured={w13_tflops:.6f} TFLOP/s "
        f"w13_h100_sxm_roofline_estimate={w13_roofline_tflops:.3f} TFLOP/s"
    )


def _make_inputs(
    *,
    ep_size: int,
    tokens_per_rank: int,
    experts_per_rank: int,
    top_k: int,
    hidden: int,
    intermediate: int,
    dtype: jnp.dtype,
    weight_init: str,
    weight_std: float | None,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array]:
    num_experts = ep_size * experts_per_rank
    k_x, k_w = jax.random.split(jax.random.key(6597))
    x_by_rank = jax.random.normal(k_x, (ep_size, tokens_per_rank, hidden), dtype=dtype)
    rank_ids = jnp.arange(ep_size, dtype=jnp.int32)[:, None, None]
    token_ids = jnp.arange(tokens_per_rank, dtype=jnp.int32)[None, :, None]
    slot_ids = jnp.arange(top_k, dtype=jnp.int32)[None, None, :]
    expert_ids = (rank_ids * experts_per_rank + token_ids + slot_ids) % num_experts
    router_weights = jax.nn.softmax(
        jnp.arange(ep_size * tokens_per_rank * top_k, dtype=jnp.float32).reshape(ep_size, tokens_per_rank, top_k),
        axis=-1,
    ).astype(dtype)
    w_shape = (ep_size, experts_per_rank, hidden, 2 * intermediate)
    if weight_init == "grug_truncated":
        std = 0.5 / math.sqrt(hidden) if weight_std is None else weight_std
        w_gate_up = (std * jax.random.truncated_normal(k_w, -3, 3, w_shape)).astype(dtype)
    elif weight_init == "standard_normal":
        w_gate_up = jax.random.normal(k_w, w_shape, dtype=dtype)
    else:
        raise ValueError(f"unknown weight_init={weight_init!r}")
    return x_by_rank, expert_ids, router_weights, w_gate_up


def _sharded(mesh: Mesh, value: jax.Array, spec: P) -> jax.Array:
    return jax.device_put(value, NamedSharding(mesh, spec))


def _pallas_dispatch_args(mesh: Mesh, prepacked) -> tuple[jax.Array, ...]:
    return (
        _sharded(mesh, prepacked.send_x_by_dst, P("expert", None, None, None)),
        _sharded(mesh, prepacked.send_row_by_dst, P("expert", None, None)),
        _sharded(mesh, prepacked.send_local_expert_by_dst, P("expert", None, None)),
        _sharded(mesh, prepacked.send_src_token_idx_by_dst, P("expert", None, None)),
        _sharded(mesh, prepacked.send_topk_slot_by_dst, P("expert", None, None)),
        _sharded(mesh, prepacked.send_router_weight_by_dst, P("expert", None, None)),
        _sharded(mesh, prepacked.send_count_by_dst, P("expert", None)),
        _sharded(mesh, prepacked.rows_per_expert, P("expert", None)),
        _sharded(mesh, prepacked.expert_base, P("expert", None)),
    )


def _pallas_dispatch_ready_args(mesh: Mesh, prepacked) -> tuple[jax.Array, ...]:
    return (
        *_pallas_dispatch_args(mesh, prepacked),
        _sharded(mesh, prepacked.send_expert_base_by_dst, P("expert", None, None)),
        _sharded(mesh, prepacked.send_expert_count_by_dst, P("expert", None, None)),
        _sharded(mesh, prepacked.recv_source_expert_base, P("expert", None, None)),
        _sharded(mesh, prepacked.recv_source_expert_count, P("expert", None, None)),
    )


def _expected_ready_count(prepacked, recv_capacity: int) -> jax.Array:
    remaining_capacity = jnp.maximum(recv_capacity - prepacked.recv_source_expert_base, 0)
    return jnp.minimum(prepacked.recv_source_expert_count, remaining_capacity)


def _expected_ready_block_count(prepacked, recv_capacity: int, block_m: int) -> jax.Array:
    total_ready_rows = jnp.minimum(jnp.sum(prepacked.rows_per_expert, axis=1, dtype=jnp.int32), recv_capacity)
    block_starts = jnp.arange(math.ceil(recv_capacity / block_m), dtype=jnp.int32) * block_m
    return jnp.minimum(jnp.maximum(total_ready_rows[:, None] - block_starts[None, :], 0), block_m)


def _pallas_dispatch_fn(
    mesh: Mesh,
    *,
    recv_capacity: int,
    copy_mode: str,
) -> Callable[..., MoeDispatchUpLayout]:
    def local_dispatch(
        send_x,
        send_row,
        send_local_expert,
        send_src_token_idx,
        send_topk_slot,
        send_router_weight,
        send_count,
        rows_per_expert,
        expert_base,
    ):
        layout = dispatch_prepacked_moe_dispatch_up_mosaic_gpu_local(
            jnp.squeeze(send_x, axis=0),
            jnp.squeeze(send_row, axis=0),
            jnp.squeeze(send_local_expert, axis=0),
            jnp.squeeze(send_src_token_idx, axis=0),
            jnp.squeeze(send_topk_slot, axis=0),
            jnp.squeeze(send_router_weight, axis=0),
            jnp.squeeze(send_count, axis=0),
            jnp.squeeze(rows_per_expert, axis=0),
            jnp.squeeze(expert_base, axis=0),
            axis_name="expert",
            recv_capacity=recv_capacity,
            copy_mode=copy_mode,
        )
        return MoeDispatchUpLayout(
            layout.recv_x[None, ...],
            layout.recv_valid[None, ...],
            layout.rows_per_expert[None, ...],
            layout.expert_base[None, ...],
            layout.recv_local_expert[None, ...],
            layout.recv_src_rank[None, ...],
            layout.recv_src_token_idx[None, ...],
            layout.recv_topk_slot[None, ...],
            layout.recv_router_weight[None, ...],
            layout.overflow_count[None],
        )

    out_specs = MoeDispatchUpLayout(
        P("expert", None, None),
        P("expert", None),
        P("expert", None),
        P("expert", None),
        P("expert", None),
        P("expert", None),
        P("expert", None),
        P("expert", None),
        P("expert", None),
        P("expert"),
    )
    fn = jax.jit(
        shard_map(
            local_dispatch,
            mesh=mesh,
            in_specs=(
                P("expert", None, None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None),
                P("expert", None),
                P("expert", None),
            ),
            out_specs=out_specs,
            check_vma=False,
        )
    )
    return fn


def _pallas_dispatch_ready_fn(
    mesh: Mesh,
    *,
    recv_capacity: int,
    ready_block_m: int,
    rows_per_program: int,
) -> Callable[..., tuple[MoeDispatchUpLayout, jax.Array, jax.Array]]:
    def local_dispatch(
        send_x,
        send_row,
        send_local_expert,
        send_src_token_idx,
        send_topk_slot,
        send_router_weight,
        send_count,
        rows_per_expert,
        expert_base,
        send_expert_base,
        send_expert_count,
        recv_source_expert_base,
        recv_source_expert_count,
    ):
        layout, ready_count, ready_block_count = dispatch_prepacked_moe_dispatch_up_mosaic_gpu_ready_local(
            jnp.squeeze(send_x, axis=0),
            jnp.squeeze(send_row, axis=0),
            jnp.squeeze(send_local_expert, axis=0),
            jnp.squeeze(send_src_token_idx, axis=0),
            jnp.squeeze(send_topk_slot, axis=0),
            jnp.squeeze(send_router_weight, axis=0),
            jnp.squeeze(send_count, axis=0),
            jnp.squeeze(rows_per_expert, axis=0),
            jnp.squeeze(expert_base, axis=0),
            jnp.squeeze(send_expert_base, axis=0),
            jnp.squeeze(send_expert_count, axis=0),
            jnp.squeeze(recv_source_expert_base, axis=0),
            jnp.squeeze(recv_source_expert_count, axis=0),
            axis_name="expert",
            recv_capacity=recv_capacity,
            ready_block_m=ready_block_m,
            rows_per_program=rows_per_program,
        )
        return (
            MoeDispatchUpLayout(
                layout.recv_x[None, ...],
                layout.recv_valid[None, ...],
                layout.rows_per_expert[None, ...],
                layout.expert_base[None, ...],
                layout.recv_local_expert[None, ...],
                layout.recv_src_rank[None, ...],
                layout.recv_src_token_idx[None, ...],
                layout.recv_topk_slot[None, ...],
                layout.recv_router_weight[None, ...],
                layout.overflow_count[None],
            ),
            ready_count[None, ...],
            ready_block_count[None, ...],
        )

    out_specs = (
        MoeDispatchUpLayout(
            P("expert", None, None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert"),
        ),
        P("expert", None, None),
        P("expert", None),
    )
    fn = jax.jit(
        shard_map(
            local_dispatch,
            mesh=mesh,
            in_specs=(
                P("expert", None, None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
            ),
            out_specs=out_specs,
            check_vma=False,
        )
    )
    return fn


def _pallas_dispatch_direct_ready_args(mesh: Mesh, prepacked) -> tuple[jax.Array, ...]:
    return (
        *_pallas_dispatch_args(mesh, prepacked),
        _sharded(mesh, prepacked.recv_source_expert_base, P("expert", None, None)),
        _sharded(mesh, prepacked.recv_source_expert_count, P("expert", None, None)),
    )


def _pallas_dispatch_direct_ready_fn(
    mesh: Mesh,
    *,
    recv_capacity: int,
    ready_block_m: int,
    rows_per_program: int,
) -> Callable[..., tuple[MoeDispatchUpLayout, jax.Array, jax.Array]]:
    def local_dispatch(
        send_x,
        send_row,
        send_local_expert,
        send_src_token_idx,
        send_topk_slot,
        send_router_weight,
        send_count,
        rows_per_expert,
        expert_base,
        recv_source_expert_base,
        recv_source_expert_count,
    ):
        layout, ready_count, ready_block_count = dispatch_prepacked_moe_dispatch_up_mosaic_gpu_direct_ready_local(
            jnp.squeeze(send_x, axis=0),
            jnp.squeeze(send_row, axis=0),
            jnp.squeeze(send_local_expert, axis=0),
            jnp.squeeze(send_src_token_idx, axis=0),
            jnp.squeeze(send_topk_slot, axis=0),
            jnp.squeeze(send_router_weight, axis=0),
            jnp.squeeze(send_count, axis=0),
            jnp.squeeze(rows_per_expert, axis=0),
            jnp.squeeze(expert_base, axis=0),
            jnp.squeeze(recv_source_expert_base, axis=0),
            jnp.squeeze(recv_source_expert_count, axis=0),
            axis_name="expert",
            recv_capacity=recv_capacity,
            ready_block_m=ready_block_m,
            rows_per_program=rows_per_program,
        )
        return (
            MoeDispatchUpLayout(
                layout.recv_x[None, ...],
                layout.recv_valid[None, ...],
                layout.rows_per_expert[None, ...],
                layout.expert_base[None, ...],
                layout.recv_local_expert[None, ...],
                layout.recv_src_rank[None, ...],
                layout.recv_src_token_idx[None, ...],
                layout.recv_topk_slot[None, ...],
                layout.recv_router_weight[None, ...],
                layout.overflow_count[None],
            ),
            ready_count[None, ...],
            ready_block_count[None, ...],
        )

    out_specs = (
        MoeDispatchUpLayout(
            P("expert", None, None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert", None),
            P("expert"),
        ),
        P("expert", None, None),
        P("expert", None),
    )
    fn = jax.jit(
        shard_map(
            local_dispatch,
            mesh=mesh,
            in_specs=(
                P("expert", None, None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P("expert", None, None),
                P("expert", None, None),
            ),
            out_specs=out_specs,
            check_vma=False,
        )
    )
    return fn


def _pallas_w13_silu_args(mesh: Mesh, layout: MoeDispatchUpLayout, w_gate_up: jax.Array) -> tuple[jax.Array, ...]:
    return (
        _sharded(mesh, layout.recv_x, P("expert", None, None)),
        _sharded(mesh, layout.rows_per_expert, P("expert", None)),
        _sharded(mesh, w_gate_up, P("expert", None, None, None)),
    )


def _pallas_ready_w13_silu_args(
    mesh: Mesh,
    layout: MoeDispatchUpLayout,
    ready_count: jax.Array,
    w_gate_up: jax.Array,
) -> tuple[jax.Array, ...]:
    return (
        _sharded(mesh, layout.recv_x, P("expert", None, None)),
        ready_count,
        _sharded(mesh, w_gate_up, P("expert", None, None, None)),
    )


def _pallas_block_ready_w13_silu_args(
    mesh: Mesh,
    layout: MoeDispatchUpLayout,
    ready_block_count: jax.Array,
    w_gate_up: jax.Array,
) -> tuple[jax.Array, ...]:
    return (
        _sharded(mesh, layout.recv_x, P("expert", None, None)),
        _sharded(mesh, layout.rows_per_expert, P("expert", None)),
        ready_block_count,
        _sharded(mesh, w_gate_up, P("expert", None, None, None)),
    )


def _pallas_w13_silu_fn(mesh: Mesh, args) -> Callable[..., jax.Array]:
    def local_w13(recv_x, rows_per_expert, local_w_gate_up):
        h = compute_moe_up_mosaic_gpu_local(
            jnp.squeeze(recv_x, axis=0),
            jnp.squeeze(rows_per_expert, axis=0),
            jnp.squeeze(local_w_gate_up, axis=0),
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            max_concurrent_steps=args.num_stages,
            grid_block_n=1,
        )
        return h[None, ...]

    fn = jax.jit(
        shard_map(
            local_w13,
            mesh=mesh,
            in_specs=(P("expert", None, None), P("expert", None), P("expert", None, None, None)),
            out_specs=P("expert", None, None),
            check_vma=False,
        )
    )
    return fn


def _pallas_block_ready_w13_silu_fn(mesh: Mesh, args) -> Callable[..., jax.Array]:
    def local_w13(recv_x, rows_per_expert, ready_block_count, local_w_gate_up):
        h = compute_moe_up_mosaic_gpu_block_ready_local(
            jnp.squeeze(recv_x, axis=0),
            jnp.squeeze(rows_per_expert, axis=0),
            jnp.squeeze(ready_block_count, axis=0),
            jnp.squeeze(local_w_gate_up, axis=0),
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            max_concurrent_steps=args.num_stages,
            grid_block_n=1,
        )
        return h[None, ...]

    fn = jax.jit(
        shard_map(
            local_w13,
            mesh=mesh,
            in_specs=(P("expert", None, None), P("expert", None), P("expert", None), P("expert", None, None, None)),
            out_specs=P("expert", None, None),
            check_vma=False,
        )
    )
    return fn


def _pallas_ready_w13_silu_fn(mesh: Mesh, args) -> Callable[..., jax.Array]:
    def local_w13(recv_x, ready_count, local_w_gate_up):
        h = compute_moe_up_mosaic_gpu_ready_local(
            jnp.squeeze(recv_x, axis=0),
            jnp.squeeze(ready_count, axis=0),
            jnp.squeeze(local_w_gate_up, axis=0),
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            max_concurrent_steps=args.num_stages,
            grid_block_n=1,
        )
        return h[None, ...]

    fn = jax.jit(
        shard_map(
            local_w13,
            mesh=mesh,
            in_specs=(P("expert", None, None), P("expert", None, None), P("expert", None, None, None)),
            out_specs=P("expert", None, None),
            check_vma=False,
        )
    )
    return fn


def _ragged_dot_w13_silu_fn(mesh: Mesh, args) -> Callable[..., jax.Array]:
    def local_w13(recv_x, rows_per_expert, local_w_gate_up):
        w13_out = ragged_dot(
            jnp.squeeze(recv_x, axis=0),
            jnp.squeeze(local_w_gate_up, axis=0),
            jnp.squeeze(rows_per_expert, axis=0),
            implementation=args.ragged_dot_impl,
        )
        gate, up = jnp.split(w13_out, 2, axis=-1)
        h = jax.nn.silu(gate.astype(jnp.float32)).astype(gate.dtype) * up
        return h[None, ...]

    fn = jax.jit(
        shard_map(
            local_w13,
            mesh=mesh,
            in_specs=(P("expert", None, None), P("expert", None), P("expert", None, None, None)),
            out_specs=P("expert", None, None),
            check_vma=False,
        )
    )
    return fn


def _ragged_a2a_dispatch_up_fn(
    mesh: Mesh,
    args,
    *,
    recv_capacity: int,
    num_experts: int,
) -> Callable[..., jax.Array]:
    def local_dispatch_up(x_local, selected_experts_local, local_w_gate_up):
        x_local = jnp.squeeze(x_local, axis=0)
        selected_experts_local = jnp.squeeze(selected_experts_local, axis=0)
        local_w_gate_up = jnp.squeeze(local_w_gate_up, axis=0)
        shard_id = lax.axis_index("expert")
        local_experts = local_w_gate_up.shape[0]
        topk = selected_experts_local.shape[1]
        assignments_per_shard = x_local.shape[0] * topk

        sorted_x, _, group_sizes = _permute_by_global_expert(
            x_local,
            selected_experts_local,
            num_experts=num_experts,
        )
        all_group_sizes = lax.all_gather(group_sizes.astype(jnp.int32), "expert")
        clipped_group_sizes = _clip_receiver_group_sizes(
            all_group_sizes,
            local_expert_size=local_experts,
            receiver_capacity=recv_capacity,
        )
        sender_group_sizes = clipped_group_sizes[shard_id]
        keep_mask = _expert_prefix_keep_mask(
            group_sizes.astype(jnp.int32),
            sender_group_sizes,
            total_size=assignments_per_shard,
        )
        compacted_x = _compact_by_keep_mask(sorted_x, keep_mask)

        ep_size = num_experts // local_experts
        all_shard_counts = jnp.sum(clipped_group_sizes.reshape(ep_size, ep_size, local_experts), axis=2)
        input_offsets, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(all_shard_counts, shard_id)
        dispatched_shape = jnp.zeros((recv_capacity, x_local.shape[1]), dtype=x_local.dtype)
        x_dispatched = lax.ragged_all_to_all(
            compacted_x,
            dispatched_shape,
            input_offsets,
            send_sizes,
            output_offsets,
            recv_sizes,
            axis_name="expert",
        )

        x_dispatch, _, ragged_group_sizes = _local_permute_from_counts(
            x_dispatched,
            clipped_group_sizes,
            local_expert_size=local_experts,
            shard_index=shard_id,
        )
        local_counts_by_sender = lax.dynamic_slice_in_dim(
            clipped_group_sizes,
            start_index=shard_id * local_experts,
            slice_size=local_experts,
            axis=1,
        )
        true_local_group_sizes = jnp.sum(local_counts_by_sender, axis=0, dtype=jnp.int32)
        valid_rows = jnp.arange(recv_capacity, dtype=jnp.int32) < jnp.sum(true_local_group_sizes, dtype=jnp.int32)

        w13_out = ragged_dot(
            x_dispatch,
            local_w_gate_up,
            ragged_group_sizes,
            implementation=args.ragged_dot_impl,
        )
        gate, up = jnp.split(w13_out, 2, axis=-1)
        h = jax.nn.silu(gate.astype(jnp.float32)).astype(gate.dtype) * up
        return jnp.where(valid_rows[:, None], h, jnp.zeros((), dtype=h.dtype))[None, ...]

    return jax.jit(
        shard_map(
            local_dispatch_up,
            mesh=mesh,
            in_specs=(P("expert", None, None), P("expert", None, None), P("expert", None, None, None)),
            out_specs=P("expert", None, None),
            check_vma=False,
        )
    )


def _padded_a2a_dispatch_up_fn(
    mesh: Mesh,
    args,
    *,
    capacity_factor: float,
    num_experts: int,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    def local_dispatch_up(x_local, selected_experts_local, local_w_gate_up):
        x_local = jnp.squeeze(x_local, axis=0)
        selected_experts_local = jnp.squeeze(selected_experts_local, axis=0)
        local_w_gate_up = jnp.squeeze(local_w_gate_up, axis=0)
        local_experts = local_w_gate_up.shape[0]

        dispatch = _dispatch_fixed_buckets(
            x_local,
            selected_experts_local,
            num_experts=num_experts,
            local_experts=local_experts,
            capacity_factor=capacity_factor,
        )
        w13_out = ragged_dot(
            dispatch.x_dispatch,
            local_w_gate_up,
            dispatch.local_group_sizes,
            implementation=args.ragged_dot_impl,
        )
        gate, up = jnp.split(w13_out, 2, axis=-1)
        h = jax.nn.silu(gate.astype(jnp.float32)).astype(gate.dtype) * up
        valid_rows = jnp.arange(dispatch.x_dispatch.shape[0], dtype=jnp.int32) < jnp.sum(
            dispatch.local_group_sizes[:-1],
            dtype=jnp.int32,
        )
        dropped_total = lax.psum(dispatch.dropped_local, "expert")
        return jnp.where(valid_rows[:, None], h, jnp.zeros((), dtype=h.dtype))[None, ...], dropped_total[None]

    return jax.jit(
        shard_map(
            local_dispatch_up,
            mesh=mesh,
            in_specs=(P("expert", None, None), P("expert", None, None), P("expert", None, None, None)),
            out_specs=(P("expert", None, None), P("expert")),
            check_vma=False,
        )
    )


def _synthetic_layout(
    mesh: Mesh,
    *,
    ep_size: int,
    recv_capacity: int,
    experts_per_rank: int,
    hidden: int,
    intermediate: int,
    dtype: jnp.dtype,
    weight_init: str,
    weight_std: float | None,
) -> tuple[MoeDispatchUpLayout, jax.Array]:
    if recv_capacity % experts_per_rank != 0:
        raise ValueError(
            f"synthetic recv_capacity={recv_capacity} must be divisible by experts_per_rank={experts_per_rank}"
        )
    recv_sharding = NamedSharding(mesh, P("expert", None, None))
    metadata_sharding = NamedSharding(mesh, P("expert", None))
    weight_sharding = NamedSharding(mesh, P("expert", None, None, None))

    k_x, k_w = jax.random.split(jax.random.key(6597))
    recv_x = jax.random.normal(
        k_x,
        (ep_size, recv_capacity, hidden),
        dtype=dtype,
        out_sharding=recv_sharding,
    )
    rows_per_expert_host = np.full((ep_size, experts_per_rank), recv_capacity // experts_per_rank, dtype=np.int32)
    rows_per_expert = jax.device_put(rows_per_expert_host, metadata_sharding)
    expert_base = jax.device_put(
        np.broadcast_to(
            np.arange(experts_per_rank, dtype=np.int32) * (recv_capacity // experts_per_rank),
            (ep_size, experts_per_rank),
        ),
        metadata_sharding,
    )
    recv_valid = jnp.ones((ep_size, recv_capacity), dtype=jnp.bool, out_sharding=metadata_sharding)
    metadata_zeros = jnp.zeros((ep_size, recv_capacity), dtype=jnp.int32, out_sharding=metadata_sharding)
    router_weights = jnp.ones((ep_size, recv_capacity), dtype=dtype, out_sharding=metadata_sharding)
    overflow_count = jnp.zeros((ep_size,), dtype=jnp.int32, out_sharding=NamedSharding(mesh, P("expert")))

    w_shape = (ep_size, experts_per_rank, hidden, 2 * intermediate)
    if weight_init == "grug_truncated":
        std = 0.5 / math.sqrt(hidden) if weight_std is None else weight_std
        w_gate_up = (
            std
            * jax.random.truncated_normal(
                k_w,
                -3,
                3,
                w_shape,
                dtype=dtype,
                out_sharding=weight_sharding,
            )
        ).astype(dtype)
    elif weight_init == "standard_normal":
        w_gate_up = jax.random.normal(k_w, w_shape, dtype=dtype, out_sharding=weight_sharding)
    else:
        raise ValueError(f"unknown weight_init={weight_init!r}")

    layout = MoeDispatchUpLayout(
        recv_x,
        recv_valid,
        rows_per_expert,
        expert_base,
        metadata_zeros,
        metadata_zeros,
        metadata_zeros,
        metadata_zeros,
        router_weights,
        overflow_count,
    )
    return layout, w_gate_up


def _resolve_recv_capacity(args, *, synthetic_layout: bool) -> tuple[int, float | None]:
    if args.recv_capacity is not None and args.recv_capacity_factor is not None:
        raise ValueError("Specify at most one of --recv-capacity and --recv-capacity-factor")

    routed_rows_per_rank = args.tokens_per_rank * args.top_k
    if args.recv_capacity is not None:
        return args.recv_capacity, None

    if args.recv_capacity_factor is not None:
        return math.ceil(args.recv_capacity_factor * routed_rows_per_rank), args.recv_capacity_factor

    if synthetic_layout:
        return routed_rows_per_rank, 1.0

    default_capacity_factor = 1.25
    return math.ceil(default_capacity_factor * routed_rows_per_rank), default_capacity_factor


def _resolve_send_capacity(args) -> tuple[int, float | None]:
    if args.send_capacity is not None and args.send_capacity_factor is not None:
        raise ValueError("Specify at most one of --send-capacity and --send-capacity-factor")

    if args.send_capacity is not None:
        return args.send_capacity, None

    if args.send_capacity_factor is not None:
        balanced_rows_per_destination = args.tokens_per_rank * args.top_k / args.ep_size
        return math.ceil(args.send_capacity_factor * balanced_rows_per_destination), args.send_capacity_factor

    return args.tokens_per_rank * args.top_k, None


def _run_synthetic_layout_benchmark(args, dtype: jnp.dtype, devices: list[jax.Device]) -> None:
    if len(devices) < args.ep_size:
        raise RuntimeError(f"Need at least {args.ep_size} local devices, found {len(devices)}")
    if args.w13_impl in ("mosaic_gpu_block_ready", "mosaic_gpu_ready"):
        raise ValueError(f"--w13-impl={args.w13_impl} requires routed dispatch readiness metadata")
    recv_capacity, recv_capacity_factor = _resolve_recv_capacity(args, synthetic_layout=True)
    mesh = Mesh(np.array(devices[: args.ep_size]), ("expert",), axis_types=(AxisType.Explicit,))
    capacity_desc = (
        f"recv_capacity_factor={recv_capacity_factor:g}"
        if recv_capacity_factor is not None
        else "recv_capacity=explicit"
    )
    print(
        "synthetic_layout: "
        f"recv_capacity={recv_capacity} {capacity_desc} rows_per_expert={recv_capacity // args.experts_per_rank} "
        f"w13_impl={args.w13_impl} ragged_dot_impl={args.ragged_dot_impl}"
    )
    with jax.set_mesh(mesh):
        layout, w_gate_up = _synthetic_layout(
            mesh,
            ep_size=args.ep_size,
            recv_capacity=recv_capacity,
            experts_per_rank=args.experts_per_rank,
            hidden=args.hidden,
            intermediate=args.intermediate,
            dtype=dtype,
            weight_init=args.weight_init,
            weight_std=args.weight_std,
        )

        timings: dict[str, float] = {}
        outputs: dict[str, jax.Array] = {}
        if args.w13_impl in ("mosaic_gpu", "both"):
            pallas_w13_fn = _pallas_w13_silu_fn(mesh, args)
            pallas_w13_args = _pallas_w13_silu_args(mesh, layout, w_gate_up)

            def run_pallas_w13():
                return pallas_w13_fn(*pallas_w13_args)

            pallas_h, _ = _time_block("w13_silu/mosaic_gpu", run_pallas_w13)
            outputs["mosaic_gpu"] = pallas_h
            if args.bench_iters > 0:
                timings["mosaic_gpu"] = _measure_steady_state(
                    "w13_silu/mosaic_gpu",
                    run_pallas_w13,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
        if args.w13_impl in ("ragged_dot", "both"):
            ragged_w13_fn = _ragged_dot_w13_silu_fn(mesh, args)
            ragged_w13_args = _pallas_w13_silu_args(mesh, layout, w_gate_up)

            def run_ragged_w13():
                return ragged_w13_fn(*ragged_w13_args)

            ragged_h, _ = _time_block("w13_silu/ragged_dot", run_ragged_w13)
            outputs["ragged_dot"] = ragged_h
            if args.bench_iters > 0:
                timings["ragged_dot"] = _measure_steady_state(
                    "w13_silu/ragged_dot",
                    run_ragged_w13,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
        if args.check_w13 and set(outputs) == {"mosaic_gpu", "ragged_dot"}:
            h_err = jnp.max(
                jnp.abs(outputs["mosaic_gpu"].astype(jnp.float32) - outputs["ragged_dot"].astype(jnp.float32))
            )
            h_err_float = float(h_err)
            print(f"w13_silu_mosaic_vs_ragged_dot_max_abs_error: {h_err_float:.6g}")
            _print_error_summary("w13_silu_mosaic_vs_ragged_dot", outputs["mosaic_gpu"], outputs["ragged_dot"])
            _check_error("w13_silu_mosaic_vs_ragged_dot_max_abs_error", h_err_float, args.w13_atol)
        for label, steady_ms in timings.items():
            _print_roofline(
                ep_size=args.ep_size,
                tokens_per_rank=recv_capacity // args.top_k,
                experts_per_rank=args.experts_per_rank,
                top_k=args.top_k,
                hidden=args.hidden,
                intermediate=args.intermediate,
                dtype_bytes=2 if dtype == jnp.bfloat16 else 4,
                dispatch_ms=None,
                w13_ms=steady_ms,
            )


def _print_error_debug(
    label: str,
    actual: jax.Array,
    expected: jax.Array,
    layout: MoeDispatchUpLayout,
    w_gate_up: jax.Array,
) -> None:
    actual_host = np.asarray(jax.device_get(actual)).astype(np.float32)
    expected_host = np.asarray(jax.device_get(expected)).astype(np.float32)
    err = np.abs(actual_host - expected_host)
    flat_idx = int(np.argmax(err))
    idx = np.unravel_index(flat_idx, err.shape)
    rank, row, col = idx
    layout_host = jax.tree.map(lambda x: np.asarray(jax.device_get(x)), layout)
    weights_host = np.asarray(jax.device_get(w_gate_up)).astype(np.float32)
    local_expert = int(layout_host.recv_local_expert[rank, row])
    intermediate = weights_host.shape[-1] // 2
    x = layout_host.recv_x[rank, row].astype(np.float32)
    gate = float(x @ weights_host[rank, local_expert, :, col])
    up = float(x @ weights_host[rank, local_expert, :, intermediate + col])
    print(
        f"{label}_max_error_at: {idx} "
        f"actual={float(actual_host[idx]):.6g} "
        f"expected={float(expected_host[idx]):.6g} "
        f"valid={bool(layout_host.recv_valid[rank, row])} "
        f"local_expert={local_expert} "
        f"rows_per_expert={layout_host.rows_per_expert[rank].tolist()} "
        f"gate={gate:.6g} up={up:.6g}"
    )


def _print_error_summary(label: str, actual: jax.Array, expected: jax.Array) -> None:
    actual_f32 = actual.astype(jnp.float32)
    expected_f32 = expected.astype(jnp.float32)
    abs_err = jnp.abs(actual_f32 - expected_f32)
    denom = jnp.maximum(jnp.abs(expected_f32), 1.0)
    rel_err = abs_err / denom
    max_abs = jnp.max(abs_err)
    max_mask = abs_err == max_abs
    expected_abs_at_max = jnp.max(jnp.where(max_mask, jnp.abs(expected_f32), 0.0))
    actual_abs_at_max = jnp.max(jnp.where(max_mask, jnp.abs(actual_f32), 0.0))
    print(
        f"{label}_error_summary: "
        f"max_abs={float(max_abs):.6g} "
        f"mean_abs={float(jnp.mean(abs_err)):.6g} "
        f"rms_abs={float(jnp.sqrt(jnp.mean(abs_err * abs_err))):.6g} "
        f"max_rel={float(jnp.max(rel_err)):.6g} "
        f"mean_rel={float(jnp.mean(rel_err)):.6g} "
        f"expected_abs_at_max={float(expected_abs_at_max):.6g} "
        f"actual_abs_at_max={float(actual_abs_at_max):.6g}"
    )


def _check_error(label: str, value: float, tolerance: float) -> None:
    if value > tolerance:
        raise AssertionError(f"{label}={value:.6g} exceeds tolerance {tolerance:.6g}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ep-size", type=int, default=8)
    parser.add_argument("--tokens-per-rank", type=int, default=16)
    parser.add_argument("--experts-per-rank", type=int, default=4)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--intermediate", type=int, default=64)
    parser.add_argument(
        "--weight-init",
        choices=("standard_normal", "grug_truncated"),
        default="standard_normal",
        help="Random expert weight distribution for W13/SiLU validation.",
    )
    parser.add_argument(
        "--weight-std",
        type=float,
        default=None,
        help="Override expert weight std. Defaults to 0.5/sqrt(hidden) for grug_truncated.",
    )
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--run-pallas", action="store_true")
    parser.add_argument(
        "--run-ragged-a2a-dispatch-up",
        action="store_true",
        help="Benchmark built-in ragged_all_to_all dispatch followed by ragged_dot W13/SiLU.",
    )
    parser.add_argument(
        "--run-padded-a2a-dispatch-up",
        action="store_true",
        help="Benchmark fixed-bucket all_to_all dispatch followed by ragged_dot W13/SiLU.",
    )
    parser.add_argument(
        "--synthetic-layout",
        action="store_true",
        help="Skip routing/prepack and benchmark W13/SiLU on a synthetic expert-major layout.",
    )
    parser.add_argument(
        "--recv-capacity",
        type=int,
        default=None,
        help="Rows per destination rank. Overrides --recv-capacity-factor.",
    )
    parser.add_argument(
        "--recv-capacity-factor",
        type=float,
        default=None,
        help=(
            "Capacity multiplier for T*K routed rows per destination rank; defaults to 1.25 routed, "
            "1.0 synthetic. Use 1.0 for no buffer and 1.1-1.25 for typical imbalance probes."
        ),
    )
    parser.add_argument(
        "--send-capacity",
        type=int,
        default=None,
        help="Rows per source/destination pair. Overrides --send-capacity-factor.",
    )
    parser.add_argument(
        "--send-capacity-factor",
        type=float,
        default=None,
        help=(
            "Capacity multiplier for balanced T*K/EP rows per source/destination pair. "
            "Defaults to T*K for conservative skew tolerance."
        ),
    )
    parser.add_argument(
        "--w13-impl",
        choices=("mosaic_gpu", "mosaic_gpu_block_ready", "mosaic_gpu_ready", "ragged_dot", "both"),
        default="mosaic_gpu",
        help="W13/SiLU implementation to benchmark.",
    )
    parser.add_argument(
        "--ragged-dot-impl",
        choices=("auto", "triton", "xla"),
        default="auto",
        help="haliax.nn.ragged_dot implementation when --w13-impl includes ragged_dot.",
    )
    parser.add_argument(
        "--check-w13",
        action="store_true",
        help="In synthetic layout mode with --w13-impl=both, compare Mosaic GPU W13 against ragged_dot.",
    )
    parser.add_argument(
        "--dispatch-copy-mode",
        choices=("scalar", "row_vector", "scratch", "scratch_ready", "direct_ready"),
        default="scratch",
        help="Mosaic GPU dispatch copy implementation.",
    )
    parser.add_argument(
        "--dispatch-rows-per-program",
        type=int,
        default=1,
        help="Send rows handled by each scratch_ready dispatch program.",
    )
    parser.add_argument(
        "--pallas-w13-from-reference-layout",
        action="store_true",
        help="Run Mosaic GPU W13/SiLU against the reference dispatch layout, skipping Mosaic dispatch.",
    )
    parser.add_argument(
        "--skip-reference-checks",
        action="store_true",
        help="Skip reference dispatch/W13 construction and correctness checks for large candidate-only perf runs.",
    )
    parser.add_argument("--debug-errors", action="store_true")
    parser.add_argument("--dispatch-atol", type=float, default=0.0)
    parser.add_argument("--w13-atol", type=float, default=2.0)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--bench-iters", type=int, default=0)
    args = parser.parse_args()

    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    devices = jax.local_devices()
    print(f"devices: {len(devices)} {[device.platform for device in devices]}")
    print(
        "shape: "
        f"EP={args.ep_size} T/rank={args.tokens_per_rank} E/rank={args.experts_per_rank} "
        f"K={args.top_k} H={args.hidden} I={args.intermediate} dtype={args.dtype} "
        f"weight_init={args.weight_init} weight_std={args.weight_std}"
    )
    if args.synthetic_layout:
        _run_synthetic_layout_benchmark(args, dtype, devices)
        return
    if args.skip_reference_checks and args.pallas_w13_from_reference_layout:
        raise ValueError("--skip-reference-checks cannot be used with --pallas-w13-from-reference-layout")

    x_by_rank, expert_ids, router_weights, w_gate_up = _make_inputs(
        ep_size=args.ep_size,
        tokens_per_rank=args.tokens_per_rank,
        experts_per_rank=args.experts_per_rank,
        top_k=args.top_k,
        hidden=args.hidden,
        intermediate=args.intermediate,
        dtype=dtype,
        weight_init=args.weight_init,
        weight_std=args.weight_std,
    )
    recv_capacity, recv_capacity_factor = _resolve_recv_capacity(args, synthetic_layout=False)
    if recv_capacity_factor is not None:
        print(f"recv_capacity: {recv_capacity} from factor={recv_capacity_factor:g}")
    else:
        print(f"recv_capacity: {recv_capacity}")
    send_capacity, send_capacity_factor = _resolve_send_capacity(args)
    if send_capacity_factor is not None:
        print(f"send_capacity: {send_capacity} from factor={send_capacity_factor:g}")
    else:
        print(f"send_capacity: {send_capacity}")
    num_experts = args.ep_size * args.experts_per_rank

    prepacked = None
    if (
        args.skip_reference_checks
        and (args.run_ragged_a2a_dispatch_up or args.run_padded_a2a_dispatch_up)
        and not args.run_pallas
    ):
        print("prepack/reference: skipped")
    else:
        prepacked, _ = _time_block(
            "prepack/reference",
            lambda: prepack_moe_dispatch_up_reference(
                x_by_rank,
                expert_ids,
                router_weights,
                num_experts=num_experts,
                recv_capacity=recv_capacity,
                send_capacity=send_capacity,
            ),
        )
    ref_dispatch_steady_ms = None
    ref_w13_steady_ms = None
    ref_layout = None
    ref_h = None
    if args.skip_reference_checks:
        print("reference_checks: skipped")
    else:
        if prepacked is None:
            raise AssertionError("reference prepack is required when reference checks are enabled")
        ref_layout, _ = _time_block(
            "dispatch/reference",
            lambda: dispatch_prepacked_moe_dispatch_up_reference(prepacked, recv_capacity=recv_capacity),
        )
        ref_h, _ = _time_block(
            "w13_silu/reference",
            lambda: compute_moe_up_from_layout_reference(ref_layout, w_gate_up),
        )
        if args.bench_iters > 0:
            ref_dispatch_fn = jax.jit(
                lambda prepacked_arg: dispatch_prepacked_moe_dispatch_up_reference(
                    prepacked_arg,
                    recv_capacity=recv_capacity,
                )
            )
            ref_w13_fn = jax.jit(compute_moe_up_from_layout_reference)

            def run_ref_dispatch():
                return ref_dispatch_fn(prepacked)

            def run_ref_w13():
                return ref_w13_fn(ref_layout, w_gate_up)

            _time_block("dispatch/reference_jit", run_ref_dispatch)
            ref_dispatch_steady_ms = _measure_steady_state(
                "dispatch/reference_jit",
                run_ref_dispatch,
                warmup_steps=args.warmup_steps,
                bench_iters=args.bench_iters,
            )
            _time_block("w13_silu/reference_jit", run_ref_w13)
            ref_w13_steady_ms = _measure_steady_state(
                "w13_silu/reference_jit",
                run_ref_w13,
                warmup_steps=args.warmup_steps,
                bench_iters=args.bench_iters,
            )

    if not args.run_pallas and not args.run_ragged_a2a_dispatch_up and not args.run_padded_a2a_dispatch_up:
        return
    if len(devices) < args.ep_size:
        raise RuntimeError(f"Need at least {args.ep_size} local devices, found {len(devices)}")

    mesh = Mesh(np.array(devices[: args.ep_size]), ("expert",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        if args.run_ragged_a2a_dispatch_up:
            ragged_a2a_fn = _ragged_a2a_dispatch_up_fn(
                mesh,
                args,
                recv_capacity=recv_capacity,
                num_experts=num_experts,
            )
            ragged_a2a_args = (
                _sharded(mesh, x_by_rank, P("expert", None, None)),
                _sharded(mesh, expert_ids, P("expert", None, None)),
                _sharded(mesh, w_gate_up, P("expert", None, None, None)),
            )

            def run_ragged_a2a_dispatch_up():
                return ragged_a2a_fn(*ragged_a2a_args)

            ragged_a2a_h, _ = _time_block("dispatch_up/ragged_a2a_ragged_dot", run_ragged_a2a_dispatch_up)
            ragged_a2a_steady_ms = None
            if args.bench_iters > 0:
                ragged_a2a_steady_ms = _measure_steady_state(
                    "dispatch_up/ragged_a2a_ragged_dot",
                    run_ragged_a2a_dispatch_up,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
            if ref_h is None:
                print("dispatch_up/ragged_a2a_ragged_dot/reference_check: skipped")
            else:
                ragged_a2a_err = jnp.max(jnp.abs(ragged_a2a_h.astype(jnp.float32) - ref_h.astype(jnp.float32)))
                ragged_a2a_err_float = float(ragged_a2a_err)
                print(f"dispatch_up_ragged_a2a_ragged_dot_max_abs_error: {ragged_a2a_err_float:.6g}")
                _print_error_summary("dispatch_up_ragged_a2a_ragged_dot", ragged_a2a_h, ref_h)
                _check_error(
                    "dispatch_up_ragged_a2a_ragged_dot_max_abs_error",
                    ragged_a2a_err_float,
                    args.w13_atol,
                )
            if ragged_a2a_steady_ms is not None:
                print(f"dispatch_up/ragged_a2a_ragged_dot_end_to_end_ms: {ragged_a2a_steady_ms:.3f}")

        if args.run_padded_a2a_dispatch_up:
            padded_capacity_factor = recv_capacity / (args.tokens_per_rank * args.top_k)
            padded_a2a_fn = _padded_a2a_dispatch_up_fn(
                mesh,
                args,
                capacity_factor=padded_capacity_factor,
                num_experts=num_experts,
            )
            padded_a2a_args = (
                _sharded(mesh, x_by_rank, P("expert", None, None)),
                _sharded(mesh, expert_ids, P("expert", None, None)),
                _sharded(mesh, w_gate_up, P("expert", None, None, None)),
            )

            def run_padded_a2a_dispatch_up():
                return padded_a2a_fn(*padded_a2a_args)

            padded_a2a_result, _ = _time_block("dispatch_up/padded_a2a_ragged_dot", run_padded_a2a_dispatch_up)
            padded_a2a_h, padded_a2a_dropped = padded_a2a_result
            padded_a2a_dropped_int = int(jnp.max(padded_a2a_dropped))
            print(f"dispatch_up/padded_a2a_dropped_total: {padded_a2a_dropped_int}")
            padded_a2a_steady_ms = None
            if args.bench_iters > 0:
                padded_a2a_steady_ms = _measure_steady_state(
                    "dispatch_up/padded_a2a_ragged_dot",
                    run_padded_a2a_dispatch_up,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
            if ref_h is None:
                print("dispatch_up/padded_a2a_ragged_dot/reference_check: skipped")
            elif padded_a2a_h.shape[1] < ref_h.shape[1]:
                raise AssertionError(
                    "padded all-to-all output has fewer rows than reference: "
                    f"{padded_a2a_h.shape[1]} < {ref_h.shape[1]}"
                )
            else:
                comparable_padded_h = padded_a2a_h[:, : ref_h.shape[1], :]
                padded_a2a_err = jnp.max(jnp.abs(comparable_padded_h.astype(jnp.float32) - ref_h.astype(jnp.float32)))
                padded_a2a_err_float = float(padded_a2a_err)
                print(f"dispatch_up_padded_a2a_ragged_dot_max_abs_error: {padded_a2a_err_float:.6g}")
                _print_error_summary("dispatch_up_padded_a2a_ragged_dot", comparable_padded_h, ref_h)
                _check_error(
                    "dispatch_up_padded_a2a_ragged_dot_max_abs_error",
                    padded_a2a_err_float,
                    args.w13_atol,
                )
            if padded_a2a_steady_ms is not None:
                print(f"dispatch_up/padded_a2a_ragged_dot_end_to_end_ms: {padded_a2a_steady_ms:.3f}")

        if not args.run_pallas:
            return

        pallas_dispatch_steady_ms = None
        pallas_ready_count = None
        pallas_ready_block_count = None
        if args.pallas_w13_from_reference_layout:
            if ref_layout is None:
                raise AssertionError("reference layout is required for --pallas-w13-from-reference-layout")
            pallas_layout = ref_layout
            print("dispatch/mosaic_gpu: skipped; using reference layout for W13/SiLU")
        else:
            if prepacked is None:
                raise AssertionError("prepacked sends are required for Mosaic GPU dispatch")
            if args.dispatch_copy_mode == "scratch_ready":
                pallas_dispatch_fn = _pallas_dispatch_ready_fn(
                    mesh,
                    recv_capacity=recv_capacity,
                    ready_block_m=args.block_m,
                    rows_per_program=args.dispatch_rows_per_program,
                )
                pallas_dispatch_args = _pallas_dispatch_ready_args(mesh, prepacked)
            elif args.dispatch_copy_mode == "direct_ready":
                pallas_dispatch_fn = _pallas_dispatch_direct_ready_fn(
                    mesh,
                    recv_capacity=recv_capacity,
                    ready_block_m=args.block_m,
                    rows_per_program=args.dispatch_rows_per_program,
                )
                pallas_dispatch_args = _pallas_dispatch_direct_ready_args(mesh, prepacked)
            else:
                pallas_dispatch_fn = _pallas_dispatch_fn(
                    mesh, recv_capacity=recv_capacity, copy_mode=args.dispatch_copy_mode
                )
                pallas_dispatch_args = _pallas_dispatch_args(mesh, prepacked)

            def run_pallas_dispatch():
                return pallas_dispatch_fn(*pallas_dispatch_args)

            pallas_dispatch_result, _ = _time_block("dispatch/mosaic_gpu", run_pallas_dispatch)
            if args.dispatch_copy_mode in ("direct_ready", "scratch_ready"):
                pallas_layout, pallas_ready_count, pallas_ready_block_count = pallas_dispatch_result
            else:
                pallas_layout = pallas_dispatch_result
            if args.bench_iters > 0:
                pallas_dispatch_steady_ms = _measure_steady_state(
                    "dispatch/mosaic_gpu",
                    run_pallas_dispatch,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
                if ref_dispatch_steady_ms is not None:
                    _print_speedup(
                        "dispatch/mosaic_gpu_vs_reference_jit",
                        ref_dispatch_steady_ms,
                        pallas_dispatch_steady_ms,
                    )
            if ref_layout is None:
                print("dispatch/reference_check: skipped")
            else:
                dispatch_err = jnp.max(
                    jnp.abs(pallas_layout.recv_x.astype(jnp.float32) - ref_layout.recv_x.astype(jnp.float32))
                )
                dispatch_err_float = float(dispatch_err)
                print(f"dispatch_max_abs_error: {dispatch_err_float:.6g}")
                valid_err = jnp.sum(pallas_layout.recv_valid != ref_layout.recv_valid, dtype=jnp.int32)
                expert_err = jnp.sum(pallas_layout.recv_local_expert != ref_layout.recv_local_expert, dtype=jnp.int32)
                src_rank_err = jnp.sum(pallas_layout.recv_src_rank != ref_layout.recv_src_rank, dtype=jnp.int32)
                valid_err_int = int(valid_err)
                expert_err_int = int(expert_err)
                src_rank_err_int = int(src_rank_err)
                print(
                    "dispatch_metadata_errors: "
                    f"valid={valid_err_int} local_expert={expert_err_int} src_rank={src_rank_err_int}"
                )
            if pallas_ready_count is not None:
                expected_ready_count = _sharded(
                    mesh, _expected_ready_count(prepacked, recv_capacity), P("expert", None, None)
                )
                ready_count_err = jnp.max(jnp.abs(pallas_ready_count - expected_ready_count))
                ready_count_err_int = int(ready_count_err)
                print(f"dispatch_ready_count_max_abs_error: {ready_count_err_int}")
                if ready_count_err_int != 0:
                    raise AssertionError(f"dispatch ready-count mismatch: max_abs={ready_count_err_int}")
            if pallas_ready_block_count is not None:
                expected_ready_block_count = _sharded(
                    mesh, _expected_ready_block_count(prepacked, recv_capacity, args.block_m), P("expert", None)
                )
                ready_block_count_err = jnp.max(jnp.abs(pallas_ready_block_count - expected_ready_block_count))
                ready_block_count_err_int = int(ready_block_count_err)
                print(f"dispatch_ready_block_count_max_abs_error: {ready_block_count_err_int}")
                if ready_block_count_err_int != 0:
                    raise AssertionError(f"dispatch ready-block-count mismatch: max_abs={ready_block_count_err_int}")
            if ref_layout is not None:
                _check_error("dispatch_max_abs_error", dispatch_err_float, args.dispatch_atol)
                if valid_err_int != 0 or expert_err_int != 0 or src_rank_err_int != 0:
                    raise AssertionError(
                        "dispatch metadata mismatch: "
                        f"valid={valid_err_int} local_expert={expert_err_int} src_rank={src_rank_err_int}"
                    )

        if args.w13_impl == "both":
            raise ValueError("--w13-impl=both is only supported with --synthetic-layout")
        if args.w13_impl == "mosaic_gpu_block_ready":
            if pallas_ready_block_count is None:
                pallas_ready_block_count = _sharded(
                    mesh,
                    _expected_ready_block_count(prepacked, recv_capacity, args.block_m),
                    P("expert", None),
                )
            pallas_w13_fn = _pallas_block_ready_w13_silu_fn(mesh, args)
            pallas_w13_args = _pallas_block_ready_w13_silu_args(
                mesh, pallas_layout, pallas_ready_block_count, w_gate_up
            )
            w13_label = "w13_silu/mosaic_gpu_block_ready"
        elif args.w13_impl == "mosaic_gpu_ready":
            if pallas_ready_count is None:
                pallas_ready_count = _sharded(
                    mesh, _expected_ready_count(prepacked, recv_capacity), P("expert", None, None)
                )
            pallas_w13_fn = _pallas_ready_w13_silu_fn(mesh, args)
            pallas_w13_args = _pallas_ready_w13_silu_args(mesh, pallas_layout, pallas_ready_count, w_gate_up)
            w13_label = "w13_silu/mosaic_gpu_ready"
        elif args.w13_impl == "ragged_dot":
            pallas_w13_fn = _ragged_dot_w13_silu_fn(mesh, args)
            pallas_w13_args = _pallas_w13_silu_args(mesh, pallas_layout, w_gate_up)
            w13_label = "w13_silu/ragged_dot"
        else:
            pallas_w13_fn = _pallas_w13_silu_fn(mesh, args)
            pallas_w13_args = _pallas_w13_silu_args(mesh, pallas_layout, w_gate_up)
            w13_label = "w13_silu/mosaic_gpu"

        def run_pallas_w13():
            return pallas_w13_fn(*pallas_w13_args)

        pallas_h, _ = _time_block(w13_label, run_pallas_w13)
        pallas_w13_steady_ms = None
        if args.bench_iters > 0:
            pallas_w13_steady_ms = _measure_steady_state(
                w13_label,
                run_pallas_w13,
                warmup_steps=args.warmup_steps,
                bench_iters=args.bench_iters,
            )
            if ref_w13_steady_ms is not None:
                _print_speedup(f"{w13_label}_vs_reference_jit", ref_w13_steady_ms, pallas_w13_steady_ms)
        if ref_h is None:
            print("w13_silu/reference_check: skipped")
        else:
            h_err = jnp.max(jnp.abs(pallas_h.astype(jnp.float32) - ref_h.astype(jnp.float32)))
            h_err_float = float(h_err)
            print(f"w13_silu_max_abs_error: {h_err_float:.6g}")
            _print_error_summary("w13_silu", pallas_h, ref_h)
            if args.debug_errors:
                _print_error_debug("w13_silu", pallas_h, ref_h, pallas_layout, w_gate_up)
            _check_error("w13_silu_max_abs_error", h_err_float, args.w13_atol)
        if pallas_w13_steady_ms is not None:
            _print_roofline(
                ep_size=args.ep_size,
                tokens_per_rank=args.tokens_per_rank,
                experts_per_rank=args.experts_per_rank,
                top_k=args.top_k,
                hidden=args.hidden,
                intermediate=args.intermediate,
                dtype_bytes=2 if dtype == jnp.bfloat16 else 4,
                dispatch_ms=pallas_dispatch_steady_ms,
                w13_ms=pallas_w13_steady_ms,
            )


if __name__ == "__main__":
    main()
