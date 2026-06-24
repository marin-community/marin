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
from levanter.grug._moe.ep_ring import _dispatch_up_ep_ring_local
from levanter.kernels.pallas.moe_dispatch_up.mosaic_gpu import (
    dispatch_prepacked_moe_dispatch_up_mosaic_gpu_direct_ready_local,
    dispatch_prepacked_moe_dispatch_up_mosaic_gpu_local,
    dispatch_prepacked_moe_dispatch_up_mosaic_gpu_ready_local,
    dispatch_prepacked_moe_dispatch_up_mosaic_gpu_source_expert_local,
    dispatch_prepacked_moe_dispatch_up_mosaic_gpu_source_expert_tiled_local,
    compute_moe_up_mosaic_gpu_source_expert_padded_local,
    compute_moe_up_mosaic_gpu_local,
    compute_moe_up_mosaic_gpu_block_ready_local,
    compute_moe_up_mosaic_gpu_ready_local,
)
from levanter.kernels.pallas.moe_dispatch_up.reference import (
    MoeDispatchUpLayout,
    MoeDispatchUpPrepackedSend,
    MoeDispatchUpSourceExpertPrepackedSend,
    dispatch_prepacked_moe_dispatch_up_reference,
    compute_moe_up_from_layout_reference,
    prepack_moe_dispatch_up_source_expert_reference,
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
            grid_block_n=args.grid_block_n,
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
            grid_block_n=args.grid_block_n,
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


def _pallas_fused_dispatch_block_ready_w13_fn(
    mesh: Mesh,
    args,
    *,
    recv_capacity: int,
    ready_block_m: int,
    rows_per_program: int,
    copy_mode: str,
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    if copy_mode not in ("scratch_ready", "direct_ready"):
        raise ValueError(f"copy_mode must be 'scratch_ready' or 'direct_ready', got {copy_mode!r}")

    def local_dispatch_up(
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
        local_w_gate_up,
    ):
        if copy_mode == "scratch_ready":
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
        else:
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
        h = compute_moe_up_mosaic_gpu_block_ready_local(
            layout.recv_x,
            layout.rows_per_expert,
            ready_block_count,
            jnp.squeeze(local_w_gate_up, axis=0),
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            max_concurrent_steps=args.num_stages,
            grid_block_n=args.grid_block_n,
        )
        return h[None, ...], ready_count[None, ...], ready_block_count[None, ...]

    return jax.jit(
        shard_map(
            local_dispatch_up,
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
                P("expert", None, None, None),
            ),
            out_specs=(P("expert", None, None), P("expert", None, None), P("expert", None)),
            check_vma=False,
        )
    )


def _compact_source_expert_args(
    mesh: Mesh, prepacked: MoeDispatchUpSourceExpertPrepackedSend
) -> tuple[jax.Array, ...]:
    return (
        _sharded(mesh, prepacked.send_x_by_dst_expert, P("expert", None, None, None, None)),
        _sharded(mesh, prepacked.send_src_token_idx_by_dst_expert, P("expert", None, None, None)),
        _sharded(mesh, prepacked.send_topk_slot_by_dst_expert, P("expert", None, None, None)),
        _sharded(mesh, prepacked.send_router_weight_by_dst_expert, P("expert", None, None, None)),
        _sharded(mesh, prepacked.send_row_base_by_dst_expert, P("expert", None, None)),
        _sharded(mesh, prepacked.send_count_by_dst_expert, P("expert", None, None)),
        _sharded(mesh, prepacked.rows_per_expert, P("expert", None)),
        _sharded(mesh, prepacked.expert_base, P("expert", None)),
        _sharded(mesh, prepacked.recv_source_expert_base, P("expert", None, None)),
        _sharded(mesh, prepacked.recv_source_expert_count, P("expert", None, None)),
    )


def _pallas_compact_source_expert_dispatch_up_fn(
    mesh: Mesh,
    args,
    *,
    recv_capacity: int,
    ready_block_m: int,
    rows_per_program: int,
    zero_recv: bool,
    copy_cols: int | None,
    copy_rows: int,
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    def local_dispatch_up(
        send_x_by_dst_expert,
        send_src_token_idx_by_dst_expert,
        send_topk_slot_by_dst_expert,
        send_router_weight_by_dst_expert,
        send_row_base_by_dst_expert,
        send_count_by_dst_expert,
        rows_per_expert,
        expert_base,
        recv_source_expert_base,
        recv_source_expert_count,
        local_w_gate_up,
    ):
        compact_prepacked = MoeDispatchUpSourceExpertPrepackedSend(
            jnp.squeeze(send_x_by_dst_expert, axis=0),
            jnp.squeeze(send_src_token_idx_by_dst_expert, axis=0),
            jnp.squeeze(send_topk_slot_by_dst_expert, axis=0),
            jnp.squeeze(send_router_weight_by_dst_expert, axis=0),
            jnp.squeeze(send_row_base_by_dst_expert, axis=0),
            jnp.squeeze(send_count_by_dst_expert, axis=0),
            jnp.squeeze(rows_per_expert, axis=0),
            jnp.squeeze(expert_base, axis=0),
            jnp.squeeze(recv_source_expert_base, axis=0),
            jnp.squeeze(recv_source_expert_count, axis=0),
            jnp.array(0, dtype=jnp.int32),
        )
        if copy_cols is None:
            layout, ready_count, ready_block_count = dispatch_prepacked_moe_dispatch_up_mosaic_gpu_source_expert_local(
                compact_prepacked,
                axis_name="expert",
                recv_capacity=recv_capacity,
                ready_block_m=ready_block_m,
                rows_per_program=rows_per_program,
                zero_recv=zero_recv,
            )
        else:
            layout, ready_count, ready_block_count = (
                dispatch_prepacked_moe_dispatch_up_mosaic_gpu_source_expert_tiled_local(
                    compact_prepacked,
                    axis_name="expert",
                    recv_capacity=recv_capacity,
                    ready_block_m=ready_block_m,
                    copy_cols=copy_cols,
                    copy_rows=copy_rows,
                )
            )
        h = compute_moe_up_mosaic_gpu_block_ready_local(
            layout.recv_x,
            layout.rows_per_expert,
            ready_block_count,
            jnp.squeeze(local_w_gate_up, axis=0),
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            max_concurrent_steps=args.num_stages,
            grid_block_n=args.grid_block_n,
        )
        return h[None, ...], ready_count[None, ...], ready_block_count[None, ...]

    return jax.jit(
        shard_map(
            local_dispatch_up,
            mesh=mesh,
            in_specs=(
                P("expert", None, None, None, None),
                P("expert", None, None, None),
                P("expert", None, None, None),
                P("expert", None, None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None),
                P("expert", None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None, None),
            ),
            out_specs=(P("expert", None, None), P("expert", None, None), P("expert", None)),
            check_vma=False,
        )
    )


def _compact_a2a_mosaic_w13_dispatch_up_fn(
    mesh: Mesh,
    args,
    *,
    recv_capacity: int,
    return_compact_output: bool,
    merge_source_groups: bool,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    def local_dispatch_up(
        send_x_by_dst_expert,
        send_count_by_dst_expert,
        recv_source_expert_base,
        local_w_gate_up,
    ):
        send_x_by_dst_expert = jnp.squeeze(send_x_by_dst_expert, axis=0)
        send_count_by_dst_expert = jnp.squeeze(send_count_by_dst_expert, axis=0)
        recv_source_expert_base = jnp.squeeze(recv_source_expert_base, axis=0)
        local_w_gate_up = jnp.squeeze(local_w_gate_up, axis=0)

        source_expert_x = lax.all_to_all(
            send_x_by_dst_expert,
            "expert",
            split_axis=0,
            concat_axis=0,
        )
        source_expert_count = lax.all_to_all(
            send_count_by_dst_expert,
            "expert",
            split_axis=0,
            concat_axis=0,
        )

        ep_size, local_experts, source_expert_capacity, hidden = source_expert_x.shape
        source_expert_groups = ep_size * local_experts
        if merge_source_groups:
            local_expert_x = jnp.swapaxes(source_expert_x, 0, 1).reshape(
                local_experts * ep_size * source_expert_capacity,
                hidden,
            )
            local_expert_rows = jnp.full(
                (local_experts,),
                ep_size * source_expert_capacity,
                dtype=jnp.int32,
            )
            local_expert_h = compute_moe_up_mosaic_gpu_local(
                local_expert_x,
                local_expert_rows,
                local_w_gate_up,
                block_m=args.block_m,
                block_n=args.block_n,
                block_k=args.block_k,
                max_concurrent_steps=args.num_stages,
                grid_block_n=args.grid_block_n,
            )
            source_expert_overflow = jnp.sum(
                jnp.maximum(source_expert_count - source_expert_capacity, 0),
                dtype=jnp.int32,
            )
            receiver_overflow = jnp.maximum(jnp.max(recv_source_expert_base + source_expert_count) - recv_capacity, 0)
            overflow_count = lax.psum(source_expert_overflow + receiver_overflow, "expert")
            return local_expert_h[None, ...], overflow_count[None]

        compact_x = source_expert_x.reshape(source_expert_groups * source_expert_capacity, hidden)
        compact_rows_per_group = jnp.full(
            (source_expert_groups,),
            source_expert_capacity,
            dtype=jnp.int32,
        )

        compact_h = compute_moe_up_mosaic_gpu_source_expert_padded_local(
            compact_x,
            compact_rows_per_group,
            local_w_gate_up,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            max_concurrent_steps=args.num_stages,
            grid_block_n=args.grid_block_n,
        ).reshape(ep_size, local_experts, source_expert_capacity, args.intermediate)

        source_expert_overflow = jnp.sum(
            jnp.maximum(source_expert_count - source_expert_capacity, 0),
            dtype=jnp.int32,
        )
        receiver_overflow = jnp.maximum(jnp.max(recv_source_expert_base + source_expert_count) - recv_capacity, 0)
        overflow_count = lax.psum(source_expert_overflow + receiver_overflow, "expert")
        if return_compact_output:
            return (
                compact_h.reshape(source_expert_groups * source_expert_capacity, args.intermediate)[None, ...],
                overflow_count[None],
            )

        h = jnp.zeros((recv_capacity, args.intermediate), dtype=compact_h.dtype)
        source_positions = jnp.arange(source_expert_capacity, dtype=jnp.int32)
        for src_rank in range(ep_size):
            for local_expert in range(local_experts):
                base = recv_source_expert_base[src_rank, local_expert]
                count = source_expert_count[src_rank, local_expert]
                rows = base + source_positions
                valid = (source_positions < count) & (rows < recv_capacity)
                safe_rows = jnp.where(valid, rows, 0)
                h = h.at[safe_rows].add(
                    jnp.where(valid[:, None], compact_h[src_rank, local_expert], jnp.zeros((), dtype=compact_h.dtype))
                )

        return h[None, ...], overflow_count[None]

    return jax.jit(
        shard_map(
            local_dispatch_up,
            mesh=mesh,
            in_specs=(
                P("expert", None, None, None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None, None),
            ),
            out_specs=(P("expert", None, None), P("expert")),
            check_vma=False,
        )
    )


def _compact_a2a_transport_fn(mesh: Mesh) -> Callable[..., tuple[jax.Array, jax.Array]]:
    def local_transport(send_x_by_dst_expert, send_count_by_dst_expert):
        send_x_by_dst_expert = jnp.squeeze(send_x_by_dst_expert, axis=0)
        send_count_by_dst_expert = jnp.squeeze(send_count_by_dst_expert, axis=0)
        source_expert_x = lax.all_to_all(
            send_x_by_dst_expert,
            "expert",
            split_axis=0,
            concat_axis=0,
        )
        source_expert_count = lax.all_to_all(
            send_count_by_dst_expert,
            "expert",
            split_axis=0,
            concat_axis=0,
        )
        return source_expert_x[None, ...], source_expert_count[None, ...]

    return jax.jit(
        shard_map(
            local_transport,
            mesh=mesh,
            in_specs=(
                P("expert", None, None, None, None),
                P("expert", None, None),
            ),
            out_specs=(
                P("expert", None, None, None, None),
                P("expert", None, None),
            ),
            check_vma=False,
        )
    )


def _compact_a2a_merged_mosaic_w13_fn(
    mesh: Mesh,
    args,
) -> Callable[..., jax.Array]:
    def local_w13(source_expert_x, local_w_gate_up):
        source_expert_x = jnp.squeeze(source_expert_x, axis=0)
        local_w_gate_up = jnp.squeeze(local_w_gate_up, axis=0)
        ep_size, local_experts, source_expert_capacity, hidden = source_expert_x.shape
        local_expert_x = jnp.swapaxes(source_expert_x, 0, 1).reshape(
            local_experts * ep_size * source_expert_capacity,
            hidden,
        )
        local_expert_rows = jnp.full(
            (local_experts,),
            ep_size * source_expert_capacity,
            dtype=jnp.int32,
        )
        h = compute_moe_up_mosaic_gpu_local(
            local_expert_x,
            local_expert_rows,
            local_w_gate_up,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            max_concurrent_steps=args.num_stages,
            grid_block_n=args.grid_block_n,
        )
        return h[None, ...]

    return jax.jit(
        shard_map(
            local_w13,
            mesh=mesh,
            in_specs=(
                P("expert", None, None, None, None),
                P("expert", None, None, None),
            ),
            out_specs=P("expert", None, None),
            check_vma=False,
        )
    )


def _pallas_compact_source_expert_dispatch_fn(
    mesh: Mesh,
    *,
    recv_capacity: int,
    ready_block_m: int,
    rows_per_program: int,
    zero_recv: bool,
    copy_cols: int | None,
    copy_rows: int,
) -> Callable[..., tuple[jax.Array, ...]]:
    def local_dispatch(
        send_x_by_dst_expert,
        send_src_token_idx_by_dst_expert,
        send_topk_slot_by_dst_expert,
        send_router_weight_by_dst_expert,
        send_row_base_by_dst_expert,
        send_count_by_dst_expert,
        rows_per_expert,
        expert_base,
        recv_source_expert_base,
        recv_source_expert_count,
    ):
        compact_prepacked = MoeDispatchUpSourceExpertPrepackedSend(
            jnp.squeeze(send_x_by_dst_expert, axis=0),
            jnp.squeeze(send_src_token_idx_by_dst_expert, axis=0),
            jnp.squeeze(send_topk_slot_by_dst_expert, axis=0),
            jnp.squeeze(send_router_weight_by_dst_expert, axis=0),
            jnp.squeeze(send_row_base_by_dst_expert, axis=0),
            jnp.squeeze(send_count_by_dst_expert, axis=0),
            jnp.squeeze(rows_per_expert, axis=0),
            jnp.squeeze(expert_base, axis=0),
            jnp.squeeze(recv_source_expert_base, axis=0),
            jnp.squeeze(recv_source_expert_count, axis=0),
            jnp.array(0, dtype=jnp.int32),
        )
        if copy_cols is None:
            layout, ready_count, ready_block_count = dispatch_prepacked_moe_dispatch_up_mosaic_gpu_source_expert_local(
                compact_prepacked,
                axis_name="expert",
                recv_capacity=recv_capacity,
                ready_block_m=ready_block_m,
                rows_per_program=rows_per_program,
                zero_recv=zero_recv,
            )
        else:
            layout, ready_count, ready_block_count = (
                dispatch_prepacked_moe_dispatch_up_mosaic_gpu_source_expert_tiled_local(
                    compact_prepacked,
                    axis_name="expert",
                    recv_capacity=recv_capacity,
                    ready_block_m=ready_block_m,
                    copy_cols=copy_cols,
                    copy_rows=copy_rows,
                )
            )
        return (
            layout.recv_x[None, ...],
            layout.recv_valid[None, ...],
            layout.recv_local_expert[None, ...],
            layout.recv_src_rank[None, ...],
            layout.recv_src_token_idx[None, ...],
            layout.recv_topk_slot[None, ...],
            layout.recv_router_weight[None, ...],
            ready_count[None, ...],
            ready_block_count[None, ...],
        )

    return jax.jit(
        shard_map(
            local_dispatch,
            mesh=mesh,
            in_specs=(
                P("expert", None, None, None, None),
                P("expert", None, None, None),
                P("expert", None, None, None),
                P("expert", None, None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None),
                P("expert", None),
                P("expert", None, None),
                P("expert", None, None),
            ),
            out_specs=(
                P("expert", None, None),
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P("expert", None),
                P("expert", None, None),
                P("expert", None),
            ),
            check_vma=False,
        )
    )


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
            grid_block_n=args.grid_block_n,
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


def _ragged_a2a_dispatch_only_fn(
    mesh: Mesh,
    *,
    recv_capacity: int,
    num_experts: int,
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    def local_dispatch(x_local, selected_experts_local):
        x_local = jnp.squeeze(x_local, axis=0)
        selected_experts_local = jnp.squeeze(selected_experts_local, axis=0)
        shard_id = lax.axis_index("expert")
        local_experts = num_experts // lax.axis_size("expert")
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
        return x_dispatch[None, ...], ragged_group_sizes[None, ...], true_local_group_sizes[None, ...]

    return jax.jit(
        shard_map(
            local_dispatch,
            mesh=mesh,
            in_specs=(P("expert", None, None), P("expert", None, None)),
            out_specs=(P("expert", None, None), P("expert", None), P("expert", None)),
            check_vma=False,
        )
    )


def _ragged_a2a_dispatched_w13_fn(mesh: Mesh, args) -> Callable[..., jax.Array]:
    def local_w13(x_dispatch, ragged_group_sizes, true_group_sizes, local_w_gate_up):
        x_dispatch = jnp.squeeze(x_dispatch, axis=0)
        ragged_group_sizes = jnp.squeeze(ragged_group_sizes, axis=0)
        true_group_sizes = jnp.squeeze(true_group_sizes, axis=0)
        local_w_gate_up = jnp.squeeze(local_w_gate_up, axis=0)

        w13_out = ragged_dot(
            x_dispatch,
            local_w_gate_up,
            ragged_group_sizes,
            implementation=args.ragged_dot_impl,
        )
        gate, up = jnp.split(w13_out, 2, axis=-1)
        h = jax.nn.silu(gate.astype(jnp.float32)).astype(gate.dtype) * up
        valid_rows = jnp.arange(x_dispatch.shape[0], dtype=jnp.int32) < jnp.sum(true_group_sizes, dtype=jnp.int32)
        return jnp.where(valid_rows[:, None], h, jnp.zeros((), dtype=h.dtype))[None, ...]

    return jax.jit(
        shard_map(
            local_w13,
            mesh=mesh,
            in_specs=(P("expert", None, None), P("expert", None), P("expert", None), P("expert", None, None, None)),
            out_specs=P("expert", None, None),
            check_vma=False,
        )
    )


def _ragged_a2a_precollective_fn(
    mesh: Mesh,
    *,
    recv_capacity: int,
    num_experts: int,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    def local_precollective(x_local, selected_experts_local):
        x_local = jnp.squeeze(x_local, axis=0)
        selected_experts_local = jnp.squeeze(selected_experts_local, axis=0)
        shard_id = lax.axis_index("expert")
        local_experts = num_experts // lax.axis_size("expert")
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
        return compacted_x[None, ...], clipped_group_sizes

    return jax.jit(
        shard_map(
            local_precollective,
            mesh=mesh,
            in_specs=(P("expert", None, None), P("expert", None, None)),
            out_specs=(P("expert", None, None), P(None, None)),
            check_vma=False,
        )
    )


def _ragged_a2a_transport_from_precollective_fn(
    mesh: Mesh,
    *,
    recv_capacity: int,
    num_experts: int,
) -> Callable[..., tuple[jax.Array, jax.Array, jax.Array]]:
    def local_transport(compacted_x, clipped_group_sizes):
        compacted_x = jnp.squeeze(compacted_x, axis=0)
        shard_id = lax.axis_index("expert")
        local_experts = num_experts // lax.axis_size("expert")

        ep_size = num_experts // local_experts
        all_shard_counts = jnp.sum(clipped_group_sizes.reshape(ep_size, ep_size, local_experts), axis=2)
        input_offsets, send_sizes, output_offsets, recv_sizes = _shard_a2a_params(all_shard_counts, shard_id)
        dispatched_shape = jnp.zeros((recv_capacity, compacted_x.shape[1]), dtype=compacted_x.dtype)
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
        return x_dispatch[None, ...], ragged_group_sizes[None, ...], true_local_group_sizes[None, ...]

    return jax.jit(
        shard_map(
            local_transport,
            mesh=mesh,
            in_specs=(P("expert", None, None), P(None, None)),
            out_specs=(P("expert", None, None), P("expert", None), P("expert", None)),
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


def _ring_gather_dispatch_up_fn(
    mesh: Mesh,
    args,
    *,
    local_capacity: int,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    num_experts = args.ep_size * args.experts_per_rank
    capacity_factor = local_capacity / (args.tokens_per_rank * args.top_k)

    def local_dispatch_up(x_local, selected_experts_local, combine_weights_local, local_w_gate_up):
        x_local = jnp.squeeze(x_local, axis=0)
        selected_experts_local = jnp.squeeze(selected_experts_local, axis=0)
        combine_weights_local = jnp.squeeze(combine_weights_local, axis=0)
        local_w_gate_up = jnp.squeeze(local_w_gate_up, axis=0)
        local_experts = local_w_gate_up.shape[0]

        dispatch = _dispatch_up_ep_ring_local(
            x_local,
            selected_experts_local,
            combine_weights_local,
            local_experts=local_experts,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )
        w13_out = ragged_dot(
            dispatch.x_dispatch,
            local_w_gate_up,
            dispatch.group_sizes,
            implementation=args.ragged_dot_impl,
        )
        gate, up = jnp.split(w13_out, 2, axis=-1)
        h = jax.nn.silu(gate.astype(jnp.float32)).astype(gate.dtype) * up
        dropped_total = lax.psum(dispatch.dropped_local, "expert")
        return jnp.where(dispatch.valid[:, None], h, jnp.zeros((), dtype=h.dtype))[None, ...], dropped_total[None]

    return jax.jit(
        shard_map(
            local_dispatch_up,
            mesh=mesh,
            in_specs=(
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None, None),
            ),
            out_specs=(P("expert", None, None), P("expert")),
            check_vma=False,
        )
    )


def _ring_gather_mosaic_w13_dispatch_up_fn(
    mesh: Mesh,
    args,
    *,
    local_capacity: int,
) -> Callable[..., tuple[jax.Array, jax.Array]]:
    num_experts = args.ep_size * args.experts_per_rank
    capacity_factor = local_capacity / (args.tokens_per_rank * args.top_k)

    def local_dispatch_up(x_local, selected_experts_local, combine_weights_local, local_w_gate_up):
        x_local = jnp.squeeze(x_local, axis=0)
        selected_experts_local = jnp.squeeze(selected_experts_local, axis=0)
        combine_weights_local = jnp.squeeze(combine_weights_local, axis=0)
        local_w_gate_up = jnp.squeeze(local_w_gate_up, axis=0)
        local_experts = local_w_gate_up.shape[0]

        dispatch = _dispatch_up_ep_ring_local(
            x_local,
            selected_experts_local,
            combine_weights_local,
            local_experts=local_experts,
            num_experts=num_experts,
            capacity_factor=capacity_factor,
        )
        h = compute_moe_up_mosaic_gpu_local(
            dispatch.x_dispatch,
            dispatch.group_sizes,
            local_w_gate_up,
            block_m=args.block_m,
            block_n=args.block_n,
            block_k=args.block_k,
            max_concurrent_steps=args.num_stages,
            grid_block_n=args.grid_block_n,
        )
        dropped_total = lax.psum(dispatch.dropped_local, "expert")
        return jnp.where(dispatch.valid[:, None], h, jnp.zeros((), dtype=h.dtype))[None, ...], dropped_total[None]

    return jax.jit(
        shard_map(
            local_dispatch_up,
            mesh=mesh,
            in_specs=(
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None),
                P("expert", None, None, None),
            ),
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


def _resolve_source_expert_capacity(args) -> tuple[int | None, float | None]:
    if args.source_expert_capacity is not None and args.source_expert_capacity_factor is not None:
        raise ValueError("Specify at most one of --source-expert-capacity and --source-expert-capacity-factor")
    if args.source_expert_capacity is not None:
        return args.source_expert_capacity, None
    if args.source_expert_capacity_factor is not None:
        balanced = args.tokens_per_rank * args.top_k / (args.ep_size * args.experts_per_rank)
        return max(1, math.ceil(args.source_expert_capacity_factor * balanced)), args.source_expert_capacity_factor
    return None, None


def _print_source_expert_load_stats(prepacked, args, *, send_capacity: int) -> None:
    if isinstance(prepacked, MoeDispatchUpPrepackedSend):
        counts = np.asarray(jax.device_get(prepacked.send_expert_count_by_dst))
    elif isinstance(prepacked, MoeDispatchUpSourceExpertPrepackedSend):
        counts = np.asarray(jax.device_get(prepacked.send_count_by_dst_expert))
    else:
        raise TypeError(f"unsupported prepack type: {type(prepacked).__name__}")
    balanced = args.tokens_per_rank * args.top_k / (args.ep_size * args.experts_per_rank)
    print(
        "source_expert_loads: "
        f"balanced={balanced:.3f} "
        f"mean={float(np.mean(counts)):.3f} "
        f"max={int(np.max(counts))} "
        f"p95={float(np.percentile(counts, 95)):.1f} "
        f"p99={float(np.percentile(counts, 99)):.1f}"
    )
    current_slots_per_source = args.ep_size * send_capacity
    compact_parts = []
    for factor in (1.25, 1.5, 2.0):
        capacity = max(1, math.ceil(factor * balanced))
        overflow = int(np.maximum(counts - capacity, 0).sum())
        slots_per_source = args.ep_size * args.experts_per_rank * capacity
        compact_parts.append(f"{factor:g}x:cap={capacity},overflow={overflow},slots/source={slots_per_source}")
    print(
        "source_expert_compact_capacity: "
        f"current_slots/source={current_slots_per_source} " + " ".join(compact_parts)
    )


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
    parser.add_argument("--block-n", type=int, default=128)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument(
        "--grid-block-n",
        type=int,
        default=1,
        help="N-axis snake-grid tile width for Mosaic W13/SiLU kernels.",
    )
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--run-pallas", action="store_true")
    parser.add_argument(
        "--run-ragged-a2a-dispatch-up",
        action="store_true",
        help="Benchmark built-in ragged_all_to_all dispatch followed by ragged_dot W13/SiLU.",
    )
    parser.add_argument(
        "--run-ragged-a2a-breakdown",
        action="store_true",
        help="Benchmark ragged_all_to_all dispatch-only and ragged_dot-on-dispatched-layout as separate steps.",
    )
    parser.add_argument(
        "--run-padded-a2a-dispatch-up",
        action="store_true",
        help="Benchmark fixed-bucket all_to_all dispatch followed by ragged_dot W13/SiLU.",
    )
    parser.add_argument(
        "--run-ring-gather-dispatch-up",
        action="store_true",
        help="Benchmark all-gather/ring-style dispatch followed by ragged_dot W13/SiLU.",
    )
    parser.add_argument(
        "--run-ring-gather-mosaic-dispatch-up",
        action="store_true",
        help="Benchmark all-gather/ring-style dispatch followed by Mosaic GPU W13/SiLU.",
    )
    parser.add_argument(
        "--run-pallas-fused-dispatch-up",
        action="store_true",
        help="Benchmark one jitted ready-dispatch plus block-ready Mosaic GPU W13/SiLU path.",
    )
    parser.add_argument(
        "--run-compact-source-expert-dispatch",
        action="store_true",
        help="Benchmark compact source/expert Mosaic dispatch without W13/SiLU.",
    )
    parser.add_argument(
        "--run-compact-source-expert-dispatch-up",
        action="store_true",
        help="Benchmark compact source/expert Mosaic dispatch followed by block-ready W13/SiLU.",
    )
    parser.add_argument(
        "--run-compact-a2a-mosaic-dispatch-up",
        action="store_true",
        help="Benchmark compact source/expert all-to-all followed by Mosaic GPU W13/SiLU.",
    )
    parser.add_argument(
        "--run-compact-a2a-breakdown",
        action="store_true",
        help="Benchmark compact source/expert all-to-all transport and merged Mosaic W13 separately.",
    )
    parser.add_argument(
        "--compact-a2a-return-compact-output",
        action="store_true",
        help="Return compact source/expert W13 output instead of scattering back to destination-major rows.",
    )
    parser.add_argument(
        "--compact-a2a-merge-source-groups",
        action="store_true",
        help="After compact all-to-all, merge source groups per local expert before Mosaic W13.",
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
        "--source-expert-capacity",
        type=int,
        default=None,
        help="Rows per source/destination/local-expert compact send group. Overrides factor.",
    )
    parser.add_argument(
        "--source-expert-capacity-factor",
        type=float,
        default=None,
        help="Compact send capacity multiplier for balanced T*K/(EP*experts_per_rank) source/expert rows.",
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
        "--compact-dispatch-skip-zero-recv",
        action="store_true",
        help="For compact source/expert dispatch, skip receive-buffer zeroing and validate only written rows.",
    )
    parser.add_argument(
        "--compact-dispatch-copy-cols",
        type=int,
        default=None,
        help="Use the tiled compact source/expert dispatch path with this hidden-column tile size.",
    )
    parser.add_argument(
        "--compact-dispatch-copy-rows",
        type=int,
        default=1,
        help="Rows per compact source/expert payload tile when --compact-dispatch-copy-cols is set.",
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
    if args.compact_a2a_merge_source_groups and not args.compact_a2a_return_compact_output:
        raise ValueError("--compact-a2a-merge-source-groups requires --compact-a2a-return-compact-output")

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
        and (
            args.run_ragged_a2a_dispatch_up
            or args.run_ragged_a2a_breakdown
            or args.run_padded_a2a_dispatch_up
            or args.run_ring_gather_dispatch_up
            or args.run_ring_gather_mosaic_dispatch_up
            or args.run_compact_a2a_mosaic_dispatch_up
            or args.run_compact_a2a_breakdown
        )
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
        _print_source_expert_load_stats(prepacked, args, send_capacity=send_capacity)
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

    if (
        not args.run_pallas
        and not args.run_ragged_a2a_dispatch_up
        and not args.run_ragged_a2a_breakdown
        and not args.run_padded_a2a_dispatch_up
        and not args.run_ring_gather_dispatch_up
        and not args.run_ring_gather_mosaic_dispatch_up
        and not args.run_pallas_fused_dispatch_up
        and not args.run_compact_source_expert_dispatch
        and not args.run_compact_source_expert_dispatch_up
        and not args.run_compact_a2a_mosaic_dispatch_up
        and not args.run_compact_a2a_breakdown
    ):
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

        if args.run_ragged_a2a_breakdown:
            ragged_dispatch_fn = _ragged_a2a_dispatch_only_fn(
                mesh,
                recv_capacity=recv_capacity,
                num_experts=num_experts,
            )
            ragged_dispatch_args = (
                _sharded(mesh, x_by_rank, P("expert", None, None)),
                _sharded(mesh, expert_ids, P("expert", None, None)),
            )

            def run_ragged_dispatch():
                return ragged_dispatch_fn(*ragged_dispatch_args)

            ragged_dispatch_result, _ = _time_block("dispatch/ragged_a2a", run_ragged_dispatch)
            x_dispatch, ragged_group_sizes, true_group_sizes = ragged_dispatch_result
            ragged_dispatch_steady_ms = None
            if args.bench_iters > 0:
                ragged_dispatch_steady_ms = _measure_steady_state(
                    "dispatch/ragged_a2a",
                    run_ragged_dispatch,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )

            ragged_dispatched_w13_fn = _ragged_a2a_dispatched_w13_fn(mesh, args)
            ragged_dispatched_w13_args = (
                x_dispatch,
                ragged_group_sizes,
                true_group_sizes,
                _sharded(mesh, w_gate_up, P("expert", None, None, None)),
            )

            def run_ragged_dispatched_w13():
                return ragged_dispatched_w13_fn(*ragged_dispatched_w13_args)

            ragged_breakdown_h, _ = _time_block(
                "w13_silu/ragged_dot_on_ragged_a2a_layout",
                run_ragged_dispatched_w13,
            )
            ragged_dispatched_w13_steady_ms = None
            if args.bench_iters > 0:
                ragged_dispatched_w13_steady_ms = _measure_steady_state(
                    "w13_silu/ragged_dot_on_ragged_a2a_layout",
                    run_ragged_dispatched_w13,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
            if ref_h is None:
                print("dispatch/ragged_a2a/reference_check: skipped")
            else:
                ragged_breakdown_err = jnp.max(
                    jnp.abs(ragged_breakdown_h.astype(jnp.float32) - ref_h.astype(jnp.float32))
                )
                ragged_breakdown_err_float = float(ragged_breakdown_err)
                print(f"dispatch_up_ragged_a2a_breakdown_max_abs_error: {ragged_breakdown_err_float:.6g}")
                _print_error_summary("dispatch_up_ragged_a2a_breakdown", ragged_breakdown_h, ref_h)
                _check_error(
                    "dispatch_up_ragged_a2a_breakdown_max_abs_error",
                    ragged_breakdown_err_float,
                    args.w13_atol,
                )
            if ragged_dispatch_steady_ms is not None and ragged_dispatched_w13_steady_ms is not None:
                print(
                    "dispatch_up/ragged_a2a_breakdown_sum_ms: "
                    f"{ragged_dispatch_steady_ms + ragged_dispatched_w13_steady_ms:.3f}"
                )

            ragged_precollective_fn = _ragged_a2a_precollective_fn(
                mesh,
                recv_capacity=recv_capacity,
                num_experts=num_experts,
            )

            def run_ragged_precollective():
                return ragged_precollective_fn(*ragged_dispatch_args)

            ragged_precollective_result, _ = _time_block(
                "dispatch/ragged_a2a_precollective",
                run_ragged_precollective,
            )
            compacted_x, clipped_group_sizes = ragged_precollective_result
            ragged_precollective_steady_ms = None
            if args.bench_iters > 0:
                ragged_precollective_steady_ms = _measure_steady_state(
                    "dispatch/ragged_a2a_precollective",
                    run_ragged_precollective,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )

            ragged_transport_fn = _ragged_a2a_transport_from_precollective_fn(
                mesh,
                recv_capacity=recv_capacity,
                num_experts=num_experts,
            )

            def run_ragged_transport():
                return ragged_transport_fn(compacted_x, clipped_group_sizes)

            ragged_transport_result, _ = _time_block(
                "dispatch/ragged_a2a_transport_local_permute",
                run_ragged_transport,
            )
            transport_x_dispatch, transport_ragged_group_sizes, transport_true_group_sizes = ragged_transport_result
            ragged_transport_steady_ms = None
            if args.bench_iters > 0:
                ragged_transport_steady_ms = _measure_steady_state(
                    "dispatch/ragged_a2a_transport_local_permute",
                    run_ragged_transport,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )

            ragged_transport_w13_args = (
                transport_x_dispatch,
                transport_ragged_group_sizes,
                transport_true_group_sizes,
                _sharded(mesh, w_gate_up, P("expert", None, None, None)),
            )

            def run_ragged_transport_w13():
                return ragged_dispatched_w13_fn(*ragged_transport_w13_args)

            ragged_transport_w13_h, _ = _time_block(
                "w13_silu/ragged_dot_on_transport_layout",
                run_ragged_transport_w13,
            )
            ragged_transport_w13_steady_ms = None
            if args.bench_iters > 0:
                ragged_transport_w13_steady_ms = _measure_steady_state(
                    "w13_silu/ragged_dot_on_transport_layout",
                    run_ragged_transport_w13,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
            if ref_h is not None:
                transport_breakdown_err = jnp.max(
                    jnp.abs(ragged_transport_w13_h.astype(jnp.float32) - ref_h.astype(jnp.float32))
                )
                transport_breakdown_err_float = float(transport_breakdown_err)
                print(f"dispatch_up_ragged_a2a_transport_breakdown_max_abs_error: {transport_breakdown_err_float:.6g}")
                _check_error(
                    "dispatch_up_ragged_a2a_transport_breakdown_max_abs_error",
                    transport_breakdown_err_float,
                    args.w13_atol,
                )
            if (
                ragged_precollective_steady_ms is not None
                and ragged_transport_steady_ms is not None
                and ragged_transport_w13_steady_ms is not None
            ):
                print(
                    "dispatch_up/ragged_a2a_deep_breakdown_sum_ms: "
                    f"{ragged_precollective_steady_ms + ragged_transport_steady_ms + ragged_transport_w13_steady_ms:.3f}"
                )

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

        if args.run_ring_gather_dispatch_up:
            ring_gather_fn = _ring_gather_dispatch_up_fn(
                mesh,
                args,
                local_capacity=recv_capacity,
            )
            ring_gather_args = (
                _sharded(mesh, x_by_rank, P("expert", None, None)),
                _sharded(mesh, expert_ids, P("expert", None, None)),
                _sharded(mesh, router_weights, P("expert", None, None)),
                _sharded(mesh, w_gate_up, P("expert", None, None, None)),
            )

            def run_ring_gather_dispatch_up():
                return ring_gather_fn(*ring_gather_args)

            ring_gather_result, _ = _time_block("dispatch_up/ring_gather_ragged_dot", run_ring_gather_dispatch_up)
            ring_gather_h, ring_gather_dropped = ring_gather_result
            ring_gather_dropped_int = int(jnp.max(ring_gather_dropped))
            print(f"dispatch_up/ring_gather_dropped_total: {ring_gather_dropped_int}")
            ring_gather_steady_ms = None
            if args.bench_iters > 0:
                ring_gather_steady_ms = _measure_steady_state(
                    "dispatch_up/ring_gather_ragged_dot",
                    run_ring_gather_dispatch_up,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
            if ref_h is None:
                print("dispatch_up/ring_gather_ragged_dot/reference_check: skipped")
            else:
                ring_gather_err = jnp.max(jnp.abs(ring_gather_h.astype(jnp.float32) - ref_h.astype(jnp.float32)))
                ring_gather_err_float = float(ring_gather_err)
                print(f"dispatch_up_ring_gather_ragged_dot_max_abs_error: {ring_gather_err_float:.6g}")
                _print_error_summary("dispatch_up_ring_gather_ragged_dot", ring_gather_h, ref_h)
                _check_error(
                    "dispatch_up_ring_gather_ragged_dot_max_abs_error",
                    ring_gather_err_float,
                    args.w13_atol,
                )
            if ring_gather_steady_ms is not None:
                print(f"dispatch_up/ring_gather_ragged_dot_end_to_end_ms: {ring_gather_steady_ms:.3f}")

        if args.run_ring_gather_mosaic_dispatch_up:
            ring_gather_mosaic_fn = _ring_gather_mosaic_w13_dispatch_up_fn(
                mesh,
                args,
                local_capacity=recv_capacity,
            )
            ring_gather_mosaic_args = (
                _sharded(mesh, x_by_rank, P("expert", None, None)),
                _sharded(mesh, expert_ids, P("expert", None, None)),
                _sharded(mesh, router_weights, P("expert", None, None)),
                _sharded(mesh, w_gate_up, P("expert", None, None, None)),
            )

            def run_ring_gather_mosaic_dispatch_up():
                return ring_gather_mosaic_fn(*ring_gather_mosaic_args)

            ring_gather_mosaic_result, _ = _time_block(
                "dispatch_up/ring_gather_mosaic_gpu",
                run_ring_gather_mosaic_dispatch_up,
            )
            ring_gather_mosaic_h, ring_gather_mosaic_dropped = ring_gather_mosaic_result
            ring_gather_mosaic_dropped_int = int(jnp.max(ring_gather_mosaic_dropped))
            print(f"dispatch_up/ring_gather_mosaic_gpu_dropped_total: {ring_gather_mosaic_dropped_int}")
            ring_gather_mosaic_steady_ms = None
            if args.bench_iters > 0:
                ring_gather_mosaic_steady_ms = _measure_steady_state(
                    "dispatch_up/ring_gather_mosaic_gpu",
                    run_ring_gather_mosaic_dispatch_up,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
            if ref_h is None:
                print("dispatch_up/ring_gather_mosaic_gpu/reference_check: skipped")
            else:
                ring_gather_mosaic_err = jnp.max(
                    jnp.abs(ring_gather_mosaic_h.astype(jnp.float32) - ref_h.astype(jnp.float32))
                )
                ring_gather_mosaic_err_float = float(ring_gather_mosaic_err)
                print(f"dispatch_up_ring_gather_mosaic_gpu_max_abs_error: {ring_gather_mosaic_err_float:.6g}")
                _print_error_summary("dispatch_up_ring_gather_mosaic_gpu", ring_gather_mosaic_h, ref_h)
                _check_error(
                    "dispatch_up_ring_gather_mosaic_gpu_max_abs_error",
                    ring_gather_mosaic_err_float,
                    args.w13_atol,
                )
            if ring_gather_mosaic_steady_ms is not None:
                print(f"dispatch_up/ring_gather_mosaic_gpu_end_to_end_ms: {ring_gather_mosaic_steady_ms:.3f}")

        if args.run_compact_a2a_mosaic_dispatch_up:
            source_expert_capacity, source_expert_capacity_factor = _resolve_source_expert_capacity(args)
            compact_prepacked, _ = _time_block(
                "prepack/source_expert_reference",
                lambda: prepack_moe_dispatch_up_source_expert_reference(
                    x_by_rank,
                    expert_ids,
                    router_weights,
                    num_experts=num_experts,
                    recv_capacity=recv_capacity,
                    source_expert_capacity=source_expert_capacity,
                ),
            )
            inferred_source_expert_capacity = compact_prepacked.send_x_by_dst_expert.shape[3]
            if source_expert_capacity_factor is None:
                print(f"source_expert_capacity: {inferred_source_expert_capacity}")
            else:
                print(
                    "source_expert_capacity: "
                    f"{inferred_source_expert_capacity} from factor={source_expert_capacity_factor:g}"
                )
            print(f"source_expert_overflow_count: {int(jax.device_get(compact_prepacked.overflow_count))}")
            _print_source_expert_load_stats(compact_prepacked, args, send_capacity=send_capacity)
            compact_a2a_mosaic_fn = _compact_a2a_mosaic_w13_dispatch_up_fn(
                mesh,
                args,
                recv_capacity=recv_capacity,
                return_compact_output=args.compact_a2a_return_compact_output,
                merge_source_groups=args.compact_a2a_merge_source_groups,
            )
            compact_a2a_mosaic_args = (
                _sharded(mesh, compact_prepacked.send_x_by_dst_expert, P("expert", None, None, None, None)),
                _sharded(mesh, compact_prepacked.send_count_by_dst_expert, P("expert", None, None)),
                _sharded(mesh, compact_prepacked.recv_source_expert_base, P("expert", None, None)),
                _sharded(mesh, w_gate_up, P("expert", None, None, None)),
            )

            def run_compact_a2a_mosaic_dispatch_up():
                return compact_a2a_mosaic_fn(*compact_a2a_mosaic_args)

            compact_a2a_mosaic_result, _ = _time_block(
                "dispatch_up/compact_a2a_mosaic_gpu",
                run_compact_a2a_mosaic_dispatch_up,
            )
            compact_a2a_mosaic_h, compact_a2a_mosaic_overflow = compact_a2a_mosaic_result
            compact_a2a_mosaic_overflow_int = int(jnp.max(compact_a2a_mosaic_overflow))
            print(f"dispatch_up/compact_a2a_mosaic_gpu_overflow_total: {compact_a2a_mosaic_overflow_int}")
            compact_a2a_mosaic_steady_ms = None
            if args.bench_iters > 0:
                compact_a2a_mosaic_steady_ms = _measure_steady_state(
                    "dispatch_up/compact_a2a_mosaic_gpu",
                    run_compact_a2a_mosaic_dispatch_up,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
            if ref_h is None:
                print("dispatch_up/compact_a2a_mosaic_gpu/reference_check: skipped")
            elif args.compact_a2a_return_compact_output:
                print("dispatch_up/compact_a2a_mosaic_gpu/reference_check: skipped_compact_output")
            else:
                compact_a2a_mosaic_err = jnp.max(
                    jnp.abs(compact_a2a_mosaic_h.astype(jnp.float32) - ref_h.astype(jnp.float32))
                )
                compact_a2a_mosaic_err_float = float(compact_a2a_mosaic_err)
                print(f"dispatch_up_compact_a2a_mosaic_gpu_max_abs_error: {compact_a2a_mosaic_err_float:.6g}")
                _print_error_summary("dispatch_up_compact_a2a_mosaic_gpu", compact_a2a_mosaic_h, ref_h)
                _check_error(
                    "dispatch_up_compact_a2a_mosaic_gpu_max_abs_error",
                    compact_a2a_mosaic_err_float,
                    args.w13_atol,
                )
            if compact_a2a_mosaic_steady_ms is not None:
                print(f"dispatch_up/compact_a2a_mosaic_gpu_end_to_end_ms: {compact_a2a_mosaic_steady_ms:.3f}")

        if args.run_compact_a2a_breakdown:
            source_expert_capacity, source_expert_capacity_factor = _resolve_source_expert_capacity(args)
            compact_prepacked, _ = _time_block(
                "prepack/source_expert_reference",
                lambda: prepack_moe_dispatch_up_source_expert_reference(
                    x_by_rank,
                    expert_ids,
                    router_weights,
                    num_experts=num_experts,
                    recv_capacity=recv_capacity,
                    source_expert_capacity=source_expert_capacity,
                ),
            )
            inferred_source_expert_capacity = compact_prepacked.send_x_by_dst_expert.shape[3]
            if source_expert_capacity_factor is None:
                print(f"source_expert_capacity: {inferred_source_expert_capacity}")
            else:
                print(
                    "source_expert_capacity: "
                    f"{inferred_source_expert_capacity} from factor={source_expert_capacity_factor:g}"
                )
            print(f"source_expert_overflow_count: {int(jax.device_get(compact_prepacked.overflow_count))}")
            _print_source_expert_load_stats(compact_prepacked, args, send_capacity=send_capacity)

            compact_transport_fn = _compact_a2a_transport_fn(mesh)
            compact_transport_args = (
                _sharded(mesh, compact_prepacked.send_x_by_dst_expert, P("expert", None, None, None, None)),
                _sharded(mesh, compact_prepacked.send_count_by_dst_expert, P("expert", None, None)),
            )

            def run_compact_a2a_transport():
                return compact_transport_fn(*compact_transport_args)

            compact_transport_result, _ = _time_block(
                "dispatch/compact_a2a_transport",
                run_compact_a2a_transport,
            )
            source_expert_x, source_expert_count = compact_transport_result
            if args.bench_iters > 0:
                compact_transport_steady_ms = _measure_steady_state(
                    "dispatch/compact_a2a_transport",
                    run_compact_a2a_transport,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
            else:
                compact_transport_steady_ms = None
            print(f"dispatch/compact_a2a_transport_count_max: {int(jnp.max(source_expert_count))}")

            compact_merged_w13_fn = _compact_a2a_merged_mosaic_w13_fn(mesh, args)
            compact_merged_w13_args = (
                source_expert_x,
                _sharded(mesh, w_gate_up, P("expert", None, None, None)),
            )

            def run_compact_merged_w13():
                return compact_merged_w13_fn(*compact_merged_w13_args)

            _compact_merged_h, _ = _time_block(
                "w13_silu/compact_a2a_merged_mosaic_gpu",
                run_compact_merged_w13,
            )
            if args.bench_iters > 0:
                compact_merged_w13_steady_ms = _measure_steady_state(
                    "w13_silu/compact_a2a_merged_mosaic_gpu",
                    run_compact_merged_w13,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
            else:
                compact_merged_w13_steady_ms = None
            if compact_transport_steady_ms is not None and compact_merged_w13_steady_ms is not None:
                print(
                    "dispatch_up/compact_a2a_breakdown_sum_ms: "
                    f"{compact_transport_steady_ms + compact_merged_w13_steady_ms:.3f}"
                )

        if args.run_compact_source_expert_dispatch:
            source_expert_capacity, source_expert_capacity_factor = _resolve_source_expert_capacity(args)
            compact_prepacked, _ = _time_block(
                "prepack/source_expert_reference",
                lambda: prepack_moe_dispatch_up_source_expert_reference(
                    x_by_rank,
                    expert_ids,
                    router_weights,
                    num_experts=num_experts,
                    recv_capacity=recv_capacity,
                    source_expert_capacity=source_expert_capacity,
                ),
            )
            inferred_source_expert_capacity = compact_prepacked.send_x_by_dst_expert.shape[3]
            if source_expert_capacity_factor is None:
                print(f"source_expert_capacity: {inferred_source_expert_capacity}")
            else:
                print(
                    "source_expert_capacity: "
                    f"{inferred_source_expert_capacity} from factor={source_expert_capacity_factor:g}"
                )
            compact_dispatch_fn = _pallas_compact_source_expert_dispatch_fn(
                mesh,
                recv_capacity=recv_capacity,
                ready_block_m=args.block_m,
                rows_per_program=args.dispatch_rows_per_program,
                zero_recv=not args.compact_dispatch_skip_zero_recv,
                copy_cols=args.compact_dispatch_copy_cols,
                copy_rows=args.compact_dispatch_copy_rows,
            )
            compact_dispatch_args = _compact_source_expert_args(mesh, compact_prepacked)

            def run_compact_source_expert_dispatch():
                return compact_dispatch_fn(*compact_dispatch_args)

            compact_dispatch_result, _ = _time_block(
                "dispatch/mosaic_gpu_compact_source_expert",
                run_compact_source_expert_dispatch,
            )
            (
                compact_recv_x,
                compact_recv_valid,
                compact_recv_local_expert,
                compact_recv_src_rank,
                compact_recv_src_token_idx,
                compact_recv_topk_slot,
                compact_recv_router_weight,
                compact_ready_count,
                compact_ready_block_count,
            ) = compact_dispatch_result
            compact_dispatch_steady_ms = None
            if args.bench_iters > 0:
                compact_dispatch_steady_ms = _measure_steady_state(
                    "dispatch/mosaic_gpu_compact_source_expert",
                    run_compact_source_expert_dispatch,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
            if prepacked is not None:
                expected_ready_count = _sharded(
                    mesh, _expected_ready_count(prepacked, recv_capacity), P("expert", None, None)
                )
                compact_ready_count_err = int(jnp.max(jnp.abs(compact_ready_count - expected_ready_count)))
                print(f"dispatch_compact_source_expert_ready_count_max_abs_error: {compact_ready_count_err}")
                if compact_ready_count_err != 0:
                    raise AssertionError(
                        f"compact source/expert ready-count mismatch: max_abs={compact_ready_count_err}"
                    )
                expected_ready_block_count = _sharded(
                    mesh, _expected_ready_block_count(prepacked, recv_capacity, args.block_m), P("expert", None)
                )
                compact_ready_block_count_err = int(
                    jnp.max(jnp.abs(compact_ready_block_count - expected_ready_block_count))
                )
                print(
                    f"dispatch_compact_source_expert_ready_block_count_max_abs_error: {compact_ready_block_count_err}"
                )
                if compact_ready_block_count_err != 0:
                    raise AssertionError(
                        "compact source/expert ready-block-count mismatch: " f"max_abs={compact_ready_block_count_err}"
                    )
            if ref_layout is None:
                print("dispatch/mosaic_gpu_compact_source_expert/reference_check: skipped")
            else:
                valid_mask = ref_layout.recv_valid
                recv_x_err = float(
                    jnp.max(
                        jnp.abs(compact_recv_x.astype(jnp.float32) - ref_layout.recv_x.astype(jnp.float32))
                        * valid_mask[:, :, None].astype(jnp.float32)
                    )
                )
                print(f"dispatch_mosaic_gpu_compact_source_expert_recv_x_max_abs_error: {recv_x_err:.6g}")
                _check_error("dispatch_mosaic_gpu_compact_source_expert_recv_x_max_abs_error", recv_x_err, 0.0)
                valid_err = int(jnp.max(jnp.abs(compact_recv_valid.astype(jnp.int32) - valid_mask.astype(jnp.int32))))
                print(f"dispatch_mosaic_gpu_compact_source_expert_recv_valid_max_abs_error: {valid_err}")
                if valid_err != 0:
                    raise AssertionError(f"compact source/expert recv_valid mismatch: max_abs={valid_err}")
                for label, actual, expected in (
                    ("recv_local_expert", compact_recv_local_expert, ref_layout.recv_local_expert),
                    ("recv_src_rank", compact_recv_src_rank, ref_layout.recv_src_rank),
                    ("recv_src_token_idx", compact_recv_src_token_idx, ref_layout.recv_src_token_idx),
                    ("recv_topk_slot", compact_recv_topk_slot, ref_layout.recv_topk_slot),
                ):
                    err = int(
                        jnp.max(
                            jnp.abs(actual.astype(jnp.int32) - expected.astype(jnp.int32))
                            * valid_mask.astype(jnp.int32)
                        )
                    )
                    print(f"dispatch_mosaic_gpu_compact_source_expert_{label}_max_abs_error: {err}")
                    if err != 0:
                        raise AssertionError(f"compact source/expert {label} mismatch: max_abs={err}")
                router_weight_err = float(
                    jnp.max(
                        jnp.abs(
                            compact_recv_router_weight.astype(jnp.float32)
                            - ref_layout.recv_router_weight.astype(jnp.float32)
                        )
                        * valid_mask.astype(jnp.float32)
                    )
                )
                print(
                    "dispatch_mosaic_gpu_compact_source_expert_recv_router_weight_max_abs_error: "
                    f"{router_weight_err:.6g}"
                )
                _check_error(
                    "dispatch_mosaic_gpu_compact_source_expert_recv_router_weight_max_abs_error",
                    router_weight_err,
                    0.0,
                )
            if compact_dispatch_steady_ms is not None:
                print(f"dispatch/mosaic_gpu_compact_source_expert_end_to_end_ms: {compact_dispatch_steady_ms:.3f}")

        if args.run_compact_source_expert_dispatch_up:
            source_expert_capacity, source_expert_capacity_factor = _resolve_source_expert_capacity(args)
            compact_prepacked, _ = _time_block(
                "prepack/source_expert_reference",
                lambda: prepack_moe_dispatch_up_source_expert_reference(
                    x_by_rank,
                    expert_ids,
                    router_weights,
                    num_experts=num_experts,
                    recv_capacity=recv_capacity,
                    source_expert_capacity=source_expert_capacity,
                ),
            )
            inferred_source_expert_capacity = compact_prepacked.send_x_by_dst_expert.shape[3]
            if source_expert_capacity_factor is None:
                print(f"source_expert_capacity: {inferred_source_expert_capacity}")
            else:
                print(
                    "source_expert_capacity: "
                    f"{inferred_source_expert_capacity} from factor={source_expert_capacity_factor:g}"
                )
            compact_dispatch_up_fn = _pallas_compact_source_expert_dispatch_up_fn(
                mesh,
                args,
                recv_capacity=recv_capacity,
                ready_block_m=args.block_m,
                rows_per_program=args.dispatch_rows_per_program,
                zero_recv=not args.compact_dispatch_skip_zero_recv,
                copy_cols=args.compact_dispatch_copy_cols,
                copy_rows=args.compact_dispatch_copy_rows,
            )
            compact_dispatch_up_args = (
                *_compact_source_expert_args(mesh, compact_prepacked),
                _sharded(mesh, w_gate_up, P("expert", None, None, None)),
            )

            def run_compact_source_expert_dispatch_up():
                return compact_dispatch_up_fn(*compact_dispatch_up_args)

            compact_result, _ = _time_block(
                "dispatch_up/mosaic_gpu_compact_source_expert_block_ready",
                run_compact_source_expert_dispatch_up,
            )
            compact_h, compact_ready_count, compact_ready_block_count = compact_result
            compact_steady_ms = None
            if args.bench_iters > 0:
                compact_steady_ms = _measure_steady_state(
                    "dispatch_up/mosaic_gpu_compact_source_expert_block_ready",
                    run_compact_source_expert_dispatch_up,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
            if prepacked is not None:
                expected_ready_count = _sharded(
                    mesh, _expected_ready_count(prepacked, recv_capacity), P("expert", None, None)
                )
                compact_ready_count_err = int(jnp.max(jnp.abs(compact_ready_count - expected_ready_count)))
                print(f"dispatch_up_compact_source_expert_ready_count_max_abs_error: {compact_ready_count_err}")
                if compact_ready_count_err != 0:
                    raise AssertionError(
                        f"compact source/expert ready-count mismatch: max_abs={compact_ready_count_err}"
                    )
                expected_ready_block_count = _sharded(
                    mesh, _expected_ready_block_count(prepacked, recv_capacity, args.block_m), P("expert", None)
                )
                compact_ready_block_count_err = int(
                    jnp.max(jnp.abs(compact_ready_block_count - expected_ready_block_count))
                )
                print(
                    "dispatch_up_compact_source_expert_ready_block_count_max_abs_error: "
                    f"{compact_ready_block_count_err}"
                )
                if compact_ready_block_count_err != 0:
                    raise AssertionError(
                        "compact source/expert ready-block-count mismatch: " f"max_abs={compact_ready_block_count_err}"
                    )
            if ref_h is None:
                print("dispatch_up/mosaic_gpu_compact_source_expert_block_ready/reference_check: skipped")
            else:
                compact_err = jnp.max(jnp.abs(compact_h.astype(jnp.float32) - ref_h.astype(jnp.float32)))
                compact_err_float = float(compact_err)
                print(
                    f"dispatch_up_mosaic_gpu_compact_source_expert_block_ready_max_abs_error: {compact_err_float:.6g}"
                )
                _print_error_summary("dispatch_up_mosaic_gpu_compact_source_expert_block_ready", compact_h, ref_h)
                _check_error(
                    "dispatch_up_mosaic_gpu_compact_source_expert_block_ready_max_abs_error",
                    compact_err_float,
                    args.w13_atol,
                )
            if compact_steady_ms is not None:
                print(
                    f"dispatch_up/mosaic_gpu_compact_source_expert_block_ready_end_to_end_ms: {compact_steady_ms:.3f}"
                )

        if args.run_pallas_fused_dispatch_up:
            if prepacked is None:
                raise AssertionError("prepacked sends are required for fused Mosaic GPU dispatch-up")
            if args.dispatch_copy_mode not in ("scratch_ready", "direct_ready"):
                raise ValueError(
                    "--run-pallas-fused-dispatch-up requires --dispatch-copy-mode=scratch_ready or direct_ready"
                )
            fused_dispatch_up_fn = _pallas_fused_dispatch_block_ready_w13_fn(
                mesh,
                args,
                recv_capacity=recv_capacity,
                ready_block_m=args.block_m,
                rows_per_program=args.dispatch_rows_per_program,
                copy_mode=args.dispatch_copy_mode,
            )
            fused_dispatch_up_args = (
                *_pallas_dispatch_ready_args(mesh, prepacked),
                _sharded(mesh, w_gate_up, P("expert", None, None, None)),
            )

            def run_pallas_fused_dispatch_up():
                return fused_dispatch_up_fn(*fused_dispatch_up_args)

            fused_dispatch_up_result, _ = _time_block(
                "dispatch_up/mosaic_gpu_fused_block_ready",
                run_pallas_fused_dispatch_up,
            )
            fused_h, fused_ready_count, fused_ready_block_count = fused_dispatch_up_result
            fused_dispatch_up_steady_ms = None
            if args.bench_iters > 0:
                fused_dispatch_up_steady_ms = _measure_steady_state(
                    "dispatch_up/mosaic_gpu_fused_block_ready",
                    run_pallas_fused_dispatch_up,
                    warmup_steps=args.warmup_steps,
                    bench_iters=args.bench_iters,
                )
            expected_ready_count = _sharded(
                mesh, _expected_ready_count(prepacked, recv_capacity), P("expert", None, None)
            )
            fused_ready_count_err = int(jnp.max(jnp.abs(fused_ready_count - expected_ready_count)))
            print(f"dispatch_up_fused_ready_count_max_abs_error: {fused_ready_count_err}")
            if fused_ready_count_err != 0:
                raise AssertionError(f"fused dispatch ready-count mismatch: max_abs={fused_ready_count_err}")
            expected_ready_block_count = _sharded(
                mesh, _expected_ready_block_count(prepacked, recv_capacity, args.block_m), P("expert", None)
            )
            fused_ready_block_count_err = int(jnp.max(jnp.abs(fused_ready_block_count - expected_ready_block_count)))
            print(f"dispatch_up_fused_ready_block_count_max_abs_error: {fused_ready_block_count_err}")
            if fused_ready_block_count_err != 0:
                raise AssertionError(
                    f"fused dispatch ready-block-count mismatch: max_abs={fused_ready_block_count_err}"
                )
            if ref_h is None:
                print("dispatch_up/mosaic_gpu_fused_block_ready/reference_check: skipped")
            else:
                fused_err = jnp.max(jnp.abs(fused_h.astype(jnp.float32) - ref_h.astype(jnp.float32)))
                fused_err_float = float(fused_err)
                print(f"dispatch_up_mosaic_gpu_fused_block_ready_max_abs_error: {fused_err_float:.6g}")
                _print_error_summary("dispatch_up_mosaic_gpu_fused_block_ready", fused_h, ref_h)
                _check_error(
                    "dispatch_up_mosaic_gpu_fused_block_ready_max_abs_error",
                    fused_err_float,
                    args.w13_atol,
                )
            if fused_dispatch_up_steady_ms is not None:
                print(f"dispatch_up/mosaic_gpu_fused_block_ready_end_to_end_ms: {fused_dispatch_up_steady_ms:.3f}")

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
