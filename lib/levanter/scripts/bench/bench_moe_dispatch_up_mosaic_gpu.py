# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate the experimental MoE dispatch-up Mosaic GPU subkernel."""

import argparse
import time
from collections.abc import Callable

import jax
import jax.numpy as jnp
import numpy as np
from jax import shard_map
from jax.sharding import AxisType, Mesh, NamedSharding, PartitionSpec as P

from levanter.kernels.pallas.moe_dispatch_up.mosaic_gpu import (
    dispatch_prepacked_moe_dispatch_up_mosaic_gpu_local,
    compute_moe_up_mosaic_gpu_local,
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
    dispatch_ms: float,
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
    dispatch_payload_gbs = dispatch_payload_bytes / (dispatch_ms / 1e3) / 1e9
    w13_tflops = w13_flops / (w13_ms / 1e3) / 1e12
    w13_intensity = w13_flops / w13_bytes
    w13_hbm_bound_tflops = 3.35e12 * w13_intensity / 1e12
    h100_bf16_peak_tflops = 989.0
    w13_roofline_tflops = min(h100_bf16_peak_tflops * ep_size, w13_hbm_bound_tflops * ep_size)
    print(
        "roofline: "
        f"routed_rows={routed_rows} "
        f"dispatch_payload={dispatch_payload_bytes / 1024:.1f} KiB "
        f"dispatch_payload_bw={dispatch_payload_gbs:.6f} GB/s "
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
    w_gate_up = jax.random.normal(k_w, (ep_size, experts_per_rank, hidden, 2 * intermediate), dtype=dtype)
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


def _pallas_dispatch_fn(
    mesh: Mesh,
    *,
    recv_capacity: int,
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


def _pallas_w13_silu_args(mesh: Mesh, layout: MoeDispatchUpLayout, w_gate_up: jax.Array) -> tuple[jax.Array, ...]:
    return (
        _sharded(mesh, layout.recv_x, P("expert", None, None)),
        _sharded(mesh, layout.rows_per_expert, P("expert", None)),
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
    parser.add_argument("--block-m", type=int, default=64)
    parser.add_argument("--block-n", type=int, default=64)
    parser.add_argument("--block-k", type=int, default=64)
    parser.add_argument("--num-stages", type=int, default=2)
    parser.add_argument("--dtype", choices=("bf16", "fp32"), default="bf16")
    parser.add_argument("--run-pallas", action="store_true")
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
        f"K={args.top_k} H={args.hidden} I={args.intermediate} dtype={args.dtype}"
    )

    x_by_rank, expert_ids, router_weights, w_gate_up = _make_inputs(
        ep_size=args.ep_size,
        tokens_per_rank=args.tokens_per_rank,
        experts_per_rank=args.experts_per_rank,
        top_k=args.top_k,
        hidden=args.hidden,
        intermediate=args.intermediate,
        dtype=dtype,
    )
    recv_capacity = args.ep_size * args.tokens_per_rank * args.top_k
    num_experts = args.ep_size * args.experts_per_rank

    prepacked, _ = _time_block(
        "prepack/reference",
        lambda: prepack_moe_dispatch_up_reference(
            x_by_rank,
            expert_ids,
            router_weights,
            num_experts=num_experts,
            recv_capacity=recv_capacity,
            send_capacity=args.tokens_per_rank * args.top_k,
        ),
    )
    ref_layout, _ = _time_block(
        "dispatch/reference",
        lambda: dispatch_prepacked_moe_dispatch_up_reference(prepacked, recv_capacity=recv_capacity),
    )
    ref_h, _ = _time_block("w13_silu/reference", lambda: compute_moe_up_from_layout_reference(ref_layout, w_gate_up))
    ref_dispatch_steady_ms = None
    ref_w13_steady_ms = None
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

    if not args.run_pallas:
        return
    if len(devices) < args.ep_size:
        raise RuntimeError(f"Need at least {args.ep_size} local devices, found {len(devices)}")

    mesh = Mesh(np.array(devices[: args.ep_size]), ("expert",), axis_types=(AxisType.Explicit,))
    with jax.set_mesh(mesh):
        pallas_dispatch_fn = _pallas_dispatch_fn(mesh, recv_capacity=recv_capacity)
        pallas_dispatch_args = _pallas_dispatch_args(mesh, prepacked)

        def run_pallas_dispatch():
            return pallas_dispatch_fn(*pallas_dispatch_args)

        pallas_layout, _ = _time_block("dispatch/mosaic_gpu", run_pallas_dispatch)
        pallas_dispatch_steady_ms = None
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
        _check_error("dispatch_max_abs_error", dispatch_err_float, args.dispatch_atol)
        if valid_err_int != 0 or expert_err_int != 0 or src_rank_err_int != 0:
            raise AssertionError(
                "dispatch metadata mismatch: "
                f"valid={valid_err_int} local_expert={expert_err_int} src_rank={src_rank_err_int}"
            )

        pallas_w13_fn = _pallas_w13_silu_fn(mesh, args)
        pallas_w13_args = _pallas_w13_silu_args(mesh, pallas_layout, w_gate_up)

        def run_pallas_w13():
            return pallas_w13_fn(*pallas_w13_args)

        pallas_h, _ = _time_block("w13_silu/mosaic_gpu", run_pallas_w13)
        pallas_w13_steady_ms = None
        if args.bench_iters > 0:
            pallas_w13_steady_ms = _measure_steady_state(
                "w13_silu/mosaic_gpu",
                run_pallas_w13,
                warmup_steps=args.warmup_steps,
                bench_iters=args.bench_iters,
            )
            if ref_w13_steady_ms is not None:
                _print_speedup("w13_silu/mosaic_gpu_vs_reference_jit", ref_w13_steady_ms, pallas_w13_steady_ms)
        h_err = jnp.max(jnp.abs(pallas_h.astype(jnp.float32) - ref_h.astype(jnp.float32)))
        h_err_float = float(h_err)
        print(f"w13_silu_max_abs_error: {h_err_float:.6g}")
        if args.debug_errors:
            _print_error_debug("w13_silu", pallas_h, ref_h, pallas_layout, w_gate_up)
        _check_error("w13_silu_max_abs_error", h_err_float, args.w13_atol)
        if pallas_dispatch_steady_ms is not None and pallas_w13_steady_ms is not None:
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
