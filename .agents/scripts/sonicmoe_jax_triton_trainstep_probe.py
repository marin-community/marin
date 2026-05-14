# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Probe raw Sonic-style Triton calls inside a train-step-shaped JAX executable.

This complements `sonicmoe_jax_triton_overhead_probe.py`: that script measures
standalone launch overhead. This one embeds the same raw Sonic-style gather/sum
body, plus a simple raw Triton backward for that op, inside larger `jax.jit`
functions so we can separate standalone dispatch cost from in-executable custom
call cost.
"""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import numpy as np
import torch
import triton
import triton.language as tl
from haliax.nn.ragged_dot import ragged_dot
from levanter.grug.grug_moe import (
    _gather_sum_reference,
    _prepare_moe_dispatch_indices_with_assignment_ids,
    split_moe_w13_output,
)
from sonicmoe_jax_triton_overhead_probe import (
    KernelConfig,
    _diff_stats,
    _launch_torch_gather,
    _make_inputs,
    _reference,
    _sonic_token_gather_sum_kernel,
    _time_torch,
    _torch_to_jax,
    jt,
)


@triton.jit
def _sonic_token_gather_sum_bwd_kernel(
    dout_ptr,  # (T, H)
    x_ptr,  # (Mtotal, H)
    w_ptr,  # (Mtotal,)
    m_perm_ptr,  # (Mtotal,) int32
    m_offset_ptr,  # unused for fixed-k, kept to match forward metadata
    dx_ptr,  # (Mtotal, H)
    dw_ptr,  # (Mtotal,)
    t: tl.constexpr,
    h: tl.constexpr,
    max_k: tl.constexpr,
    stride_doutt: tl.constexpr,
    stride_douth: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xh: tl.constexpr,
    block_h: tl.constexpr,
):
    assignment = tl.program_id(axis=0)
    token = assignment // max_k
    h_idx = tl.arange(0, block_h).to(tl.int64)
    h_mask = h_idx < h

    perm_idx = tl.load(m_perm_ptr + assignment).to(tl.int64)
    weight = tl.load(w_ptr + assignment).to(tl.float32)

    dout = tl.load(dout_ptr + token * stride_doutt + h_idx * stride_douth, mask=h_mask, other=0.0).to(tl.float32)
    x = tl.load(x_ptr + perm_idx * stride_xm + h_idx * stride_xh, mask=h_mask, other=0.0).to(tl.float32)

    tl.store(dx_ptr + perm_idx * stride_xm + h_idx * stride_xh, dout * weight, mask=h_mask)
    dw = tl.sum(dout * x, axis=0)
    tl.store(dw_ptr + assignment, dw)


@dataclass(frozen=True)
class StepTimingResult:
    label: str
    mode: str
    count: int | None
    wall_ms: float
    xla_baseline_wall_ms: float | None
    delta_vs_xla_ms: float | None
    per_raw_call_delta_ms: float | None
    max_abs_vs_xla: float | None
    mean_abs_vs_xla: float | None


@dataclass(frozen=True)
class TorchTimingResult:
    label: str
    torch_event_ms: float
    torch_wall_ms: float
    max_abs_dx: float
    mean_abs_dx: float
    max_abs_dw: float
    mean_abs_dw: float


def _block_until_ready(value):
    return jax.tree_util.tree_map(lambda leaf: leaf.block_until_ready(), value)


def _time_jax_pytree(
    fn: Callable[..., object],
    args: tuple[object, ...],
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        _block_until_ready(fn(*args))

    start = time.perf_counter()
    for _ in range(repeats):
        _block_until_ready(fn(*args))
    end = time.perf_counter()
    return (end - start) * 1000.0 / repeats


def _jax_xla_gather_sum(
    x: jax.Array,
    w: jax.Array,
    perm: jax.Array,
    *,
    tokens: int,
    topk: int,
) -> jax.Array:
    gathered = jnp.take(x, perm.reshape(tokens, topk), axis=0).astype(jnp.float32)
    weighted = gathered * w.reshape(tokens, topk, 1).astype(jnp.float32)
    return weighted.sum(axis=1).astype(x.dtype)


def _make_jax_raw_gather(
    *,
    tokens: int,
    hidden: int,
    topk: int,
    config: KernelConfig,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    out_shape = jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16)

    def raw_gather(x: jax.Array, w: jax.Array, perm: jax.Array, offset: jax.Array) -> jax.Array:
        return jt.triton_call(
            x,
            w,
            perm,
            offset,
            kernel=_sonic_token_gather_sum_kernel,
            out_shape=out_shape,
            grid=(tokens,),
            num_warps=config.num_warps,
            num_stages=config.num_stages,
            t=tokens,
            h=hidden,
            max_k=topk,
            stride_xm=hidden,
            stride_xh=1,
            stride_outt=hidden,
            stride_outh=1,
            block_h=config.block_h,
            block_k=config.block_k,
            w_is_none=False,
            is_varlen_k=False,
        )

    return raw_gather


def _make_jax_raw_gather_with_vjp(
    *,
    tokens: int,
    hidden: int,
    topk: int,
    config: KernelConfig,
    bwd_block_h: int,
    bwd_num_warps: int,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    raw_gather_impl = _make_jax_raw_gather(tokens=tokens, hidden=hidden, topk=topk, config=config)
    dx_shape = jax.ShapeDtypeStruct((tokens * topk, hidden), jnp.bfloat16)
    dw_shape = jax.ShapeDtypeStruct((tokens * topk,), jnp.float32)

    def raw_bwd_impl(
        dout: jax.Array,
        x: jax.Array,
        w: jax.Array,
        perm: jax.Array,
        offset: jax.Array,
    ) -> tuple[jax.Array, jax.Array]:
        return jt.triton_call(
            dout,
            x,
            w,
            perm,
            offset,
            kernel=_sonic_token_gather_sum_bwd_kernel,
            out_shape=(dx_shape, dw_shape),
            grid=(tokens * topk,),
            num_warps=bwd_num_warps,
            num_stages=4,
            t=tokens,
            h=hidden,
            max_k=topk,
            stride_doutt=hidden,
            stride_douth=1,
            stride_xm=hidden,
            stride_xh=1,
            block_h=bwd_block_h,
        )

    @jax.custom_vjp
    def raw_gather(x: jax.Array, w: jax.Array, perm: jax.Array, offset: jax.Array) -> jax.Array:
        return raw_gather_impl(x, w, perm, offset)

    def raw_gather_fwd(
        x: jax.Array,
        w: jax.Array,
        perm: jax.Array,
        offset: jax.Array,
    ) -> tuple[jax.Array, tuple[jax.Array, jax.Array, jax.Array, jax.Array]]:
        out = raw_gather_impl(x, w, perm, offset)
        return out, (x, w, perm, offset)

    def raw_gather_bwd(
        residuals: tuple[jax.Array, jax.Array, jax.Array, jax.Array],
        dout: jax.Array,
    ) -> tuple[jax.Array, jax.Array, None, None]:
        x, w, perm, offset = residuals
        dx, dw = raw_bwd_impl(dout, x, w, perm, offset)
        return dx, dw, None, None

    raw_gather.defvjp(raw_gather_fwd, raw_gather_bwd)
    return raw_gather


def _make_forward_loss_step(
    gather_fn: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
    *,
    tokens: int,
    topk: int,
    use_xla_reference: bool,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    @jax.jit
    def step(x: jax.Array, w: jax.Array, perm: jax.Array, offset: jax.Array, target: jax.Array) -> jax.Array:
        if use_xla_reference:
            y = _jax_xla_gather_sum(x, w, perm, tokens=tokens, topk=topk)
        else:
            y = gather_fn(x, w, perm, offset)
        diff = y.astype(jnp.float32) - target.astype(jnp.float32)
        return jnp.mean(diff * diff)

    return step


def _make_train_step(
    gather_fn: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
    *,
    tokens: int,
    topk: int,
    use_xla_reference: bool,
) -> Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], tuple[jax.Array, jax.Array, jax.Array]
]:
    def loss_fn(
        x: jax.Array,
        w: jax.Array,
        perm: jax.Array,
        offset: jax.Array,
        target: jax.Array,
    ) -> jax.Array:
        if use_xla_reference:
            y = _jax_xla_gather_sum(x, w, perm, tokens=tokens, topk=topk)
        else:
            y = gather_fn(x, w, perm, offset)
        diff = y.astype(jnp.float32) - target.astype(jnp.float32)
        return jnp.mean(diff * diff)

    @jax.jit
    def step(
        x: jax.Array,
        w: jax.Array,
        perm: jax.Array,
        offset: jax.Array,
        target: jax.Array,
        lr: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        loss, (dx, dw) = jax.value_and_grad(loss_fn, argnums=(0, 1))(x, w, perm, offset, target)
        next_x = (x - lr.astype(x.dtype) * dx).astype(x.dtype)
        next_w = w - lr.astype(w.dtype) * dw
        return loss, next_x, next_w

    return step


def _make_multi_forward_loss_step(
    gather_fn: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
    *,
    tokens: int,
    hidden: int,
    topk: int,
    count: int,
    use_xla_reference: bool,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    @jax.jit
    def step(x: jax.Array, ws: jax.Array, perm: jax.Array, offset: jax.Array, target: jax.Array) -> jax.Array:
        acc = jnp.zeros((tokens, hidden), dtype=jnp.float32)
        for i in range(count):
            if use_xla_reference:
                y = _jax_xla_gather_sum(x, ws[i], perm, tokens=tokens, topk=topk)
            else:
                y = gather_fn(x, ws[i], perm, offset)
            acc += y.astype(jnp.float32)
        diff = acc / float(count) - target.astype(jnp.float32)
        return jnp.mean(diff * diff)

    return step


def _moe_forward(
    x: jax.Array,
    selected_experts: jax.Array,
    combine_weights: jax.Array,
    w13_interleaved: jax.Array,
    w2: jax.Array,
    raw_gather_fn: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
    *,
    use_xla_reference: bool,
) -> jax.Array:
    num_experts = w13_interleaved.shape[0]
    tokens, topk = selected_experts.shape
    token_ids_sort, dispatch_positions, group_sizes, _sorted_assignment_ids = (
        _prepare_moe_dispatch_indices_with_assignment_ids(
            selected_experts,
            num_experts=num_experts,
        )
    )
    x_dispatch = x[token_ids_sort]
    w13_out = ragged_dot(x_dispatch, w13_interleaved, group_sizes)
    gate, up = split_moe_w13_output(w13_out, intermediate_dim=w2.shape[1], interleaved=True)
    hidden = jax.nn.silu(gate) * up
    out_dispatch = ragged_dot(hidden, w2, group_sizes)
    if use_xla_reference:
        return _gather_sum_reference(out_dispatch, dispatch_positions, combine_weights)

    offset = jnp.arange(0, tokens * topk + 1, topk, dtype=jnp.int32)
    return raw_gather_fn(
        out_dispatch,
        combine_weights.reshape(tokens * topk).astype(jnp.float32),
        dispatch_positions.reshape(tokens * topk).astype(jnp.int32),
        offset,
    )


def _make_moe_forward_loss_step(
    raw_gather_fn: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
    *,
    use_xla_reference: bool,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    @jax.jit
    def step(
        x: jax.Array,
        selected_experts: jax.Array,
        combine_weights: jax.Array,
        w13_interleaved: jax.Array,
        w2: jax.Array,
        target: jax.Array,
    ) -> jax.Array:
        y = _moe_forward(
            x,
            selected_experts,
            combine_weights,
            w13_interleaved,
            w2,
            raw_gather_fn,
            use_xla_reference=use_xla_reference,
        )
        diff = y.astype(jnp.float32) - target.astype(jnp.float32)
        return jnp.mean(diff * diff)

    return step


def _make_moe_train_step(
    raw_gather_fn: Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array],
    *,
    use_xla_reference: bool,
) -> Callable[
    [jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
    tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array],
]:
    def loss_fn(
        x: jax.Array,
        selected_experts: jax.Array,
        combine_weights: jax.Array,
        w13_interleaved: jax.Array,
        w2: jax.Array,
        target: jax.Array,
    ) -> jax.Array:
        y = _moe_forward(
            x,
            selected_experts,
            combine_weights,
            w13_interleaved,
            w2,
            raw_gather_fn,
            use_xla_reference=use_xla_reference,
        )
        diff = y.astype(jnp.float32) - target.astype(jnp.float32)
        return jnp.mean(diff * diff)

    @jax.jit
    def step(
        x: jax.Array,
        selected_experts: jax.Array,
        combine_weights: jax.Array,
        w13_interleaved: jax.Array,
        w2: jax.Array,
        target: jax.Array,
        lr: jax.Array,
    ) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
        loss, (dx, d_combine, d_w13, d_w2) = jax.value_and_grad(loss_fn, argnums=(0, 2, 3, 4))(
            x,
            selected_experts,
            combine_weights,
            w13_interleaved,
            w2,
            target,
        )
        next_x = (x - lr.astype(x.dtype) * dx).astype(x.dtype)
        next_combine = combine_weights - lr.astype(combine_weights.dtype) * d_combine
        next_w13 = (w13_interleaved - lr.astype(w13_interleaved.dtype) * d_w13).astype(w13_interleaved.dtype)
        next_w2 = (w2 - lr.astype(w2.dtype) * d_w2).astype(w2.dtype)
        return loss, next_x, next_combine, next_w13, next_w2

    return step


def _compare_jax_outputs(raw_value, xla_value) -> tuple[float, float]:
    raw_leaves = jax.tree_util.tree_leaves(raw_value)
    xla_leaves = jax.tree_util.tree_leaves(xla_value)
    max_abs = 0.0
    total_abs = 0.0
    total_size = 0
    for raw_leaf, xla_leaf in zip(raw_leaves, xla_leaves, strict=True):
        raw_np = np.asarray(raw_leaf, dtype=np.float32)
        xla_np = np.asarray(xla_leaf, dtype=np.float32)
        diff = np.abs(raw_np - xla_np)
        max_abs = max(max_abs, float(diff.max()))
        total_abs += float(diff.sum())
        total_size += diff.size
    return max_abs, total_abs / total_size


def _torch_bwd_reference(
    dout: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    perm: torch.Tensor,
    *,
    tokens: int,
    topk: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_ids = torch.arange(tokens, device=dout.device, dtype=torch.int64).repeat_interleave(topk)
    dout_by_assignment = dout[token_ids]
    x_by_assignment = x[perm.to(torch.long)]
    dx = torch.empty_like(x)
    dx[perm.to(torch.long)] = (dout_by_assignment.float() * w[:, None]).to(x.dtype)
    dw = (dout_by_assignment.float() * x_by_assignment.float()).sum(dim=1)
    return dx, dw


def _launch_torch_gather_bwd(
    dout: torch.Tensor,
    x: torch.Tensor,
    w: torch.Tensor,
    perm: torch.Tensor,
    offset: torch.Tensor,
    *,
    tokens: int,
    hidden: int,
    topk: int,
    block_h: int,
    num_warps: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    dx = torch.empty_like(x)
    dw = torch.empty_like(w)
    _sonic_token_gather_sum_bwd_kernel[(tokens * topk,)](
        dout,
        x,
        w,
        perm,
        offset,
        dx,
        dw,
        t=tokens,
        h=hidden,
        max_k=topk,
        stride_doutt=hidden,
        stride_douth=1,
        stride_xm=hidden,
        stride_xh=1,
        block_h=block_h,
        num_warps=num_warps,
        num_stages=4,
    )
    return dx, dw


def _run_torch_raw_fwd_bwd(
    x: torch.Tensor,
    w: torch.Tensor,
    perm: torch.Tensor,
    offset: torch.Tensor,
    dout: torch.Tensor,
    *,
    tokens: int,
    hidden: int,
    topk: int,
    config: KernelConfig,
    bwd_block_h: int,
    bwd_num_warps: int,
    warmup: int,
    repeats: int,
) -> TorchTimingResult:
    dx_ref, dw_ref = _torch_bwd_reference(dout, x, w, perm, tokens=tokens, topk=topk)
    dx, dw = _launch_torch_gather_bwd(
        dout,
        x,
        w,
        perm,
        offset,
        tokens=tokens,
        hidden=hidden,
        topk=topk,
        block_h=bwd_block_h,
        num_warps=bwd_num_warps,
    )
    torch.cuda.synchronize()
    max_abs_dx, mean_abs_dx = _diff_stats(dx, dx_ref)
    max_abs_dw, mean_abs_dw = _diff_stats(dw, dw_ref)

    def fwd_bwd() -> torch.Tensor:
        out = _launch_torch_gather(
            x,
            w,
            perm,
            offset,
            tokens=tokens,
            hidden=hidden,
            topk=topk,
            config=config,
        )
        dx_out, _dw_out = _launch_torch_gather_bwd(
            dout,
            x,
            w,
            perm,
            offset,
            tokens=tokens,
            hidden=hidden,
            topk=topk,
            block_h=bwd_block_h,
            num_warps=bwd_num_warps,
        )
        return out if dx_out is None else dx_out

    event_ms, wall_ms = _time_torch(fwd_bwd, warmup=warmup, repeats=repeats)
    return TorchTimingResult(
        label="direct_triton_raw_gather_fwd_bwd",
        torch_event_ms=event_ms,
        torch_wall_ms=wall_ms,
        max_abs_dx=max_abs_dx,
        mean_abs_dx=mean_abs_dx,
        max_abs_dw=max_abs_dw,
        mean_abs_dw=mean_abs_dw,
    )


def _run_pair(
    *,
    label: str,
    mode: str,
    raw_step: Callable[..., object],
    xla_step: Callable[..., object],
    args: tuple[object, ...],
    warmup: int,
    repeats: int,
    count: int | None = None,
) -> StepTimingResult:
    raw_value = _block_until_ready(raw_step(*args))
    xla_value = _block_until_ready(xla_step(*args))
    max_abs, mean_abs = _compare_jax_outputs(raw_value, xla_value)

    raw_wall = _time_jax_pytree(raw_step, args, warmup=warmup, repeats=repeats)
    xla_wall = _time_jax_pytree(xla_step, args, warmup=warmup, repeats=repeats)
    delta = raw_wall - xla_wall
    return StepTimingResult(
        label=label,
        mode=mode,
        count=count,
        wall_ms=raw_wall,
        xla_baseline_wall_ms=xla_wall,
        delta_vs_xla_ms=delta,
        per_raw_call_delta_ms=delta / count if count is not None else None,
        max_abs_vs_xla=max_abs,
        mean_abs_vs_xla=mean_abs,
    )


def _print_result(result) -> None:
    print(json.dumps(asdict(result), sort_keys=True))


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _make_moe_inputs(
    *,
    tokens: int,
    hidden: int,
    intermediate: int,
    num_experts: int,
    topk: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    token_ids = torch.arange(tokens, device="cuda", dtype=torch.int32)
    selected_experts = torch.stack([(token_ids + expert_offset) % num_experts for expert_offset in range(topk)], dim=1)
    combine_weights = torch.softmax(torch.randn((tokens, topk), device="cuda", dtype=torch.float32), dim=-1)
    scale = 0.02
    x = scale * torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    w13_interleaved = scale * torch.randn((num_experts, hidden, 2 * intermediate), device="cuda", dtype=torch.bfloat16)
    w2 = scale * torch.randn((num_experts, intermediate, hidden), device="cuda", dtype=torch.bfloat16)
    target = scale * torch.randn((tokens, hidden), device="cuda", dtype=torch.bfloat16)
    return x, selected_experts, combine_weights, w13_interleaved, w2, target


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-h", type=int, default=2048)
    parser.add_argument("--block-k", type=int, default=1)
    parser.add_argument("--num-warps", type=int, default=8)
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--bwd-block-h", type=int, default=0)
    parser.add_argument("--bwd-num-warps", type=int, default=8)
    parser.add_argument("--multi-counts", default="1,2,4,8")
    parser.add_argument("--run-moe", action="store_true")
    parser.add_argument("--moe-repeats", type=int, default=0)
    parser.add_argument("--intermediate", type=int, default=3072)
    parser.add_argument("--num-experts", type=int, default=8)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this probe")

    bwd_block_h = args.bwd_block_h or int(triton.next_power_of_2(args.hidden))
    config = KernelConfig(
        block_h=args.block_h,
        block_k=args.block_k,
        num_warps=args.num_warps,
        num_stages=args.num_stages,
    )

    print(
        json.dumps(
            {
                "kind": "versions",
                "torch": torch.__version__,
                "triton": triton.__version__,
                "jax": jax.__version__,
                "jax_triton": getattr(jt, "__version__", "unknown"),
                "device": torch.cuda.get_device_name(),
            },
            sort_keys=True,
        )
    )
    print(
        json.dumps(
            {
                "kind": "shape",
                "tokens": args.tokens,
                "hidden": args.hidden,
                "topk": args.topk,
                "warmup": args.warmup,
                "repeats": args.repeats,
                "config": asdict(config),
                "bwd_block_h": bwd_block_h,
                "bwd_num_warps": args.bwd_num_warps,
            },
            sort_keys=True,
        )
    )

    x_t, w_t, perm_t, offset_t = _make_inputs(
        tokens=args.tokens,
        hidden=args.hidden,
        topk=args.topk,
        seed=args.seed,
    )
    target_t = torch.randn((args.tokens, args.hidden), device="cuda", dtype=torch.bfloat16)
    dout_t = torch.randn((args.tokens, args.hidden), device="cuda", dtype=torch.bfloat16)

    ref_t = _reference(x_t, w_t, perm_t, tokens=args.tokens, topk=args.topk)
    raw_t = _launch_torch_gather(
        x_t,
        w_t,
        perm_t,
        offset_t,
        tokens=args.tokens,
        hidden=args.hidden,
        topk=args.topk,
        config=config,
    )
    torch.cuda.synchronize()
    max_abs, mean_abs = _diff_stats(raw_t, ref_t)
    print(json.dumps({"kind": "forward_direct_correctness", "max_abs": max_abs, "mean_abs": mean_abs}, sort_keys=True))

    torch_result = _run_torch_raw_fwd_bwd(
        x_t,
        w_t,
        perm_t,
        offset_t,
        dout_t,
        tokens=args.tokens,
        hidden=args.hidden,
        topk=args.topk,
        config=config,
        bwd_block_h=bwd_block_h,
        bwd_num_warps=args.bwd_num_warps,
        warmup=args.warmup,
        repeats=args.repeats,
    )
    _print_result(torch_result)

    x_j = _torch_to_jax(x_t)
    w_j = _torch_to_jax(w_t)
    perm_j = _torch_to_jax(perm_t)
    offset_j = _torch_to_jax(offset_t)
    target_j = _torch_to_jax(target_t)
    lr_j = jnp.asarray(1.0e-3, dtype=jnp.float32)

    raw_gather_no_vjp = _make_jax_raw_gather(
        tokens=args.tokens,
        hidden=args.hidden,
        topk=args.topk,
        config=config,
    )
    raw_gather_with_vjp = _make_jax_raw_gather_with_vjp(
        tokens=args.tokens,
        hidden=args.hidden,
        topk=args.topk,
        config=config,
        bwd_block_h=bwd_block_h,
        bwd_num_warps=args.bwd_num_warps,
    )

    forward_raw_step = _make_forward_loss_step(
        raw_gather_no_vjp,
        tokens=args.tokens,
        topk=args.topk,
        use_xla_reference=False,
    )
    forward_xla_step = _make_forward_loss_step(
        raw_gather_no_vjp,
        tokens=args.tokens,
        topk=args.topk,
        use_xla_reference=True,
    )
    _print_result(
        _run_pair(
            label="embedded_raw_sonic_gather_forward_loss",
            mode="forward",
            raw_step=forward_raw_step,
            xla_step=forward_xla_step,
            args=(x_j, w_j, perm_j, offset_j, target_j),
            warmup=args.warmup,
            repeats=args.repeats,
        )
    )

    train_raw_step = _make_train_step(
        raw_gather_with_vjp,
        tokens=args.tokens,
        topk=args.topk,
        use_xla_reference=False,
    )
    train_xla_step = _make_train_step(
        raw_gather_with_vjp,
        tokens=args.tokens,
        topk=args.topk,
        use_xla_reference=True,
    )
    _print_result(
        _run_pair(
            label="embedded_raw_sonic_gather_train_step",
            mode="fwd_bwd_update",
            raw_step=train_raw_step,
            xla_step=train_xla_step,
            args=(x_j, w_j, perm_j, offset_j, target_j, lr_j),
            warmup=args.warmup,
            repeats=args.repeats,
        )
    )

    max_multi_count = max(_parse_csv_ints(args.multi_counts), default=1)
    ws_t = torch.stack([w_t + 0.0001 * float(i) for i in range(max_multi_count)])
    ws_j = _torch_to_jax(ws_t)
    for count in _parse_csv_ints(args.multi_counts):
        multi_raw_step = _make_multi_forward_loss_step(
            raw_gather_no_vjp,
            tokens=args.tokens,
            hidden=args.hidden,
            topk=args.topk,
            count=count,
            use_xla_reference=False,
        )
        multi_xla_step = _make_multi_forward_loss_step(
            raw_gather_no_vjp,
            tokens=args.tokens,
            hidden=args.hidden,
            topk=args.topk,
            count=count,
            use_xla_reference=True,
        )
        _print_result(
            _run_pair(
                label="embedded_multi_raw_sonic_gather_forward_loss",
                mode="multi_forward",
                count=count,
                raw_step=multi_raw_step,
                xla_step=multi_xla_step,
                args=(x_j, ws_j, perm_j, offset_j, target_j),
                warmup=args.warmup,
                repeats=args.repeats,
            )
        )

    if not args.run_moe:
        return

    moe_repeats = args.moe_repeats or args.repeats
    moe_x_t, selected_t, combine_t, w13_t, w2_t, moe_target_t = _make_moe_inputs(
        tokens=args.tokens,
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_experts=args.num_experts,
        topk=args.topk,
        seed=args.seed + 1,
    )
    moe_x_j = _torch_to_jax(moe_x_t)
    selected_j = _torch_to_jax(selected_t)
    combine_j = _torch_to_jax(combine_t)
    w13_j = _torch_to_jax(w13_t)
    w2_j = _torch_to_jax(w2_t)
    moe_target_j = _torch_to_jax(moe_target_t)

    moe_raw_forward_step = _make_moe_forward_loss_step(raw_gather_no_vjp, use_xla_reference=False)
    moe_xla_forward_step = _make_moe_forward_loss_step(raw_gather_no_vjp, use_xla_reference=True)
    _print_result(
        _run_pair(
            label="full_moe_raw_sonic_gather_forward_loss",
            mode="moe_forward",
            raw_step=moe_raw_forward_step,
            xla_step=moe_xla_forward_step,
            args=(moe_x_j, selected_j, combine_j, w13_j, w2_j, moe_target_j),
            warmup=args.warmup,
            repeats=moe_repeats,
        )
    )

    moe_raw_train_step = _make_moe_train_step(raw_gather_with_vjp, use_xla_reference=False)
    moe_xla_train_step = _make_moe_train_step(raw_gather_with_vjp, use_xla_reference=True)
    _print_result(
        _run_pair(
            label="full_moe_raw_sonic_gather_train_step",
            mode="moe_fwd_bwd_update",
            raw_step=moe_raw_train_step,
            xla_step=moe_xla_train_step,
            args=(moe_x_j, selected_j, combine_j, w13_j, w2_j, moe_target_j, lr_j),
            warmup=args.warmup,
            repeats=moe_repeats,
        )
    )


if __name__ == "__main__":
    main()
