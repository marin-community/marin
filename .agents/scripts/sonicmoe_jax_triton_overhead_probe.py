# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Measure jax-triton overhead for SonicMoE's token gather/sum kernel.

This is a research probe, not production code. It clones the SonicMoE
`token_gather_sum_kernel` body at commit cfbd65f and launches the same Triton
kernel body from both PyTorch/Triton and jax-triton.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import numpy as np
import torch
import triton
import triton.language as tl

try:
    import jax
    import jax.numpy as jnp
    import jax_triton as jt
except ModuleNotFoundError as exc:
    raise SystemExit(
        "Missing jax-triton/JAX dependency. Run with: "
        "uv run --package marin --extra gpu --group dev --with jax-triton==0.3.1 python "
        ".agents/scripts/sonicmoe_jax_triton_overhead_probe.py"
    ) from exc

if not os.environ.get("TRITON_CACHE_DIR"):
    os.environ["TRITON_CACHE_DIR"] = "/tmp/triton-cache"
os.makedirs(os.environ["TRITON_CACHE_DIR"], exist_ok=True)


@triton.jit
def _sonic_token_gather_sum_kernel(
    x_ptr,  # (Mtotal, H)
    w_ptr,  # (Mtotal,)
    m_perm_ptr,  # (Mtotal,) int32
    m_offset_ptr,  # (T+1,) int32
    out_ptr,  # (T, H)
    t: tl.constexpr,
    h: tl.constexpr,
    max_k: tl.constexpr,
    stride_xm: tl.constexpr,
    stride_xh: tl.constexpr,
    stride_outt: tl.constexpr,
    stride_outh: tl.constexpr,
    block_h: tl.constexpr,
    block_k: tl.constexpr,
    w_is_none: tl.constexpr,
    is_varlen_k: tl.constexpr,
):
    pid_t = tl.program_id(axis=0)
    t_idx = pid_t.to(tl.int64)

    if is_varlen_k:
        ms = tl.load(m_offset_ptr + t_idx).to(tl.int64)
        me = tl.load(m_offset_ptr + t_idx + 1).to(tl.int64)
        k_this_token = me - ms
    else:
        ms = max_k * t_idx
        k_this_token: tl.constexpr = max_k

    for h_tile in tl.static_range(triton.cdiv(h, block_h)):
        h_idx = (h_tile * block_h + tl.arange(0, block_h)).to(tl.int64)
        h_mask = h_idx < h
        acc = tl.zeros([block_h], dtype=tl.float32)

        for k_tile in tl.range(tl.cdiv(k_this_token, block_k)):
            k_offset = k_tile * block_k
            k_idx = (k_offset + tl.arange(0, block_k)).to(tl.int64)
            k_mask = k_idx < k_this_token
            m_abs = ms + k_idx
            perm_idx = tl.load(m_perm_ptr + m_abs, mask=k_mask, other=0).to(tl.int64)

            x_ptrs = x_ptr + perm_idx[:, None] * stride_xm + h_idx[None, :] * stride_xh
            x_mask = k_mask[:, None] & h_mask[None, :]
            x_vals = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

            if w_is_none:
                acc += tl.sum(x_vals, axis=0)
            else:
                w_vals = tl.load(w_ptr + m_abs, mask=k_mask, other=0.0).to(tl.float32)
                acc += tl.sum(x_vals * w_vals[:, None], axis=0)

        out_ptrs = out_ptr + t_idx * stride_outt + h_idx * stride_outh
        tl.store(out_ptrs, acc, mask=h_mask)


@triton.jit
def _tiny_store_kernel(out_ptr):
    tl.store(out_ptr + 0, tl.full((), 1, dtype=tl.int32))


@triton.jit
def _increment_kernel(x_ptr, out_ptr):
    value = tl.load(x_ptr + 0).to(tl.int32)
    tl.store(out_ptr + 0, value + 1)


@dataclass(frozen=True)
class KernelConfig:
    block_h: int
    block_k: int
    num_warps: int
    num_stages: int


@dataclass(frozen=True)
class TimingResult:
    label: str
    block_h: int | None
    block_k: int | None
    num_warps: int | None
    torch_event_ms: float | None
    torch_wall_ms: float | None
    jax_wall_ms: float | None
    jax_minus_torch_wall_ms: float | None
    jax_over_torch_wall: float | None
    max_abs_vs_ref: float | None
    mean_abs_vs_ref: float | None
    chain_length: int | None = None


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def _torch_to_jax(tensor: torch.Tensor) -> jax.Array:
    return jax.dlpack.from_dlpack(tensor.contiguous())


def _make_jax_gather(
    *,
    tokens: int,
    hidden: int,
    topk: int,
    config: KernelConfig,
) -> Callable[[jax.Array, jax.Array, jax.Array, jax.Array], jax.Array]:
    out_shape = jax.ShapeDtypeStruct((tokens, hidden), jnp.bfloat16)

    @jax.jit
    def gather(x: jax.Array, w: jax.Array, perm: jax.Array, offset: jax.Array) -> jax.Array:
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

    return gather


def _make_jax_tiny_store() -> Callable[[], jax.Array]:
    out_shape = jax.ShapeDtypeStruct((1,), jnp.int32)

    @jax.jit
    def tiny_store() -> jax.Array:
        return jt.triton_call(
            kernel=_tiny_store_kernel,
            out_shape=out_shape,
            grid=(1,),
            num_warps=1,
            num_stages=1,
        )

    return tiny_store


def _make_jax_increment_chain(chain_length: int) -> Callable[[jax.Array], jax.Array]:
    out_shape = jax.ShapeDtypeStruct((1,), jnp.int32)

    @jax.jit
    def increment_chain(x: jax.Array) -> jax.Array:
        y = x
        for _ in range(chain_length):
            y = jt.triton_call(
                y,
                kernel=_increment_kernel,
                out_shape=out_shape,
                grid=(1,),
                num_warps=1,
                num_stages=1,
            )
        return y

    return increment_chain


def _launch_torch_gather(
    x: torch.Tensor,
    w: torch.Tensor,
    perm: torch.Tensor,
    offset: torch.Tensor,
    *,
    tokens: int,
    hidden: int,
    topk: int,
    config: KernelConfig,
) -> torch.Tensor:
    out = torch.empty((tokens, hidden), device=x.device, dtype=x.dtype)
    _sonic_token_gather_sum_kernel[(tokens,)](
        x,
        w,
        perm,
        offset,
        out,
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
        num_warps=config.num_warps,
        num_stages=config.num_stages,
    )
    return out


def _launch_torch_tiny_store(out: torch.Tensor) -> torch.Tensor:
    _tiny_store_kernel[(1,)](out, num_warps=1, num_stages=1)
    return out


def _launch_torch_increment_chain(buffers: tuple[torch.Tensor, torch.Tensor], chain_length: int) -> torch.Tensor:
    src, dst = buffers
    for _ in range(chain_length):
        _increment_kernel[(1,)](src, dst, num_warps=1, num_stages=1)
        src, dst = dst, src
    return src


def _time_torch(
    fn: Callable[[], torch.Tensor],
    *,
    warmup: int,
    repeats: int,
) -> tuple[float, float]:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)
    start_wall = time.perf_counter()
    start_event.record()
    for _ in range(repeats):
        fn()
    end_event.record()
    torch.cuda.synchronize()
    end_wall = time.perf_counter()
    return start_event.elapsed_time(end_event) / repeats, (end_wall - start_wall) * 1000.0 / repeats


def _time_jax(
    fn: Callable[..., jax.Array],
    args: tuple[jax.Array, ...],
    *,
    warmup: int,
    repeats: int,
) -> float:
    for _ in range(warmup):
        fn(*args).block_until_ready()

    start = time.perf_counter()
    for _ in range(repeats):
        fn(*args).block_until_ready()
    end = time.perf_counter()
    return (end - start) * 1000.0 / repeats


def _make_inputs(
    *,
    tokens: int,
    hidden: int,
    topk: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    torch.manual_seed(seed)
    assignments = tokens * topk
    x = torch.randn((assignments, hidden), device="cuda", dtype=torch.bfloat16)
    w = torch.randn((assignments,), device="cuda", dtype=torch.float32)
    perm = torch.randperm(assignments, device="cuda", dtype=torch.int32)
    offset = torch.arange(0, assignments + 1, topk, device="cuda", dtype=torch.int32)
    return x, w, perm, offset


def _reference(
    x: torch.Tensor,
    w: torch.Tensor,
    perm: torch.Tensor,
    *,
    tokens: int,
    topk: int,
) -> torch.Tensor:
    gathered = x[perm.reshape(tokens, topk).to(torch.long)].float()
    weighted = gathered * w.reshape(tokens, topk, 1)
    return weighted.sum(dim=1).to(x.dtype)


def _diff_stats(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    diff = (actual.float() - expected.float()).abs()
    return float(diff.max().item()), float(diff.mean().item())


def _run_gather_config(
    x_t: torch.Tensor,
    w_t: torch.Tensor,
    perm_t: torch.Tensor,
    offset_t: torch.Tensor,
    ref_t: torch.Tensor,
    *,
    tokens: int,
    hidden: int,
    topk: int,
    config: KernelConfig,
    warmup: int,
    repeats: int,
) -> TimingResult:
    x_j = _torch_to_jax(x_t)
    w_j = _torch_to_jax(w_t)
    perm_j = _torch_to_jax(perm_t)
    offset_j = _torch_to_jax(offset_t)

    jax_gather = _make_jax_gather(tokens=tokens, hidden=hidden, topk=topk, config=config)

    def torch_fn() -> torch.Tensor:
        return _launch_torch_gather(
            x_t,
            w_t,
            perm_t,
            offset_t,
            tokens=tokens,
            hidden=hidden,
            topk=topk,
            config=config,
        )

    torch_out = torch_fn()
    torch.cuda.synchronize()
    torch_max_abs, torch_mean_abs = _diff_stats(torch_out, ref_t)
    if torch_max_abs != 0.0:
        print(
            f"WARNING torch kernel differs from reference for {config}: "
            f"max_abs={torch_max_abs} mean_abs={torch_mean_abs}",
            file=sys.stderr,
        )

    jax_out = jax_gather(x_j, w_j, perm_j, offset_j).block_until_ready()
    jax_out_t = torch.utils.dlpack.from_dlpack(jax_out)
    jax_max_abs, jax_mean_abs = _diff_stats(jax_out_t, ref_t)

    torch_event_ms, torch_wall_ms = _time_torch(torch_fn, warmup=warmup, repeats=repeats)
    jax_wall_ms = _time_jax(jax_gather, (x_j, w_j, perm_j, offset_j), warmup=warmup, repeats=repeats)
    return TimingResult(
        label="sonic_token_gather_sum_fixed",
        block_h=config.block_h,
        block_k=config.block_k,
        num_warps=config.num_warps,
        torch_event_ms=torch_event_ms,
        torch_wall_ms=torch_wall_ms,
        jax_wall_ms=jax_wall_ms,
        jax_minus_torch_wall_ms=jax_wall_ms - torch_wall_ms,
        jax_over_torch_wall=jax_wall_ms / torch_wall_ms,
        max_abs_vs_ref=jax_max_abs,
        mean_abs_vs_ref=jax_mean_abs,
    )


def _run_tiny_store(*, warmup: int, repeats: int) -> TimingResult:
    torch_out = torch.empty((1,), device="cuda", dtype=torch.int32)
    jax_tiny_store = _make_jax_tiny_store()
    jax_tiny_store().block_until_ready()
    torch_event_ms, torch_wall_ms = _time_torch(
        lambda: _launch_torch_tiny_store(torch_out),
        warmup=warmup,
        repeats=repeats,
    )
    jax_wall_ms = _time_jax(jax_tiny_store, (), warmup=warmup, repeats=repeats)
    return TimingResult(
        label="tiny_triton_store",
        block_h=None,
        block_k=None,
        num_warps=1,
        torch_event_ms=torch_event_ms,
        torch_wall_ms=torch_wall_ms,
        jax_wall_ms=jax_wall_ms,
        jax_minus_torch_wall_ms=jax_wall_ms - torch_wall_ms,
        jax_over_torch_wall=jax_wall_ms / torch_wall_ms,
        max_abs_vs_ref=None,
        mean_abs_vs_ref=None,
        chain_length=1,
    )


def _run_increment_chain(
    *,
    chain_length: int,
    warmup: int,
    repeats: int,
) -> TimingResult:
    torch_buffers = (
        torch.zeros((1,), device="cuda", dtype=torch.int32),
        torch.empty((1,), device="cuda", dtype=torch.int32),
    )
    jax_input = jnp.zeros((1,), dtype=jnp.int32)
    jax_chain = _make_jax_increment_chain(chain_length)

    jax_out = jax_chain(jax_input).block_until_ready()
    expected = chain_length
    actual = int(np.asarray(jax_out)[0])
    if actual != expected:
        raise AssertionError(f"jax increment chain expected {expected}, got {actual}")

    torch_out = _launch_torch_increment_chain(torch_buffers, chain_length)
    torch.cuda.synchronize()
    actual_torch = int(torch_out.cpu().item())
    if actual_torch != expected:
        raise AssertionError(f"torch increment chain expected {expected}, got {actual_torch}")

    torch_event_ms, torch_wall_ms = _time_torch(
        lambda: _launch_torch_increment_chain(torch_buffers, chain_length),
        warmup=warmup,
        repeats=repeats,
    )
    jax_wall_ms = _time_jax(jax_chain, (jax_input,), warmup=warmup, repeats=repeats)
    return TimingResult(
        label="increment_chain",
        block_h=None,
        block_k=None,
        num_warps=1,
        torch_event_ms=torch_event_ms,
        torch_wall_ms=torch_wall_ms,
        jax_wall_ms=jax_wall_ms,
        jax_minus_torch_wall_ms=jax_wall_ms - torch_wall_ms,
        jax_over_torch_wall=jax_wall_ms / torch_wall_ms,
        max_abs_vs_ref=None,
        mean_abs_vs_ref=None,
        chain_length=chain_length,
    )


def _print_result(result: TimingResult) -> None:
    print(json.dumps(asdict(result), sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=8192)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--topk", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--block-h", default="512,1024,2048")
    parser.add_argument("--block-k", default="1,2")
    parser.add_argument("--num-warps", default="4,8")
    parser.add_argument("--num-stages", type=int, default=4)
    parser.add_argument("--chain-lengths", default="")
    parser.add_argument("--skip-gather", action="store_true")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required for this probe")

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
            },
            sort_keys=True,
        )
    )

    tiny_result = _run_tiny_store(warmup=args.warmup, repeats=args.repeats)
    _print_result(tiny_result)

    for chain_length in _parse_csv_ints(args.chain_lengths):
        chain_result = _run_increment_chain(chain_length=chain_length, warmup=args.warmup, repeats=args.repeats)
        _print_result(chain_result)

    if args.skip_gather:
        return

    x_t, w_t, perm_t, offset_t = _make_inputs(
        tokens=args.tokens,
        hidden=args.hidden,
        topk=args.topk,
        seed=args.seed,
    )
    ref_t = _reference(x_t, w_t, perm_t, tokens=args.tokens, topk=args.topk)
    torch.cuda.synchronize()

    for block_h in _parse_csv_ints(args.block_h):
        if block_h > triton.next_power_of_2(args.hidden):
            continue
        for block_k in _parse_csv_ints(args.block_k):
            if block_k > triton.next_power_of_2(args.topk):
                continue
            if block_h * block_k > 32768:
                continue
            if min(args.hidden * args.topk, 1024) > block_h * block_k:
                continue
            for num_warps in _parse_csv_ints(args.num_warps):
                config = KernelConfig(
                    block_h=block_h,
                    block_k=block_k,
                    num_warps=num_warps,
                    num_stages=args.num_stages,
                )
                result = _run_gather_config(
                    x_t,
                    w_t,
                    perm_t,
                    offset_t,
                    ref_t,
                    tokens=args.tokens,
                    hidden=args.hidden,
                    topk=args.topk,
                    config=config,
                    warmup=args.warmup,
                    repeats=args.repeats,
                )
                _print_result(result)


if __name__ == "__main__":
    main()
