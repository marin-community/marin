# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
import os
import subprocess
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass

import jax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np

from haliax.nn import ragged_dot
from haliax.quantization import Fp8RaggedDotOp


@dataclass(frozen=True)
class Shape:
    experts: int
    tokens_per_expert: int
    k: int
    n: int
    name: str

    @property
    def tokens(self) -> int:
        return self.experts * self.tokens_per_expert

    @property
    def fwd_flops(self) -> int:
        return 2 * self.experts * self.tokens_per_expert * self.k * self.n

    @property
    def fwd_bwd_flops(self) -> int:
        return 3 * self.fwd_flops


def _block_until_ready(tree):
    jax.tree_util.tree_map(lambda x: x.block_until_ready() if hasattr(x, "block_until_ready") else x, tree)
    return tree


def _time_jit(fn: Callable, *args, warmups: int, iters: int) -> tuple[float, float, object]:
    compiled = jax.jit(fn)
    start = time.perf_counter()
    out = _block_until_ready(compiled(*args))
    compile_time = time.perf_counter() - start

    for _ in range(warmups):
        out = _block_until_ready(compiled(*args))

    start = time.perf_counter()
    for _ in range(iters):
        out = _block_until_ready(compiled(*args))
    steady = (time.perf_counter() - start) / iters
    return compile_time, steady, out


def _make_inputs(shape: Shape, dtype):
    lhs_key, rhs_key, cot_key = jrandom.split(jrandom.PRNGKey(shape.experts * 100_000 + shape.tokens_per_expert), 3)
    lhs = jrandom.normal(lhs_key, (shape.tokens, shape.k), dtype=dtype)
    rhs = jrandom.normal(rhs_key, (shape.experts, shape.k, shape.n), dtype=dtype) / jnp.sqrt(shape.k).astype(dtype)
    cot = jrandom.normal(cot_key, (shape.tokens, shape.n), dtype=dtype)
    group_sizes = jnp.asarray(_nonuniform_group_sizes(shape), dtype=jnp.int32)
    return lhs, rhs, cot, group_sizes


def _nonuniform_group_sizes(shape: Shape) -> np.ndarray:
    """Return deterministic capacity-limited MoE-like counts summing to the fixed total."""
    rng = np.random.default_rng(shape.experts * 10_000 + shape.tokens_per_expert)
    noise = rng.normal(loc=0.0, scale=0.18, size=shape.experts)
    counts = np.rint(shape.tokens_per_expert * (1.0 + noise)).astype(np.int32)
    low = max(1, int(round(shape.tokens_per_expert * 0.65)))
    high = max(low + 1, int(round(shape.tokens_per_expert * 1.35)))
    counts = np.clip(counts, low, high)
    delta = int(shape.tokens - np.sum(counts))
    while delta != 0:
        if delta > 0:
            candidates = np.flatnonzero(counts < high)
            step = min(delta, candidates.size)
            order = candidates[np.argsort(counts[candidates])[:step]]
            counts[order] += 1
            delta -= step
        else:
            candidates = np.flatnonzero(counts > low)
            step = min(-delta, candidates.size)
            order = candidates[np.argsort(counts[candidates])[::-1][:step]]
            counts[order] -= 1
            delta += step
    assert np.sum(counts) == shape.tokens
    assert np.max(counts) > shape.tokens_per_expert
    assert np.min(counts) < shape.tokens_per_expert
    return counts


def _relative_frobenius(got, want) -> float:
    got = got.astype(jnp.float32)
    want = want.astype(jnp.float32)
    return float((jnp.linalg.norm(got - want) / jnp.linalg.norm(want)).block_until_ready())


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True).strip()
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None


def _bench_shape(
    shape: Shape,
    *,
    warmups: int,
    iters: int,
    amax_history_length: int,
    fp8_block_m: int | None,
    fp8_block_n: int | None,
    fp8_block_k: int | None,
    fp8_implementation: str,
    mode: str,
) -> list[dict]:
    lhs, rhs, cot, group_sizes = _make_inputs(shape, jnp.bfloat16)
    max_group_size = int(jnp.max(group_sizes))
    min_group_size = int(jnp.min(group_sizes))
    fp8_op = Fp8RaggedDotOp.init(
        amax_history_length=amax_history_length,
        block_m=fp8_block_m,
        block_n=fp8_block_n,
        block_k=fp8_block_k,
    )

    def bf16_fwd(lhs, rhs):
        return ragged_dot(lhs, rhs, group_sizes, implementation="triton")

    def fp8_fwd(lhs, rhs):
        return ragged_dot(
            lhs,
            rhs,
            group_sizes,
            implementation=fp8_implementation,
            fp8_dot=fp8_op,
            max_group_size=max_group_size,
        )

    def bf16_fwd_bwd(lhs, rhs, cot):
        def loss(lhs, rhs):
            out = ragged_dot(lhs, rhs, group_sizes, implementation="triton")
            return jnp.sum(out.astype(jnp.float32) * cot.astype(jnp.float32))

        return jax.value_and_grad(loss, argnums=(0, 1))(lhs, rhs)

    def fp8_fwd_bwd(lhs, rhs, cot):
        def loss(lhs, rhs):
            out = ragged_dot(
                lhs,
                rhs,
                group_sizes,
                implementation=fp8_implementation,
                fp8_dot=fp8_op,
                max_group_size=max_group_size,
            )
            return jnp.sum(out.astype(jnp.float32) * cot.astype(jnp.float32))

        return jax.value_and_grad(loss, argnums=(0, 1))(lhs, rhs)

    fwd_rel_error = None
    timings = []
    if mode in ("all", "fwd"):
        bf16_compile_fwd, bf16_fwd_time, bf16_out = _time_jit(bf16_fwd, lhs, rhs, warmups=warmups, iters=iters)
        fp8_compile_fwd, fp8_fwd_time, fp8_out = _time_jit(fp8_fwd, lhs, rhs, warmups=warmups, iters=iters)
        fwd_rel_error = _relative_frobenius(fp8_out, bf16_out)
        timings.extend(
            [
                ("fwd", "bf16_triton", bf16_compile_fwd, bf16_fwd_time, shape.fwd_flops),
                ("fwd", f"fp8_{fp8_implementation}", fp8_compile_fwd, fp8_fwd_time, shape.fwd_flops),
            ]
        )

    if mode in ("all", "fwd_bwd"):
        bf16_compile_fb, bf16_fb_time, _ = _time_jit(bf16_fwd_bwd, lhs, rhs, cot, warmups=warmups, iters=iters)
        fp8_compile_fb, fp8_fb_time, _ = _time_jit(fp8_fwd_bwd, lhs, rhs, cot, warmups=warmups, iters=iters)
        timings.extend(
            [
                ("fwd_bwd", "bf16_triton", bf16_compile_fb, bf16_fb_time, shape.fwd_bwd_flops),
                ("fwd_bwd", f"fp8_{fp8_implementation}", fp8_compile_fb, fp8_fb_time, shape.fwd_bwd_flops),
            ]
        )

    base = {
        "kernel": "ragged_dot",
        "shape": asdict(shape),
        "dtype": "bf16",
        "backend": jax.default_backend(),
        "device_type": jax.devices()[0].device_kind,
        "device_count": jax.device_count(),
        "block_sizes": {
            "bf16_triton": {"block_m": 128, "block_n": 128, "block_k": 32},
            f"fp8_{fp8_implementation}": {
                "block_m": fp8_block_m or 128,
                "block_n": fp8_block_n or 128,
                "block_k": fp8_block_k or (64 if fp8_implementation in ("mosaic", "triton_native") else 32),
                "max_group_size": max_group_size,
            },
        },
        "group_sizes": {
            "distribution": "bounded_normal_sigma_0.18_capacity_1.35",
            "min": min_group_size,
            "max": max_group_size,
            "sum": shape.tokens,
            "average": shape.tokens_per_expert,
        },
        "git_sha": _git_sha(),
        "xla_flags": os.environ.get("XLA_FLAGS", ""),
        "backend_env": "h100 helper",
        "error": None,
    }

    rows = []
    for row_mode, implementation, compile_time, steady_state_time, flops in timings:
        rows.append(
            {
                **base,
                "mode": row_mode,
                "implementation": implementation,
                "compile_time": compile_time,
                "steady_state_time": steady_state_time,
                "tflops": flops / steady_state_time / 1e12,
                "forward_relative_frobenius_error": fwd_rel_error,
            }
        )

    speeds = {}
    for row_mode in ("fwd", "fwd_bwd"):
        mode_rows = [row for row in rows if row["mode"] == row_mode]
        if len(mode_rows) == 2:
            bf16 = next(row for row in mode_rows if row["implementation"] == "bf16_triton")
            fp8 = next(row for row in mode_rows if row["implementation"].startswith("fp8_"))
            speeds[row_mode] = bf16["steady_state_time"] / fp8["steady_state_time"]
    print(
        f"{shape.name} E={shape.experts} tpe={shape.tokens_per_expert} K={shape.k} N={shape.n}: "
        f"group[min={min_group_size}, max={max_group_size}] "
        f"fwd speedup={speeds.get('fwd', float('nan')):.3f} "
        f"fwd+bwd speedup={speeds.get('fwd_bwd', float('nan')):.3f} "
        f"fp8 fwd err={fwd_rel_error if fwd_rel_error is not None else float('nan'):.4f}"
    )
    for row in rows:
        print(json.dumps(row, sort_keys=True))
    return rows


def _parse_csv_ints(value: str) -> list[int]:
    return [int(part) for part in value.split(",") if part]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens-per-expert", default="512,1024,2048,4096")
    parser.add_argument("--experts", default="16,32,64")
    parser.add_argument("--warmups", type=int, default=2)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--amax-history-length", type=int, default=1024)
    parser.add_argument("--fp8-block-m", type=int)
    parser.add_argument("--fp8-block-n", type=int)
    parser.add_argument("--fp8-block-k", type=int)
    parser.add_argument(
        "--fp8-implementation",
        default="mosaic",
        choices=("mosaic", "padded_dense", "triton", "triton_native", "xla"),
    )
    parser.add_argument("--mode", default="all", choices=("all", "fwd", "fwd_bwd"))
    parser.add_argument("--quick", action="store_true", help="Run only the operating point E=64,tpe=1024.")
    args = parser.parse_args()

    tokens_per_expert = [1024] if args.quick else _parse_csv_ints(args.tokens_per_expert)
    experts = [64] if args.quick else _parse_csv_ints(args.experts)

    print(f"devices={jax.devices()}")
    print("shape set includes w13 K=N=2560 and w2 K=1280,N=2560")

    all_rows = []
    for experts_count in experts:
        for tpe in tokens_per_expert:
            for name, k, n in (("w13", 2560, 2560), ("w2", 1280, 2560)):
                all_rows.extend(
                    _bench_shape(
                        Shape(experts=experts_count, tokens_per_expert=tpe, k=k, n=n, name=name),
                        warmups=args.warmups,
                        iters=args.iters,
                        amax_history_length=args.amax_history_length,
                        fp8_block_m=args.fp8_block_m,
                        fp8_block_n=args.fp8_block_n,
                        fp8_block_k=args.fp8_block_k,
                        fp8_implementation=args.fp8_implementation,
                        mode=args.mode,
                    )
                )

    target_mode = "fwd" if args.mode == "fwd" else "fwd_bwd"
    target = [
        row
        for row in all_rows
        if row["shape"]["experts"] == 64
        and row["shape"]["tokens_per_expert"] == 1024
        and row["mode"] == target_mode
    ]
    for name in ("w13", "w2"):
        bf16 = next(row for row in target if row["shape"]["name"] == name and row["implementation"] == "bf16_triton")
        fp8 = next(row for row in target if row["shape"]["name"] == name and row["implementation"].startswith("fp8_"))
        print(f"TARGET {name} {target_mode} speedup={bf16['steady_state_time'] / fp8['steady_state_time']:.3f}")


if __name__ == "__main__":
    main()
