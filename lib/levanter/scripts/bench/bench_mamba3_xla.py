# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from levanter.kernels.pallas.mamba3 import (
    intra_chunk_log_alpha_cumsum,
    local_log_alpha,
    mamba3_attentionish_forward,
    mamba3_chunked_forward,
)


ApiVariant = Literal["chunked_public", "attentionish_public"]


@dataclass(frozen=True, slots=True)
class Shape:
    seq_len: int
    batch_head_groups: int
    chunk_size: int
    state_dim: int
    value_dim: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    api: ApiVariant
    seq_len: int
    batch_head_groups: int
    chunk_size: int
    num_chunks: int
    groups: int
    state_dim: int
    value_dim: int
    dtype: str
    sharded: bool
    mode: str
    compile_s: float
    steady_s: float
    tokens_per_s: float
    estimated_tflops: float


DEFAULT_SHAPES = (
    Shape(seq_len=2048, batch_head_groups=16, chunk_size=128, state_dim=128, value_dim=512),
    Shape(seq_len=8192, batch_head_groups=16, chunk_size=128, state_dim=128, value_dim=512),
    Shape(seq_len=16384, batch_head_groups=16, chunk_size=128, state_dim=128, value_dim=512),
)


def _dtype_from_name(name: str) -> jnp.dtype:
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float32":
        return jnp.float32
    raise ValueError(f"Unsupported dtype: {name}.")


def _build_inputs(
    shape: Shape, dtype: jnp.dtype
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    key = jax.random.PRNGKey(0)
    k_dt, k_lam, k_a, k_b, k_c, k_x = jax.random.split(key, 6)
    num_chunks = math.ceil(shape.seq_len / shape.chunk_size)
    leading_shape = (shape.batch_head_groups,)
    token_shape = leading_shape + (num_chunks, shape.chunk_size)
    dt = (0.01 + 0.1 * jax.random.uniform(k_dt, token_shape, dtype=jnp.float32)).astype(dtype)
    lam = jax.random.uniform(k_lam, token_shape, dtype=jnp.float32).astype(dtype)
    a = (-0.5 - jax.random.uniform(k_a, token_shape, dtype=jnp.float32)).astype(dtype)
    b = jax.random.normal(k_b, token_shape + (shape.state_dim,), dtype=dtype)
    c = jax.random.normal(k_c, token_shape + (shape.state_dim,), dtype=dtype)
    x = jax.random.normal(k_x, token_shape + (shape.value_dim,), dtype=dtype)
    return dt, lam, a, b, c, x


def _estimate_forward_flops(shape: Shape) -> float:
    num_chunks = math.ceil(shape.seq_len / shape.chunk_size)
    groups = shape.batch_head_groups
    local_cb = 2.0 * groups * num_chunks * shape.chunk_size * shape.chunk_size * shape.state_dim
    local_emit = 2.0 * groups * num_chunks * shape.chunk_size * shape.chunk_size * shape.value_dim
    chunk_state = 2.0 * groups * num_chunks * shape.chunk_size * shape.state_dim * shape.value_dim
    prefix_emit = 2.0 * groups * num_chunks * shape.chunk_size * shape.state_dim * shape.value_dim
    return local_cb + local_emit + chunk_state + prefix_emit


def _shard_spec(ndim: int, *, shard_groups: bool) -> P:
    return P("data", *([None] * (ndim - 1))) if shard_groups else P(*([None] * ndim))


def _maybe_shard_inputs(
    inputs: tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array], *, shard_groups: bool
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    if not shard_groups:
        return inputs
    mesh = Mesh(np.array(jax.devices()), ("data",))

    def put(x: jax.Array) -> jax.Array:
        return jax.device_put(x, NamedSharding(mesh, _shard_spec(x.ndim, shard_groups=True)))

    return tuple(put(x) for x in inputs)


def _attentionish_inputs(
    dt: jax.Array,
    lam: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    x: jax.Array,
) -> tuple[jax.Array, ...]:
    groups, num_chunks, chunk_size = dt.shape
    seq_len = num_chunks * chunk_size
    state_dim = b.shape[-1]
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    eps = jnp.finfo(lam.dtype).eps
    clipped_lam = jnp.clip(lam, eps, 1 - eps)
    trap = (jnp.log(clipped_lam) - jnp.log1p(-clipped_lam)).reshape(groups, 1, seq_len)
    q = c.reshape(groups, seq_len, 1, state_dim)
    k = b.reshape(groups, seq_len, 1, state_dim)
    v = x.reshape(groups, seq_len, 1, x.shape[-1])
    q_bias = jnp.zeros((1, state_dim), dtype=dt.dtype)
    k_bias = jnp.zeros((1, state_dim), dtype=dt.dtype)
    da_cs = a_log_cumsum.reshape(groups, 1, seq_len)
    dt_seq = dt.reshape(groups, 1, seq_len)
    return q, k, v, q_bias, k_bias, da_cs, dt_seq, trap


def _time_jitted(fn, *args, steps: int, warmup: int) -> tuple[float, float]:
    start = time.perf_counter()
    out = fn(*args)
    jax.block_until_ready(out)
    compile_s = time.perf_counter() - start

    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)

    start = time.perf_counter()
    for _ in range(steps):
        out = fn(*args)
        jax.block_until_ready(out)
    steady_s = (time.perf_counter() - start) / steps
    return compile_s, steady_s


def _capture_profile(
    fn,
    *args,
    profile_dir: str,
    profile_steps: int,
) -> None:
    output_dir = Path(profile_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(output_dir), create_perfetto_trace=True)
    try:
        for _ in range(profile_steps):
            out = fn(*args)
            jax.block_until_ready(out)
    finally:
        jax.profiler.stop_trace()


def _benchmark(
    shape: Shape,
    *,
    dtype: jnp.dtype,
    api: ApiVariant,
    shard_groups: bool,
    steps: int,
    warmup: int,
    profile_dir: str | None,
    profile_mode: str,
    profile_steps: int,
) -> list[BenchmarkResult]:
    inputs = _maybe_shard_inputs(_build_inputs(shape, dtype), shard_groups=shard_groups)
    num_chunks = math.ceil(shape.seq_len / shape.chunk_size)
    groups = shape.batch_head_groups
    tokens = groups * shape.seq_len
    flops = _estimate_forward_flops(shape)

    if api == "chunked_public":
        forward_fn = jax.jit(lambda *xs: mamba3_chunked_forward(*xs, implementation="xla")[0])
        backward_fn = jax.jit(
            jax.grad(
                lambda *xs: jnp.sum(mamba3_chunked_forward(*xs, implementation="xla")[0].astype(jnp.float32) ** 2),
                argnums=5,
            )
        )
        forward_args = inputs
        backward_args = inputs
    else:
        attentionish_inputs = _attentionish_inputs(*inputs)

        def forward_attentionish(q, k, v, q_bias, k_bias, da_cs, dt_seq, trap):
            return mamba3_attentionish_forward(
                q,
                k,
                v,
                q_bias=q_bias,
                k_bias=k_bias,
                da_cs=da_cs,
                dt=dt_seq,
                trap=trap,
                chunk_size=shape.chunk_size,
                implementation="xla",
            )

        def backward_attentionish(q, k, v, q_bias, k_bias, da_cs, dt_seq, trap):
            return jnp.sum(forward_attentionish(q, k, v, q_bias, k_bias, da_cs, dt_seq, trap).astype(jnp.float32) ** 2)

        forward_fn = jax.jit(forward_attentionish)
        backward_fn = jax.jit(jax.grad(backward_attentionish, argnums=2))
        forward_args = attentionish_inputs
        backward_args = attentionish_inputs

    forward_compile, forward_steady = _time_jitted(forward_fn, *forward_args, steps=steps, warmup=warmup)
    backward_compile, backward_steady = _time_jitted(backward_fn, *backward_args, steps=steps, warmup=warmup)

    if profile_dir is not None:
        if profile_mode == "forward":
            _capture_profile(forward_fn, *forward_args, profile_dir=profile_dir, profile_steps=profile_steps)
        elif profile_mode == "backward":
            _capture_profile(backward_fn, *backward_args, profile_dir=profile_dir, profile_steps=profile_steps)
        else:
            raise ValueError(f"Unsupported profile mode: {profile_mode}.")

    return [
        BenchmarkResult(
            api=api,
            seq_len=shape.seq_len,
            batch_head_groups=shape.batch_head_groups,
            chunk_size=shape.chunk_size,
            num_chunks=num_chunks,
            groups=groups,
            state_dim=shape.state_dim,
            value_dim=shape.value_dim,
            dtype=str(dtype),
            sharded=shard_groups,
            mode="forward",
            compile_s=forward_compile,
            steady_s=forward_steady,
            tokens_per_s=tokens / forward_steady,
            estimated_tflops=flops / forward_steady / 1e12,
        ),
        BenchmarkResult(
            api=api,
            seq_len=shape.seq_len,
            batch_head_groups=shape.batch_head_groups,
            chunk_size=shape.chunk_size,
            num_chunks=num_chunks,
            groups=groups,
            state_dim=shape.state_dim,
            value_dim=shape.value_dim,
            dtype=str(dtype),
            sharded=shard_groups,
            mode="backward",
            compile_s=backward_compile,
            steady_s=backward_steady,
            tokens_per_s=tokens / backward_steady,
            estimated_tflops=flops / backward_steady / 1e12,
        ),
    ]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Mamba-3 reference and XLA chunked paths.")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--api", type=str, default="chunked_public,attentionish_public")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seq-lens", type=str, default="")
    parser.add_argument("--batch-head-groups", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--state-dim", type=int, default=128)
    parser.add_argument("--value-dim", type=int, default=512)
    parser.add_argument("--shard-groups", action="store_true")
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--profile-dir", type=str, default="")
    parser.add_argument("--profile-mode", type=str, default="backward")
    parser.add_argument("--profile-steps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dtype = _dtype_from_name(args.dtype)
    apis = tuple(item.strip() for item in args.api.split(",") if item.strip())
    shapes = DEFAULT_SHAPES
    if args.seq_lens:
        shapes = tuple(
            Shape(
                seq_len=int(item),
                batch_head_groups=args.batch_head_groups,
                chunk_size=args.chunk_size,
                state_dim=args.state_dim,
                value_dim=args.value_dim,
            )
            for item in args.seq_lens.split(",")
        )

    rows: list[BenchmarkResult] = []
    for shape in shapes:
        for api in apis:
            rows.extend(
                _benchmark(
                    shape,
                    dtype=dtype,
                    api=api,  # type: ignore[arg-type]
                    shard_groups=args.shard_groups,
                    steps=args.steps,
                    warmup=args.warmup,
                    profile_dir=args.profile_dir or None,
                    profile_mode=args.profile_mode,
                    profile_steps=args.profile_steps,
                )
            )

    print(f"backend={jax.default_backend()} device={jax.devices()[0].device_kind}")
    print(f"devices={len(jax.devices())} shard_groups={args.shard_groups}")
    for row in rows:
        print(
            f"api={row.api:17s} mode={row.mode:8s} seq={row.seq_len:5d} "
            f"groups={row.groups:3d} chunk={row.chunk_size:3d} state={row.state_dim:3d} value={row.value_dim:3d} "
            f"sharded={str(row.sharded):5s} "
            f"compile_s={row.compile_s:.4f} steady_s={row.steady_s:.6f} "
            f"tokens_per_s={row.tokens_per_s:.2f} est_tflops={row.estimated_tflops:.2f}"
        )
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))


if __name__ == "__main__":
    main()
