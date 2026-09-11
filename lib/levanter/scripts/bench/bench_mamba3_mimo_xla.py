# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Literal

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P

from levanter.kernels.pallas.mamba3 import (
    mamba3_mimo_attentionish_forward,
    mamba3_mimo_chunked_forward,
    prepare_mamba3_chunked_scales,
)
from levanter.kernels.pallas.mamba3.reference import (
    mamba3_mimo_apply_gate_and_collapse_chunked,
    mamba3_mimo_rank_expand_chunked,
)
from levanter.kernels.pallas.mamba3.xla import mamba3_mimo_chunked_forward_ranked_xla_batched
from levanter.kernels.pallas.ssd import intra_chunk_log_alpha_cumsum, local_log_alpha


@dataclass(frozen=True, slots=True)
class Shape:
    state_dim: int
    value_dim: int


@dataclass(frozen=True, slots=True)
class BenchmarkResult:
    seq_len: int
    groups: int
    chunk_size: int
    rank: int
    state_dim: int
    value_dim: int
    dtype: str
    sharded: bool
    api: str
    mode: str
    compile_s: float
    steady_s: float
    tokens_per_s: float
    estimated_tflops: float


DEFAULT_SHAPES = (
    Shape(state_dim=128, value_dim=512),
    Shape(state_dim=512, value_dim=1024),
    Shape(state_dim=1024, value_dim=512),
)
ApiVariant = Literal["ranked_core", "chunked_public", "attentionish_public"]


def _dtype_from_name(name: str) -> jnp.dtype:
    if name == "bfloat16":
        return jnp.bfloat16
    if name == "float32":
        return jnp.float32
    raise ValueError(f"Unsupported dtype: {name}.")


def _build_inputs(
    *,
    seq_len: int,
    groups: int,
    chunk_size: int,
    state_dim: int,
    value_dim: int,
    rank: int,
    dtype: jnp.dtype,
) -> tuple[jax.Array, ...]:
    key = jax.random.PRNGKey(17)
    keys = jax.random.split(key, 10)
    num_chunks = seq_len // chunk_size
    token_shape = (groups, num_chunks, chunk_size)
    dt = (0.01 + 0.04 * jax.random.uniform(keys[0], token_shape, dtype=jnp.float32)).astype(dtype)
    lam = jax.random.uniform(keys[1], token_shape, minval=0.05, maxval=0.95, dtype=jnp.float32).astype(dtype)
    a = (-0.01 - jax.random.uniform(keys[2], (groups, num_chunks), minval=0.0, maxval=1.0, dtype=jnp.float32)).astype(
        dtype
    )
    b = (0.02 * jax.random.normal(keys[3], token_shape + (state_dim, rank), dtype=jnp.float32)).astype(dtype)
    c = (0.02 * jax.random.normal(keys[4], token_shape + (state_dim, rank), dtype=jnp.float32)).astype(dtype)
    x_base = (0.02 * jax.random.normal(keys[5], token_shape + (value_dim,), dtype=jnp.float32)).astype(dtype)
    z_base = (0.02 * jax.random.normal(keys[6], token_shape + (value_dim,), dtype=jnp.float32)).astype(dtype)
    w_x = (0.02 * jax.random.normal(keys[7], (value_dim, rank), dtype=jnp.float32)).astype(dtype)
    w_z = (0.02 * jax.random.normal(keys[8], (value_dim, rank), dtype=jnp.float32)).astype(dtype)
    w_o = (0.02 * jax.random.normal(keys[9], (value_dim, rank), dtype=jnp.float32)).astype(dtype)
    return dt, lam, a, b, c, x_base, z_base, w_x, w_z, w_o


def _estimate_forward_flops(
    seq_len: int, groups: int, chunk_size: int, state_dim: int, value_dim: int, rank: int
) -> float:
    num_chunks = seq_len // chunk_size
    local_bc = 2.0 * groups * num_chunks * chunk_size * chunk_size * state_dim * rank * rank
    local_emit = 2.0 * groups * num_chunks * chunk_size * chunk_size * value_dim * rank * rank
    chunk_state = 2.0 * groups * num_chunks * chunk_size * state_dim * value_dim * rank
    prefix_emit = 2.0 * groups * num_chunks * chunk_size * state_dim * value_dim * rank
    correction = 2.0 * groups * num_chunks * chunk_size * (state_dim + value_dim) * rank * rank
    return local_bc + local_emit + chunk_state + prefix_emit + correction


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


def _capture_profile(fn, *args, profile_dir: str, profile_steps: int) -> None:
    output_dir = Path(profile_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jax.profiler.start_trace(str(output_dir), create_perfetto_trace=True)
    try:
        for _ in range(profile_steps):
            out = fn(*args)
            jax.block_until_ready(out)
    finally:
        jax.profiler.stop_trace()


def _attentionish_public_inputs(
    dt: jax.Array,
    lam: jax.Array,
    a: jax.Array,
    b: jax.Array,
    c: jax.Array,
    x_base: jax.Array,
    z_base: jax.Array,
    w_x: jax.Array,
    w_z: jax.Array,
    w_o: jax.Array,
) -> tuple[jax.Array, ...]:
    groups, num_chunks, chunk_size = dt.shape
    seq_len = num_chunks * chunk_size
    state_dim = b.shape[-2]
    rank = b.shape[-1]
    value_dim = x_base.shape[-1]
    a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
    eps = jnp.finfo(lam.dtype).eps
    clipped_lam = jnp.clip(lam, eps, 1 - eps)
    trap = (jnp.log(clipped_lam) - jnp.log1p(-clipped_lam)).reshape(groups, 1, seq_len)
    q = c.transpose(0, 1, 2, 4, 3).reshape(groups, seq_len, rank, 1, state_dim)
    k = b.transpose(0, 1, 2, 4, 3).reshape(groups, seq_len, rank, 1, state_dim)
    v = x_base.reshape(groups, seq_len, 1, value_dim)
    z = z_base.reshape(groups, seq_len, 1, value_dim)
    q_bias = jnp.zeros((1, rank, state_dim), dtype=dt.dtype)
    k_bias = jnp.zeros((1, rank, state_dim), dtype=dt.dtype)
    mimo_v = jnp.broadcast_to(jnp.swapaxes(w_x, -1, -2)[None, ...], (1, rank, value_dim))
    mimo_z = jnp.broadcast_to(jnp.swapaxes(w_z, -1, -2)[None, ...], (1, rank, value_dim))
    mimo_o = jnp.broadcast_to(jnp.swapaxes(w_o, -1, -2)[None, ...], (1, rank, value_dim))
    da_cs = a_log_cumsum.reshape(groups, 1, seq_len)
    dt_seq = dt.reshape(groups, 1, seq_len)
    return q, k, v, mimo_v, mimo_o, q_bias, k_bias, z, mimo_z, da_cs, dt_seq, trap


def _shard_spec(ndim: int, *, shard_groups: bool) -> P:
    return P("data", *([None] * (ndim - 1))) if shard_groups else P(*([None] * ndim))


def _maybe_shard_inputs(inputs: tuple[jax.Array, ...], *, shard_groups: bool) -> tuple[jax.Array, ...]:
    if not shard_groups:
        return inputs
    devices = np.array(jax.devices())
    mesh = Mesh(devices, ("data",))

    def put(x: jax.Array, *, shard_leading: bool) -> jax.Array:
        return jax.device_put(x, NamedSharding(mesh, _shard_spec(x.ndim, shard_groups=shard_leading)))

    dt, lam, a, b, c, x_base, z_base, w_x, w_z, w_o = inputs
    return (
        put(dt, shard_leading=True),
        put(lam, shard_leading=True),
        put(a, shard_leading=True),
        put(b, shard_leading=True),
        put(c, shard_leading=True),
        put(x_base, shard_leading=True),
        put(z_base, shard_leading=True),
        put(w_x, shard_leading=False),
        put(w_z, shard_leading=False),
        put(w_o, shard_leading=False),
    )


def _benchmark(
    *,
    seq_len: int,
    groups: int,
    chunk_size: int,
    rank: int,
    shape: Shape,
    dtype: jnp.dtype,
    shard_groups: bool,
    api: ApiVariant,
    steps: int,
    warmup: int,
    profile_dir: str | None,
    profile_mode: str,
    profile_steps: int,
) -> list[BenchmarkResult]:
    inputs = _maybe_shard_inputs(
        _build_inputs(
            seq_len=seq_len,
            groups=groups,
            chunk_size=chunk_size,
            state_dim=shape.state_dim,
            value_dim=shape.value_dim,
            rank=rank,
            dtype=dtype,
        ),
        shard_groups=shard_groups,
    )
    dt, lam, a, b, c, x_base, z_base, w_x, w_z, w_o = inputs
    flops = _estimate_forward_flops(seq_len, groups, chunk_size, shape.state_dim, shape.value_dim, rank)
    tokens = groups * seq_len

    if api == "ranked_core":
        src_scale, out_correction = prepare_mamba3_chunked_scales(dt, lam)
        a_log_cumsum = intra_chunk_log_alpha_cumsum(local_log_alpha(dt, a))
        x_ranked = mamba3_mimo_rank_expand_chunked(x_base, w_x)
        z_ranked = mamba3_mimo_rank_expand_chunked(z_base, w_z)

        def forward_core(
            a_log_cumsum_in, src_scale_in, out_correction_in, b_in, c_in, x_ranked_in, z_ranked_in, w_o_in
        ):
            y_ranked, _ = mamba3_mimo_chunked_forward_ranked_xla_batched(
                a_log_cumsum_in,
                src_scale_in,
                out_correction_in,
                b_in,
                c_in,
                x_ranked_in,
            )
            return mamba3_mimo_apply_gate_and_collapse_chunked(y_ranked, z_ranked_in, w_o_in)

        def backward_loss(
            a_log_cumsum_in, src_scale_in, out_correction_in, b_in, c_in, x_ranked_in, z_ranked_in, w_o_in
        ):
            return jnp.sum(
                forward_core(
                    a_log_cumsum_in, src_scale_in, out_correction_in, b_in, c_in, x_ranked_in, z_ranked_in, w_o_in
                ).astype(jnp.float32)
                ** 2
            )

        forward_fn = jax.jit(forward_core)
        backward_fn = jax.jit(jax.grad(backward_loss, argnums=5))
        forward_args = (a_log_cumsum, src_scale, out_correction, b, c, x_ranked, z_ranked, w_o)
        backward_args = forward_args
    elif api == "chunked_public":

        def forward_public(dt_in, lam_in, a_in, b_in, c_in, x_base_in, z_base_in, w_x_in, w_z_in, w_o_in):
            return mamba3_mimo_chunked_forward(
                dt_in,
                lam_in,
                a_in,
                b_in,
                c_in,
                x_base_in,
                z_base_in,
                w_x_in,
                w_z_in,
                w_o_in,
                implementation="xla",
            )[0]

        def backward_public(dt_in, lam_in, a_in, b_in, c_in, x_base_in, z_base_in, w_x_in, w_z_in, w_o_in):
            return jnp.sum(
                forward_public(dt_in, lam_in, a_in, b_in, c_in, x_base_in, z_base_in, w_x_in, w_z_in, w_o_in).astype(
                    jnp.float32
                )
                ** 2
            )

        forward_fn = jax.jit(forward_public)
        backward_fn = jax.jit(jax.grad(backward_public, argnums=5))
        forward_args = inputs
        backward_args = inputs
    elif api == "attentionish_public":
        attentionish_inputs = _attentionish_public_inputs(dt, lam, a, b, c, x_base, z_base, w_x, w_z, w_o)
        q, k, v, mimo_v, mimo_o, q_bias, k_bias, z, mimo_z, da_cs, dt_seq, trap = attentionish_inputs

        def forward_attentionish(
            q_in,
            k_in,
            v_in,
            mimo_v_in,
            mimo_o_in,
            q_bias_in,
            k_bias_in,
            z_in,
            mimo_z_in,
            da_cs_in,
            dt_seq_in,
            trap_in,
        ):
            return mamba3_mimo_attentionish_forward(
                q_in,
                k_in,
                v_in,
                mimo_v_in,
                mimo_o_in,
                q_bias=q_bias_in,
                k_bias=k_bias_in,
                z=z_in,
                mimo_z=mimo_z_in,
                da_cs=da_cs_in,
                dt=dt_seq_in,
                trap=trap_in,
                chunk_size=chunk_size,
                implementation="xla",
            )

        def backward_attentionish(
            q_in,
            k_in,
            v_in,
            mimo_v_in,
            mimo_o_in,
            q_bias_in,
            k_bias_in,
            z_in,
            mimo_z_in,
            da_cs_in,
            dt_seq_in,
            trap_in,
        ):
            return jnp.sum(
                forward_attentionish(
                    q_in,
                    k_in,
                    v_in,
                    mimo_v_in,
                    mimo_o_in,
                    q_bias_in,
                    k_bias_in,
                    z_in,
                    mimo_z_in,
                    da_cs_in,
                    dt_seq_in,
                    trap_in,
                ).astype(jnp.float32)
                ** 2
            )

        forward_fn = jax.jit(forward_attentionish)
        backward_fn = jax.jit(jax.grad(backward_attentionish, argnums=2))
        forward_args = attentionish_inputs
        backward_args = attentionish_inputs
    else:
        raise ValueError(f"Unsupported benchmark API: {api}.")

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
            seq_len=seq_len,
            groups=groups,
            chunk_size=chunk_size,
            rank=rank,
            state_dim=shape.state_dim,
            value_dim=shape.value_dim,
            dtype=str(dtype),
            sharded=shard_groups,
            api=api,
            mode="forward",
            compile_s=forward_compile,
            steady_s=forward_steady,
            tokens_per_s=tokens / forward_steady,
            estimated_tflops=flops / forward_steady / 1e12,
        ),
        BenchmarkResult(
            seq_len=seq_len,
            groups=groups,
            chunk_size=chunk_size,
            rank=rank,
            state_dim=shape.state_dim,
            value_dim=shape.value_dim,
            dtype=str(dtype),
            sharded=shard_groups,
            api=api,
            mode="backward",
            compile_s=backward_compile,
            steady_s=backward_steady,
            tokens_per_s=tokens / backward_steady,
            estimated_tflops=flops / backward_steady / 1e12,
        ),
    ]


def _parse_shapes(value: str) -> tuple[Shape, ...]:
    if not value:
        return DEFAULT_SHAPES
    out = []
    for item in value.split(","):
        state_dim, value_dim = item.split("x")
        out.append(Shape(state_dim=int(state_dim), value_dim=int(value_dim)))
    return tuple(out)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark MIMO Mamba-3 XLA path.")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--seq-len", type=int, default=16384)
    parser.add_argument("--groups", type=int, default=16)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--rank", type=int, default=4)
    parser.add_argument("--shapes", type=str, default="128x512,512x1024,1024x512")
    parser.add_argument("--shard-groups", action="store_true")
    parser.add_argument(
        "--api",
        type=str,
        default="ranked_core",
        choices=("ranked_core", "chunked_public", "attentionish_public"),
    )
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--json", action="store_true")
    parser.add_argument("--profile-dir", type=str, default="")
    parser.add_argument("--profile-mode", type=str, default="backward")
    parser.add_argument("--profile-steps", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dtype = _dtype_from_name(args.dtype)
    shapes = _parse_shapes(args.shapes)
    rows: list[BenchmarkResult] = []
    for shape in shapes:
        rows.extend(
            _benchmark(
                seq_len=args.seq_len,
                groups=args.groups,
                chunk_size=args.chunk_size,
                rank=args.rank,
                shape=shape,
                dtype=dtype,
                shard_groups=args.shard_groups,
                api=args.api,
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
            f"mode={row.mode:8s} seq={row.seq_len:5d} groups={row.groups:3d} chunk={row.chunk_size:4d} rank={row.rank:2d} "
            f"state={row.state_dim:4d} value={row.value_dim:4d} api={row.api:18s} sharded={str(row.sharded):5s} "
            f"compile_s={row.compile_s:.4f} steady_s={row.steady_s:.6f} "
            f"tokens_per_s={row.tokens_per_s:.2f} est_tflops={row.estimated_tflops:.2f}"
        )
    if args.json:
        print(json.dumps([asdict(row) for row in rows], indent=2))


if __name__ == "__main__":
    main()
