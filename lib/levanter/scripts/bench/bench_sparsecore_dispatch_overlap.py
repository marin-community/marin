# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
import os
import sys
import time

import jax
import jax.numpy as jnp

import levanter.tracker
from levanter.callbacks import profile_ctx
from levanter.grug.grug_moe import _dense_moe_up_down, _take_rows_impl
from levanter.tracker import NoopTracker
from levanter.utils.activation import ActivationFunctionEnum


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _append_xla_flag(flag: str) -> None:
    current = os.environ.get("XLA_FLAGS", "")
    parts = current.split()
    if flag in parts:
        return
    os.environ["XLA_FLAGS"] = (current + " " + flag).strip()


def _configure_xla_dump_dir(xla_dump_dir: str | None) -> str | None:
    if xla_dump_dir is None:
        return None
    resolved = os.path.abspath(xla_dump_dir)
    os.makedirs(resolved, exist_ok=True)
    _append_xla_flag(f"--xla_dump_to={resolved}")
    _append_xla_flag("--xla_dump_hlo_as_text")
    return resolved


@contextmanager
def _tee_stdio(log_path: str | None):
    if log_path is None:
        yield
        return
    resolved = os.path.abspath(log_path)
    parent = os.path.dirname(resolved)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(resolved, "a", encoding="utf-8") as handle:
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = _TeeStream(old_stdout, handle)
        sys.stderr = _TeeStream(old_stderr, handle)
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def _make_routing(
    *,
    num_tokens: int,
    num_chunks: int,
    chunk_tokens: int,
) -> tuple[jax.Array, jax.Array]:
    key_ids, key_w = jax.random.split(jax.random.key(1))
    token_ids = jax.random.randint(key_ids, (num_chunks, chunk_tokens), 0, num_tokens, dtype=jnp.int32)
    weights = jax.random.uniform(key_w, (num_chunks, chunk_tokens), minval=0.0, maxval=1.0, dtype=jnp.float32)
    return token_ids, weights


def _make_inputs(
    *,
    num_tokens: int,
    hidden: int,
    intermediate: int,
    num_chunks: int,
    chunk_tokens: int,
    dtype: jnp.dtype,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    keys = jax.random.split(jax.random.key(0), 4)
    x = jax.random.normal(keys[0], (num_tokens, hidden), dtype=dtype)
    token_ids, weights = _make_routing(num_tokens=num_tokens, num_chunks=num_chunks, chunk_tokens=chunk_tokens)
    w13 = jax.random.normal(keys[1], (hidden, 2 * intermediate), dtype=dtype)
    w2 = jax.random.normal(keys[2], (intermediate, hidden), dtype=dtype)
    return x, token_ids, weights, w13, w2


def _serial_dispatch(
    x: jax.Array,
    token_ids: jax.Array,
    weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    *,
    gather_impl: str,
) -> jax.Array:
    activation_fn = ActivationFunctionEnum.silu.to_jax_fn()
    out = jnp.zeros_like(x)
    num_chunks = int(token_ids.shape[0])
    for chunk_idx in range(num_chunks):
        ids = token_ids[chunk_idx]
        chunk_weights = weights[chunk_idx]
        with jax.named_scope(f"gather_chunk_{chunk_idx}"):
            x_chunk = _take_rows_impl(x, ids, implementation=gather_impl)
        with jax.named_scope(f"mlp_chunk_{chunk_idx}"):
            out_chunk = _dense_moe_up_down(x_chunk, w13, w2, activation_fn=activation_fn)
        with jax.named_scope(f"scatter_chunk_{chunk_idx}"):
            out = out.at[ids].add(out_chunk * chunk_weights[:, None], mode="drop")
    return out


def _pipeline_sc_prefetch(
    x: jax.Array,
    token_ids: jax.Array,
    weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    *,
    barrier: bool,
) -> jax.Array:
    activation_fn = ActivationFunctionEnum.silu.to_jax_fn()
    out = jnp.zeros_like(x)
    num_chunks = int(token_ids.shape[0])

    with jax.named_scope("gather_chunk_0"):
        current_ids = token_ids[0]
        current_gather = _take_rows_impl(x, current_ids, implementation="sparsecore")

    for chunk_idx in range(num_chunks - 1):
        next_ids = token_ids[chunk_idx + 1]
        current_weights = weights[chunk_idx]

        with jax.named_scope(f"gather_chunk_{chunk_idx + 1}"):
            next_gather = _take_rows_impl(x, next_ids, implementation="sparsecore")

        if barrier:
            current_gather = jax.lax.optimization_barrier(current_gather)
            current_weights = jax.lax.optimization_barrier(current_weights)

        with jax.named_scope(f"mlp_chunk_{chunk_idx}"):
            out_chunk = _dense_moe_up_down(current_gather, w13, w2, activation_fn=activation_fn)

        if barrier:
            out_chunk = jax.lax.optimization_barrier(out_chunk)

        with jax.named_scope(f"scatter_chunk_{chunk_idx}"):
            out = out.at[current_ids].add(out_chunk * current_weights[:, None], mode="drop")

        current_ids = next_ids
        current_gather = next_gather

    final_weights = weights[-1]
    if barrier:
        current_gather = jax.lax.optimization_barrier(current_gather)
        final_weights = jax.lax.optimization_barrier(final_weights)
    with jax.named_scope(f"mlp_chunk_{num_chunks - 1}"):
        out_chunk = _dense_moe_up_down(current_gather, w13, w2, activation_fn=activation_fn)
    if barrier:
        out_chunk = jax.lax.optimization_barrier(out_chunk)
    with jax.named_scope(f"scatter_chunk_{num_chunks - 1}"):
        out = out.at[current_ids].add(out_chunk * final_weights[:, None], mode="drop")
    return out


def _timeit(fn, *args, warmup: int, iters: int) -> float:
    out = fn(*args)
    jax.block_until_ready(out)
    for _ in range(warmup):
        out = fn(*args)
        jax.block_until_ready(out)
    start = time.perf_counter()
    for _ in range(iters):
        out = fn(*args)
        jax.block_until_ready(out)
    return (time.perf_counter() - start) / iters


def _run_impl(
    implementation: str,
    x: jax.Array,
    token_ids: jax.Array,
    weights: jax.Array,
    w13: jax.Array,
    w2: jax.Array,
    *,
    barrier: bool,
    warmup: int,
    iters: int,
    profile_dir: str | None,
) -> float:
    if implementation == "serial_xla":
        fn = jax.jit(lambda x, ids, w, w13, w2: _serial_dispatch(x, ids, w, w13, w2, gather_impl="xla"))
    elif implementation == "serial_sc":
        fn = jax.jit(lambda x, ids, w, w13, w2: _serial_dispatch(x, ids, w, w13, w2, gather_impl="sparsecore"))
    elif implementation == "pipeline_sc":
        fn = jax.jit(lambda x, ids, w, w13, w2: _pipeline_sc_prefetch(x, ids, w, w13, w2, barrier=barrier))
    else:
        raise ValueError(f"Unknown implementation {implementation!r}")

    out = fn(x, token_ids, weights, w13, w2)
    jax.block_until_ready(out)
    for _ in range(warmup):
        out = fn(x, token_ids, weights, w13, w2)
        jax.block_until_ready(out)

    prof_ctx = nullcontext()
    if profile_dir is not None:
        prof_ctx = profile_ctx(profile_dir, create_perfetto_link=False)

    with levanter.tracker.current_tracker(NoopTracker()):
        with prof_ctx:
            start = time.perf_counter()
            for _ in range(iters):
                out = fn(x, token_ids, weights, w13, w2)
                jax.block_until_ready(out)
            return (time.perf_counter() - start) / iters


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=40960)
    parser.add_argument("--hidden", type=int, default=2048)
    parser.add_argument("--intermediate", type=int, default=1536)
    parser.add_argument("--num-chunks", type=int, default=32)
    parser.add_argument("--chunk-tokens", type=int, default=5120)
    parser.add_argument("--dtype", choices=("bf16", "f32"), default="bf16")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--iters", type=int, default=5)
    parser.add_argument("--barrier", action="store_true")
    parser.add_argument("--implementation", choices=("serial_xla", "serial_sc", "pipeline_sc"), required=True)
    parser.add_argument("--profile-dir", type=str, default=None)
    parser.add_argument("--xla-dump-dir", type=str, default=None)
    parser.add_argument("--compiler-log-path", type=str, default=None)
    args = parser.parse_args()

    dump_dir = _configure_xla_dump_dir(args.xla_dump_dir)
    dtype = jnp.bfloat16 if args.dtype == "bf16" else jnp.float32
    x, token_ids, weights, w13, w2 = _make_inputs(
        num_tokens=args.tokens,
        hidden=args.hidden,
        intermediate=args.intermediate,
        num_chunks=args.num_chunks,
        chunk_tokens=args.chunk_tokens,
        dtype=dtype,
    )

    with _tee_stdio(args.compiler_log_path):
        total_chunk_assignments = args.num_chunks * args.chunk_tokens
        print("device_kind", jax.devices()[0].device_kind)
        print("implementation", args.implementation)
        print("tokens", args.tokens)
        print("hidden", args.hidden)
        print("intermediate", args.intermediate)
        print("num_chunks", args.num_chunks)
        print("chunk_tokens", args.chunk_tokens)
        print("total_chunk_assignments", total_chunk_assignments)
        print("barrier", args.barrier)
        print("LIBTPU_INIT_ARGS", os.environ.get("LIBTPU_INIT_ARGS", ""))
        print("XLA_FLAGS", os.environ.get("XLA_FLAGS", ""))
        if dump_dir is not None:
            print("xla_dump_dir", dump_dir)
        steady_s = _run_impl(
            args.implementation,
            x,
            token_ids,
            weights,
            w13,
            w2,
            barrier=args.barrier,
            warmup=args.warmup,
            iters=args.iters,
            profile_dir=args.profile_dir,
        )
        print(f"steady_s={steady_s:.6f}")


if __name__ == "__main__":
    main()
