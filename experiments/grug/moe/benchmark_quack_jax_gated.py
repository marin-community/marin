# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Compile QuACK's fused variable-length gated GEMM as a JAX TVM-FFI call."""

import argparse
import json
import time

import cutlass.cute as cute
import jax
import jax.numpy as jnp
import numpy as np
from cuda.bindings import driver as cuda
from cutlass import BFloat16, Float32, Int32
from quack.activation import gate_fn_map
from quack.compile_utils import make_fake_tensor
from quack.cute_dsl_utils import get_max_active_clusters
from quack.gemm_act import GemmActMixin, GemmGatedSm90
from quack.jax_utils import TvmFfiKernel
from quack.tile_scheduler import TileSchedulerOptions
from quack.varlen_utils import VarlenArguments

_TILE_SHAPE = (128, 192)
_CLUSTER_SHAPE = (2, 1, 1)


class _GatedVarlenFfi:
    def __init__(self) -> None:
        self.gemm = GemmGatedSm90(
            Float32,
            BFloat16,
            _TILE_SHAPE,
            _CLUSTER_SHAPE,
            pingpong=True,
            is_persistent=True,
        )
        self.max_active_clusters = get_max_active_clusters(_CLUSTER_SHAPE[0] * _CLUSTER_SHAPE[1])

    @cute.jit
    def __call__(
        self,
        a: cute.Tensor,
        b_storage: cute.Tensor,
        cu_seqlens: cute.Tensor,
        preact: cute.Tensor,
        postact: cute.Tensor,
        stream: cuda.CUstream,
    ) -> None:
        # JAX supplies compact [E, N, K] storage. QuACK consumes the zero-copy
        # [N, K, E] view used by its grouped GEMM implementation.
        b = cute.make_tensor(
            b_storage.iterator,
            cute.make_layout(
                (b_storage.shape[1], b_storage.shape[2], b_storage.shape[0]),
                stride=(b_storage.shape[2], 1, b_storage.shape[1] * b_storage.shape[2]),
            ),
        )
        epilogue = GemmActMixin.EpilogueArguments(
            postact,
            gate_fn_map["swiglu"],
        )
        scheduler = TileSchedulerOptions(
            max_active_clusters=Int32(self.max_active_clusters),
            max_swizzle_size=Int32(8),
        )
        varlen = VarlenArguments(mCuSeqlensM=cu_seqlens)
        self.gemm(a, b, preact, None, epilogue, scheduler, varlen, stream)


def _compile_gated_varlen():
    total_tokens = cute.sym_int()
    hidden_dim = cute.sym_int()
    output_dim = cute.sym_int()
    postact_dim = cute.sym_int()
    num_experts = cute.sym_int()
    num_offsets = cute.sym_int()
    alignment = 8

    a = make_fake_tensor(BFloat16, (total_tokens, hidden_dim), leading_dim=1, divisibility=alignment)
    b = make_fake_tensor(
        BFloat16,
        (num_experts, output_dim, hidden_dim),
        leading_dim=2,
        divisibility=alignment,
    )
    cu_seqlens = make_fake_tensor(Int32, (num_offsets,), leading_dim=0, divisibility=4)
    preact = make_fake_tensor(BFloat16, (total_tokens, output_dim), leading_dim=1, divisibility=alignment)
    postact = make_fake_tensor(BFloat16, (total_tokens, postact_dim), leading_dim=1, divisibility=alignment)
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    return cute.compile(
        _GatedVarlenFfi(),
        a,
        b,
        cu_seqlens,
        preact,
        postact,
        stream,
        options="--enable-tvm-ffi",
    )


_GATED_VARLEN = TvmFfiKernel(
    "marin_quack_gated_varlen_fwd",
    _compile_gated_varlen,
    allow_cuda_graph=True,
)


def quack_gated_varlen(
    x: jax.Array,
    weights: jax.Array,
    cu_seqlens: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    if x.dtype != jnp.bfloat16 or weights.dtype != jnp.bfloat16:
        raise TypeError("QuACK gated GEMM currently requires bfloat16 inputs")
    if x.ndim != 2 or weights.ndim != 3 or cu_seqlens.ndim != 1:
        raise ValueError("expected x [M,H], weights [E,2I,H], and offsets [E+1]")
    if weights.shape[2] != x.shape[1] or weights.shape[1] % 2:
        raise ValueError(f"incompatible x and weight shapes: {x.shape} and {weights.shape}")
    preact_shape = jax.ShapeDtypeStruct((x.shape[0], weights.shape[1]), x.dtype)
    postact_shape = jax.ShapeDtypeStruct((x.shape[0], weights.shape[1] // 2), x.dtype)
    return _GATED_VARLEN(
        x,
        weights,
        cu_seqlens,
        key=(),
        output_shape_dtype=(preact_shape, postact_shape),
    )


def _reference(x: jax.Array, weights: jax.Array, offsets: np.ndarray) -> tuple[jax.Array, jax.Array]:
    pieces = []
    for expert in range(weights.shape[0]):
        pieces.append(x[offsets[expert] : offsets[expert + 1]] @ weights[expert].T)
    preact = jnp.concatenate(pieces, axis=0)
    gate, up = jnp.split(preact, 2, axis=1)
    return preact, jax.nn.silu(gate) * up


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokens", type=int, default=16_384 * 4)
    parser.add_argument("--hidden-dim", type=int, default=2_560)
    parser.add_argument("--intermediate-dim", type=int, default=1_280)
    parser.add_argument("--experts", type=int, default=64)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=5)
    args = parser.parse_args()

    if jax.default_backend() != "gpu":
        raise RuntimeError("QuACK JAX benchmark requires a GPU")
    if args.tokens % args.experts:
        raise ValueError("tokens must be divisible by experts for this balanced correctness probe")

    key_x, key_w = jax.random.split(jax.random.key(0))
    x = jax.random.normal(key_x, (args.tokens, args.hidden_dim), dtype=jnp.bfloat16)
    weights = (
        jax.random.normal(
            key_w,
            (args.experts, 2 * args.intermediate_dim, args.hidden_dim),
            dtype=jnp.bfloat16,
        )
        * 0.02
    )
    offsets_np = np.arange(0, args.tokens + 1, args.tokens // args.experts, dtype=np.int32)
    offsets = jnp.asarray(offsets_np)
    fn = jax.jit(quack_gated_varlen)

    preact, postact = fn(x, weights, offsets)
    jax.block_until_ready((preact, postact))
    reference_preact, reference_postact = _reference(x, weights, offsets_np)
    max_preact_error = float(jnp.max(jnp.abs(preact.astype(jnp.float32) - reference_preact.astype(jnp.float32))))
    max_postact_error = float(jnp.max(jnp.abs(postact.astype(jnp.float32) - reference_postact.astype(jnp.float32))))

    for _ in range(args.warmup):
        jax.block_until_ready(fn(x, weights, offsets))
    durations = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        jax.block_until_ready(fn(x, weights, offsets))
        durations.append(time.perf_counter() - start)

    durations.sort()
    print(
        json.dumps(
            {
                "tokens": args.tokens,
                "hidden_dim": args.hidden_dim,
                "intermediate_dim": args.intermediate_dim,
                "experts": args.experts,
                "mean_duration": sum(durations) / len(durations),
                "median_duration": durations[len(durations) // 2],
                "min_duration": durations[0],
                "max_preact_error": max_preact_error,
                "max_postact_error": max_postact_error,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
