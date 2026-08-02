# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""CuTeDSL->JAX bridge for QuACK's SM100 symmetric GEMM (D = X @ X^T, full symmetric via
in-kernel mirror), used to accelerate the two symmetric products in the Grug Muon Newton-Schulz.

A thin ``@cute.jit`` launcher over ``GemmSymmetricSm100`` wrapped with ``cutlass.jax.cutlass_call``
(the same bridge the FA4 attention backend uses, so the kernel runs on XLA's stream and is ordered
correctly with the surrounding graph). The kernel computes ``A @ B^T`` with a triangular tile
scheduler that writes each lower tile to D and its mirror to ``D.mT`` (same storage), so D comes out
fully symmetric with ~half the matmul FLOPs.

Config matches QuACK's own SM100 path exactly: ``mma_tiler (256, 256)``, ``cluster (2, 1, 1)``, and
**static** persistence (``use_clc_persistence=False``). The dynamic CLC scheduler (with its GMEM
tile-count semaphore) races between clusters and returned intermittent garbage on tiny-magnitude and
square inputs; static persistence is deterministic and bit-exact on all inputs (gram, tiny gram,
square A@A). Validated ~1.7x vs dense on the symmetric products (GB200, 128 experts).

Lazy-imported (quack/cutlass-dsl is Blackwell-only), like the sonic MoE backend.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
from cutlass import Float32
from quack.gemm import make_scheduler_args
from quack.gemm_symmetric import GemmSymmetricMixin, GemmSymmetricSm100

try:
    from quack.cute_dsl_utils import get_max_active_clusters
except Exception:  # pragma: no cover - name moved across quack versions
    from quack.gemm_act import get_max_active_clusters

_ACC = cutlass.Float32
_JAX_TO_CUTE = {
    jnp.dtype(jnp.bfloat16): cutlass.BFloat16,
    jnp.dtype(jnp.float16): cutlass.Float16,
    jnp.dtype(jnp.float32): cutlass.Float32,
}

# QuACK's SM100 symmetric config (_symmetric_gemm_config): tile (256, 256), cluster_M 2, static.
# NOTE: (256, 128) benchmarks 1.62x faster on the isolated [96, 2560, 5120] expert gram, but it
# BREAKS live training (loss stuck at init, run never progresses) -- the NS also orthogonalizes
# non-expert matrices whose shapes (256, 128) does not handle correctly, and the isolated gram
# microbench does not cover them. (256, 256) is validated clean end-to-end (30-step GB200 run,
# loss 11.8 -> 5.94). Do not narrow tile_N without validating on the full set of live NS shapes.
_DEFAULT_MMA_TILER = (256, 256)
_DEFAULT_CLUSTER = (2, 1, 1)
_DEFAULT_SWIZZLE = 8
_FALLBACK_MAX_ACTIVE_CLUSTERS = 148


def _cute_dtype(dt):
    return _JAX_TO_CUTE[jnp.dtype(dt)]


def _transpose_mn(mD):
    """aux = D^T (same storage): swap the two M modes of cute [M, M, L], keep batch L last."""
    return cute.make_tensor(mD.iterator, cute.select(mD.layout, mode=[1, 0, 2]))


def _build_launcher(*, a_dtype, mma_tiler_mnk, cluster_mnk, mac, max_swizzle):
    @cute.jit
    def launcher(stream, mA, mB, mD):
        # Static persistence (use_clc_persistence=False): no tile-count semaphore, deterministic
        # tile scheduling. The dynamic CLC path raced and corrupted tiny/square outputs.
        gemm = GemmSymmetricSm100(_ACC, a_dtype, mma_tiler_mnk, cluster_mnk, use_clc_persistence=False)
        # aux = D.mT (transposed view of the single output): the kernel writes each lower tile to D
        # and its mirror to D.mT, so D is fully symmetric. alpha/beta must be set (D = a*acc + b*C).
        epi_args = GemmSymmetricMixin.EpilogueArguments(
            _transpose_mn(mD), act_fn=None, alpha=Float32(1.0), beta=Float32(1.0)
        )
        scheduler_args = make_scheduler_args(mac, max_swizzle, None)
        gemm(mA, mB, mD, None, epi_args, scheduler_args, None, stream)

    return launcher


def quack_symmetric_gemm(
    X: jax.Array,
    *,
    mma_tiler_mnk: tuple[int, int] = _DEFAULT_MMA_TILER,
    cluster_mnk: tuple[int, int, int] = _DEFAULT_CLUSTER,
    max_swizzle: int = _DEFAULT_SWIZZLE,
) -> jax.Array:
    """Batched symmetric GEMM: ``X[L, M, K] -> X @ X^T [L, M, M]`` (full symmetric, bit-exact).

    ``X`` must be device-local (no cross-device sharding on any axis) — call inside a shard_map.
    The kernel computes ``A @ B^T``, so both operands are ``X``, k-major ([M, K, L], mode (1,2,0)).
    """
    L, M, K = X.shape
    a_dtype = _cute_dtype(X.dtype)
    try:
        mac = get_max_active_clusters(cluster_mnk[0] * cluster_mnk[1])
    except Exception:
        mac = _FALLBACK_MAX_ACTIVE_CLUSTERS
    launcher = _build_launcher(
        a_dtype=a_dtype, mma_tiler_mnk=mma_tiler_mnk, cluster_mnk=cluster_mnk, mac=mac, max_swizzle=max_swizzle
    )
    ts = cjax.TensorSpec
    spec = ts(mode=(1, 2, 0), divisibility=(1, 1, 8), static=False)
    call = cjax.cutlass_call(
        launcher,
        output_shape_dtype=(jax.ShapeDtypeStruct((L, M, M), X.dtype),),
        input_spec=(spec, spec),
        output_spec=(spec,),
        use_static_tensors=False,
    )
    (d,) = call(X, X)
    return d


__all__ = ["quack_symmetric_gemm"]
