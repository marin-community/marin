# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Mosaic-GPU FP8 cast-transpose: one read of a bf16/f32 tile -> rowwise + transposed f8 stores.

The f8 weight-gradient consumes token-contiguous (transposed) f8 operands. Producing that transpose
through XLA (``swapaxes`` on the f8 result) is an uncoalesced 1-byte transpose; producing it through
a second ``quantize`` re-reads the bf16 source (logbook GFP8-030). This kernel reads each tile of the
high-precision source **once**, quantizes it, and TMA-stores both the rowwise tile (``q[M,K]``) and
its transpose (``qT[K,M]``) — the transpose is a coalesced shared-memory store, the cast is fused in.

H100-only (Hopper, Warpgroup lowering). The bit-exact reference and the public dispatcher live in
``fp8_cast_transpose.py``. See logbook GFP8-033.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu

# Transposed-store strategies, A/B'd on H100 (the transposed store is the perf/lowering risk):
#   "smem_swz":   transpose into a SWIZZLED qt_smem then TMA store (the matmul-epilogue idiom: the
#                 tiling+swizzle transforms let the store use the dialect transpose path instead of
#                 the collapse_shape that rejects a strided/transposed plain SMEM view)
#   "smem_plain": transpose into a plain (un-transformed) qt_smem — known to fail lowering; kept to
#                 confirm the swizzle transforms are what unlock the store
_STORE_STRATEGIES = ("smem_swz", "smem_plain")


def cast_transpose_mgpu(
    x: jax.Array,  # [M, K] bf16/f32
    scale: jax.Array,  # (1,) per-tensor scale
    *,
    out_dtype: jnp.dtype = jnp.float8_e4m3fn,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    block_m: int = 128,
    block_k: int = 128,
    store_strategy: str = "smem_swz",
) -> tuple[jax.Array, jax.Array]:
    """One read of ``x`` -> ``(q[M,K], qT[K,M])`` in ``out_dtype`` (Mosaic-GPU, Hopper f8).

    ``q`` is ``quantize(x, out_dtype, scale, compute_dtype)``; ``qT`` is its transpose, produced by a
    coalesced SMEM transpose (a store into a ``transpose_ref`` view) rather than a second cast or an
    XLA f8 transpose.
    """
    if x.ndim != 2:
        raise ValueError(f"cast_transpose expects a 2D array, got shape {x.shape}")
    if store_strategy not in _STORE_STRATEGIES:
        raise ValueError(f"store_strategy must be one of {_STORE_STRATEGIES}, got {store_strategy!r}")
    m, k = x.shape
    if m % block_m != 0:
        raise ValueError(f"m={m} must be a multiple of block_m={block_m}")
    if k % block_k != 0:
        raise ValueError(f"k={k} must be a multiple of block_k={block_k}")

    # Quantize in f32: Mosaic supports only f16->f8 / f32->f8 casts (a bf16->f8 cast errors, and it
    # collapses an explicit bf16->f32->f8 back to the unsupported bf16->f8), so the divide/clip run in
    # f32 and cast f32->f8. Inputs are bf16 (exact in f32), so this matches the bf16 reference whenever
    # the division is exact (e.g. power-of-two scales); for other scales it differs by at most an f8 ULP.
    del compute_dtype
    dtype_max = float(jnp.finfo(out_dtype).max)  # plain float bound (no f8 promotion in clip)

    def body(scale_gmem, x_gmem, q_gmem, qt_gmem):
        grid_m = pl.cdiv(m, block_m)
        grid_k = pl.cdiv(k, block_k)
        scale_f = scale_gmem[0].astype(jnp.float32)

        @plgpu.nd_loop((grid_m * grid_k,), collective_axes="sm")
        def mk_loop(loop_info: plgpu.NDLoopInfo):
            tile = loop_info.index[0]
            m_i = tile // grid_k
            k_i = tile % grid_k

            # qt_smem holds the transpose q.T[bk,bm]. A swizzle transform makes its TMA store legal.
            qt_transforms = ()
            if store_strategy == "smem_swz":
                swizzle = plgpu.find_swizzle(block_m * jnp.dtype(out_dtype).itemsize * 8)
                swizzle_elems = swizzle // jnp.dtype(out_dtype).itemsize
                qt_transforms = (plgpu.TilingTransform((8, swizzle_elems)), plgpu.SwizzleTransform(swizzle))

            @functools.partial(
                pl.run_scoped,
                x_smem=plgpu.SMEM((block_m, block_k), dtype=x.dtype),
                q_smem=plgpu.SMEM((block_m, block_k), dtype=out_dtype),
                qt_smem=plgpu.SMEM((block_k, block_m), dtype=out_dtype, transforms=qt_transforms),
                barrier=plgpu.Barrier(),
            )
            def scope(x_smem, q_smem, qt_smem, barrier):
                plgpu.copy_gmem_to_smem(
                    x_gmem.at[pl.ds(m_i * block_m, block_m), pl.ds(k_i * block_k, block_k)],
                    x_smem,
                    barrier,
                )
                plgpu.barrier_wait(barrier)

                x_tile = x_smem[...].astype(jnp.float32)
                q = jnp.clip(x_tile / scale_f, -dtype_max, dtype_max).astype(out_dtype)

                # Rowwise store: q[bm,bk] -> q_gmem[m_i, k_i].
                q_smem[...] = q
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    q_smem, q_gmem.at[pl.ds(m_i * block_m, block_m), pl.ds(k_i * block_k, block_k)]
                )

                # Transposed store -> qt_gmem[k_i, m_i] (a [bk, bm] tile holding q.T): write q[bm,bk]
                # into a transposed view of qt_smem so it physically holds q.T, then TMA.
                plgpu.transpose_ref(qt_smem, (1, 0))[...] = q
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    qt_smem, qt_gmem.at[pl.ds(k_i * block_k, block_k), pl.ds(m_i * block_m, block_m)]
                )
                plgpu.wait_smem_to_gmem(0, wait_read_only=True)

    num_sms = jax.devices()[0].core_count
    kernel = plgpu.kernel(
        body,
        out_shape=(
            jax.ShapeDtypeStruct((m, k), out_dtype),
            jax.ShapeDtypeStruct((k, m), out_dtype),
        ),
        grid=(num_sms,),
        grid_names=("sm",),
        compiler_params=plgpu.CompilerParams(lowering_semantics=plgpu.LoweringSemantics.Warpgroup),
    )
    return kernel(scale, x)
