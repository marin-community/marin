# Copyright The Levanter Authors
#
# SPDX-License-Identifier: Apache-2.0

"""Mosaic-GPU FP8 cast-transpose: one read of a bf16/f32 tile -> rowwise + transposed f8 stores.

The f8 weight-gradient consumes token-contiguous (transposed) f8 operands. Producing that transpose
through XLA (``swapaxes`` on the f8 result) is an uncoalesced 1-byte transpose. This kernel reads each
tile of the high-precision source **once**, quantizes it, and TMA-stores both the rowwise tile
(``q[M,K]``) and its transpose (``qT[K,M]``) — the transpose is materialized in-register via the
``WGMMA`` -> ``WGMMA_TRANSPOSED`` layout cast (the tensor-core register transpose), so the SMEM store
stays coalesced.

Lowering notes (jax 0.10.0, hard-won — logbook GFP8-033):
- The transpose must be a *register layout cast*, not a memref transpose. ``transpose_ref(qt_smem,(1,0))``
  on a swizzled SMEM fails ("Can't transpose the swizzled dimension"); a transposed GMEM TMA descriptor
  is unimplemented; a register ``q.T`` is unimplemented in Warpgroup semantics.
- ``layout_cast(.., WGMMA_TRANSPOSED)`` is only defined for the *same dtype*, and f8 has its own packed
  WGMMA_8BIT tiling that won't convert — so the cast is done on the **f32** quant, then f32->f8 at the
  store. The store target ``qt_smem`` is PLAIN (no swizzle): the pre-transposed register layout matches
  the transposed view, so it lowers; the TMA then stores the now-contiguous ``qt_smem``.
This is the same idiom as JAX's own ``test_transposed_load_store``. H100-only (Hopper, Warpgroup). The
bit-exact reference + the public dispatcher live in ``fp8_cast_transpose.py``.
"""

import functools

import jax
import jax.numpy as jnp
from jax.experimental import pallas as pl
from jax.experimental.pallas import mosaic_gpu as plgpu


def cast_transpose_mgpu(
    x: jax.Array,  # [M, K] bf16/f32
    scale: jax.Array,  # (1,) per-tensor scale
    *,
    out_dtype: jnp.dtype = jnp.float8_e4m3fn,
    compute_dtype: jnp.dtype = jnp.bfloat16,
    block_m: int = 128,
    block_k: int = 128,
) -> tuple[jax.Array, jax.Array]:
    """One read of ``x`` -> ``(q[M,K], qT[K,M])`` in ``out_dtype`` (Mosaic-GPU, Hopper f8).

    ``q`` is ``quantize(x, out_dtype, scale, compute_dtype)``; ``qT`` is its transpose, produced by the
    ``WGMMA`` -> ``WGMMA_TRANSPOSED`` register layout cast (on the f32 quant, since the cast is only
    defined for same-dtype layouts) rather than a second cast or an XLA f8 transpose.
    """
    if x.ndim != 2:
        raise ValueError(f"cast_transpose expects a 2D array, got shape {x.shape}")
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

            @functools.partial(
                pl.run_scoped,
                x_smem=plgpu.SMEM((block_m, block_k), dtype=x.dtype),
                q_smem=plgpu.SMEM((block_m, block_k), dtype=out_dtype),
                qt_smem=plgpu.SMEM((block_k, block_m), dtype=out_dtype),
                barrier=plgpu.Barrier(),
            )
            def scope(x_smem, q_smem, qt_smem, barrier):
                plgpu.copy_gmem_to_smem(
                    x_gmem.at[pl.ds(m_i * block_m, block_m), pl.ds(k_i * block_k, block_k)],
                    x_smem,
                    barrier,
                )
                plgpu.barrier_wait(barrier)

                # Load + quantize in the WGMMA register layout so the transpose is a layout cast.
                x_tile = plgpu.load(x_smem, (), layout=plgpu.Layout.WGMMA, optimized=False).astype(jnp.float32)
                qf = jnp.clip(x_tile / scale_f, -dtype_max, dtype_max)  # f32, WGMMA layout (not yet f8)

                # Rowwise store: q[bm,bk] -> q_gmem[m_i, k_i].
                q_smem[...] = qf.astype(out_dtype)
                plgpu.commit_smem()
                plgpu.copy_smem_to_gmem(
                    q_smem, q_gmem.at[pl.ds(m_i * block_m, block_m), pl.ds(k_i * block_k, block_k)]
                )

                # Transposed store -> qt_gmem[k_i, m_i]: transpose via the WGMMA->WGMMA_TRANSPOSED layout
                # cast (defined for same-dtype, so on the f32 quant), cast f32->f8, and write into a
                # transposed view of a PLAIN qt_smem (pre-transposed register data matches the strided
                # view, so it lowers without a swizzle). qt_smem then holds q.T contiguously for the TMA.
                qt_view = plgpu.transpose_ref(qt_smem, (1, 0))  # [bk,bm] smem -> [bm,bk] logical view
                qf_t = plgpu.layout_cast(qf, plgpu.Layout.WGMMA_TRANSPOSED)
                qt_view[...] = qf_t.astype(out_dtype)
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
