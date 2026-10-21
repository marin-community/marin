# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Can a standalone MXFP8 token quantizer overlap a weight-gradient GEMM on one GPU?

The ceiling of a Dist-MoE-style "quantize once in the consuming kernel" design is the quantizer hidden under the
compute-bound GEMM. Without writing the fused kernel, this measures the spatial-split proxy: the varlen-k MXFP8
GEMM persistent on fewer clusters (``max_active_clusters``) and an independent quantizer issued on a second XLA
stream (``compute_on("gpu_stream:1")``, needs ``--xla_gpu_experimental_stream_annotation=true``), against both
back to back on one stream. Hero EP64 chunk shapes (262144/301568 rows, three experts, 3072 x 4608).
"""

import statistics
import time

import jax
import jax.numpy as jnp
import numpy as np
from jax.experimental.compute_on import compute_on

from levanter.cutlass_kernel_cache import cutlass_call
from levanter.grug._moe.quack_mxfp8 import (
    _INDEX_SPEC,
    _SCALE_SPEC,
    _TOKEN_MAJOR,
    _WGRAD_OUT_SPEC,
    _build_wgrad_launcher,
    blocked_scales,
    quantize_token_blocks,
    token_scale_blocks,
)
from levanter.grug.mxfp8_pallas import quantize_rows, requantize_row_blocks


def _gemm(lhs, rhs, cu, *, mac, cluster=(2, 1, 1)):
    (a, ae), (b, be) = lhs, rhs
    launcher = _build_wgrad_launcher(tile_mn=(256, 256), cluster_mnk=cluster, max_active_clusters=mac, max_swizzle=8)
    call = cutlass_call(
        launcher,
        output_shape_dtype=jax.ShapeDtypeStruct((cu.shape[0] - 1, a.shape[1], b.shape[1]), jnp.bfloat16),
        input_spec=(_TOKEN_MAJOR, _TOKEN_MAJOR, _SCALE_SPEC, _SCALE_SPEC, _INDEX_SPEC),
        output_spec=(_WGRAD_OUT_SPEC,),
        use_static_tensors=False,
    )
    return call(a, b, blocked_scales(ae.T[None]), blocked_scales(be.T[None]), cu)


def _time(fn, args, iters=10, reps=5):
    jitted = jax.jit(fn)
    jax.block_until_ready(jitted(*args))
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        for _ in range(iters):
            out = jitted(*args)
        jax.block_until_ready(out)
        samples.append((time.perf_counter() - start) / iters)
    return statistics.median(samples) * 1e3


def main():
    rows, cap, e, h, i = 262_144, 301_568, 3, 3072, 4608
    sizes = np.full(e, rows // e)
    sizes[-1] += rows - sizes.sum()
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(sizes)]), jnp.int32)
    blocks = token_scale_blocks(cu, cap)
    k = jax.random.split(jax.random.PRNGKey(0), 3)
    x = jax.random.normal(k[0], (cap, h), jnp.bfloat16)
    d = jax.random.normal(k[1], (cap, i), jnp.bfloat16)
    other = jax.random.normal(k[2], (cap, i), jnp.bfloat16)
    qx, qd = jax.block_until_ready(
        jax.jit(lambda a, b: (quantize_token_blocks(a, cu), quantize_token_blocks(b, cu)))(x, d)
    )
    rows_q = jax.block_until_ready(jax.jit(quantize_rows)(other))

    def quant(v, s):
        return requantize_row_blocks(v, s, *blocks)

    print(jax.devices()[0].device_kind, flush=True)
    t_q = _time(quant, rows_q)
    print(f"requant d_pre alone (one stream, all SMs): {t_q:.3f} ms", flush=True)
    for mac in (76, 72, 68, 64, 56):

        def seq(qx, qd, v, s, mac=mac):
            return _gemm(qx, qd, cu, mac=mac), quant(v, s)

        def gemm_only(qx, qd, mac=mac):
            return _gemm(qx, qd, cu, mac=mac)

        def par(qx, qd, v, s, mac=mac):
            q = compute_on(quant, compute_type="gpu_stream:1", out_memory_spaces=jax.memory.Space.Device)(v, s)
            return _gemm(qx, qd, cu, mac=mac), q

        t_g = _time(gemm_only, (qx, qd))
        t_s = _time(seq, (qx, qd, *rows_q))
        try:
            t_p = _time(par, (qx, qd, *rows_q))
        except Exception as exc:
            print(f"stream annotation failed: {type(exc).__name__}: {str(exc)[:500]}", flush=True)
            t_p = float("nan")
        print(
            f"dw_up gemm (2,1,1) mac {mac:3d}: gemm {t_g:.3f}  gemm;quant {t_s:.3f}  gemm||quant {t_p:.3f} ms "
            f"(hidden {t_s - t_p:+.3f})",
            flush=True,
        )


if __name__ == "__main__":
    main()
