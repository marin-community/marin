# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""MXFP8-002c debug: bisect which kernel construct trips the pod's libNVVM.

The dual quantizer compiles clean on a GPU-less x86 box (PTX-only conditional
SASS) but fails on the GB200 pod with ``NVVM_ERROR_COMPILATION: unsupported
operation`` (jobs 002c-g1/g2). The vendored GEMM kernel compiles fine on the
same pod, so one of the quantizer's constructs is the culprit. This script
compiles+runs a ladder of stripped variants, each adding one construct:

  v1_cvt        bf16 read -> f32 -> e4m3 cvt -> store (packfloat path)
  v2_shuffle    + scalar shuffle_sync_bfly / fmax combine
  v3_e8m0       + f32<->i32 bitcasts and integer bit-twiddle + scaled store
  v4_scalebyte  + predicated 1-D scale byte store via sliced local_tile
  v5_full       the real dual-quantize kernel

Prints PASS/FAIL per variant; run on 1x GB200.
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp
from mxfp8_grouped.adapter import _as_gmem_tensor, ensure_blackwell_arch
from mxfp8_grouped.quantize_cute import _e8m0_round_up, _group_max, dual_quantize_mxfp8_cute

M, K = 512, 128
TILE = 8  # per-thread tile; 256 threads as (32, 8)


def _probe_launcher(level: int):
    class ProbeKernel:
        @cute.jit
        def __call__(self, stream, mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor):
            mX = _as_gmem_tensor(mX)
            mQ = _as_gmem_tensor(mQ)
            mS = _as_gmem_tensor(mS)
            grid = (mX.shape[1] // 64, mX.shape[0] // 256, 1)
            self.kernel(mX, mQ, mS).launch(grid=grid, block=[256, 1, 1], stream=stream)

        @cute.kernel
        def kernel(self, mX: cute.Tensor, mQ: cute.Tensor, mS: cute.Tensor):
            tidx, _, _ = cute.arch.thread_idx()
            bn, bm, _ = cute.arch.block_idx()
            tr = tidx // 8
            tc = tidx % 8
            row_blk = bm * 32 + tr
            col_blk = bn * 8 + tc

            gX = cute.local_tile(mX, (TILE, TILE), (row_blk, col_blk))
            frag_x = cute.make_fragment_like(gX)
            cute.autovec_copy(gX, frag_x)
            xf = frag_x.load().to(cutlass.Float32)

            scale = cutlass.Float32(1.0)
            if cutlass.const_expr(level >= 2):
                rp = xf.reduce(cute.ReductionOp.MAX, 0.0, (None, 1))
                r_amax = cute.make_fragment(TILE, cutlass.Float32)
                r_amax.store(rp)
                scale = _group_max(r_amax[0], (1, 2))
            if cutlass.const_expr(level >= 3):
                e = _e8m0_round_up(scale)
                scale = ((254 - e) << 23).bitcast(cutlass.Float32)
            if cutlass.const_expr(level >= 4):
                if tc % 4 == 0:
                    sb = cute.make_fragment(TILE, cutlass.Uint8)
                    for i in cutlass.range_constexpr(TILE):
                        sb[i] = cutlass.Uint8(1)
                    s_row = mS[(bn * 2 + tc // 4, None)]
                    gS = cute.local_tile(s_row, (TILE,), (row_blk,))
                    cute.autovec_copy(sb, gS)

            q = xf * scale
            gQ = cute.local_tile(mQ, (TILE, TILE), (row_blk, col_blk))
            frag_q = cute.make_fragment_like(gQ)
            frag_q.store(q.to(frag_q.element_type))
            cute.autovec_copy(frag_q, gQ)

    return ProbeKernel()


def run_probe(level: int, name: str):
    x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.bfloat16)
    ts = cjax.TensorSpec
    call = cjax.cutlass_call(
        _probe_launcher(level),
        output_shape_dtype=(
            jax.ShapeDtypeStruct((M, K), jnp.float8_e4m3fn),
            jax.ShapeDtypeStruct((K // 32, M), jnp.uint8),
        ),
        input_spec=(ts(mode=(0, 1), divisibility=(1, 16), static=True),),
        output_spec=(ts(mode=(0, 1), divisibility=(1, 16), static=True), ts(mode=(0, 1), static=True)),
        use_static_tensors=True,
        compile_options=(cute.GPUArch("sm_100a"),),
    )
    try:
        out = jax.block_until_ready(jax.jit(lambda t: call(t))(x))
        print(f"PASS {name}: q mean {float(jnp.mean(jnp.abs(out[0].astype(jnp.float32)))):.4f}", flush=True)
    except Exception as e:  # noqa: BLE001 - diagnostic ladder must continue
        msg = str(e).splitlines()
        head = " / ".join(msg[:3])
        print(f"FAIL {name}: {type(e).__name__}: {head}", flush=True)
        if "--verbose" in sys.argv:
            traceback.print_exc()


def _fail_detail(e: BaseException) -> str:
    """Walk the cause chain for the extracted libNVVM error."""
    seen = []
    cur: BaseException | None = e
    while cur is not None:
        nvvm = getattr(cur, "nvvm_error", None)
        if nvvm:
            seen.append(str(nvvm).splitlines()[0])
        cur = cur.__cause__ or cur.__context__
    return " | ".join(seen) if seen else " / ".join(str(e).splitlines()[:3])


def run_full(m: int, k: int):
    x = jax.random.normal(jax.random.PRNGKey(0), (m, k), dtype=jnp.bfloat16)
    try:
        out = jax.block_until_ready(jax.jit(dual_quantize_mxfp8_cute)(x))
        print(f"PASS v5_full ({m}x{k}): outputs {[o.shape for o in out]}", flush=True)
    except Exception as e:  # noqa: BLE001 - diagnostic ladder must continue
        print(f"FAIL v5_full ({m}x{k}): {type(e).__name__}: {_fail_detail(e)}", flush=True)


def main():
    ensure_blackwell_arch()
    print(f"device: {jax.devices()[0].device_kind}, cutlass {cutlass.__version__}", flush=True)
    for level, name in ((1, "v1_cvt"), (2, "v2_shuffle"), (3, "v3_e8m0"), (4, "v4_scalebyte")):
        run_probe(level, name)
    # Shape scan: g2 failed at (8192, 2560) while (512, 128) passes (job g3).
    for m, k in ((512, 128), (512, 2560), (8192, 128), (2048, 2560), (8192, 1280), (8192, 2560), (262144, 2560)):
        run_full(m, k)


if __name__ == "__main__":
    main()
