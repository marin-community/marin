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
import socket
import subprocess
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cutlass
import cutlass.cute as cute
import cutlass.jax as cjax
import jax
import jax.numpy as jnp
from mxfp8_grouped.adapter import _as_gmem_tensor, ensure_blackwell_arch, quantize_mxfp8, quantize_mxfp8_tokens
from mxfp8_grouped.quantize_cute import (
    _e8m0_round_up,
    _group_max,
    _store_ssa_as_e4m3,
    dual_quantize_mxfp8_cute,
)

M, K = 512, 128
TILE = 8  # per-thread tile; 256 threads as (32, 8)


def _probe_launcher(level: int, use_asm: bool = True):
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
            if cutlass.const_expr(use_asm):
                _store_ssa_as_e4m3(q, frag_q)
            else:
                frag_q.store(q.to(frag_q.element_type))
            cute.autovec_copy(frag_q, gQ)

    return ProbeKernel()


def run_probe(level: int, name: str, use_asm: bool = True):
    x = jax.random.normal(jax.random.PRNGKey(0), (M, K), dtype=jnp.bfloat16)
    ts = cjax.TensorSpec
    call = cjax.cutlass_call(
        _probe_launcher(level, use_asm),
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
        if level == 1:
            # Byte-order check for the inline-asm e4m3x2 pack (scale == 1).
            ref = x.astype(jnp.float32).astype(jnp.float8_e4m3fn)
            u_out = jax.lax.bitcast_convert_type(out[0], jnp.uint8)
            u_ref = jax.lax.bitcast_convert_type(ref, jnp.uint8)
            n_diff = int(jnp.sum(u_out != u_ref))
            n_diff_sw = int(jnp.sum(u_out != u_ref.reshape(-1, 2)[:, ::-1].reshape(u_ref.shape)))
            print(f"  {name} vs jnp cvt: {n_diff}/{u_ref.size} differ (pair-swapped: {n_diff_sw})", flush=True)
    except Exception as e:
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
        q_row, s_row_t, q_col, s_col = out
        qr_ref, sr_ref = jax.jit(quantize_mxfp8)(x)
        qc_ref, sc_ref = jax.jit(quantize_mxfp8_tokens)(x)
        diffs = {
            "q_row": int(
                jnp.sum(
                    jax.lax.bitcast_convert_type(q_row, jnp.uint8) != jax.lax.bitcast_convert_type(qr_ref, jnp.uint8)
                )
            ),
            "s_row": int(jnp.sum(s_row_t.T != sr_ref)),
            "q_col": int(
                jnp.sum(
                    jax.lax.bitcast_convert_type(q_col, jnp.uint8) != jax.lax.bitcast_convert_type(qc_ref, jnp.uint8)
                )
            ),
            "s_col": int(jnp.sum(s_col != sc_ref)),
        }
        print(f"PASS v5_full ({m}x{k}): byte diffs {diffs}", flush=True)
    except Exception as e:
        print(f"FAIL v5_full ({m}x{k}): {type(e).__name__}: {_fail_detail(e)}", flush=True)


def print_env():
    """Node/toolchain diagnostics: the same probe PASSes and FAILs across jobs
    (g3 all-pass vs g4 v1-fail), so suspicion is node- or toolchain-resolution
    dependence of the libNVVM the DSL invokes."""
    print(f"hostname: {socket.gethostname()}", flush=True)
    for cmd in (
        ["nvidia-smi", "--query-gpu=driver_version,name", "--format=csv,noheader"],
        ["sh", "-c", "ls -d /usr/local/cuda* 2>/dev/null"],
        ["sh", "-c", "echo LD_LIBRARY_PATH=$LD_LIBRARY_PATH; echo CUDA_TOOLKIT_PATH=$CUDA_TOOLKIT_PATH"],
        ["sh", "-c", "which ptxas; ptxas --version | tail -1"],
    ):
        try:
            print(subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip(), flush=True)
        except Exception as e:
            print(f"(diag {cmd[0]} failed: {e})", flush=True)
    try:
        from cuda import pathfinder  # noqa: PLC0415 - GPU-only optional dep

        for lib in ("nvvm",):
            print(f"libnvvm: {pathfinder.load_nvidia_dynamic_lib(lib).abs_path}", flush=True)
    except Exception as e:
        print(f"(pathfinder failed: {e})", flush=True)


def main():
    ensure_blackwell_arch()
    print(f"device: {jax.devices()[0].device_kind}, cutlass {cutlass.__version__}", flush=True)
    print_env()
    run_probe(1, "v1_cvt_intrinsic", use_asm=False)  # control: expected FAIL on cluster libNVVM
    for level, name in ((1, "v1_asm"), (2, "v2_shuffle"), (3, "v3_e8m0"), (4, "v4_scalebyte")):
        run_probe(level, name)
    for m, k in ((512, 128), (8192, 2560), (262144, 2560)):
        run_full(m, k)


if __name__ == "__main__":
    main()
