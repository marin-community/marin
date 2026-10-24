# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""One GB200: kernel-only (CUPTI via torch.profiler) times of dense vs 2:4-sparse GEMMs at the lc1 GB200 expert shapes,
plus the operand-building (quantize / prune / compress) kernels. Prints one ``RESULT {json}`` line per measurement."""

import json
import sys
import traceback

import sp24_triton as sp
import torch
from torch.profiler import ProfilerActivity, profile

DEV = "cuda"
BF = torch.bfloat16
FP8 = torch.float8_e4m3fn
ITERS = 20
L, F, H = 3072, 4608, 6144  # latent (expert in/out), routed d_ff, full width (latent_out_full_layers)
M_LIST = (4096, 16384, 65536, 109312)  # rows per expert; 109312 ~ 655360 / 6 routed rows per GPU / experts per GPU


def emit(**kw):
    print("RESULT " + json.dumps(kw), flush=True)


def kernel_us(fn, iters=ITERS):
    """Mean summed device-kernel time per call and the top kernels."""
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CUDA]) as prof:
        for _ in range(iters):
            fn()
        torch.cuda.synchronize()
    total, names = 0.0, {}
    for e in prof.events():
        if e.device_type == torch.autograd.DeviceType.CUDA:
            total += e.device_time
            names[e.name[:80]] = names.get(e.name[:80], 0.0) + e.device_time
    top = sorted(names.items(), key=lambda kv: -kv[1])[:3]
    return total / iters, [(n, round(v / iters, 1)) for n, v in top]


def to_blocked(s):
    rows, cols = s.shape
    rb, cb = -(-rows // 128), -(-cols // 4)
    p = torch.zeros(rb * 128, cb * 4, dtype=s.dtype, device=s.device)
    p[:rows, :cols] = s
    blocks = p.view(rb, 128, cb, 4).permute(0, 2, 1, 3)
    return blocks.reshape(-1, 4, 32, 4).transpose(1, 2).reshape(-1, 32, 16).flatten()


def run(name, fn, **meta):
    try:
        us, top = kernel_us(fn)
        emit(name=name, us=us, top=top, **meta)
        return us
    except Exception as e:
        emit(name=name, error=f"{type(e).__name__}: {e}"[:400], **meta)
        return None


def masked_rows(a):
    return a * sp.ref_mask_rows(a)


def bench_gemm(shape_name, Mg, Kg, Ng):
    """``A[Mg, Kg] @ B[Kg, Ng]``; the sparse variants are 2:4 along Kg on A."""
    meta = dict(shape=shape_name, M=Mg, K=Kg, N=Ng)
    a = torch.relu(torch.randn(Mg, Kg, device=DEV, dtype=BF)).square()
    a = masked_rows(a)
    b = torch.randn(Kg, Ng, device=DEV, dtype=BF) / Kg**0.5
    flops = 2 * Mg * Kg * Ng
    out = {}
    out["bf16_dense"] = run("bf16_dense", lambda: a @ b, **meta)
    a8 = a.to(FP8)
    b8_col = b.t().contiguous().to(FP8).t()  # column-major B for cuBLASLt fp8
    one = torch.ones((), device=DEV, dtype=torch.float32)
    out["fp8_dense_tensorwise"] = run(
        "fp8_dense_tensorwise", lambda: torch._scaled_mm(a8, b8_col, one, one, out_dtype=BF), **meta
    )
    sa = to_blocked(torch.full((Mg, Kg // 32), 127, device=DEV, dtype=torch.uint8)).view(torch.float8_e8m0fnu)
    sb = to_blocked(torch.full((Ng, Kg // 32), 127, device=DEV, dtype=torch.uint8)).view(torch.float8_e8m0fnu)
    out["mxfp8_dense_cublas"] = run(
        "mxfp8_dense_cublas", lambda: torch._scaled_mm(a8, b8_col, sa, sb, out_dtype=BF), **meta
    )
    for dt_name, aa, bb in (("bf16", a, b), ("fp8", a8, b8_col)):
        try:
            comp = torch._cslt_compress(aa)
            try:
                alg = torch._cslt_sparse_mm_search(comp, bb)
            except Exception:
                alg = 0
            kw = dict(out_dtype=BF) if dt_name == "fp8" else {}
            fn = lambda comp=comp, bb=bb, alg=alg, kw=kw: torch._cslt_sparse_mm(comp, bb, alg_id=alg, **kw)  # noqa: E731
            ref = a.float() @ b.float()
            err = float((fn().float() - ref).abs().max() / ref.abs().max())
            out[f"{dt_name}_sparse_cslt"] = run(f"{dt_name}_sparse_cslt", fn, alg=alg, relerr=err, **meta)
        except Exception as e:
            emit(name=f"{dt_name}_sparse_cslt", error=f"{type(e).__name__}: {e}"[:400], **meta)
    try:
        import cslt_mx  # noqa: PLC0415

        ref = a.float() @ b.float()
        for kind in ("bf16", "fp8", "mxfp8"):
            try:
                mm = cslt_mx.SparseMM(Mg, Kg, Ng, kind)
                mm.load(a, b)
                try:
                    mm.search()
                except Exception as e:
                    emit(name=f"nvm_{kind}_search", error=f"{type(e).__name__}: {e}"[:300], **meta)
                err = float((mm().float() - ref).abs().max() / ref.abs().max())
                out[f"nvm_{kind}_sparse"] = run(f"nvm_{kind}_sparse", mm, relerr=err, **meta)
                out[f"nvm_{kind}_compress"] = run(f"nvm_{kind}_compress", mm.compress, **meta)
                del mm
            except Exception as e:
                emit(name=f"nvm_{kind}_sparse", error=f"{type(e).__name__}: {e}"[:400], **meta)
    except Exception as e:
        emit(name="nvm_import", error=f"{type(e).__name__}: {e}"[:400])
    emit(name="gemm_summary", flops=flops, **meta, **{k: v for k, v in out.items()})
    torch.cuda.empty_cache()


def check_triton():
    torch.manual_seed(0)
    pats = sp.patterns_tensor(DEV)
    x = torch.randn(256, 512, device=DEV, dtype=BF) * 3
    v, e = sp.quant_rows(x)
    e_ref = sp.ref_exponent(x.float().view(256, 16, 32).abs().amax(-1))
    v_ref = (x.float().view(256, 16, 32) / torch.exp2(e_ref.float() - 127)[..., None]).to(FP8).view(256, 512)
    ok_q = bool((e.int() == e_ref).all()) and bool((v.view(torch.uint8) == v_ref.view(torch.uint8)).all())
    (rv, re), (_cv, _ce) = sp.quant_rows_cols(x)
    ok_rc = bool((rv.view(torch.uint8) == v_ref.view(torch.uint8)).all()) and bool((re.int() == e_ref).all())
    v2, m2, e2 = sp.sp24_rows(x)
    dense = sp.decompress_rows(v2, m2, e2, 512)
    mref = sp.ref_mask_rows(x)
    xr = x.float() * mref
    ok_sp_mask = bool(((dense != 0) <= mref).all())
    relerr_sp = float((dense - xr).norm() / xr.norm())
    (tv, tm, te), (tcv, tcm, tce) = sp.sp24t(x, pats)
    dense_t = sp.decompress_rows(tv, tm, te, 512)
    mt = sp.ref_mask_t(x, pats)
    xt = x.float() * mt
    relerr_t_rows = float((dense_t - xt).norm() / xt.norm())
    # cols: decompress the transposed problem
    dense_tc = sp.decompress_rows(tcv.t().contiguous(), tcm.t().contiguous(), tce.t().contiguous(), 256).t()
    relerr_t_cols = float((dense_tc - xt).norm() / xt.norm())
    emit(
        name="triton_check",
        quant_rows_bitexact=ok_q,
        rows_cols_bitexact=ok_rc,
        sp24_mask_subset=ok_sp_mask,
        sp24_relerr=relerr_sp,
        sp24t_rows_relerr=relerr_t_rows,
        sp24t_cols_relerr=relerr_t_cols,
        sp24t_rows_24=bool(((mt.view(256, -1, 4).sum(-1)) == 2).all()),
        sp24t_cols_24=bool(((mt.t().reshape(512, -1, 4).sum(-1)) == 2).all()),
    )


def bench_compress(R):
    """Operand building on post-activations [R, F] bf16 (bandwidth bound)."""
    meta = dict(R=R, K=F)
    x = torch.relu(torch.randn(R, F, device=DEV, dtype=BF)).square()
    pats = sp.patterns_tensor(DEV)
    gb_in = R * F * 2 / 1e9
    res = {}
    res["triton_quant_rows"] = run("triton_quant_rows", lambda: sp.quant_rows(x), **meta)
    res["triton_quant_rows_cols"] = run("triton_quant_rows_cols", lambda: sp.quant_rows_cols(x), **meta)
    for bm, bk, nw in ((16, 512, 4), (32, 256, 4), (64, 128, 4), (16, 256, 2)):
        res[f"triton_sp24_rows_{bm}x{bk}w{nw}"] = run(
            f"triton_sp24_rows_{bm}x{bk}w{nw}", lambda bm=bm, bk=bk, nw=nw: sp.sp24_rows(x, bm, bk, nw), **meta
        )
    for bm, bk, nw in ((64, 128, 4), (64, 64, 4), (64, 128, 8), (128, 64, 8)):
        res[f"triton_sp24t_{bm}x{bk}w{nw}"] = run(
            f"triton_sp24t_{bm}x{bk}w{nw}", lambda bm=bm, bk=bk, nw=nw: sp.sp24t(x, pats, bm, bk, nw), **meta
        )
    xm = masked_rows(x) if R <= 131072 else None
    if xm is not None:
        res["cslt_compress_bf16"] = run("cslt_compress_bf16", lambda: torch._cslt_compress(xm), **meta)
        x8 = xm.to(FP8)
        res["cslt_compress_fp8"] = run("cslt_compress_fp8", lambda: torch._cslt_compress(x8), **meta)
        res["torch_tile_prune_both"] = run(
            "torch_tile_prune_both",
            lambda: torch._sparse_semi_structured_tile(x, algorithm="", use_cutlass=False),
            **meta,
        )
        res["copy_bf16"] = run("copy_bf16", lambda: x.clone(), **meta)
    emit(name="compress_summary", gb_in=gb_in, **meta, **res)
    torch.cuda.empty_cache()


def main():
    emit(
        name="env",
        torch=torch.__version__,
        cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(),
        cusparselt=torch.backends.cusparselt.version(),
        triton=__import__("triton").__version__,
    )
    try:
        import cslt_mx  # noqa: PLC0415

        emit(name="nvmath_cusparselt", version=cslt_mx.version())
    except Exception as e:
        emit(name="nvmath_cusparselt", error=f"{type(e).__name__}: {e}"[:400])
    try:
        check_triton()
    except Exception:
        emit(name="triton_check", error=traceback.format_exc()[-2500:])
    for R in (24576, 98304, 393216, 655872):
        try:
            bench_compress(R)
        except Exception:
            emit(name="compress_summary", R=R, error=traceback.format_exc()[-2500:])
    for M in M_LIST:
        for shape_name, dims in (("up", (M, L, F)), ("down", (M, F, L)), ("wgrad", (F, M, L))):
            try:
                bench_gemm(shape_name, *dims)
            except Exception:
                emit(name="gemm_summary", shape=shape_name, M=M, error=traceback.format_exc()[-1500:])
    M = M_LIST[-1]
    for shape_name, dims in (("up_full", (M, H, F)), ("down_full", (M, F, H)), ("wgrad_full", (F, M, H))):
        try:
            bench_gemm(shape_name, *dims)
        except Exception:
            emit(name="gemm_summary", shape=shape_name, M=M, error=traceback.format_exc()[-1500:])


if __name__ == "__main__":
    sys.exit(main())
