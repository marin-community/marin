# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""v2 fused MXFP8 + 2:4 compress kernels (strided loads, generated straight-line code): correctness and kernel time."""

import sys
import traceback

import sp24_triton as sp
import sp24_v2 as v2
import torch
from bench_torch import emit, run

DEV = "cuda"
BF = torch.bfloat16
F = 4608


def decompress(v, meta, e, K, packed_meta):
    """2:4-along-K compressed (64-element scales) -> dense fp32 [M, K]."""
    M = v.shape[0]
    vals = (v.float().view(M, K // 64, 32) * torch.exp2(e.float() - 127)[:, :, None]).view(M, K // 4, 2)
    nib = torch.stack([meta & 0xF, meta >> 4], dim=-1).view(M, K // 4) if packed_meta else meta.view(M, K // 4)
    nib = nib.long()
    out = torch.zeros(M, K // 4, 4, device=v.device)
    out.scatter_(2, (nib & 3)[..., None], vals[..., :1])
    out.scatter_(2, (nib >> 2)[..., None], vals[..., 1:])
    return out.view(M, K)


def check():
    torch.manual_seed(0)
    pats = sp.patterns_tensor(DEV)
    for name, x in (
        ("relu2", torch.relu(torch.randn(512, 1024, device=DEV, dtype=BF)).square()),
        ("signed", torch.randn(512, 1024, device=DEV, dtype=BF) * 3),
    ):
        v, m, e = v2.sp24_rows(x)
        dense = decompress(v, m, e, 1024, True)
        ref = x.float() * sp.ref_mask_rows(x)
        (rv, rm, re), (cv, cm, ce) = v2.sp24t(x)
        dr = decompress(rv, rm, re, 1024, False)
        dc = decompress(cv.t().contiguous(), cm.t().contiguous(), ce.t().contiguous(), 512, False).t()
        mt = sp.ref_mask_t(x, pats)
        reft = x.float() * mt
        kept_opt = float((x.float().abs() * mt).sum())
        kept_kernel = float((x.float().abs() * (dr != 0)).sum())
        emit(
            name="v2_check",
            data=name,
            rows_relerr=float((dense - ref).norm() / ref.norm()),
            rows_mask_match=float(((dense != 0) == (ref != 0)).float().mean()),
            t_rows_relerr=float((dr - reft).norm() / reft.norm()),
            t_cols_relerr=float((dc - reft).norm() / reft.norm()),
            t_rows_eq_cols=float(((dr != 0) == (dc != 0)).float().mean()),
            t_rows24=bool(((dr != 0).view(512, -1, 4).sum(-1) <= 2).all()),
            t_cols24=bool(((dc != 0).t().reshape(1024, -1, 4).sum(-1) <= 2).all()),
            t_kept_mass_kernel_over_opt=kept_kernel / kept_opt,
        )


def bench(R):
    meta = dict(R=R, K=F)
    x = torch.relu(torch.randn(R, F, device=DEV, dtype=BF)).square()
    res = {}
    res["triton_quant_rows"] = run("triton_quant_rows", lambda: sp.quant_rows(x), **meta)
    res["triton_quant_rows_cols"] = run("triton_quant_rows_cols", lambda: sp.quant_rows_cols(x), **meta)
    for bm, nw in ((32, 4), (64, 4), (128, 4), (128, 8)):
        res[f"v2_sp24_rows_bm{bm}w{nw}"] = run(
            f"v2_sp24_rows_bm{bm}w{nw}", lambda bm=bm, nw=nw: v2.sp24_rows(x, bm, nw), **meta
        )
    for nw in (4, 8):
        res[f"v2_sp24t_w{nw}"] = run(f"v2_sp24t_w{nw}", lambda nw=nw: v2.sp24t(x, nw), **meta)
    res["copy_bf16"] = run("copy_bf16", lambda: x.clone(), **meta)
    emit(name="v2_summary", **meta, **res)
    torch.cuda.empty_cache()


def main():
    try:
        check()
    except Exception:
        emit(name="v2_check", error=traceback.format_exc()[-2500:])
    for R in (24576, 98304, 393216, 655872):
        try:
            bench(R)
        except Exception:
            emit(name="v2_summary", R=R, error=traceback.format_exc()[-2500:])


if __name__ == "__main__":
    sys.exit(main())
