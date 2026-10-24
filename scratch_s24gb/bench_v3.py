# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""v3 (contiguous loads + in-register splits) vs v2 fused MXFP8 + 2:4 kernels: equality check and kernel time."""

import sys
import traceback

import sp24_triton as sp
import sp24_v2 as v2
import torch
from bench_torch import emit, run

DEV = "cuda"
F = 4608


def check():
    x = torch.randn(512, 1024, device=DEV, dtype=torch.bfloat16) * 3
    a = v2.sp24_rows(x)
    b = v2.sp24_rows(x, kernel=v2.sp24_rows_v3_kernel)
    eq_rows = all(bool((p.view(torch.uint8) == q.view(torch.uint8)).all()) for p, q in zip(a, b, strict=True))
    (a1, a2), (b1, b2) = v2.sp24t(x), v2.sp24t(x, kernel=v2.sp24t_v3_kernel)
    eq_t = all(bool((p.view(torch.uint8) == q.view(torch.uint8)).all()) for p, q in zip(a1 + a2, b1 + b2, strict=True))
    emit(name="v3_check", rows_equal_v2=eq_rows, t_equal_v2=eq_t)


def bench(R):
    meta = dict(R=R, K=F)
    x = torch.relu(torch.randn(R, F, device=DEV, dtype=torch.bfloat16)).square()
    res = {"triton_quant_rows": run("triton_quant_rows", lambda: sp.quant_rows(x), **meta)}
    for bm, nw in ((32, 4), (64, 4), (64, 8), (128, 8)):
        res[f"v3_sp24_rows_bm{bm}w{nw}"] = run(
            f"v3_sp24_rows_bm{bm}w{nw}",
            lambda bm=bm, nw=nw: v2.sp24_rows(x, bm, nw, kernel=v2.sp24_rows_v3_kernel),
            **meta,
        )
    for nw in (2, 4, 8):
        res[f"v3_sp24t_w{nw}"] = run(f"v3_sp24t_w{nw}", lambda nw=nw: v2.sp24t(x, nw, kernel=v2.sp24t_v3_kernel), **meta)
    res["v2_sp24t_w2"] = run("v2_sp24t_w2", lambda: v2.sp24t(x, 2), **meta)
    emit(name="v3_summary", **meta, **res)


def main():
    try:
        check()
    except Exception:
        emit(name="v3_check", error=traceback.format_exc()[-2500:])
    for R in (98304, 655872):
        try:
            bench(R)
        except Exception:
            emit(name="v3_summary", R=R, error=traceback.format_exc()[-2500:])


if __name__ == "__main__":
    sys.exit(main())
