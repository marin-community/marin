# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Which rounding sequence do the short-conv implementations follow? (one GPU, no segments)"""

import json

import jax
import jax.numpy as jnp
import ml_dtypes
import numpy as np
from levanter.kernels.pallas.short_conv import short_conv

BF16 = ml_dtypes.bfloat16


def _r(a):
    return np.asarray(a, np.float32).astype(BF16).astype(np.float32)


def _shift(x, lag):
    out = np.zeros_like(x)
    out[:, lag:] = x[:, : x.shape[1] - lag]
    return out


def main():
    batch, seq, channels = 2, 64, 1024
    keys = jax.random.split(jax.random.key(0), 2)
    x = jax.random.normal(keys[0], (batch, seq, channels), jnp.bfloat16)
    w = (jax.random.normal(keys[1], (4, channels)) * 0.5).astype(jnp.bfloat16)
    xf = np.asarray(x, np.float32)
    wf = np.asarray(w, np.float32)
    per_op = _r(wf[0] * xf)
    f32_sum_rounded_products = _r(wf[0] * xf)
    all_f32 = wf[0] * xf
    fma_chain = wf[0] * xf
    for lag in range(1, 4):
        per_op = _r(per_op + _r(wf[lag] * _shift(xf, lag)))
        f32_sum_rounded_products = f32_sum_rounded_products + _r(wf[lag] * _shift(xf, lag))
        all_f32 = all_f32 + wf[lag] * _shift(xf, lag)
        fma_chain = _r(fma_chain) + wf[lag] * _shift(xf, lag)
    candidates = dict(
        per_op=per_op,
        f32_sum_rounded_products=_r(f32_sum_rounded_products),
        all_f32=_r(all_f32),
        unrounded_product_then_round=_r(fma_chain),
    )
    for impl in ("reference", "pallas_gpu", "triton_gpu"):
        out = np.asarray(jax.jit(lambda w, x, i=impl: short_conv(w, x, None, implementation=i))(w, x), np.float32)
        print(json.dumps({impl: {k: int(np.sum(out != v)) for k, v in candidates.items()}}), flush=True)


if __name__ == "__main__":
    main()
