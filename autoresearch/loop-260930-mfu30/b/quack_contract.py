# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Check that QuACK's grouped GEMMs never read rows past the last group boundary.

The ragged EP backend hands the expert MLP buffers whose rows past ``cu[-1]`` hold unspecified,
possibly non-finite, values. Each grouped GEMM here runs twice, once with those rows zero and
once with them NaN, and the active output rows (or the weight gradients) must match bitwise.

Usage (one SM100 GPU): python autoresearch/loop-260930-mfu30/b/quack_contract.py
"""

import json
import sys

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug._moe.quack_moe_cute import quack_gated_grouped_gemm, quack_grouped_gemm, quack_grouped_wgrad
from levanter.grug._moe.sonic_cute import _QUACK_GATED_KW, _QUACK_GROUPED_KW, _QUACK_WGRAD_KW

ROWS, HIDDEN, INTER, EXPERTS = 4096, 512, 512, 3
GROUPS = np.array([1000, 1357, 1100], dtype=np.int32)  # 3457 active rows of 4096


def _poison(a, active):
    return a.at[active:].set(jnp.nan), a.at[active:].set(0)


def main():
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(GROUPS)]), dtype=jnp.int32)
    active = int(cu[-1])
    keys = jax.random.split(jax.random.key(0), 6)
    bf16 = jnp.bfloat16
    x = jax.random.normal(keys[0], (ROWS, HIDDEN), bf16)
    w13 = (jax.random.normal(keys[1], (EXPERTS, HIDDEN, 2 * INTER)) * 0.05).astype(bf16)
    w2 = (jax.random.normal(keys[2], (EXPERTS, INTER, HIDDEN)) * 0.05).astype(bf16)
    dy = jax.random.normal(keys[3], (ROWS, HIDDEN), bf16)
    h = jax.random.normal(keys[4], (ROWS, INTER), bf16)
    d_gu = jax.random.normal(keys[5], (ROWS, 2 * INTER), bf16)

    checks = {}

    def active_rows_equal(name, fn, operand):
        nan_in, zero_in = _poison(operand, active)
        a, b = np.asarray(fn(nan_in), np.float32)[:active], np.asarray(fn(zero_in), np.float32)[:active]
        checks[name] = bool(np.array_equal(a, b) and np.isfinite(a).all())

    def all_equal(name, fn, *operands):
        poisoned = [_poison(op, active) for op in operands]
        a = np.asarray(fn(*[p[0] for p in poisoned]), np.float32)
        b = np.asarray(fn(*[p[1] for p in poisoned]), np.float32)
        checks[name] = bool(np.array_equal(a, b) and np.isfinite(a).all())

    gated = jax.jit(lambda x: quack_gated_grouped_gemm(x, w13, cu, return_preact=True, **_QUACK_GATED_KW)[1])
    active_rows_equal("gated_forward", gated, x)
    down = jax.jit(lambda h: quack_grouped_gemm(h, w2, cu, b_major="n", **_QUACK_GROUPED_KW))
    active_rows_equal("down_forward", down, h)
    dh = jax.jit(lambda dy: quack_grouped_gemm(dy, w2, cu, b_major="k", **_QUACK_GROUPED_KW))
    active_rows_equal("dh", dh, dy)
    dx = jax.jit(lambda d_gu: quack_grouped_gemm(d_gu, w13, cu, b_major="k", **_QUACK_GROUPED_KW))
    active_rows_equal("dx", dx, d_gu)
    all_equal("dw2", jax.jit(lambda h, dy: quack_grouped_wgrad(h, dy, cu, **_QUACK_WGRAD_KW)), h, dy)
    all_equal("dw13", jax.jit(lambda x, d_gu: quack_grouped_wgrad(x, d_gu, cu, **_QUACK_WGRAD_KW)), x, d_gu)

    print(json.dumps(dict(quack_contract=checks)), flush=True)
    sys.exit(0 if all(checks.values()) else 1)


if __name__ == "__main__":
    main()
