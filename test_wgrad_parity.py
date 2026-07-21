"""Parity: quack_grouped_wgrad_gemm vs the XLA ragged_dot weight-grad it replaces.

Run on 1 B200. Covers even groups, heavily ragged groups (odd sizes incl. 1),
and empty experts (kernel must clear the accumulator, cf. gemm_sm100 clear_acc).
"""
import traceback

import jax
import jax.numpy as jnp
import numpy as np
from haliax.nn.ragged_dot import ragged_dot

from levanter.grug._moe.quack_moe_cute import quack_grouped_wgrad_gemm


def xla_ref(acts, grads, group_sizes, E, M, N):
    """The exact XLA weight-grad this replaces: vjp of ragged_dot w.r.t. w."""
    w0 = jnp.zeros((E, M, N), acts.dtype)
    (dw,) = jax.vjp(lambda w: ragged_dot(acts, w, group_sizes), w0)[1](grads)
    return dw


CASES = [
    ("even/dw13-shape", [2048] * 8, 2560, 5120),
    ("ragged", [1, 8191, 777, 4096, 33, 3000, 190, 96], 2560, 5120),
    ("empty-experts/dw2-shape", [4096, 0, 2048, 0, 8192, 512, 1024, 512], 1280, 2560),
]

failures = 0
for name, gs, M, N in CASES:
    TK, E = sum(gs), len(gs)
    k1, k2 = jax.random.split(jax.random.PRNGKey(abs(hash(name)) % 2**31))
    acts = (jax.random.normal(k1, (TK, M)) * 0.1).astype(jnp.bfloat16)
    grads = (jax.random.normal(k2, (TK, N)) * 0.1).astype(jnp.bfloat16)
    group_sizes = jnp.asarray(gs, jnp.int32)
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(gs)]), jnp.int32)
    try:
        dw = quack_grouped_wgrad_gemm(acts, grads, cu, E)
        ref = xla_ref(acts, grads, group_sizes, E, M, N)
        jax.block_until_ready((dw, ref))
        err = float(jnp.abs(dw.astype(jnp.float32) - ref.astype(jnp.float32)).max())
        rel = err / (float(jnp.abs(ref).max()) + 1e-6)
        status = "ok" if rel < 0.02 else "FAIL"
        if status == "FAIL":
            failures += 1
        print(f"{name}: max_abs {err:.5f} rel {rel:.5f} {status}", flush=True)
        # empty experts must be exactly zero
        for e, g in enumerate(gs):
            if g == 0:
                z = float(jnp.abs(dw[e]).max())
                print(f"  expert {e} empty: max|dw|={z} {'ok' if z == 0.0 else 'FAIL'}", flush=True)
                if z != 0.0:
                    failures += 1
    except Exception:
        traceback.print_exc()
        failures += 1
        print(f"{name}: EXCEPTION", flush=True)

print("WGRAD_PARITY_OK" if failures == 0 else f"WGRAD_PARITY_FAIL ({failures})", flush=True)
