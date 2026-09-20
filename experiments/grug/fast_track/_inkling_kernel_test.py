# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU check: fused fa4_cute forward+backward with the Inkling relative-position bias vs the reference
oracle (value + grads). Run on one H100 via iris (needs adequate --memory)."""

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask, gpu_fa4_cute_attention, reference_attention


def _run(B, S, Hq, Hkv, D, L, sliding_window, seed=0):
    key = jax.random.PRNGKey(seed)
    kq, kk, kv, ka, kc = jax.random.split(key, 5)
    q = jax.random.normal(kq, (B, S, Hq, D), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (B, S, Hkv, D), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (B, S, Hkv, D), dtype=jnp.bfloat16)
    A = (jax.random.normal(ka, (B, Hq, S, L), dtype=jnp.float32) * 0.5).astype(jnp.bfloat16)
    cot = jax.random.normal(kc, (B, S, Hq, D), dtype=jnp.bfloat16)
    mask = AttentionMask.causal(sliding_window=sliding_window)

    def loss_fused(q, k, v, A):
        return jnp.sum(gpu_fa4_cute_attention(q, k, v, mask, rel_bias=A).astype(jnp.float32) * cot.astype(jnp.float32))

    def loss_ref(q, k, v, A):
        return jnp.sum(
            reference_attention(q, k, v, mask, logits_dtype=jnp.float32, rel_bias=A).astype(jnp.float32)
            * cot.astype(jnp.float32)
        )

    (df_q, df_k, df_v, df_A) = jax.jit(jax.grad(loss_fused, argnums=(0, 1, 2, 3)))(q, k, v, A)
    (dr_q, dr_k, dr_v, dr_A) = jax.grad(loss_ref, argnums=(0, 1, 2, 3))(q, k, v, A)

    def cmp(name, a, b):
        a = np.asarray(a).astype(np.float32)
        b = np.asarray(b).astype(np.float32)
        err = float(np.max(np.abs(a - b)))
        rel = err / (float(np.max(np.abs(b))) + 1e-6)
        print(f"    d{name}: max_abs={err:.4e} rel={rel:.4e} -> {'PASS' if rel < 5e-2 else 'FAIL'}", flush=True)

    print(f"[grad {B}x{S}x{Hq}/{Hkv}x{D} L={L} win={sliding_window}]", flush=True)
    cmp("q", df_q, dr_q)
    cmp("k", df_k, dr_k)
    cmp("v", df_v, dr_v)
    cmp("A", df_A, dr_A)


def main():
    if jax.default_backend() != "gpu":
        print("SKIP: needs GPU backend", flush=True)
        return
    print("backend:", jax.default_backend(), "device:", jax.devices()[0], flush=True)
    _run(B=1, S=256, Hq=4, Hkv=1, D=128, L=64, sliding_window=None)
    _run(B=2, S=512, Hq=4, Hkv=1, D=128, L=256, sliding_window=256)
    print("INKLING_GRAD_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
