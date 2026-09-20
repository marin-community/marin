# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU check: fused fa4_cute forward with the Inkling relative-position bias vs the reference oracle.
Run on one H100 via iris (needs adequate --memory)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask, reference_attention
from levanter.grug.attention._fa4_cute import _segmented_kernel_config, _simple_causal_lower_bounds
from levanter.grug.attention._fa4_cute_backend import segmented_flash_attention_forward


def _run(B, S, Hq, Hkv, D, L, sliding_window, seed=0):
    key = jax.random.PRNGKey(seed)
    kq, kk, kv, ka = jax.random.split(key, 4)
    q = jax.random.normal(kq, (B, S, Hq, D), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (B, S, Hkv, D), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (B, S, Hkv, D), dtype=jnp.bfloat16)
    A = (jax.random.normal(ka, (B, Hq, S, L), dtype=jnp.float32) * 0.5).astype(jnp.bfloat16)
    scale = 1.0 / math.sqrt(D)
    lower_bounds, valid = _simple_causal_lower_bounds(batch_size=B, seq_len=S, sliding_window=sliding_window)
    cfg = _segmented_kernel_config(D)

    def fwd(q, k, v, A):
        out, _ = segmented_flash_attention_forward(
            q, k, v, lower_bounds, valid, A, softmax_scale=scale, kernel_config=cfg
        )
        return out

    out = np.asarray(jax.block_until_ready(jax.jit(fwd)(q, k, v, A))).astype(np.float32)
    mask = AttentionMask.causal(sliding_window=sliding_window)
    exp = np.asarray(reference_attention(q, k, v, mask, logits_dtype=jnp.float32, rel_bias=A)).astype(np.float32)
    err = float(np.max(np.abs(out - exp)))
    rel = err / (float(np.max(np.abs(exp))) + 1e-6)
    print(
        f"[bias {B}x{S}x{Hq}/{Hkv}x{D} L={L} win={sliding_window}] max_abs={err:.4e} rel={rel:.4e} "
        f"-> {'PASS' if err < 7e-2 else 'FAIL'}",
        flush=True,
    )
    return err


def main():
    if jax.default_backend() != "gpu":
        print("SKIP: needs GPU backend", flush=True)
        return
    print("backend:", jax.default_backend(), "device:", jax.devices()[0], flush=True)
    _run(B=1, S=256, Hq=4, Hkv=1, D=128, L=64, sliding_window=None)
    _run(B=1, S=256, Hq=4, Hkv=1, D=128, L=128, sliding_window=128)
    _run(B=2, S=512, Hq=4, Hkv=1, D=128, L=256, sliding_window=None)
    print("INKLING_FWD_BIAS_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
