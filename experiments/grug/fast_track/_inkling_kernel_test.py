# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""GPU isolation check for the Inkling relative-position fused forward. Run on one H100 via iris."""


import jax
import jax.numpy as jnp
import numpy as np
from levanter.grug.attention import AttentionMask, gpu_fa4_cute_attention, reference_attention


def _base_case():
    key = jax.random.PRNGKey(0)
    kq, kk, kv = jax.random.split(key, 3)
    q = jax.random.normal(kq, (1, 256, 4, 128), dtype=jnp.bfloat16)
    k = jax.random.normal(kk, (1, 256, 1, 128), dtype=jnp.bfloat16)
    v = jax.random.normal(kv, (1, 256, 1, 128), dtype=jnp.bfloat16)
    mask = AttentionMask.causal()
    out = np.asarray(jax.block_until_ready(gpu_fa4_cute_attention(q, k, v, mask))).astype(np.float32)
    exp = np.asarray(reference_attention(q, k, v, mask, logits_dtype=jnp.float32)).astype(np.float32)
    err = float(np.max(np.abs(out - exp)))
    print(f"[base no-bias via wrapper] max_abs={err:.4e} -> {'PASS' if err < 7e-2 else 'FAIL'}", flush=True)


def main():
    if jax.default_backend() != "gpu":
        print("SKIP: needs GPU backend", flush=True)
        return
    print("backend:", jax.default_backend(), "device:", jax.devices()[0], flush=True)
    _base_case()
    print("INKLING_BASE_KERNEL_TEST_DONE", flush=True)


if __name__ == "__main__":
    main()
