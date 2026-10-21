# Copyright The Levanter Authors
# SPDX-License-Identifier: Apache-2.0

"""Interleaved A/B of the MXFP8 varlen-k wgrad GEMM cluster shape at the hero EP64 chunk (262144/301568 rows,
three experts, 3072 x 4608): rounds of (2,2,1) vs (4,1,1) vs (2,1,1), medians over rounds."""

import time

import jax
import jax.numpy as jnp
import numpy as np

from levanter.grug._moe.quack_mxfp8 import mxfp8_token_grouped_gemm, quantize_token_blocks


def main():
    rows, cap, e, h, i = 262_144, 301_568, 3, 3072, 4608
    sizes = np.full(e, rows // e)
    sizes[-1] += rows - sizes.sum()
    cu = jnp.asarray(np.concatenate([[0], np.cumsum(sizes)]), jnp.int32)
    k = jax.random.split(jax.random.PRNGKey(0), 4)
    ops = [jax.random.normal(kk, (cap, w), jnp.bfloat16) for kk, w in zip(k, (h, i, i, h))]
    qx, qd, qp, qdy = jax.block_until_ready(jax.jit(lambda *a: [quantize_token_blocks(t, cu) for t in a])(*ops))
    clusters = [(2, 2, 1), (4, 1, 1), (2, 1, 1), (4, 2, 1)]
    fns = {}
    for c in clusters:
        for name, a, b in (("dw_up", qx, qd), ("dw_down", qp, qdy)):
            f = jax.jit(lambda a, b, c=c: mxfp8_token_grouped_gemm(a, b, cu, cluster_mnk=c))
            jax.block_until_ready(f(a, b))
            fns[(name, c)] = (f, a, b)
    samples = {key: [] for key in fns}
    for _ in range(15):
        for key, (f, a, b) in fns.items():
            jax.block_until_ready(f(a, b))
            start = time.perf_counter()
            for _ in range(10):
                out = f(a, b)
            jax.block_until_ready(out)
            samples[key].append((time.perf_counter() - start) / 10 * 1e3)
    print(jax.devices()[0].device_kind)
    for key, s in samples.items():
        q = np.percentile(s, [25, 50, 75])
        print(f"{key[0]:8s} cluster {key[1]}: median {q[1]:.3f} ms (IQR {q[0]:.3f}-{q[2]:.3f})", flush=True)


if __name__ == "__main__":
    main()
