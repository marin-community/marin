# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Where the Triton short conv differs from the Pallas kernel and the reference (small shape, one GPU)."""

import json

import jax
import jax.numpy as jnp
import numpy as np
from levanter.kernels.pallas.short_conv import short_conv, short_conv_reference


def main():
    batch, seq, channels = 2, 256, 2048
    keys = jax.random.split(jax.random.key(0), 3)
    x = jax.random.normal(keys[0], (batch, seq, channels), jnp.bfloat16)
    weight = (jax.random.normal(keys[1], (4, channels)) * 0.5).astype(jnp.bfloat16)
    ct = jax.random.normal(keys[2], (batch, seq, channels), jnp.bfloat16)
    outs = {}
    for impl in ("reference", "pallas_gpu", "triton_gpu"):
        f = lambda w, x, i=impl: short_conv(w, x, None, implementation=i)  # noqa: E731
        out, pull = jax.vjp(f, weight, x)
        dw, dx = pull(ct)
        outs[impl] = [np.asarray(a, np.float32) for a in (out, dx, dw)]
    eager_ref = np.asarray(short_conv_reference(weight, x, None), np.float32)
    for a, b in (("triton_gpu", "pallas_gpu"), ("triton_gpu", "reference"), ("pallas_gpu", "reference")):
        for k, name in enumerate(("out", "dx", "dw")):
            u, v = outs[a][k], outs[b][k]
            diff = np.abs(u - v)
            bad = np.argwhere(diff > 0)
            print(
                json.dumps(
                    dict(
                        pair=f"{a} vs {b}",
                        tensor=name,
                        max_abs=float(diff.max()),
                        max_ref=float(np.abs(v).max()),
                        mismatches=len(bad),
                        first=bad[:5].tolist(),
                    )
                ),
                flush=True,
            )
    print(json.dumps(dict(eager_reference_vs_jit_reference=float(np.abs(eager_ref - outs["reference"][0]).max()))))
    u = outs["triton_gpu"][0]
    v = outs["pallas_gpu"][0]
    rows = np.unique(np.argwhere(np.abs(u - v) > 0)[:, 1])
    print(json.dumps(dict(mismatched_rows=rows[:40].tolist(), n_rows=len(rows))))


if __name__ == "__main__":
    main()
