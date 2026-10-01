# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Size and source of the holdback's gradient differences in the model smoke.

Runs C's model smoke (reference attention) with 0 and 1 held-back shared experts in one process, prints
the relative difference of every gradient leaf that differs, and lists the shared-expert GEMMs of each
compiled step (count and operand shapes), to see whether XLA merges the shared experts' GEMMs differently.

Usage (GB200x4, hero env, from a checkout with the holdback): python autoresearch/loop-260930-mfu30/b/holdback_diff.py
"""

import collections
import dataclasses
import json
import sys

sys.path.insert(0, "autoresearch/loop-260930-mfu30/stack")
import jax
import jax.numpy as jnp
import model_smoke as ms
import numpy as np
from jax.sharding import AxisType, Mesh, NamedSharding
from jax.sharding import PartitionSpec as P

from experiments.grug.moe_hero_ep import model


def _shared_gemms(hlo_text):
    counts = collections.Counter()
    for line in hlo_text.splitlines():
        if "DenseMLP" not in line or not ("__cublas$gemm" in line or "__cublas$lt" in line or "triton_gemm" in line):
            continue
        result = line.split("=", 1)[1].strip().split(" ")[0]
        counts[result[:60]] += 1
    return dict(counts)


def main():
    devices = np.asarray(jax.devices()).reshape(1, 1, 1, len(jax.devices()), 1)
    mesh = Mesh(devices, ("replica_dcn", "data", "context", "expert", "model"), axis_types=(AxisType.Explicit,) * 5)
    grads = {}
    for held in (0, 1):
        cfg = dataclasses.replace(ms.CFG, attention_implementation="reference", held_back_shared_experts=held)
        with jax.set_mesh(mesh):
            m = model.Transformer.init(cfg, key=jax.random.key(0))
            m = jax.tree.map(
                lambda a: (
                    a.astype(jnp.bfloat16) if isinstance(a, jax.Array) and jnp.issubdtype(a.dtype, jnp.floating) else a
                ),
                m,
            )
            spec = NamedSharding(mesh, P(("replica_dcn", "data", "expert"), None))
            tokens = jax.device_put(jax.random.randint(jax.random.key(1), (ms.BATCH, ms.SEQ), 0, cfg.vocab_size), spec)
            weight = jax.device_put(jnp.ones((ms.BATCH, ms.SEQ), jnp.float32), spec)
            step = jax.jit(
                jax.value_and_grad(
                    lambda mm, tokens=tokens, weight=weight: mm.next_token_loss(tokens, weight), argnums=0
                )
            )
            compiled = step.lower(m).compile()
            print(json.dumps(dict(held=held, shared_gemms=_shared_gemms(compiled.as_text()))), flush=True)
            _, g = compiled(m)
            grads[held] = {
                jax.tree_util.keystr(p): np.asarray(jax.device_get(leaf), np.float32)
                for p, leaf in jax.tree_util.tree_leaves_with_path(g)
            }
    for key, a in grads[0].items():
        b = grads[1][key]
        if np.array_equal(a, b):
            continue
        diff = np.abs(a - b)
        scale = float(np.max(np.abs(a))) or 1.0
        nonzero = np.abs(a) > 0
        print(
            json.dumps(
                dict(
                    leaf=key,
                    max_rel=float(diff.max()) / scale,
                    median_rel=float(np.median(diff[nonzero] / np.abs(a[nonzero]))) if nonzero.any() else 0.0,
                    fraction_differing=float(np.mean(a != b)),
                )
            ),
            flush=True,
        )


if __name__ == "__main__":
    main()
