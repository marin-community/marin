# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Hashes of the model smoke's loss, router metrics (QB statistics included) and gradients.

Run from two checkouts and compare the printed hashes: a change that only moves where XLA computes the QB
statistics must leave every hash equal. Uses reference attention so the backward is deterministic, and checks
that by running the step twice.

Usage (GB200x4, hero env, from a stack checkout): python autoresearch/loop-260930-mfu30/b/qb_values.py
"""

import dataclasses
import hashlib
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


def _digest(tree):
    out = {}
    for path, leaf in jax.tree_util.tree_leaves_with_path(tree):
        array = np.asarray(jax.device_get(leaf))
        out[jax.tree_util.keystr(path)] = hashlib.sha256(array.tobytes()).hexdigest()[:16]
    return out


def main():
    cfg = dataclasses.replace(ms.CFG, attention_implementation="reference")
    devices = np.asarray(jax.devices()).reshape(1, 1, 1, len(jax.devices()), 1)
    mesh = Mesh(devices, ("replica_dcn", "data", "context", "expert", "model"), axis_types=(AxisType.Explicit,) * 5)
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
            jax.value_and_grad(lambda mm: mm.next_token_loss(tokens, weight, return_router_metrics=True), has_aux=True)
        )
        runs = [step(m) for _ in range(3)]
    for i, ((loss, metrics), grads) in enumerate(runs):
        values = {}
        for path, leaf in jax.tree_util.tree_leaves_with_path(metrics):
            array = np.asarray(jax.device_get(leaf), np.float64)
            values[jax.tree_util.keystr(path)] = [float(array.sum()), float(np.abs(array).max()) if array.size else 0.0]
        print(
            "RUN"
            + str(i)
            + " "
            + json.dumps(dict(loss=float(loss), metrics=values, grads=_digest(grads)), sort_keys=True),
            flush=True,
        )


if __name__ == "__main__":
    main()
