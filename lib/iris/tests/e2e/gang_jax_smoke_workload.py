#!/usr/bin/env python3
# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Multi-host JAX workload for the gang smoke (tests/e2e/gpu_gang_smoke.py).

Submitted as a normal Iris job (one task per gang member) and run from the job
bundle, so it can ``import jax`` (installed via the job's ``gpu``/``cpu`` extra)
and use ``iris.runtime.jax_init.initialize_jax`` for coordinator discovery via
the controller's endpoint registry — no hand-rolled rendezvous.

It joins one JAX mesh across the whole gang, runs an explicit cross-host
all-reduce sanity check, then trains a small causal transformer data-parallel
across every device. The gradient all-reduce is the real inter-host collective
(NCCL over InfiniBand on H100; gRPC/gloo on CPU kind). Knobs come from the env:

    GANG_SMOKE_STEPS  training steps (default 20)
    GANG_SMOKE_PDB    per-device batch (default 8)
"""

from __future__ import annotations

import math
import os

import jax
import jax.numpy as jnp
import numpy as np
from iris.runtime.jax_init import initialize_jax
from jax.experimental import multihost_utils
from jax.sharding import Mesh, NamedSharding
from jax.sharding import PartitionSpec as P


def log(msg: str) -> None:
    print(f"[gang-jax] {msg}", flush=True)


def main() -> None:
    steps = int(os.environ.get("GANG_SMOKE_STEPS", "20"))
    per_device_batch = int(os.environ.get("GANG_SMOKE_PDB", "8"))

    # Coordinator discovery + jax.distributed.initialize via the Iris registry.
    initialize_jax()

    n_dev = jax.device_count()
    log(
        f"jax {jax.__version__}: global_devices={n_dev} local_devices={jax.local_device_count()} "
        f"process_index={jax.process_index()}/{jax.process_count()} host={os.uname().nodename}"
    )

    mesh = Mesh(np.asarray(jax.devices()), ("data",))
    replicated = NamedSharding(mesh, P())
    sharded = NamedSharding(mesh, P("data"))

    # Cross-host all-reduce sanity: device d on process p contributes (p+1); the
    # global sum forces an inter-host collective. Verifies the mesh spans hosts.
    local_vals = np.full((jax.local_device_count(),), float(jax.process_index() + 1), np.float32)
    g = jax.make_array_from_process_local_data(sharded, local_vals, (n_dev,))
    total = float(jax.jit(jnp.sum, out_shardings=replicated)(g))
    log(f"all-reduce check: sum over {n_dev} devices = {total}")

    # ---- small causal-transformer LM, data-parallel across the whole mesh ----
    SEQ, D, H, L, V = 128, 256, 4, 2, 256
    LR = 0.1
    global_batch = n_dev * per_device_batch
    rng = np.random.default_rng(0)

    def randn(*shape: int) -> np.ndarray:
        return (rng.standard_normal(shape) / math.sqrt(shape[0])).astype(np.float32)

    params = {
        "embed": randn(V, D),
        "blocks": [
            {"qkv": randn(D, 3 * D), "out": randn(D, D), "w1": randn(D, 4 * D), "w2": randn(4 * D, D)} for _ in range(L)
        ],
        "unembed": randn(D, V),
    }
    params = jax.device_put(params, replicated)

    def block(p, h):
        B, T, _ = h.shape
        q, k, v = jnp.split(h @ p["qkv"], 3, axis=-1)
        shp = (B, T, H, D // H)
        q = q.reshape(shp).transpose(0, 2, 1, 3)
        k = k.reshape(shp).transpose(0, 2, 1, 3)
        v = v.reshape(shp).transpose(0, 2, 1, 3)
        att = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(D // H)
        mask = jnp.tril(jnp.ones((T, T), jnp.float32))
        att = jax.nn.softmax(jnp.where(mask[None, None] > 0, att, -1e9), axis=-1)
        o = (att @ v).transpose(0, 2, 1, 3).reshape(B, T, D)
        h = h + o @ p["out"]
        h = h + jax.nn.gelu(h @ p["w1"]) @ p["w2"]
        return h

    def loss_fn(params, x, y):
        h = params["embed"][x]
        for p in params["blocks"]:
            h = block(p, h)
        logp = jax.nn.log_softmax(h @ params["unembed"], axis=-1)
        onehot = jax.nn.one_hot(y, V, dtype=logp.dtype)
        return -jnp.mean(jnp.sum(onehot * logp, axis=-1))

    @jax.jit
    def train_step(params, x, y):
        loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
        params = jax.tree.map(lambda w, gd: w - LR * gd, params, grads)
        return params, loss

    # One fixed synthetic batch (so the loss visibly drops as the model fits it).
    local_rows = global_batch // jax.process_count()
    lx = rng.integers(0, V, size=(local_rows, SEQ)).astype(np.int32)
    ly = np.roll(lx, -1, axis=1)
    x = jax.make_array_from_process_local_data(sharded, lx, (global_batch, SEQ))
    y = jax.make_array_from_process_local_data(sharded, ly, (global_batch, SEQ))
    log(f"training transformer(L={L},D={D},H={H}) global_batch={global_batch} steps={steps}")

    loss = None
    for step in range(steps):
        params, loss = train_step(params, x, y)
        if jax.process_index() == 0:
            log(f"step {step + 1}/{steps} loss={float(loss):.4f}")
    log(f"final loss={float(loss):.4f}")

    multihost_utils.sync_global_devices("gang-jax-done")
    log("DONE")


if __name__ == "__main__":
    main()
