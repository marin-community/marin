# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Capacity sweep: fixed data (x=64, n=100k unique random binary inputs, random 0/1 labels),
scale hidden width k and measure the converged training loss. Full-batch Adam, GPU.

Prints a JSON block to stdout (between markers). Meant to run as a one-GPU iris job.
"""
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

X_DIM = 64
N = 100_000
K_LIST = [4, 8, 16, 32, 64, 128, 256, 512, 1024]
STEPS = 50_000
LOG_EVERY = 1000
SEED = 0


def make_unique_data(x, n, seed):
    rng = np.random.default_rng(seed)
    seen, rows = set(), []
    while len(rows) < n:
        cand = rng.integers(0, 2, size=(2 * (n - len(rows)) + 16, x), dtype=np.int8)
        for r in cand:
            b = r.tobytes()
            if b not in seen:
                seen.add(b)
                rows.append(r)
                if len(rows) == n:
                    break
    Xnp = np.array(rows, dtype=np.float32)
    assert len({r.tobytes() for r in Xnp.astype(np.int8)}) == n, "inputs not unique"
    y = rng.integers(0, 2, size=n).astype(np.int32)
    return jnp.asarray(Xnp), jnp.asarray(y)


def loss_fn(p, X, y):
    h = jnp.maximum(X @ p["W1"] + p["b1"], 0.0)
    return optax.softmax_cross_entropy_with_integer_labels(h @ p["W2"] + p["b2"], y).mean()


def train_one(x, k, X, y, seed):
    rng = np.random.default_rng(1000 + seed)
    p = {
        "W1": jnp.asarray(rng.standard_normal((x, k)).astype(np.float32) / np.sqrt(x)),
        "b1": jnp.zeros((k,), jnp.float32),
        "W2": jnp.asarray(rng.standard_normal((k, 2)).astype(np.float32) / np.sqrt(k)),
        "b2": jnp.zeros((2,), jnp.float32),
    }
    sched = optax.warmup_cosine_decay_schedule(0.0, 5e-3, 1000, STEPS, 1e-4)
    opt = optax.adam(sched)
    os = opt.init(p)

    @jax.jit
    def step(p, os):
        loss, g = jax.value_and_grad(loss_fn)(p, X, y)
        u, os = opt.update(g, os, p)
        return optax.apply_updates(p, u), os, loss

    traj = []
    for i in range(STEPS):
        p, os, loss = step(p, os)
        if i % LOG_EVERY == 0 or i == STEPS - 1:
            traj.append((i, float(loss)))
    return traj


def main():
    print("backend:", jax.default_backend(), jax.devices(), flush=True)
    X, y = make_unique_data(X_DIM, N, SEED)
    print(f"data: n={N} x={X_DIM} unique=OK  class balance={float(y.mean()):.3f}", flush=True)
    results = []
    t0 = time.time()
    for k in K_LIST:
        ts = time.time()
        traj = train_one(X_DIM, k, X, y, SEED)
        final = traj[-1][1]
        tail = [l for (s, l) in traj if s >= STEPS - 5 * LOG_EVERY]
        plateau = max(tail) - min(tail)  # loss change over last 5 log points -> convergence check
        params = X_DIM * k + k + k * 2 + 2
        results.append(
            {
                "k": k,
                "params": params,
                "final_loss": final,
                "min_loss": min(l for _, l in traj),
                "plateau_delta": plateau,
                "traj": traj,
            }
        )
        print(
            f"k={k:<5} params={params:<7} final_loss={final:.4f} plateau_Δ={plateau:.4f} " f"({time.time()-ts:.0f}s)",
            flush=True,
        )
    print(f"total {time.time()-t0:.0f}s", flush=True)
    print("CAP_SWEEP_JSON_BEGIN")
    print(json.dumps({"x": X_DIM, "n": N, "steps": STEPS, "results": results}))
    print("CAP_SWEEP_JSON_END")


if __name__ == "__main__":
    main()
