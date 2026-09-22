# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Structured-target scaling in the ONLINE (infinite-data) regime: regress a fractal field on the
unit hypersphere, resampling a fresh batch every step. With fresh data each step there is no
train/test gap and no overfitting -- the reported loss is the population risk, limited only by model
capacity, so loss-vs-params isolates the pure model-scaling exponent.

Inputs: uniform on S^(x-1). Target: multi-scale random-Fourier field, power-law spectrum
a_s = 2^(-s*beta), standardized to unit variance. Model: x -> k (ReLU) -> 1, MSE.

Sweeps k (and beta). Prints a JSON block. One-GPU iris job (needs jax[cuda13] via --extra pipeline).
"""
import json
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

X_DIM = 64
K_LIST = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
BETA_LIST = [0.5, 1.0, 2.0]
N_SCALES = 8
COMPS_PER_SCALE = 128
BATCH = 8192  # fresh samples per step
STEPS = 50_000
EVAL_EVERY = 2000
N_EVAL = 65_536  # fixed held-out batch for a low-variance population-risk estimate
SEED = 0
FIELD_SEED = 777


def make_field(x, beta):
    """Frozen field params (G[M,x], phi[M], a[M]) plus standardization (mu, sd) from a big sample."""
    rng = np.random.default_rng(FIELD_SEED)
    freqs, phases, amps = [], [], []
    for s in range(N_SCALES):
        rho = 2.0**s
        freqs.append(rng.standard_normal((COMPS_PER_SCALE, x)).astype(np.float32) * (rho / np.sqrt(x)))
        phases.append(rng.uniform(0, 2 * np.pi, COMPS_PER_SCALE).astype(np.float32))
        amps.append(np.full(COMPS_PER_SCALE, 2.0 ** (-s * beta), dtype=np.float32))
    G = jnp.asarray(np.concatenate(freqs, 0))
    phi = jnp.asarray(np.concatenate(phases, 0))
    a = jnp.asarray(np.concatenate(amps, 0))
    ref = rng.standard_normal((200_000, x)).astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    f = np.cos(np.asarray(ref) @ np.asarray(G).T + np.asarray(phi)) @ np.asarray(a)
    return (G, phi, a), float(f.mean()), float(f.std())


def sphere(key, b, x):
    v = jr.normal(key, (b, x))
    return v / jnp.linalg.norm(v, axis=1, keepdims=True)


def init(x, k, seed):
    rng = np.random.default_rng(1000 + seed)
    return {
        "W1": jnp.asarray(rng.standard_normal((x, k)).astype(np.float32) / np.sqrt(x)),
        "b1": jnp.zeros((k,), jnp.float32),
        "W2": jnp.asarray(rng.standard_normal((k, 1)).astype(np.float32) / np.sqrt(k)),
        "b2": jnp.zeros((1,), jnp.float32),
    }


def predict(p, V):
    return (jnp.maximum(V @ p["W1"] + p["b1"], 0.0) @ p["W2"] + p["b2"])[:, 0]


def train_online(x, k, field, mu, sd, Veval, yeval, seed):
    G, phi, a = field

    def target(V):
        return (jnp.cos(V @ G.T + phi) @ a - mu) / sd

    def loss_on(p, V):
        return jnp.mean((predict(p, V) - target(V)) ** 2)

    def eval_mse(p):
        return jnp.mean((predict(p, Veval) - yeval) ** 2)

    sched = optax.warmup_cosine_decay_schedule(0.0, 3e-3, 1000, STEPS, 1e-4)
    opt = optax.adam(sched)
    p = init(x, k, seed)
    os = opt.init(p)

    @jax.jit
    def run_chunk(p, os, key):
        def step(carry, k_):
            p, os = carry
            V = sphere(k_, BATCH, x)
            _, g = jax.value_and_grad(loss_on)(p, V)
            u, os = opt.update(g, os, p)
            return (optax.apply_updates(p, u), os), None

        (p, os), _ = jax.lax.scan(step, (p, os), jr.split(key, EVAL_EVERY))
        return p, os

    key = jr.PRNGKey(seed)
    traj = []
    for _c in range(STEPS // EVAL_EVERY):
        key, ck = jr.split(key)
        p, os = run_chunk(p, os, ck)
        traj.append(float(eval_mse(p)))
    return traj


def main():
    print("backend:", jax.default_backend(), jax.devices(), flush=True)
    results = []
    t0 = time.time()
    for beta in BETA_LIST:
        field, mu, sd = make_field(X_DIM, beta)
        Veval = sphere(jr.PRNGKey(12345), N_EVAL, X_DIM)
        yeval = (jnp.cos(Veval @ field[0].T + field[1]) @ field[2] - mu) / sd
        print(f"beta={beta}: field ready (mu={mu:.3f} sd={sd:.3f})", flush=True)
        for k in K_LIST:
            ts = time.time()
            traj = train_online(X_DIM, k, field, mu, sd, Veval, yeval, SEED)
            params = X_DIM * k + k + k + 1
            results.append({"beta": beta, "k": k, "params": params, "loss_final": traj[-1], "loss_min": min(traj)})
            print(
                f"  beta={beta} k={k:<5} params={params:<7} pop_loss={traj[-1]:.4f} "
                f"min={min(traj):.4f} ({time.time()-ts:.0f}s)",
                flush=True,
            )
    print(f"total {time.time()-t0:.0f}s", flush=True)
    print("FRACTAL_JSON_BEGIN")
    print(
        json.dumps(
            {
                "x": X_DIM,
                "batch": BATCH,
                "scales": N_SCALES,
                "comps_per_scale": COMPS_PER_SCALE,
                "steps": STEPS,
                "online": True,
                "results": results,
            }
        )
    )
    print("FRACTAL_JSON_END")


if __name__ == "__main__":
    main()
