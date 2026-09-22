# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Structured-target scaling: regress a fractal field on the unit hypersphere, scale width k.

Inputs: uniform on S^(x-1). Target: a multi-scale random-Fourier field with a power-law amplitude
spectrum (a_s = 2^(-s*beta) over dyadic scales), standardized to unit variance. Model: x -> k (ReLU)
-> 1, MSE. We measure held-out TEST loss vs k (and vs beta) -- the quantity expected to follow a
power law (diminishing returns), in contrast to the random-label memorization cliff.

Prints a JSON block between markers. One-GPU iris job (needs jax[cuda13] via --extra pipeline).
"""
import json
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

X_DIM = 64
N_TRAIN = 100_000
N_TEST = 20_000
K_LIST = [4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
BETA_LIST = [0.5, 1.0, 2.0]
N_SCALES = 8  # dyadic scales s = 0..7, oscillation scale rho_s = 2^s
COMPS_PER_SCALE = 128  # random-Fourier components per scale
STEPS = 50_000
EVAL_EVERY = 1000
SEED = 0


def sphere(n, x, rng):
    v = rng.standard_normal((n, x)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def fractal_field(V, beta, seed):
    """Sum of dyadic random-Fourier features with power-law amplitudes; g scaled so g.v ~ N(0, rho_s^2)."""
    rng = np.random.default_rng(seed)
    x = V.shape[1]
    freqs, phases, amps = [], [], []
    for s in range(N_SCALES):
        rho = 2.0**s
        g = rng.standard_normal((COMPS_PER_SCALE, x)).astype(np.float32) * (rho / np.sqrt(x))
        freqs.append(g)
        phases.append(rng.uniform(0, 2 * np.pi, COMPS_PER_SCALE).astype(np.float32))
        amps.append(np.full(COMPS_PER_SCALE, 2.0 ** (-s * beta), dtype=np.float32))
    G = np.concatenate(freqs, 0)  # [M, x]
    ph = np.concatenate(phases, 0)  # [M]
    a = np.concatenate(amps, 0)  # [M]
    f = np.cos(V @ G.T + ph) @ a  # [n]
    return f.astype(np.float32)


def make_data(x, beta, seed):
    rng = np.random.default_rng(seed)
    Vtr, Vte = sphere(N_TRAIN, x, rng), sphere(N_TEST, x, rng)
    ftr = fractal_field(Vtr, beta, 777)  # field frozen (same seed) so train/test share the function
    fte = fractal_field(Vte, beta, 777)
    mu, sd = ftr.mean(), ftr.std()  # standardize with train stats -> unit-variance target
    ytr, yte = (ftr - mu) / sd, (fte - mu) / sd
    return jnp.asarray(Vtr), jnp.asarray(ytr), jnp.asarray(Vte), jnp.asarray(yte)


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


def mse(p, V, y):
    return jnp.mean((predict(p, V) - y) ** 2)


def train_one(x, k, data, seed):
    Vtr, ytr, Vte, yte = data
    sched = optax.warmup_cosine_decay_schedule(0.0, 3e-3, 1000, STEPS, 1e-4)
    opt = optax.adam(sched)
    p = init(x, k, seed)
    os = opt.init(p)

    @jax.jit
    def run_chunk(p, os):
        def step(carry, _):
            p, os = carry
            _, g = jax.value_and_grad(mse)(p, Vtr, ytr)
            u, os = opt.update(g, os, p)
            return (optax.apply_updates(p, u), os), None

        (p, os), _ = jax.lax.scan(step, (p, os), None, length=EVAL_EVERY)
        return p, os

    traj = []
    for _ in range(STEPS // EVAL_EVERY):
        p, os = run_chunk(p, os)
        traj.append((float(mse(p, Vtr, ytr)), float(mse(p, Vte, yte))))
    return traj


def main():
    print("backend:", jax.default_backend(), jax.devices(), flush=True)
    results = []
    t0 = time.time()
    for beta in BETA_LIST:
        data = make_data(X_DIM, beta, SEED)
        print(f"beta={beta}: target std=1.0 (standardized), n_train={N_TRAIN} n_test={N_TEST}", flush=True)
        for k in K_LIST:
            ts = time.time()
            traj = train_one(X_DIM, k, data, SEED)
            tr_final, te_final = traj[-1]
            te_min = min(t for _, t in traj)
            params = X_DIM * k + k + k + 1
            results.append(
                {
                    "beta": beta,
                    "k": k,
                    "params": params,
                    "train_mse": tr_final,
                    "test_mse_final": te_final,
                    "test_mse_min": te_min,
                }
            )
            print(
                f"  beta={beta} k={k:<5} params={params:<7} train={tr_final:.4f} "
                f"test={te_final:.4f} test_min={te_min:.4f} ({time.time()-ts:.0f}s)",
                flush=True,
            )
    print(f"total {time.time()-t0:.0f}s", flush=True)
    print("FRACTAL_JSON_BEGIN")
    print(
        json.dumps(
            {
                "x": X_DIM,
                "n_train": N_TRAIN,
                "n_test": N_TEST,
                "scales": N_SCALES,
                "comps_per_scale": COMPS_PER_SCALE,
                "steps": STEPS,
                "results": results,
            }
        )
    )
    print("FRACTAL_JSON_END")


if __name__ == "__main__":
    main()
