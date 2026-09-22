# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Data scaling law (online): fix parameter count (k), vary the number of training steps = data seen
(tokens = steps * batch, fresh each step), and measure converged loss vs data. Complements the
parameter law (fix data, vary k). Each budget is cosine-annealed to its own length (Chinchilla-style),
and the full eval trajectory + a tail-plateau delta are logged so saturation is verifiable.

Prints a JSON block. One-GPU iris job (needs jax[cuda13] via --extra pipeline).
"""
import json
import time

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax

import experiments.grug.fast_track.fractal_regression_sweep as F

X_DIM = 64
BETA = 1.0
K_LIST = [128, 512, 2048]  # fixed parameter counts (one data-scaling curve each)
STEPS_LIST = [1000, 2000, 4000, 8000, 16000, 32000, 64000, 128000]
BATCH = 8192
N_EVAL = 65_536
N_SEEDS = 4  # average final loss over seeds to smooth single-run SGD noise


def train_n(x, k, field, mu, sd, Veval, yeval, steps, seed):
    G, phi, a = field

    def target(V):
        return (jnp.cos(V @ G.T + phi) @ a - mu) / sd

    def loss_on(p, V):
        return jnp.mean((F.predict(p, V) - target(V)) ** 2)

    def eval_mse(p):
        return jnp.mean((F.predict(p, Veval) - yeval) ** 2)

    warmup = max(1, min(1000, steps // 10))
    eval_every = max(1, steps // 20)
    sched = optax.warmup_cosine_decay_schedule(0.0, 3e-3, warmup, steps, 1e-4)
    opt = optax.adam(sched)
    p = F.init(x, k, seed)
    os = opt.init(p)

    @jax.jit
    def run_chunk(p, os, key):
        def step(carry, k_):
            p, os = carry
            V = F.sphere(k_, BATCH, x)
            _, g = jax.value_and_grad(loss_on)(p, V)
            u, os = opt.update(g, os, p)
            return (optax.apply_updates(p, u), os), None

        (p, os), _ = jax.lax.scan(step, (p, os), jr.split(key, eval_every))
        return p, os

    key = jr.PRNGKey(seed)
    traj = []
    done = 0
    while done < steps:
        key, ck = jr.split(key)
        p, os = run_chunk(p, os, ck)
        done += eval_every
        traj.append((done, float(eval_mse(p))))
    return traj


def main():
    print("backend:", jax.default_backend(), jax.devices(), flush=True)
    field, mu, sd = F.make_field(X_DIM, BETA)
    Veval = F.sphere(jr.PRNGKey(12345), N_EVAL, X_DIM)
    yeval = (jnp.cos(Veval @ field[0].T + field[1]) @ field[2] - mu) / sd
    print(f"field ready (mu={mu:.3f} sd={sd:.3f}); batch={BATCH}", flush=True)
    results = []
    t0 = time.time()
    for k in K_LIST:
        params = X_DIM * k + k + k + 1
        for steps in STEPS_LIST:
            ts = time.time()
            finals, plateaus = [], []
            for seed in range(N_SEEDS):
                traj = train_n(X_DIM, k, field, mu, sd, Veval, yeval, steps, seed)
                losses = [l for _, l in traj]
                finals.append(losses[-1])
                tail0 = losses[max(0, int(len(losses) * 0.8)) - 1]
                plateaus.append(abs(losses[-1] - tail0) / losses[-1])
            tokens = steps * BATCH
            results.append(
                {
                    "k": k,
                    "params": params,
                    "steps": steps,
                    "tokens": tokens,
                    "loss_mean": float(np.mean(finals)),
                    "loss_std": float(np.std(finals)),
                    "loss_final": float(np.mean(finals)),
                    "n_seeds": N_SEEDS,
                    "plateau_rel": float(np.mean(plateaus)),
                }
            )
            print(
                f"  k={k:<5} steps={steps:<7} tokens={tokens/1e6:7.1f}M "
                f"loss={np.mean(finals):.4f}±{np.std(finals):.4f} "
                f"plateau_rel={np.mean(plateaus)*100:4.1f}% ({time.time()-ts:.0f}s)",
                flush=True,
            )
    print(f"total {time.time()-t0:.0f}s", flush=True)
    print("DATASCALE_JSON_BEGIN")
    print(json.dumps({"x": X_DIM, "beta": BETA, "batch": BATCH, "online": True, "results": results}))
    print("DATASCALE_JSON_END")


if __name__ == "__main__":
    main()
