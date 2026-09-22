# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Input-dimension scaling (ONLINE regime): fix the fractal target (beta=1), sweep sphere dim x and
width k with fresh-batch resampling each step, and measure how the model-scaling exponent
alpha (population loss ~ params^-alpha) depends on x -- the toy's stand-in for the intrinsic
data-manifold dimension in the spectral theory of scaling laws.

Reuses the online field/model helpers from fractal_regression_sweep. One-GPU iris job (--extra pipeline).
"""
import json
import time

import jax
import jax.random as jr

import experiments.grug.fast_track.fractal_regression_sweep as F

X_LIST = [8, 16, 32, 64, 128, 256]
K_LIST = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
BETA = 1.0


def main():
    print("backend:", jax.default_backend(), jax.devices(), flush=True)
    results = []
    t0 = time.time()
    for x in X_LIST:
        field, mu, sd = F.make_field(x, BETA)
        Veval = F.sphere(jr.PRNGKey(12345), F.N_EVAL, x)
        yeval = (jax.numpy.cos(Veval @ field[0].T + field[1]) @ field[2] - mu) / sd
        print(f"x={x}: field ready (mu={mu:.3f} sd={sd:.3f})", flush=True)
        for k in K_LIST:
            ts = time.time()
            traj = F.train_online(x, k, field, mu, sd, Veval, yeval, 0)
            params = x * k + k + k + 1
            results.append({"x": x, "k": k, "params": params, "loss_final": traj[-1], "loss_min": min(traj)})
            print(
                f"  x={x:<4} k={k:<5} params={params:<7} pop_loss={traj[-1]:.4f} "
                f"min={min(traj):.4f} ({time.time()-ts:.0f}s)",
                flush=True,
            )
    print(f"total {time.time()-t0:.0f}s", flush=True)
    print("DIM_JSON_BEGIN")
    print(json.dumps({"beta": BETA, "steps": F.STEPS, "batch": F.BATCH, "online": True, "results": results}))
    print("DIM_JSON_END")


if __name__ == "__main__":
    main()
