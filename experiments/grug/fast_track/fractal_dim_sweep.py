# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Input-dimension scaling: fix the fractal target (beta=1), sweep sphere dim x and width k, and
measure how the model-scaling exponent alpha (test_MSE ~ params^-alpha) depends on x -- the toy's
stand-in for the intrinsic data-manifold dimension in the spectral theory of scaling laws.

Reuses the field/model helpers from fractal_regression_sweep. One-GPU iris job (--extra pipeline).
"""
import json
import time

import jax

import experiments.grug.fast_track.fractal_regression_sweep as F

X_LIST = [8, 16, 32, 64, 128, 256]
K_LIST = [8, 16, 32, 64, 128, 256, 512, 1024, 2048]
BETA = 1.0
F.N_TRAIN = 100_000
F.N_TEST = 20_000
F.STEPS = 50_000
F.EVAL_EVERY = 1000


def main():
    print("backend:", jax.default_backend(), jax.devices(), flush=True)
    results = []
    t0 = time.time()
    for x in X_LIST:
        data = F.make_data(x, BETA, 0)
        print(f"x={x}: data ready", flush=True)
        for k in K_LIST:
            ts = time.time()
            traj = F.train_one(x, k, data, 0)
            tr, te = traj[-1]
            te_min = min(t for _, t in traj)
            params = x * k + k + k + 1
            results.append(
                {"x": x, "k": k, "params": params, "train_mse": tr, "test_mse_final": te, "test_mse_min": te_min}
            )
            print(
                f"  x={x:<4} k={k:<5} params={params:<7} train={tr:.4f} test={te:.4f} "
                f"test_min={te_min:.4f} ({time.time()-ts:.0f}s)",
                flush=True,
            )
    print(f"total {time.time()-t0:.0f}s", flush=True)
    print("DIM_JSON_BEGIN")
    print(json.dumps({"beta": BETA, "steps": F.STEPS, "results": results}))
    print("DIM_JSON_END")


if __name__ == "__main__":
    main()
