# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["pandas", "numpy", "scipy", "scikit-learn", "plotly"]
# ///
"""Extract fitted DS-RE-CEQ parameters on 3-phase StarCoder data.

Fits the full model on all data (no CV), extracts parameter values,
and finds the predicted optimum via grid search.

Usage:
  uv run experiments/domain_phase_mix/exploratory/extract_dsre_ceq_params.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from scipy.optimize import minimize

from general_scaling_models import _fit_dsre_ceq
from three_phase_visualization import load_spec

DEFAULT_TARGET = "eval/paloma/dolma_100_programing_languages/bpb"
DOMAIN_NAMES = ["nemotron_full", "starcoder"]


def _softmax(logits):
    z = logits - np.max(logits)
    e = np.exp(z)
    return e / e.sum()


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def unpack_dsre_ceq(p, M, N):
    """Unpack flat parameter vector into named parameters."""
    idx = 0
    c0 = float(p[idx])
    logA = float(p[idx + 1])
    logB = float(p[idx + 2])
    idx += 3

    logits_a = np.zeros(M)
    if M > 1:
        logits_a[: M - 1] = p[idx : idx + (M - 1)]
        idx += M - 1
    a = _softmax(logits_a)

    rho = float(np.clip(5.0 * np.tanh(p[idx]), -10.0, 0.99))
    idx += 1

    # Per-domain phase importance weights
    pi_dom = np.zeros((M, N))
    for d in range(M):
        if N > 1:
            logits_free = p[idx : idx + (N - 1)]
            idx += N - 1
        else:
            logits_free = np.array([])
        full = np.zeros(N)
        if N > 1:
            full[: N - 1] = logits_free
        pi_dom[d] = _softmax(full)

    # Per-domain per-phase interference lambda: phases 1..N-1
    lam = np.zeros((N, M))
    if N > 1:
        raw = p[idx : idx + (N - 1) * M].reshape(N - 1, M)
        idx += (N - 1) * M
        lam[1:] = np.exp(np.clip(raw, -8.0, 8.0))

    # Per-domain satiety memory phi_d via sigmoid
    phi = np.array([_sigmoid(float(p[idx + d])) for d in range(M)])
    idx += M

    # Conflict gate g_k via sigmoid; g_0 forced to 0
    g = np.array([_sigmoid(float(p[idx + k])) for k in range(N)])
    idx += N
    g[0] = 0.0

    tau = float(np.exp(np.clip(p[idx], -8.0, 8.0)))
    idx += 1

    A = float(np.exp(np.clip(logA, -10.0, 10.0)))
    B = float(np.exp(np.clip(logB, -10.0, 10.0)))

    return {
        "c0": c0,
        "A": A,
        "B": B,
        "a": a,
        "rho": rho,
        "pi": pi_dom,
        "lambda": lam,
        "phi": phi,
        "g": g,
        "tau": tau,
    }


def main():
    spec, _df = load_spec(DEFAULT_TARGET)
    M = spec.M
    N = spec.N
    R = spec.R
    print(f"Data: R={R}, N={N}, M={M}")
    print(f"Domains: {DOMAIN_NAMES}")
    print()

    # Fit with multiple seeds to see parameter stability
    n_seeds = 5
    all_params = []
    all_predict_fns = []

    for seed in range(n_seeds):
        print(f"Fitting seed {seed}...", end=" ", flush=True)
        predict_fn, info = _fit_dsre_ceq(spec, seed=seed, n_restarts=8, maxiter=500)
        params = unpack_dsre_ceq(info["final_p"], M, N)
        all_params.append(params)
        all_predict_fns.append(predict_fn)

        # In-sample RMSE
        preds = predict_fn(spec.weights)
        rmse = float(np.sqrt(np.mean((preds - spec.y) ** 2)))
        print(f"RMSE={rmse:.4f}")

    print()

    # Show parameter values (mean and range across seeds)
    print("=" * 70)
    print("FITTED PARAMETER VALUES")
    print("=" * 70)

    # Scalar parameters
    for name in ["c0", "A", "B", "rho", "tau"]:
        vals = [p[name] for p in all_params]
        mean = np.mean(vals)
        lo, hi = np.min(vals), np.max(vals)
        if abs(hi - lo) < 1e-4:
            print(f"  {name:>6s} = {mean:.6f}")
        else:
            print(f"  {name:>6s} = {mean:.6f}  [{lo:.6f}, {hi:.6f}]")

    # Domain weights a
    print()
    print("  CES domain weights a:")
    for d in range(M):
        vals = [p["a"][d] for p in all_params]
        mean = np.mean(vals)
        lo, hi = np.min(vals), np.max(vals)
        if abs(hi - lo) < 1e-4:
            print(f"    a[{DOMAIN_NAMES[d]}] = {mean:.6f}")
        else:
            print(f"    a[{DOMAIN_NAMES[d]}] = {mean:.6f}  [{lo:.6f}, {hi:.6f}]")

    # Satiety phi
    print()
    print("  Satiety memory phi:")
    for d in range(M):
        vals = [p["phi"][d] for p in all_params]
        mean = np.mean(vals)
        lo, hi = np.min(vals), np.max(vals)
        if abs(hi - lo) < 1e-4:
            print(f"    phi[{DOMAIN_NAMES[d]}] = {mean:.6f}")
        else:
            print(f"    phi[{DOMAIN_NAMES[d]}] = {mean:.6f}  [{lo:.6f}, {hi:.6f}]")

    # Phase weights pi
    print()
    print("  Phase weights pi[domain, phase]:")
    phase_names = ["phase_0", "phase_1", "phase_2"]
    for d in range(M):
        for k in range(N):
            vals = [p["pi"][d, k] for p in all_params]
            mean = np.mean(vals)
            lo, hi = np.min(vals), np.max(vals)
            if abs(hi - lo) < 1e-4:
                print(f"    pi[{DOMAIN_NAMES[d]}, {phase_names[k]}] = {mean:.6f}")
            else:
                print(f"    pi[{DOMAIN_NAMES[d]}, {phase_names[k]}] = {mean:.6f}  [{lo:.6f}, {hi:.6f}]")

    # Interference lambda
    print()
    print("  Interference rate lambda[phase, domain]:")
    for k in range(N):
        for d in range(M):
            vals = [p["lambda"][k, d] for p in all_params]
            mean = np.mean(vals)
            lo, hi = np.min(vals), np.max(vals)
            if abs(hi - lo) < 1e-4:
                print(f"    lambda[{phase_names[k]}, {DOMAIN_NAMES[d]}] = {mean:.6f}")
            else:
                print(f"    lambda[{phase_names[k]}, {DOMAIN_NAMES[d]}] = {mean:.6f}  [{lo:.6f}, {hi:.6f}]")

    # Conflict gate g
    print()
    print("  Conflict gate g:")
    for k in range(N):
        vals = [p["g"][k] for p in all_params]
        mean = np.mean(vals)
        lo, hi = np.min(vals), np.max(vals)
        if abs(hi - lo) < 1e-4:
            print(f"    g[{phase_names[k]}] = {mean:.6f}")
        else:
            print(f"    g[{phase_names[k]}] = {mean:.6f}  [{lo:.6f}, {hi:.6f}]")

    # Find predicted optimum via multi-start L-BFGS-B
    print()
    print("=" * 70)
    print("PREDICTED OPTIMUM (L-BFGS-B, multi-start, no clamping)")
    print("=" * 70)

    def find_optimum_lbfgsb(predict_fn, n_restarts=64, rng_seed=42):
        """Find optimal StarCoder weights per phase via bounded optimization."""
        opt_rng = np.random.default_rng(rng_seed)
        bounds = [(0.0, 1.0)] * N  # StarCoder weight per phase

        def obj(sc_weights):
            W = np.zeros((1, N, M))
            W[0, :, 1] = sc_weights
            W[0, :, 0] = 1.0 - sc_weights
            return float(predict_fn(W)[0])

        best_val, best_x = np.inf, None
        for _ in range(n_restarts):
            x0 = opt_rng.uniform(0.0, 1.0, N)
            try:
                res = minimize(obj, x0, method="L-BFGS-B", bounds=bounds, options={"maxiter": 500, "ftol": 1e-12})
                if np.isfinite(res.fun) and res.fun < best_val:
                    best_val, best_x = float(res.fun), res.x.copy()
            except Exception:
                continue
        return best_x, best_val

    best_x, best_val = find_optimum_lbfgsb(all_predict_fns[0])
    print(f"  StarCoder weights by phase: ({best_x[0]:.10f}, {best_x[1]:.10f}, {best_x[2]:.10f})")
    print(f"  Predicted objective: {best_val:.10f}")

    # Show range across seeds
    print()
    print("  Optima across seeds:")
    for seed_i, fn in enumerate(all_predict_fns):
        x_s, ov = find_optimum_lbfgsb(fn, rng_seed=42 + seed_i)
        print(f"    seed {seed_i}: ({x_s[0]:.10f}, {x_s[1]:.10f}, {x_s[2]:.10f})  obj={ov:.10f}")


if __name__ == "__main__":
    main()
