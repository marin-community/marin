# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
# ]
# ///
"""Can deployment regularization rescue a model with a bad raw optimum?

The claim under test is that a better-fitting model with an implausible raw
optimum may be preferable to a worse-fitting model with a sane one, because a
deployment penalty pulls the solution back toward a reasonable prior anyway. If
that holds, raw-optimum plausibility should not gate model selection.

The penalty is the phase-weighted divergence to the token-proportional policy,

    R(p0, p1) = alpha KL(p0 || prop) + (1 - alpha) KL(p1 || prop),

which reduces to KL(a || prop) for a tied policy and also charges for phase
separation. Optimization minimizes ``L_hat + lambda R`` over both simplices.

Three prespecified predictions, written before the results were read:

P1  At matched lambda, the models with fantasy raw optima (bucket-resolved family
    GRP, hierarchical phase replay) retain more weight on dolma3_wikipedia than
    compact retained state and crs_plus. If regularization corrected the surface
    rather than merely shortening the step, this gap would close.
P2  Cross-model disagreement between constrained optima does not fall to near zero
    until lambda is large enough that every optimum is essentially proportional,
    so any convergence is trivial rather than informative.
P3  Ranking models by accuracy in a neighbourhood of their own constrained optimum
    tracks low-predicted-tail RMSE more closely than pooled RMSE.

P1 and P2 test whether the penalty fixes direction or only magnitude. P3 tests
which fit metric predicts accuracy where a regularized optimizer actually lands.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from benchmark_swarm39_crossscale_20260725 import fit_with_fixed_shape, select_shape_cross_scale
from scipy.optimize import minimize
from scipy.stats import spearmanr
from swarm39_harness_20260725 import REFERENCE_OUTPUTS, TABLE9, UNCHEATABLE, Fit, Model, Panel, load_scale
from swarm39_models_20260725 import crs_plus_extensions, nested_candidates, observatory_baselines

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "deployment_regularization_20260725"
SCALES = ("60m", "300m", "delphi_3e18")
TARGETS = (UNCHEATABLE, TABLE9)
LAMBDA_GRID = (0.0, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0)
RESTARTS = 8
SEED = 20260725
NEIGHBOURHOOD = 50
EPSILON = 1e-12
WIKIPEDIA = "dolma3_wikipedia"
COMPARED = (
    "compact_retained_state",
    "bucket_family_grp",
    "hierarchical_phase_replay",
    "effective_exposure_dsp",
    "crs_plus",
)


def simplex(vector: np.ndarray) -> np.ndarray:
    shifted = vector - vector.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def single_row(reference: Panel, phase0: np.ndarray, phase1: np.ndarray) -> Panel:
    return Panel(
        scale=reference.scale,
        split="candidate",
        alpha=reference.alpha,
        buckets=reference.buckets,
        c0=reference.c0,
        c1=reference.c1,
        family_index=reference.family_index,
        family_names=reference.family_names,
        phase0=phase0.reshape(1, -1),
        phase1=phase1.reshape(1, -1),
        targets={t: np.array([np.nan]) for t in TARGETS},
        series=np.array(["candidate"]),
        policy_class=np.array(["two_phase"]),
        group=np.array(["candidate"]),
        row_id=np.array(["candidate"]),
    )


def penalty(phase0: np.ndarray, phase1: np.ndarray, prior: np.ndarray, alpha: float) -> float:
    def kl(p: np.ndarray) -> float:
        safe = np.clip(p, EPSILON, None)
        return float((safe * np.log(safe / prior)).sum())

    return alpha * kl(phase0) + (1.0 - alpha) * kl(phase1)


def constrained_optimum(fit: Fit, model: Model, reference: Panel, lam: float, rng: np.random.Generator) -> dict:
    n = len(reference.buckets)
    prior = reference.proportional

    def objective(z: np.ndarray) -> float:
        phase0, phase1 = simplex(z[:n]), simplex(z[n:])
        panel = single_row(reference, phase0, phase1)
        return float(fit.predict(panel, model)[0]) + lam * penalty(phase0, phase1, prior, reference.alpha)

    best = None
    log_prior = np.log(prior)
    for restart in range(RESTARTS):
        start = np.concatenate([log_prior, log_prior]) if restart == 0 else rng.normal(0.0, 1.5, 2 * n)
        result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 400})
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    phase0, phase1 = simplex(best.x[:n]), simplex(best.x[n:])
    panel = single_row(reference, phase0, phase1)
    return {
        "phase0": phase0,
        "phase1": phase1,
        "aggregate": panel.aggregate[0],
        "epochs": panel.epochs[0],
        "kl_to_proportional": penalty(phase0, phase1, prior, reference.alpha),
    }


def neighbourhood_rmse(
    fit: Fit, model: Model, heldout: Panel, target: str, aggregate: np.ndarray
) -> tuple[float, float]:
    """Model accuracy on the heldout policies nearest its own constrained optimum."""
    distance = np.abs(heldout.aggregate - aggregate).sum(axis=1)
    order = np.argsort(distance)[:NEIGHBOURHOOD]
    subset = heldout.subset(np.isin(np.arange(len(heldout)), order))
    predicted = fit.predict(subset, model)
    observed = subset.targets[target]
    return float(np.sqrt(np.mean((predicted - observed) ** 2))), float(distance[order].max())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scales", nargs="*", default=["delphi_3e18"])
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    panels = {scale: load_scale(scale) for scale in SCALES}
    catalogue = {
        m.name: m for m in observatory_baselines(panels["delphi_3e18"][0]) + nested_candidates() + crs_plus_extensions()
    }

    rows, disagreement_rows = [], []
    for target in TARGETS:
        usable = [s for s in SCALES if np.isfinite(panels[s][0].targets[target]).any()]
        fit_panels = {s: panels[s][0] for s in usable}
        shapes = {name: select_shape_cross_scale(fit_panels, catalogue[name], target)[0] for name in COMPARED}
        for scale in args.scales:
            fit_panel, heldout = panels[scale]
            evaluation = heldout.subset(np.isfinite(heldout.targets[target]))
            wiki = fit_panel.buckets.index(WIKIPEDIA)
            fits = {name: fit_with_fixed_shape(fit_panel, catalogue[name], target, shapes[name]) for name in COMPARED}
            for lam in LAMBDA_GRID:
                optima = {}
                for name in COMPARED:
                    optimum = constrained_optimum(fits[name], catalogue[name], fit_panel, lam, rng)
                    optima[name] = optimum
                    rmse, radius = neighbourhood_rmse(
                        fits[name], catalogue[name], evaluation, target, optimum["aggregate"]
                    )
                    rows.append(
                        {
                            "scale": scale,
                            "target": "uncheatable" if target == UNCHEATABLE else "table9",
                            "model": name,
                            "lambda": lam,
                            "wikipedia_weight": float(optimum["aggregate"][wiki]),
                            "wikipedia_epochs": float(optimum["epochs"][wiki]),
                            "max_epochs": float(optimum["epochs"].max()),
                            "max_bucket_weight": float(optimum["aggregate"].max()),
                            "effective_buckets": float(1.0 / (optimum["aggregate"] ** 2).sum()),
                            "kl_to_proportional": optimum["kl_to_proportional"],
                            "l1_to_proportional": float(np.abs(optimum["aggregate"] - fit_panel.proportional).sum()),
                            "neighbourhood_rmse": rmse,
                            "neighbourhood_radius": radius,
                        }
                    )
                names = list(COMPARED)
                pairwise = [
                    float(np.abs(optima[a]["aggregate"] - optima[b]["aggregate"]).sum())
                    for i, a in enumerate(names)
                    for b in names[i + 1 :]
                ]
                fantasy = ["bucket_family_grp", "hierarchical_phase_replay"]
                sane = ["compact_retained_state", "crs_plus"]
                disagreement_rows.append(
                    {
                        "scale": scale,
                        "target": "uncheatable" if target == UNCHEATABLE else "table9",
                        "lambda": lam,
                        "mean_pairwise_l1": float(np.mean(pairwise)),
                        "max_pairwise_l1": float(np.max(pairwise)),
                        "fantasy_mean_wikipedia": float(np.mean([optima[m]["aggregate"][wiki] for m in fantasy])),
                        "sane_mean_wikipedia": float(np.mean([optima[m]["aggregate"][wiki] for m in sane])),
                        "mean_l1_to_proportional": float(
                            np.mean([np.abs(optima[m]["aggregate"] - fit_panel.proportional).sum() for m in names])
                        ),
                    }
                )
    frame = pd.DataFrame(rows)
    disagreement = pd.DataFrame(disagreement_rows)
    frame.to_csv(output / "constrained_optima.csv", index=False)
    disagreement.to_csv(output / "cross_model_disagreement.csv", index=False)

    # P3: does neighbourhood accuracy rank like low-tail or like pooled RMSE?
    metrics = pd.read_csv(REFERENCE_OUTPUTS / "swarm39_crossscale_20260725" / "heldout_metrics.csv")
    pooled = metrics[metrics["stratum_type"] == "pooled"]
    p3_rows = []
    for (scale, target, lam), block in frame.groupby(["scale", "target", "lambda"]):
        column = UNCHEATABLE if target == "uncheatable" else TABLE9
        reference = pooled[(pooled["scale"] == scale) & (pooled["target"] == column)]
        merged = block.merge(reference[["model", "rmse", "low_tail_rmse"]], on="model", how="inner")
        if len(merged) < 4:
            continue
        p3_rows.append(
            {
                "scale": scale,
                "target": target,
                "lambda": lam,
                "spearman_vs_pooled": float(spearmanr(merged["neighbourhood_rmse"], merged["rmse"]).statistic),
                "spearman_vs_low_tail": float(
                    spearmanr(merged["neighbourhood_rmse"], merged["low_tail_rmse"]).statistic
                ),
                "n_models": len(merged),
            }
        )
    p3 = pd.DataFrame(p3_rows)
    p3.to_csv(output / "neighbourhood_metric_agreement.csv", index=False)

    (output / "protocol.json").write_text(
        json.dumps(
            {
                "penalty": "alpha KL(p0||proportional) + (1-alpha) KL(p1||proportional)",
                "lambda_grid": list(LAMBDA_GRID),
                "restarts": RESTARTS,
                "seed": SEED,
                "neighbourhood_size": NEIGHBOURHOOD,
                "predictions_registered_before_results": ["P1", "P2", "P3"],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    pd.set_option("display.width", 250)
    print("=== P1/P2: cross-model disagreement and wikipedia tilt versus lambda ===")
    print(disagreement.to_string(index=False))
    print("\n=== P3: does neighbourhood accuracy rank like pooled or like low-tail RMSE? ===")
    print(p3.to_string(index=False))
    print("\n=== constrained optima, selected columns ===")
    print(
        frame[
            ["target", "model", "lambda", "wikipedia_weight", "max_epochs", "effective_buckets", "neighbourhood_rmse"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
