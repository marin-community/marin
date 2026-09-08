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
"""Qualitative audit of the raw two-phase optimum for each 39-bucket model.

A model is only useful for mixture design if optimizing it yields a policy a
practitioner would actually run. This script optimizes each fitted model over
both phase simplices with no deployment penalty, then reports the diagnostics
that decide plausibility:

* predicted BPB and the gap to the best observed heldout policy;
* maximum bucket weight and maximum simulated epochs;
* phase total variation;
* nearest-neighbour support distance to the fit panel;
* how much aggregate mass lands on buckets that are already repeated;
* the top buckets by weight, for qualitative inspection.

An optimum that predicts far below every observed policy while sitting well
outside the fit panel's support is a fantasy, regardless of its heldout metrics.
The saturation and ridge grids are also extended here, because the benchmark
selected the top of both grids in every configuration and a boundary selection
has to be checked rather than reported.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from swarm39_harness_20260725 import (
    REFERENCE_OUTPUTS,
    TABLE9,
    UNCHEATABLE,
    Model,
    Panel,
    fit_model,
    load_scale,
    provenance,
)
from swarm39_models_20260725 import (
    build_bounded_saturation,
    candidates,
    observatory_baselines,
)

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "swarm39_raw_optima_20260725"
TARGETS = (UNCHEATABLE, TABLE9)
RESTARTS = 12
OPTIMIZER_SEED = 20260725
MAX_ITERATIONS = 400


def single_row_panel(reference: Panel, phase0: np.ndarray, phase1: np.ndarray) -> Panel:
    """Wrap one candidate policy in a Panel so any model can score it."""
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


def _simplex(vector: np.ndarray) -> np.ndarray:
    """Softmax parameterization keeps the optimizer inside the simplex interior."""
    shifted = vector - vector.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def optimize(fit, model, reference: Panel, rng: np.random.Generator) -> dict:
    """Minimize predicted BPB over both phase mixtures from several restarts."""
    n = len(reference.buckets)

    def objective(z: np.ndarray) -> float:
        phase0 = _simplex(z[:n])
        phase1 = _simplex(z[n:])
        panel = single_row_panel(reference, phase0, phase1)
        return float(fit.predict(panel, model)[0])

    best = None
    for restart in range(RESTARTS):
        if restart == 0:
            start = np.zeros(2 * n)
        elif restart == 1:
            # Start from the token-proportional policy, the natural neutral point.
            base = np.log(reference.proportional)
            start = np.concatenate([base, base])
        else:
            start = rng.normal(0.0, 1.5, 2 * n)
        result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": MAX_ITERATIONS})
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None
    phase0 = _simplex(best.x[:n])
    phase1 = _simplex(best.x[n:])
    return {"phase0": phase0, "phase1": phase1, "predicted": float(best.fun)}


def describe(optimum: dict, reference: Panel, heldout: Panel, target: str) -> dict:
    phase0, phase1 = optimum["phase0"], optimum["phase1"]
    panel = single_row_panel(reference, phase0, phase1)
    aggregate = panel.aggregate[0]
    epochs = panel.epochs[0]
    oversampling = panel.oversampling[0]
    fit_aggregate = reference.aggregate
    distance = float(np.abs(fit_aggregate - aggregate).sum(axis=1).min())
    fit_radius = float(np.quantile(np.abs(fit_aggregate[:, None, :] - fit_aggregate[None, :, :]).sum(axis=2), 0.95))
    observed = heldout.targets[target]
    best_observed = float(np.nanmin(observed))
    repeated = epochs > 1.0
    order = np.argsort(-aggregate)
    return {
        "predicted_bpb": optimum["predicted"],
        "best_observed_heldout_bpb": best_observed,
        "predicted_minus_best_observed": optimum["predicted"] - best_observed,
        "predicts_below_every_observation": bool(optimum["predicted"] < best_observed),
        "phase_tv": float(0.5 * np.abs(phase1 - phase0).sum()),
        "max_bucket_weight": float(aggregate.max()),
        "max_simulated_epochs": float(epochs.max()),
        "max_oversampling_ratio": float(oversampling.max()),
        "mass_on_repeated_buckets": float(aggregate[repeated].sum()),
        "buckets_over_one_epoch": int(repeated.sum()),
        "effective_bucket_count": float(1.0 / (aggregate**2).sum()),
        "support_distance_l1": distance,
        "fit_panel_p95_pairwise_l1": fit_radius,
        "support_distance_over_p95": distance / fit_radius,
        "top_buckets": "; ".join(f"{reference.buckets[i]}={aggregate[i]:.3f}(E={epochs[i]:.1f})" for i in order[:5]),
    }


def extended_grid_check(scale: str, output: Path) -> pd.DataFrame:
    """Re-select the bounded-saturation shape over a widened grid.

    The benchmark chose the largest saturation scale and the largest ridge in
    every configuration. If the widened grid keeps moving outward, the constraint
    is fighting the data rather than describing it, and that has to be stated.
    """

    def wide_shapes():
        for saturation_epochs in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 64.0):
            for power in (0.7, 1.0):
                for late_multiplier in (1.0, 2.0):
                    for overload_threshold in (1.0, 2.0, 4.0):
                        yield {
                            "saturation_epochs": saturation_epochs,
                            "power": power,
                            "late_multiplier": late_multiplier,
                            "forgetting_rate": 0.25,
                            "overload_threshold": overload_threshold,
                        }

    model = Model(
        "bounded_saturation_wide",
        build_bounded_saturation,
        wide_shapes,
        l2_grid=(0.1, 1.0, 10.0, 100.0),
    )
    fit_panel, _ = load_scale(scale)
    rows = []
    for target in TARGETS:
        if not np.isfinite(fit_panel.targets[target]).any():
            continue
        fit = fit_model(fit_panel, model, target)
        rows.append(
            {
                "scale": scale,
                "target": target,
                "selected_saturation_epochs": fit.shape["saturation_epochs"],
                "selected_power": fit.shape["power"],
                "selected_overload_threshold": fit.shape["overload_threshold"],
                "selected_l2": fit.l2,
                "oof_rmse": fit.oof_rmse,
                "saturation_at_grid_top": bool(fit.shape["saturation_epochs"] == 64.0),
                "l2_at_grid_top": bool(fit.l2 == 100.0),
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--scales", nargs="*", default=["delphi_3e18"])
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(OPTIMIZER_SEED)

    grid_rows, optimum_rows = [], []
    for scale in args.scales:
        grid_rows.append(extended_grid_check(scale, output))
        fit_panel, heldout = load_scale(scale)
        models = observatory_baselines(fit_panel) + candidates()
        for target in TARGETS:
            if not np.isfinite(fit_panel.targets[target]).any():
                continue
            for model in models:
                fit = fit_model(fit_panel, model, target)
                optimum = optimize(fit, model, fit_panel, rng)
                optimum_rows.append(
                    {
                        "scale": scale,
                        "target": target,
                        "model": model.name,
                        "oof_rmse": fit.oof_rmse,
                        **describe(optimum, fit_panel, heldout, target),
                    }
                )
    grid = pd.concat(grid_rows, ignore_index=True)
    optima = pd.DataFrame(optimum_rows)
    grid.to_csv(output / "extended_grid_check.csv", index=False)
    optima.to_csv(output / "raw_optima.csv", index=False)

    protocol = {
        "restarts": RESTARTS,
        "optimizer": "L-BFGS-B over a softmax parameterization of both phase simplices",
        "optimizer_seed": OPTIMIZER_SEED,
        "no_deployment_penalty": True,
        "sealed_targeted_pairwise_panel_accessed": False,
        "provenance_sha256": provenance(),
    }
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")

    pd.set_option("display.width", 260)
    print("=== extended grid check (does the boundary selection persist?) ===")
    print(grid.to_string(index=False))
    print("\n=== raw two-phase optima, no deployment penalty ===")
    columns = [
        "target",
        "model",
        "predicted_bpb",
        "best_observed_heldout_bpb",
        "predicted_minus_best_observed",
        "phase_tv",
        "max_bucket_weight",
        "max_simulated_epochs",
        "mass_on_repeated_buckets",
        "effective_bucket_count",
        "support_distance_over_p95",
    ]
    for target in TARGETS:
        block = optima[optima["target"] == target]
        if block.empty:
            continue
        print(f"\n-- {target} --")
        print(block.sort_values("predicted_minus_best_observed", ascending=False)[columns].to_string(index=False))
    print("\n=== top buckets at each optimum ===")
    for row in optima.itertuples():
        print(f"{row.target:18s} {row.model:26s} {row.top_buckets}")


if __name__ == "__main__":
    main()
