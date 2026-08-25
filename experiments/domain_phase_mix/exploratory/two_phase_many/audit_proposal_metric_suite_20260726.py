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
"""Score every candidate surrogate on the full proposal-metric suite.

Runs the five measures in ``proposal_metrics_20260726`` over the 3e18 panel. The
one that decides the project is ``phase_decision_skill``: the two ranking metrics
cannot see whether a model calls the two-phase-versus-tied decision correctly, so
a model could top the leaderboard on both while having no phase skill at all.

Phase skill is scored out of fold. The aggregate-matched pairs live in the fit
panel itself, so an in-sample number would be meaningless; each fold refits on
four fifths of the pairs and predicts ``Delta`` on the held-out fifth.

Stability holds the nonlinear shape fixed and resamples the panel, so it isolates
head instability. Shape-selection instability is a separate and larger effect
measured by the frozen-versus-refit convergence arms.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd
from audit_proposal_regret_convergence_20260726 import fit_dsp_frozen, kl_to_proportional
from benchmark_swarm39_crossscale_20260725 import fit_with_fixed_shape
from dsp_exact_baseline_20260726 import MODEL_NAME as DSP_NAME
from dsp_exact_baseline_20260726 import _as_fit, _fit_once, dsp_exact_model
from phase_order_spine_20260725 import load_paired_panel
from proposal_metrics_20260726 import (
    DEFAULT_TOP_K,
    optimum_sanity,
    phase_decision_skill,
    proposal_stability,
    regret_in_sigma,
    top_k_percentile,
)
from scipy.optimize import minimize
from swarm39_harness_20260725 import (
    REFERENCE_OUTPUTS,
    TABLE9,
    UNCHEATABLE,
    Panel,
    fit_model,
    grouped_splits,
    load_scale,
    provenance,
    support_distance,
)
from swarm39_models_20260725 import crs_plus_extensions, nested_candidates, observatory_baselines

DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "proposal_metric_suite_20260726"
SCALE = "delphi_3e18"
TARGETS = (UNCHEATABLE, TABLE9)
KAPPA_GRID = (0.25, 0.5, 1.0)
RUN_SIGMA = {UNCHEATABLE: 0.000963, TABLE9: 0.003121}
GRID_MODELS = ("compact_retained_state", "crs_plus", "bucket_family_grp", "hierarchical_phase_replay")
# Matches the delphi_3e18 tied panel's row ids; the spine asserts correspondence.
TIED_PREFIX = r"^singleavg_fit_\d+_"
BOOTSTRAP_DRAWS = 50
SEED = 20260726
OPTIMUM_LAMBDAS = (0.0, 0.1)
OPTIMUM_RESTARTS = 4
EPSILON = 1e-12


def tied_twin(panel: Panel) -> Panel:
    """The aggregate-matched single-phase counterpart of every row."""
    aggregate = panel.aggregate
    return dataclasses.replace(panel, phase0=aggregate.copy(), phase1=aggregate.copy())


def paired_subset(reference: Panel, paired) -> tuple[Panel, np.ndarray]:
    """Reorder the harness panel onto the spine's paired rows."""
    position = {row: i for i, row in enumerate(reference.row_id)}
    order = np.asarray([position[row] for row in paired.row_id], dtype=int)
    mask = np.zeros(len(reference), dtype=bool)
    mask[order] = True
    subset = reference.subset(mask)
    # subset preserves panel order, so reorder the spine arrays to match it.
    rank = {row: i for i, row in enumerate(subset.row_id)}
    return subset, np.asarray([rank[row] for row in paired.row_id], dtype=int)


def simplex(vector: np.ndarray) -> np.ndarray:
    shifted = vector - vector.max()
    weights = np.exp(shifted)
    return weights / weights.sum()


def constrained_optimum(fit, model, reference: Panel, lam: float, rng: np.random.Generator) -> tuple:
    n = len(reference.buckets)
    prior = reference.proportional
    log_prior = np.log(prior)

    def kl(p: np.ndarray) -> float:
        safe = np.clip(p, EPSILON, None)
        return float((safe * np.log(safe / prior)).sum())

    def objective(z: np.ndarray) -> float:
        p0, p1 = simplex(z[:n]), simplex(z[n:])
        row = dataclasses.replace(
            reference.subset(np.arange(len(reference)) == 0),
            phase0=p0.reshape(1, -1),
            phase1=p1.reshape(1, -1),
        )
        return float(fit.predict(row, model)[0]) + lam * (reference.alpha * kl(p0) + (1.0 - reference.alpha) * kl(p1))

    best = None
    for restart in range(OPTIMUM_RESTARTS):
        start = np.concatenate([log_prior, log_prior]) if restart == 0 else rng.normal(0.0, 1.5, 2 * n)
        result = minimize(objective, start, method="L-BFGS-B", options={"maxiter": 400})
        if best is None or result.fun < best.fun:
            best = result
    assert best is not None, "optimizer produced no result"
    p0, p1 = simplex(best.x[:n]), simplex(best.x[n:])
    row = dataclasses.replace(
        reference.subset(np.arange(len(reference)) == 0),
        phase0=p0.reshape(1, -1),
        phase1=p1.reshape(1, -1),
    )
    return row.aggregate[0], row.epochs[0], row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    args = parser.parse_args()
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    fit_panel, heldout = load_scale(SCALE)
    catalogue = {m.name: m for m in observatory_baselines(fit_panel) + nested_candidates() + crs_plus_extensions()}
    paired = load_paired_panel(SCALE, TIED_PREFIX)
    names = [*GRID_MODELS, DSP_NAME]

    ranking, phase, stability, gate = [], [], [], []
    for target in TARGETS:
        sigma = RUN_SIGMA[target]
        archive = heldout.subset(np.isfinite(heldout.targets[target]))
        observed = archive.targets[target]
        kl = kl_to_proportional(archive.aggregate, fit_panel.proportional)
        full = fit_panel.subset(np.isfinite(fit_panel.targets[target]))

        fits, shapes = {}, {}
        for name in GRID_MODELS:
            fits[name] = fit_model(full, catalogue[name], target)
            shapes[name] = fits[name].shape
        dsp_fitted, dsp_shape = _fit_once(full, target)
        fits[DSP_NAME] = _as_fit(dsp_fitted, dsp_shape, full, oof_rmse=float("nan"))
        shapes[DSP_NAME] = dsp_shape
        model_of = {name: catalogue[name] for name in GRID_MODELS} | {DSP_NAME: dsp_exact_model()}

        # --- ranking: top-k percentile and regret in sigma, swept over kappa ---
        for name in names:
            predicted = fits[name].predict(archive, model_of[name])
            for kappa in (*KAPPA_GRID, float("inf")):
                feasible = kl <= kappa
                if feasible.sum() < 20:
                    continue
                ranking.append(
                    {
                        "target": target,
                        "model": name,
                        "kappa": kappa,
                        **top_k_percentile(predicted, observed, feasible, args.top_k),
                        **regret_in_sigma(predicted, observed, feasible, sigma, args.top_k),
                    }
                )

        # --- phase decision skill, out of fold on aggregate-matched pairs ---
        pair_panel, reorder = paired_subset(full, paired)
        observed_delta = paired.delta[target][np.argsort(reorder)]
        usable = np.isfinite(observed_delta)
        for name in names:
            predicted_delta = np.full(len(pair_panel), np.nan)
            for train, test in grouped_splits(pair_panel, 5, 0):
                train_panel = pair_panel.subset(train)
                if name == DSP_NAME:
                    fitted, shape = _fit_once(train_panel, target)
                    fold_fit = _as_fit(fitted, shape, train_panel, oof_rmse=float("nan"))
                else:
                    fold_fit = fit_model(train_panel, catalogue[name], target)
                held = pair_panel.subset(test)
                predicted_delta[test] = fold_fit.predict(held, model_of[name]) - fold_fit.predict(
                    tied_twin(held), model_of[name]
                )
            keep = usable & np.isfinite(predicted_delta)
            phase.append(
                {
                    "target": target,
                    "model": name,
                    **phase_decision_skill(predicted_delta[keep], observed_delta[keep]),
                }
            )
            print(f"phase skill {target} {name}: done", flush=True)

        # --- proposal stability under panel bootstrap, shape held fixed ---
        for name in names:
            picks, percentiles = [], []
            feasible = kl <= 0.5
            for _ in range(BOOTSTRAP_DRAWS):
                draw = rng.integers(0, len(full), len(full))
                mask = np.zeros(len(full), dtype=bool)
                mask[np.unique(draw)] = True
                resampled = full.subset(mask)
                boot = (
                    fit_dsp_frozen(resampled, target, shapes[DSP_NAME])
                    if name == DSP_NAME
                    else fit_with_fixed_shape(resampled, catalogue[name], target, shapes[name])
                )
                predicted = boot.predict(archive, model_of[name])
                order = np.flatnonzero(feasible)[np.argsort(predicted[feasible])[: args.top_k]]
                picks.append(order)
                realized = float(np.min(observed[order]))
                percentiles.append(float((observed[feasible] < realized).mean()))
            stability.append({"target": target, "model": name, "kappa": 0.5, **proposal_stability(picks, percentiles)})
            print(f"stability {target} {name}: done", flush=True)

        # --- optimum sanity gate ---
        for name in names:
            for lam in OPTIMUM_LAMBDAS:
                aggregate, epochs, row = constrained_optimum(fits[name], model_of[name], full, lam, rng)
                distance = float(support_distance(full, row)[0])
                gate.append(
                    {"target": target, "model": name, "lambda": lam, **optimum_sanity(aggregate, epochs, distance)}
                )
            print(f"gate {target} {name}: done", flush=True)

    frames = {
        "ranking": pd.DataFrame(ranking),
        "phase_skill": pd.DataFrame(phase),
        "stability": pd.DataFrame(stability),
        "optimum_gate": pd.DataFrame(gate),
    }
    for key, frame in frames.items():
        frame.to_csv(output / f"{key}.csv", index=False)
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "scale": SCALE,
                "top_k": args.top_k,
                "kappa_grid": [*KAPPA_GRID, "inf"],
                "phase_skill": "out-of-fold on exact aggregate-matched pairs; 5 grouped folds",
                "stability": f"{BOOTSTRAP_DRAWS} panel bootstraps, nonlinear shape held fixed",
                "optimum_gate": {"lambdas": list(OPTIMUM_LAMBDAS), "restarts": OPTIMUM_RESTARTS},
                "run_sigma": RUN_SIGMA,
                "provenance_sha256": provenance(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )

    pd.set_option("display.width", 240)
    print("\n=== PHASE DECISION SKILL (1 = oracle, 0 = no better than always tied) ===")
    print(
        frames["phase_skill"][
            [
                "target",
                "model",
                "n_pairs",
                "base_rate_two_phase_wins",
                "phase_skill_score",
                "decision_accuracy",
                "always_tied_accuracy",
                "delta_correlation",
                "model_realized_bpb",
                "oracle_realized_bpb",
            ]
        ].to_string(index=False)
    )
    print("\n=== RANKING: top-k percentile by kappa ===")
    print(
        frames["ranking"]
        .pivot_table(index=["target", "model"], columns="kappa", values="picked_percentile")
        .round(4)
        .to_string()
    )
    print("\n=== STABILITY under panel bootstrap ===")
    print(
        frames["stability"][
            ["target", "model", "mean_pairwise_jaccard", "modal_pick_frequency", "percentile_mean", "percentile_sd"]
        ].to_string(index=False)
    )
    print("\n=== OPTIMUM SANITY GATE ===")
    print(
        frames["optimum_gate"][
            ["target", "model", "lambda", "effective_buckets", "max_simulated_epochs", "support_distance", "passes_gate"]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
