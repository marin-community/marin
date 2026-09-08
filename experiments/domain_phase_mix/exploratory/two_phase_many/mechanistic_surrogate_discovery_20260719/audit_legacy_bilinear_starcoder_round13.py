# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "matplotlib>=3.10",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Audit the legacy bilinear phase-state model on current StarCoder data."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


from experiments.domain_phase_mix.exploratory.two_phase_many import (
    benchmark_low_rank_phase_state_interaction as legacy,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import export_mixture_fit_observatory as observatory
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    audit_starcoder_hessian_equilibrium_round11 as scalar_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    screen_paired_dynamics_round1 as metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round13_legacy_bilinear_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
RANKS = (0, 1, 2)
INTERACTION_L2 = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0)
N_STARTS = 8
MAX_ITERATIONS = 60
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[legacy.Config]:
    return [legacy.Config(0, legacy.BASE_L2)] + [
        legacy.Config(rank, l2) for rank in RANKS if rank > 0 for l2 in INTERACTION_L2
    ]


def interaction_matrix(model: legacy.FittedModel) -> np.ndarray:
    if model.left.size == 0:
        return np.zeros((model.num_domains, model.num_domains), dtype=float)
    return model.left.T @ model.right


def normalized_matrix(model: legacy.FittedModel) -> np.ndarray:
    matrix = interaction_matrix(model)
    norm = np.linalg.norm(matrix)
    return matrix / norm if norm > 1e-12 else matrix


def matrix_cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1e-12:
        return 0.0
    return float(np.sum(left * right) / denominator)


def fit_predict(
    dataset: Any,
    train: np.ndarray,
    test: np.ndarray,
    config: legacy.Config,
) -> tuple[np.ndarray, legacy.FittedModel]:
    model = legacy.fit_model(
        dataset,
        train,
        config,
        max_iterations=MAX_ITERATIONS,
        starts=N_STARTS,
    )
    return legacy.predict(model, dataset, test), model


def inner_selection(
    dataset: Any,
    panel: paired.PairedPanel,
    outer_train: np.ndarray,
    all_configs: list[legacy.Config],
    seed: int,
) -> tuple[legacy.Config, pd.DataFrame]:
    folds = scalar_audit.stratified_folds(panel, outer_train, 4, seed)
    rows = []
    for config in all_configs:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in folds:
            prediction[test], _model = fit_predict(dataset, train, test, config)
        score = metrics.scalar_metrics(dataset.y[outer_train], prediction[outer_train])
        rows.append({"config": config.name, **asdict(config), **score})
    table = pd.DataFrame(rows).sort_values(["rmse", "regret_at_1", "worst_optimism", "config"])
    selected_row = table.iloc[0]
    selected = next(config for config in all_configs if config.name == selected_row["config"])
    return selected, table


def nested_audit(
    dataset: Any,
    panel: paired.PairedPanel,
    all_configs: list[legacy.Config],
) -> tuple[np.ndarray, pd.DataFrame, list[legacy.FittedModel]]:
    prediction = np.full(dataset.n, np.nan, dtype=float)
    rows = []
    models = []
    for outer_fold, (train, test) in enumerate(starcoder.surface_folds(panel)):
        selected, inner = inner_selection(dataset, panel, train, all_configs, 20260719 + outer_fold)
        prediction[test], model = fit_predict(dataset, train, test, selected)
        models.append(model)
        rows.append(
            {
                "surface": dataset.name,
                "outer_fold": outer_fold,
                "selected_config": selected.name,
                "selected_rank": selected.rank,
                "selected_l2": selected.interaction_l2,
                "inner_rmse": float(inner.iloc[0]["rmse"]),
                "matrix_norm": float(np.linalg.norm(interaction_matrix(model))),
                "matrix_00": float(interaction_matrix(model)[0, 0]),
                "matrix_01": float(interaction_matrix(model)[0, 1]),
                "matrix_10": float(interaction_matrix(model)[1, 0]),
                "matrix_11": float(interaction_matrix(model)[1, 1]),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested predictions for {dataset.name}")
    return prediction, pd.DataFrame(rows), models


def full_oof_selection(
    dataset: Any,
    panel: paired.PairedPanel,
    all_configs: list[legacy.Config],
) -> tuple[legacy.Config, np.ndarray, pd.DataFrame]:
    rows = []
    predictions: dict[str, np.ndarray] = {}
    folds = starcoder.surface_folds(panel)
    for config in all_configs:
        prediction = np.full(dataset.n, np.nan, dtype=float)
        for train, test in folds:
            prediction[test], _model = fit_predict(dataset, train, test, config)
        predictions[config.name] = prediction
        rows.append(
            {
                "surface": dataset.name,
                "config": config.name,
                **asdict(config),
                **metrics.scalar_metrics(dataset.y, prediction),
            }
        )
    table = pd.DataFrame(rows).sort_values(["rmse", "regret_at_1", "worst_optimism", "config"])
    selected = next(config for config in all_configs if config.name == table.iloc[0]["config"])
    return selected, predictions[selected.name], table


def leave_region_out(dataset: Any, panel: paired.PairedPanel, config: legacy.Config) -> pd.DataFrame:
    contrast = dataset.weights[:, 1, 1] - dataset.weights[:, 0, 1]
    regions = {
        "late_rare_enriched": contrast > 0.1,
        "early_rare_enriched": contrast < -0.1,
        "near_phase_tied": np.abs(contrast) <= 0.1,
    }
    rows = []
    for name, mask in regions.items():
        test = np.flatnonzero(mask)
        train = np.flatnonzero(~mask)
        prediction, _model = fit_predict(dataset, train, test, config)
        rows.append(
            {
                "surface": dataset.name,
                "region": name,
                "n_train": len(train),
                "n_test": len(test),
                **metrics.scalar_metrics(dataset.y[test], prediction),
            }
        )
    return pd.DataFrame(rows)


def model_from_start(dataset: Any, config: legacy.Config, seed: int) -> tuple[legacy.FittedModel, float]:
    indices = np.arange(dataset.n)
    exposure0, exposure1 = legacy.pooled.phase_exposures(dataset, indices)
    target = dataset.y
    mu0 = legacy.pooled.selected_mu(exposure0, target)
    mu1 = legacy.pooled.selected_mu(exposure1, target)
    rho0 = legacy.recency.channel_base(exposure0).rho
    rho1 = legacy.recency.channel_base(exposure1).rho
    base_design = np.hstack([legacy.pooled.bowl_design(exposure0, mu0), legacy.pooled.bowl_design(exposure1, mu1)])
    state0 = legacy.states(exposure0, rho0)
    state1 = legacy.states(exposure1, rho1)
    intercept, base_coef, left, right, objective = legacy.fit_from_start(
        base_design,
        state0,
        state1,
        target,
        config,
        legacy.initial_factors(config.rank, dataset.m, seed),
        MAX_ITERATIONS,
    )
    return (
        legacy.FittedModel(config, mu0, mu1, rho0, rho1, intercept, base_coef, left, right),
        objective,
    )


def start_stability(dataset: Any, config: legacy.Config) -> pd.DataFrame:
    if config.rank == 0:
        return pd.DataFrame()
    fits = [model_from_start(dataset, config, seed) for seed in range(20)]
    best = min(objective for _model, objective in fits)
    rows = []
    for seed, (model, objective) in enumerate(fits):
        rows.append(
            {
                "surface": dataset.name,
                "seed": seed,
                "objective": objective,
                "relative_objective": objective / max(best, 1e-12) - 1.0,
                "matrix_norm": float(np.linalg.norm(interaction_matrix(model))),
                "cosine_to_best": matrix_cosine(
                    normalized_matrix(model),
                    normalized_matrix(min(fits, key=lambda item: item[1])[0]),
                ),
            }
        )
    return pd.DataFrame(rows)


def fold_stability(models: list[legacy.FittedModel], surface: str) -> dict[str, float]:
    matrices = [interaction_matrix(model) for model in models]
    norms = np.asarray([np.linalg.norm(matrix) for matrix in matrices], dtype=float)
    cosines = [matrix_cosine(left, right) for left, right in combinations(matrices, 2)]
    return {
        "surface": surface,
        "minimum_pairwise_matrix_cosine": float(min(cosines)),
        "mean_pairwise_matrix_cosine": float(np.mean(cosines)),
        "matrix_norm_cv": float(np.std(norms, ddof=1) / max(np.mean(norms), 1e-12)),
        "nonzero_rank_fold_count": int(sum(model.config.rank > 0 for model in models)),
        "rank1_fold_count": int(sum(model.config.rank == 1 for model in models)),
    }


def optimum(dataset: Any, config: legacy.Config) -> tuple[dict[str, Any], pd.DataFrame]:
    model = legacy.fit_model(dataset, np.arange(dataset.n), config, max_iterations=MAX_ITERATIONS, starts=20)
    grid = np.linspace(0.0, 1.0, 201)
    p0, p1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [np.column_stack([1.0 - p0.ravel(), p0.ravel()]), np.column_stack([1.0 - p1.ravel(), p1.ravel()])],
        axis=1,
    )
    synthetic = legacy.pooled.Dataset(
        name=dataset.name,
        frame=pd.DataFrame(index=np.arange(len(weights))),
        y=np.zeros(len(weights)),
        weights=weights,
        c0=dataset.c0,
        c1=dataset.c1,
        domain_names=dataset.domain_names,
    )
    prediction = legacy.predict(model, synthetic, np.arange(len(weights)))
    best = int(np.argmin(prediction))
    observed_best = int(np.argmin(dataset.y))
    summary = {
        "surface": dataset.name,
        "selected_config": config.name,
        "phase0_rare": float(p0.ravel()[best]),
        "phase1_rare": float(p1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "observed_best_phase0_rare": float(dataset.weights[observed_best, 0, 1]),
        "observed_best_phase1_rare": float(dataset.weights[observed_best, 1, 1]),
        "observed_best_bpb": float(dataset.y[observed_best]),
        "distance_to_observed_best": float(
            np.hypot(
                p0.ravel()[best] - dataset.weights[observed_best, 0, 1],
                p1.ravel()[best] - dataset.weights[observed_best, 1, 1],
            )
        ),
        "matrix_norm": float(np.linalg.norm(interaction_matrix(model))),
    }
    surface = pd.DataFrame({"phase0_rare": p0.ravel(), "phase1_rare": p1.ravel(), "predicted_bpb": prediction})
    return summary, surface


def render_surface(dataset: Any, surface: pd.DataFrame, summary: dict[str, Any], output: Path) -> None:
    size = round(np.sqrt(len(surface)))
    x = surface["phase0_rare"].to_numpy().reshape(size, size)[:, 0]
    z = surface["predicted_bpb"].to_numpy().reshape(size, size)
    figure = go.Figure(
        [
            go.Surface(x=x, y=x, z=z.T, colorscale="RdYlGn_r", opacity=0.72, name="Predicted"),
            go.Scatter3d(
                x=dataset.weights[:, 0, 1],
                y=dataset.weights[:, 1, 1],
                z=dataset.y,
                mode="markers",
                marker={"size": 4, "color": dataset.y, "colorscale": "RdYlGn_r"},
                name="Observed",
            ),
            go.Scatter3d(
                x=[summary["phase0_rare"]],
                y=[summary["phase1_rare"]],
                z=[summary["predicted_bpb"]],
                mode="markers",
                marker={"size": 9, "symbol": "diamond", "color": "#111827"},
                name="Predicted optimum",
            ),
        ]
    )
    figure.update_layout(
        title=f"{dataset.name}: legacy bilinear phase-state interaction",
        template="plotly_white",
        scene={
            "xaxis_title": "Phase 0 StarCoder weight",
            "yaxis_title": "Phase 1 StarCoder weight",
            "zaxis_title": "BPB",
        },
        height=850,
    )
    figure.write_html(output, include_plotlyjs="cdn", config=PLOT_CONFIG)


def update_status(status: str, evidence: str, output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    mask = registry["id"].eq("LPSI")
    registry.loc[mask, "status"] = status
    registry.loc[mask, "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_13_legacy_updated_surface_gate",
        "candidate_id": "LPSI",
        "candidate_family": "Legacy low-rank phase-state interaction",
        "hyperparameters": "Frozen preregistered rank/l2 grid with nested selection and invariant-matrix stability",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_13_legacy_audit_preregistration",
        "novelty_class": "Legacy diagnostic only",
        "evaluation_status": status,
        "evidence_path": str(output_dir.relative_to(OUTPUT_ROOT)),
        "notes": evidence,
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosine = observatory.load_cosine_starcoder()
    datasets = [cosine, starcoder.load_refined_wsd80(cosine)]
    all_configs = configs()
    metric_rows = []
    selection_tables = []
    fold_tables = []
    stability_rows = []
    start_tables = []
    region_tables = []
    optimum_rows = []
    for dataset in datasets:
        panel = starcoder.panel_from_dataset(dataset)
        selected, oof_prediction, selection = full_oof_selection(dataset, panel, all_configs)
        nested_prediction, folds, models = nested_audit(dataset, panel, all_configs)
        metric_rows.append(
            {
                "surface": dataset.name,
                "selected_config": selected.name,
                "selected_rank": selected.rank,
                "selected_l2": selected.interaction_l2,
                "nominal_parameter_count": (4 + 2 * selected.rank) * dataset.m + 1,
                **{
                    f"selection_{key}": value for key, value in metrics.scalar_metrics(dataset.y, oof_prediction).items()
                },
                **{
                    f"nested_{key}": value for key, value in metrics.scalar_metrics(dataset.y, nested_prediction).items()
                },
            }
        )
        selection_tables.append(selection)
        fold_tables.append(folds)
        stability_rows.append(fold_stability(models, dataset.name))
        start_tables.append(start_stability(dataset, selected))
        region_tables.append(leave_region_out(dataset, panel, selected))
        summary, surface = optimum(dataset, selected)
        optimum_rows.append(summary)
        surface.to_csv(args.output_dir / f"{dataset.name}__surface.csv", index=False)
        render_surface(dataset, surface, summary, args.output_dir / f"{dataset.name}__surface.html")

    metrics_table = pd.DataFrame(metric_rows)
    selection_table = pd.concat(selection_tables, ignore_index=True)
    fold_table = pd.concat(fold_tables, ignore_index=True)
    stability = pd.DataFrame(stability_rows)
    starts = pd.concat(start_tables, ignore_index=True)
    regions = pd.concat(region_tables, ignore_index=True)
    optima = pd.DataFrame(optimum_rows)
    prior = pd.read_csv(OUTPUT_ROOT / "round1_starcoder_shape_refined107/surface_oof_metrics.csv")
    prior_best = prior.groupby("surface", as_index=False)["rmse"].min().rename(columns={"rmse": "prior_best_rmse"})
    comparison = metrics_table[["surface", "nested_rmse"]].merge(prior_best, on="surface")
    comparison["relative_rmse"] = comparison["nested_rmse"] / comparison["prior_best_rmse"] - 1.0

    metrics_table.to_csv(args.output_dir / "surface_metrics.csv", index=False)
    selection_table.to_csv(args.output_dir / "config_oof_grid.csv", index=False)
    fold_table.to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    stability.to_csv(args.output_dir / "interaction_matrix_stability.csv", index=False)
    starts.to_csv(args.output_dir / "start_stability.csv", index=False)
    regions.to_csv(args.output_dir / "leave_region_out.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    comparison.to_csv(args.output_dir / "prior_comparison.csv", index=False)

    near_best_starts = starts.loc[starts["relative_objective"] <= 0.01]
    rank1_global = bool(metrics_table["selected_rank"].eq(1).all())
    rank1_folds = bool((stability["rank1_fold_count"] >= 3).all())
    fold_stable = bool(
        (stability["minimum_pairwise_matrix_cosine"] >= 0.8).all() and (stability["matrix_norm_cv"] <= 0.5).all()
    )
    start_stable = bool(
        not near_best_starts.empty and (near_best_starts.groupby("surface")["cosine_to_best"].min() >= 0.95).all()
    )
    shape_ok = bool((comparison["relative_rmse"] <= 0.05).all())
    optimum_ok = bool((optima["distance_to_observed_best"] <= 0.15).all())
    passed = rank1_global and rank1_folds and fold_stable and start_stable and shape_ok and optimum_ok
    status = "legacy_shape_diagnostic_supported" if passed else "legacy_result_rejected"
    evidence = (
        f"rank1_global={rank1_global}; rank1_folds={rank1_folds}; fold_matrix_stable={fold_stable}; "
        f"start_matrix_stable={start_stable}; within_5pct_prior_shape={shape_ok}; optimum_distance_ok={optimum_ok}."
    )
    update_status(status, evidence, args.output_dir)
    report = [
        "# Round 13: legacy bilinear phase-state audit",
        "",
        "This is an audit of an unresolved legacy result, not a promoted model. The invariant interaction is "
        r"$M=\sum_j u_jv_j^\top$; factor signs, scales, and rotations are not interpreted.",
        "",
        "## Nested surface metrics",
        "",
        metrics_table.to_markdown(index=False),
        "",
        "## Existing shape frontier",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Interaction-matrix stability",
        "",
        stability.to_markdown(index=False),
        "",
        "## Leave-region-out",
        "",
        regions.to_markdown(index=False),
        "",
        "## Raw optima",
        "",
        optima.to_markdown(index=False),
        "",
        "## Gate decision",
        "",
        f"Status: **{status}**. {evidence}",
        "",
        "No multi-swarm, historical-heldout, or adversarial prediction was evaluated.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(metrics_table.to_string(index=False))
    print(comparison.to_string(index=False))
    print(stability.to_string(index=False))
    print(optima.to_string(index=False))
    print(status, evidence)


if __name__ == "__main__":
    main()
