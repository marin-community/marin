# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Falsify Hessian-weighted equilibrium relaxation on both StarCoder surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.model_selection import KFold

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    hessian_equilibrium_models as hessian,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round11_hessian_equilibrium_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
SEED = 20260719
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
CURVATURE_GRID = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 10.0)
RELAXATION_GRID = (0.1, 0.3, 1.0, 3.0, 10.0, 30.0)
INITIAL_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
EVALUATION_GRID = tuple(np.round(np.arange(0.05, 1.0, 0.05), 10))
L2_GRID = (0.0, 0.1, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[hessian.HessianConfig]:
    return [
        hessian.HessianConfig(curvature, relaxation, initial, evaluation, l2)
        for curvature in CURVATURE_GRID
        for relaxation in RELAXATION_GRID
        for initial in INITIAL_GRID
        for evaluation in EVALUATION_GRID
        for l2 in L2_GRID
    ]


def feature_matrix(panel: paired.PairedPanel, all_configs: list[hessian.HessianConfig]) -> np.ndarray:
    return np.asarray(
        [hessian.response_feature(panel.weights, panel.alpha0, config) for config in all_configs],
        dtype=float,
    )


def fit_predict_all(
    features: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    l2_values: np.ndarray,
) -> np.ndarray:
    train_features = features[:, train]
    means = train_features.mean(axis=1)
    scales = np.sqrt(np.mean((train_features - means[:, None]) ** 2, axis=1))
    scales = np.maximum(scales, 1e-8)
    standardized = (train_features - means[:, None]) / scales[:, None]
    target_mean = float(np.mean(target[train]))
    centered_target = target[train] - target_mean
    numerator = standardized @ centered_target
    denominator = np.sum(standardized**2, axis=1) + l2_values
    coefficients = np.maximum(numerator / np.maximum(denominator, 1e-12), 0.0)
    return target_mean + coefficients[:, None] * (features[:, test] - means[:, None]) / scales[:, None]


def score_configs(
    features: np.ndarray,
    target: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
    l2_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full(features.shape, np.nan, dtype=float)
    for train, test in folds:
        predictions[:, test] = fit_predict_all(features, target, train, test, l2_values)
    if not np.isfinite(predictions).all():
        raise RuntimeError("Incomplete vectorized OOF predictions")
    rmse = np.sqrt(np.mean((predictions - target[None, :]) ** 2, axis=1))
    return rmse, predictions


def surface_selection(
    panel: paired.PairedPanel,
    all_configs: list[hessian.HessianConfig],
    features: np.ndarray,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    l2_values = np.asarray([config.l2 for config in all_configs], dtype=float)
    rmse, predictions = score_configs(features, panel.two_phase_target, starcoder.surface_folds(panel), l2_values)
    best = int(np.argmin(rmse))
    table = pd.DataFrame(
        [
            {"surface": panel.name, "config": config.key, **asdict(config), "rmse": float(rmse[index])}
            for index, config in enumerate(all_configs)
        ]
    ).sort_values("rmse")
    return best, predictions[best], table


def stratified_folds(
    panel: paired.PairedPanel, indices: np.ndarray, n_splits: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    tied = indices[panel.paired_mask[indices]]
    untied = indices[~panel.paired_mask[indices]]
    if len(tied) < 2 or len(untied) < 2:
        splitter = KFold(min(n_splits, len(indices)), shuffle=True, random_state=seed)
        return [(indices[train], indices[test]) for train, test in splitter.split(indices)]
    count = min(n_splits, len(tied), len(untied))
    tied_folds = list(KFold(count, shuffle=True, random_state=seed).split(tied))
    untied_folds = list(KFold(count, shuffle=True, random_state=seed + 1).split(untied))
    return [
        (
            np.sort(np.concatenate([tied[tied_folds[fold][0]], untied[untied_folds[fold][0]]])),
            np.sort(np.concatenate([tied[tied_folds[fold][1]], untied[untied_folds[fold][1]]])),
        )
        for fold in range(count)
    ]


def nested_surface_selection(
    panel: paired.PairedPanel,
    all_configs: list[hessian.HessianConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    l2_values = np.asarray([config.l2 for config in all_configs], dtype=float)
    prediction = np.full(panel.n, np.nan, dtype=float)
    rows = []
    for outer_fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner_folds = stratified_folds(panel, outer_train, 4, SEED + 100 * outer_fold)
        inner_rmse, _inner_predictions = score_configs(
            features[:, outer_train],
            panel.two_phase_target[outer_train],
            [
                (
                    np.flatnonzero(np.isin(outer_train, train)),
                    np.flatnonzero(np.isin(outer_train, test)),
                )
                for train, test in inner_folds
            ],
            l2_values,
        )
        selected_index = int(np.argmin(inner_rmse))
        selected = all_configs[selected_index]
        prediction[outer_test] = fit_predict_all(
            features[[selected_index]],
            panel.two_phase_target,
            outer_train,
            outer_test,
            np.asarray([selected.l2]),
        )[0]
        rows.append(
            {
                "surface": panel.name,
                "outer_fold": outer_fold,
                "selected_config": selected.key,
                "inner_rmse": float(inner_rmse[selected_index]),
                **asdict(selected),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested HWER prediction for {panel.name}")
    return prediction, pd.DataFrame(rows)


def independent_tied_selection(
    panel: paired.PairedPanel,
    all_configs: list[hessian.HessianConfig],
    features: np.ndarray,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    tied = np.flatnonzero(panel.paired_mask)
    folds = [(train, test) for train, test in KFold(min(5, len(tied)), shuffle=True, random_state=SEED + 77).split(tied)]
    l2_values = np.asarray([config.l2 for config in all_configs], dtype=float)
    rmse, predictions = score_configs(features[:, tied], panel.two_phase_target[tied], folds, l2_values)
    best = int(np.argmin(rmse))
    table = pd.DataFrame(
        [
            {"surface": panel.name, "config": config.key, **asdict(config), "rmse": float(rmse[index])}
            for index, config in enumerate(all_configs)
        ]
    ).sort_values("rmse")
    return best, predictions[best], table


def leave_region_out(
    panel: paired.PairedPanel,
    config: hessian.HessianConfig,
    feature: np.ndarray,
) -> list[dict[str, Any]]:
    contrast = panel.weights[:, 1, 1] - panel.weights[:, 0, 1]
    regions = {
        "late_rare_enriched": contrast > 0.1,
        "early_rare_enriched": contrast < -0.1,
        "near_phase_tied": np.abs(contrast) <= 0.1,
    }
    rows = []
    for region, test_mask in regions.items():
        train = np.flatnonzero(~test_mask)
        test = np.flatnonzero(test_mask)
        predictions = fit_predict_all(
            feature[None, :],
            panel.two_phase_target,
            train,
            test,
            np.asarray([config.l2]),
        )[0]
        rows.append(
            {
                "surface": panel.name,
                "region": region,
                "n_train": len(train),
                "n_test": len(test),
                **paired_screen.scalar_metrics(panel.two_phase_target[test], predictions),
            }
        )
    return rows


def algebraic_audit() -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    p = rng.uniform(size=128)
    z = rng.uniform(size=128)
    first = rng.uniform(size=128)
    second = 1.0 - first
    semigroup_errors = []
    swap_errors = []
    for curvature in CURVATURE_GRID:
        for relaxation in RELAXATION_GRID:
            split = hessian.relaxation_update(z, p, first, curvature, relaxation)
            split = hessian.relaxation_update(split, p, second, curvature, relaxation)
            whole = hessian.relaxation_update(z, p, 1.0, curvature, relaxation)
            semigroup_errors.append(float(np.max(np.abs(split - whole))))
            original = hessian.relaxation_update(z, p, 1.0, curvature, relaxation)
            swapped = hessian.relaxation_update(
                1.0 - z,
                1.0 - p,
                1.0,
                1.0 / curvature,
                relaxation * curvature,
            )
            swap_errors.append(float(np.max(np.abs((1.0 - original) - swapped))))
    return {
        "maximum_tied_semigroup_error": max(semigroup_errors),
        "maximum_label_swap_error": max(swap_errors),
    }


def optimum_and_surface(
    panel: paired.PairedPanel,
    config: hessian.HessianConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model = hessian.fit_model(panel.weights, panel.two_phase_target, np.arange(panel.n), panel.alpha0, config)
    grid = np.linspace(0.0, 1.0, 201)
    rare0, rare1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - rare0.ravel(), rare0.ravel()]),
            np.column_stack([1.0 - rare1.ravel(), rare1.ravel()]),
        ],
        axis=1,
    )
    prediction = model.predict(weights)
    best = int(np.argmin(prediction))
    observed_best = int(np.argmin(panel.two_phase_target))
    return (
        {
            "surface": panel.name,
            "phase0_rare": float(rare0.ravel()[best]),
            "phase1_rare": float(rare1.ravel()[best]),
            "predicted_bpb": float(prediction[best]),
            "natural_response_curvature": model.head.natural_curvature,
            "observed_best_phase0_rare": float(panel.weights[observed_best, 0, 1]),
            "observed_best_phase1_rare": float(panel.weights[observed_best, 1, 1]),
            "observed_best_bpb": float(panel.two_phase_target[observed_best]),
            "distance_to_observed_best": float(
                np.hypot(
                    rare0.ravel()[best] - panel.weights[observed_best, 0, 1],
                    rare1.ravel()[best] - panel.weights[observed_best, 1, 1],
                )
            ),
        },
        pd.DataFrame({"phase0_rare": rare0.ravel(), "phase1_rare": rare1.ravel(), "predicted_bpb": prediction}),
    )


def render_surface(panel: paired.PairedPanel, surface: pd.DataFrame, optimum: dict[str, Any], output: Path) -> None:
    grid_size = round(np.sqrt(len(surface)))
    axis = surface["phase0_rare"].to_numpy().reshape(grid_size, grid_size)[:, 0]
    z = surface["predicted_bpb"].to_numpy().reshape(grid_size, grid_size)
    figure = go.Figure(
        [
            go.Surface(x=axis, y=axis, z=z.T, colorscale="RdYlGn_r", opacity=0.72, name="Predicted"),
            go.Scatter3d(
                x=panel.weights[:, 0, 1],
                y=panel.weights[:, 1, 1],
                z=panel.two_phase_target,
                mode="markers",
                marker={"size": 4, "color": panel.two_phase_target, "colorscale": "RdYlGn_r"},
                name="Observed",
            ),
            go.Scatter3d(
                x=[optimum["phase0_rare"]],
                y=[optimum["phase1_rare"]],
                z=[optimum["predicted_bpb"]],
                mode="markers",
                marker={"size": 9, "symbol": "diamond", "color": "#111827"},
                name="Predicted optimum",
            ),
        ]
    )
    figure.update_layout(
        title=f"{panel.name}: Hessian-weighted equilibrium relaxation",
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
    mask = registry["id"].eq("HWER")
    registry.loc[mask, "status"] = status
    registry.loc[mask, "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_11_starcoder_gate",
        "candidate_id": "HWER",
        "candidate_family": "Hessian-weighted equilibrium relaxation",
        "hyperparameters": "Frozen preregistered grid; nested StarCoder selection",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_11_preregistration",
        "novelty_class": "Exact quadratic multi-task gradient-flow transition",
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
    algebra = algebraic_audit()
    cosine = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine),
        starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine)),
    ]
    all_configs = configs()
    metric_rows = []
    grid_tables = []
    nested_tables = []
    tied_tables = []
    restriction_rows = []
    region_rows = []
    optimum_rows = []
    for panel in panels:
        features = feature_matrix(panel, all_configs)
        selected_index, oof_prediction, grid = surface_selection(panel, all_configs, features)
        selected = all_configs[selected_index]
        nested_prediction, nested = nested_surface_selection(panel, all_configs, features)
        tied_index, tied_prediction, tied_grid = independent_tied_selection(panel, all_configs, features)
        tied_config = all_configs[tied_index]
        tied = panel.paired_mask
        metric_rows.append(
            {
                "surface": panel.name,
                "selected_config": selected.key,
                "total_parameter_count": 6,
                "curvature_boundary": selected.curvature_ratio in (min(CURVATURE_GRID), max(CURVATURE_GRID)),
                "relaxation_boundary": selected.relaxation in (min(RELAXATION_GRID), max(RELAXATION_GRID)),
                "initial_boundary": selected.initial_state in (min(INITIAL_GRID), max(INITIAL_GRID)),
                "evaluation_boundary": selected.evaluation_optimum in (min(EVALUATION_GRID), max(EVALUATION_GRID)),
                **{
                    f"selection_{key}": value
                    for key, value in paired_screen.scalar_metrics(panel.two_phase_target, oof_prediction).items()
                },
                **{
                    f"nested_{key}": value
                    for key, value in paired_screen.scalar_metrics(panel.two_phase_target, nested_prediction).items()
                },
            }
        )
        restriction_rows.append(
            {
                "surface": panel.name,
                "n_tied": int(tied.sum()),
                "two_phase_selected_config": selected.key,
                "independent_tied_selected_config": tied_config.key,
                **{
                    f"algebraic_two_phase_oof_{key}": value
                    for key, value in paired_screen.scalar_metrics(
                        panel.two_phase_target[tied], oof_prediction[tied]
                    ).items()
                },
                **{
                    f"independent_tied_oof_{key}": value
                    for key, value in paired_screen.scalar_metrics(panel.two_phase_target[tied], tied_prediction).items()
                },
            }
        )
        region_rows.extend(leave_region_out(panel, selected, features[selected_index]))
        optimum, surface = optimum_and_surface(panel, selected)
        optimum_rows.append(optimum)
        surface.to_csv(args.output_dir / f"{panel.name}__surface.csv", index=False)
        render_surface(panel, surface, optimum, args.output_dir / f"{panel.name}__surface.html")
        grid_tables.append(grid)
        nested_tables.append(nested)
        tied_tables.append(tied_grid)

    metrics = pd.DataFrame(metric_rows)
    regions = pd.DataFrame(region_rows)
    restrictions = pd.DataFrame(restriction_rows)
    optima = pd.DataFrame(optimum_rows)
    pd.concat(grid_tables, ignore_index=True).to_csv(args.output_dir / "hyperparameter_grid.csv", index=False)
    pd.concat(nested_tables, ignore_index=True).to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    pd.concat(tied_tables, ignore_index=True).to_csv(
        args.output_dir / "independent_tied_hyperparameter_grid.csv", index=False
    )
    metrics.to_csv(args.output_dir / "surface_oof_metrics.csv", index=False)
    regions.to_csv(args.output_dir / "leave_region_out.csv", index=False)
    restrictions.to_csv(args.output_dir / "single_phase_restriction.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    (args.output_dir / "algebraic_audit.json").write_text(json.dumps(algebra, indent=2) + "\n")

    prior = pd.read_csv(OUTPUT_ROOT / "round1_starcoder_shape_refined107/surface_oof_metrics.csv")
    strongest_prior = prior.groupby("surface", as_index=False)["rmse"].min().rename(columns={"rmse": "prior_best_rmse"})
    comparison = metrics[["surface", "nested_rmse"]].merge(strongest_prior, on="surface")
    comparison["relative_rmse"] = comparison["nested_rmse"] / comparison["prior_best_rmse"] - 1.0
    comparison.to_csv(args.output_dir / "prior_comparison.csv", index=False)

    algebra_ok = max(algebra.values()) < 1e-10
    boundary_columns = ["curvature_boundary", "relaxation_boundary", "initial_boundary", "evaluation_boundary"]
    boundaries_ok = not metrics[boundary_columns].to_numpy().any()
    shape_ok = bool((comparison["relative_rmse"] <= 0.05).all())
    optimum_ok = bool((optima["distance_to_observed_best"] <= 0.15).all())
    passed = algebra_ok and boundaries_ok and shape_ok and optimum_ok
    status = "promoted_to_multi_swarm" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"algebra_ok={algebra_ok}; no_nonlinear_boundaries={boundaries_ok}; "
        f"within_5pct_prior_shape={shape_ok}; optimum_distance_ok={optimum_ok}."
    )
    update_status(status, evidence, args.output_dir)
    report = [
        "# Round 11: Hessian-weighted equilibrium relaxation",
        "",
        "## Frozen candidate",
        "",
        r"For rare-data mass $p$, $H(p)=1-p+rp$, $m(p)=rp/H(p)$, and $z'=m(p)+(z-m(p))\exp[-kH(p)\Delta]$. Evaluation is $Y=b+A(z-\nu)^2$ with $A\ge0$.",
        "",
        "## Algebraic audit",
        "",
        f"- Maximum tied-semigroup error: `{algebra['maximum_tied_semigroup_error']:.3e}`.",
        f"- Maximum label-swap error: `{algebra['maximum_label_swap_error']:.3e}`.",
        "",
        "## Nested StarCoder results",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Existing-shape comparison",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Leave-region-out",
        "",
        regions.to_markdown(index=False),
        "",
        "## Single-phase restriction",
        "",
        restrictions.to_markdown(index=False),
        "",
        "## Raw optima",
        "",
        optima.to_markdown(index=False),
        "",
        "## Gate decision",
        "",
        f"Status: **{status}**. {evidence}",
        "",
        "No historical or adversarial candidate predictions were evaluated.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(metrics.to_string(index=False))
    print(comparison.to_string(index=False))
    print(optima.to_string(index=False))
    print(status, evidence)


if __name__ == "__main__":
    main()
