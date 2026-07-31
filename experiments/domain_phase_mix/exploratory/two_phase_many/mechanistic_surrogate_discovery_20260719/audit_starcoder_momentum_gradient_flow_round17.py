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
"""Falsify optimizer-momentum gradient flow on both StarCoder surfaces."""

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

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_hessian_equilibrium_round11 as scalar_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    momentum_gradient_flow_models as momentum,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round17_momentum_gradient_flow_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
CURVATURE_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
RELAXATION_GRID = (0.5, 1.0, 2.0, 4.0, 8.0)
DAMPING_GRID = (0.25, 0.5, 1.0, 2.0)
EVALUATION_GRID = (-0.4, -0.2, 0.0, 0.2, 0.4)
L2_GRID = (0.0, 0.1, 1.0)
SEED = 20260719
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[momentum.MomentumConfig]:
    result = []
    for curvature in CURVATURE_GRID:
        for relaxation in RELAXATION_GRID:
            for evaluation in EVALUATION_GRID:
                for l2 in L2_GRID:
                    result.append(
                        momentum.MomentumConfig(
                            momentum.Dynamics.FIRST_ORDER,
                            curvature,
                            relaxation,
                            0.0,
                            evaluation,
                            l2,
                        )
                    )
                    for damping in DAMPING_GRID:
                        result.append(
                            momentum.MomentumConfig(
                                momentum.Dynamics.MOMENTUM,
                                curvature,
                                relaxation,
                                damping,
                                evaluation,
                                l2,
                            )
                        )
    return result


def feature_matrix(panel: paired.PairedPanel, all_configs: list[momentum.MomentumConfig]) -> np.ndarray:
    cache: dict[tuple[str, float, float, float], np.ndarray] = {}
    rows = []
    for config in all_configs:
        key = (config.dynamics.value, config.curvature_ratio, config.relaxation, config.damping_ratio)
        if key not in cache:
            position, _velocity = momentum.terminal_state(panel.weights, panel.alpha0, config)
            cache[key] = position
        rows.append((cache[key] - config.evaluation_center) ** 2)
    return np.asarray(rows, dtype=float)


def select_surface(
    panel: paired.PairedPanel,
    all_configs: list[momentum.MomentumConfig],
    features: np.ndarray,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    l2 = np.asarray([config.l2 for config in all_configs], dtype=float)
    rmse, predictions = scalar_audit.score_configs(features, panel.two_phase_target, starcoder.surface_folds(panel), l2)
    best = int(np.argmin(rmse))
    table = pd.DataFrame(
        [
            {
                "surface": panel.name,
                "config": config.key,
                **{**asdict(config), "dynamics": config.dynamics.value},
                "rmse": float(rmse[index]),
            }
            for index, config in enumerate(all_configs)
        ]
    ).sort_values("rmse")
    return best, predictions[best], table


def nested_selection(
    panel: paired.PairedPanel,
    all_configs: list[momentum.MomentumConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    l2 = np.asarray([config.l2 for config in all_configs], dtype=float)
    rows = []
    for fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner = scalar_audit.stratified_folds(panel, outer_train, 4, SEED + fold)
        local_folds = [
            (np.flatnonzero(np.isin(outer_train, train)), np.flatnonzero(np.isin(outer_train, test)))
            for train, test in inner
        ]
        scores, _inner_predictions = scalar_audit.score_configs(
            features[:, outer_train],
            panel.two_phase_target[outer_train],
            local_folds,
            l2,
        )
        selected_index = int(np.argmin(scores))
        selected = all_configs[selected_index]
        prediction[outer_test] = scalar_audit.fit_predict_all(
            features[[selected_index]],
            panel.two_phase_target,
            outer_train,
            outer_test,
            np.asarray([selected.l2]),
        )[0]
        rows.append(
            {
                "surface": panel.name,
                "outer_fold": fold,
                "selected_config": selected.key,
                "inner_rmse": float(scores[selected_index]),
                **{**asdict(selected), "dynamics": selected.dynamics.value},
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested prediction for {panel.name}")
    return prediction, pd.DataFrame(rows)


def tied_selection(
    panel: paired.PairedPanel,
    all_configs: list[momentum.MomentumConfig],
    features: np.ndarray,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    tied = np.flatnonzero(panel.paired_mask)
    folds = list(KFold(min(5, len(tied)), shuffle=True, random_state=SEED + 77).split(tied))
    l2 = np.asarray([config.l2 for config in all_configs], dtype=float)
    rmse, predictions = scalar_audit.score_configs(features[:, tied], panel.two_phase_target[tied], folds, l2)
    best = int(np.argmin(rmse))
    table = pd.DataFrame(
        [
            {
                "surface": panel.name,
                "config": config.key,
                **{**asdict(config), "dynamics": config.dynamics.value},
                "rmse": float(rmse[index]),
            }
            for index, config in enumerate(all_configs)
        ]
    ).sort_values("rmse")
    return best, predictions[best], table


def leave_region_out(
    panel: paired.PairedPanel,
    config: momentum.MomentumConfig,
    feature: np.ndarray,
) -> pd.DataFrame:
    contrast = panel.weights[:, 1, 1] - panel.weights[:, 0, 1]
    regions = {
        "late_rare_enriched": contrast > 0.1,
        "early_rare_enriched": contrast < -0.1,
        "near_phase_tied": np.abs(contrast) <= 0.1,
    }
    rows = []
    for region, mask in regions.items():
        train = np.flatnonzero(~mask)
        test = np.flatnonzero(mask)
        prediction = scalar_audit.fit_predict_all(
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
                **metrics.scalar_metrics(panel.two_phase_target[test], prediction),
            }
        )
    return pd.DataFrame(rows)


def algebraic_audit(all_configs: list[momentum.MomentumConfig]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    position = rng.uniform(-0.5, 0.5, size=64)
    velocity = rng.uniform(-0.25, 0.25, size=64)
    weight = rng.uniform(size=64)
    maximum_error = 0.0
    for config in all_configs[:: max(1, len(all_configs) // 100)]:
        for split_time in (0.2, 0.5, 0.8):
            split_position, split_velocity = momentum.phase_update(
                position,
                velocity,
                weight,
                split_time,
                config,
            )
            split_position, split_velocity = momentum.phase_update(
                split_position,
                split_velocity,
                weight,
                1.0 - split_time,
                config,
            )
            whole_position, whole_velocity = momentum.phase_update(position, velocity, weight, 1.0, config)
            maximum_error = max(
                maximum_error,
                float(np.max(np.abs(split_position - whole_position))),
                float(np.max(np.abs(split_velocity - whole_velocity))),
            )
    return {"maximum_tied_semigroup_error": maximum_error}


def optimum_and_surface(
    panel: paired.PairedPanel,
    config: momentum.MomentumConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model = momentum.fit_model(panel.weights, panel.two_phase_target, np.arange(panel.n), panel.alpha0, config)
    grid = np.linspace(0.0, 1.0, 201)
    p0, p1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - p0.ravel(), p0.ravel()]),
            np.column_stack([1.0 - p1.ravel(), p1.ravel()]),
        ],
        axis=1,
    )
    prediction = model.predict(weights)
    best = int(np.argmin(prediction))
    observed_best = int(np.argmin(panel.two_phase_target))
    position, velocity = momentum.terminal_state(weights[[best]], panel.alpha0, config)
    return (
        {
            "surface": panel.name,
            "phase0_rare": float(p0.ravel()[best]),
            "phase1_rare": float(p1.ravel()[best]),
            "predicted_bpb": float(prediction[best]),
            "terminal_position": float(position[0]),
            "terminal_velocity": float(velocity[0]),
            "observed_best_phase0_rare": float(panel.weights[observed_best, 0, 1]),
            "observed_best_phase1_rare": float(panel.weights[observed_best, 1, 1]),
            "observed_best_bpb": float(panel.two_phase_target[observed_best]),
            "distance_to_observed_best": float(
                np.hypot(
                    p0.ravel()[best] - panel.weights[observed_best, 0, 1],
                    p1.ravel()[best] - panel.weights[observed_best, 1, 1],
                )
            ),
            "natural_response_curvature": model.head.natural_curvature,
        },
        pd.DataFrame({"phase0_rare": p0.ravel(), "phase1_rare": p1.ravel(), "predicted_bpb": prediction}),
    )


def render_surface(panel: paired.PairedPanel, surface: pd.DataFrame, output: Path) -> None:
    size = round(np.sqrt(len(surface)))
    axis = surface["phase0_rare"].to_numpy().reshape(size, size)[:, 0]
    z = surface["predicted_bpb"].to_numpy().reshape(size, size)
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
        ]
    )
    figure.update_layout(
        title=f"{panel.name}: optimizer-momentum gradient flow",
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
    registry.loc[registry["id"].eq("OMGF"), "status"] = status
    registry.loc[registry["id"].eq("OMGF"), "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_17_starcoder_gate",
        "candidate_id": "OMGF",
        "candidate_family": "Optimizer-momentum gradient flow",
        "hyperparameters": "Frozen preregistered grid; nested StarCoder selection",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_17_preregistration",
        "novelty_class": "Signed optimizer-velocity state with exact damped transition",
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
    all_configs = configs()
    cosine = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine),
        starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine)),
    ]
    algebra = algebraic_audit(all_configs)
    metric_rows = []
    grid_tables = []
    nested_tables = []
    tied_tables = []
    restriction_rows = []
    region_tables = []
    optimum_rows = []
    ablation_rows = []
    for panel in panels:
        features = feature_matrix(panel, all_configs)
        selected_index, oof_prediction, grid = select_surface(panel, all_configs, features)
        selected = all_configs[selected_index]
        nested_prediction, nested = nested_selection(panel, all_configs, features)
        tied_index, tied_prediction, tied_grid = tied_selection(panel, all_configs, features)
        tied_config = all_configs[tied_index]
        tied = panel.paired_mask
        metric_rows.append(
            {
                "surface": panel.name,
                "selected_config": selected.key,
                "nominal_parameter_count": 7,
                **{
                    f"selection_{key}": value
                    for key, value in metrics.scalar_metrics(panel.two_phase_target, oof_prediction).items()
                },
                **{
                    f"nested_{key}": value
                    for key, value in metrics.scalar_metrics(panel.two_phase_target, nested_prediction).items()
                },
            }
        )
        restriction_rows.append(
            {
                "surface": panel.name,
                "n_tied": int(tied.sum()),
                "two_phase_config": selected.key,
                "independent_tied_config": tied_config.key,
                **{
                    f"algebraic_{key}": value
                    for key, value in metrics.scalar_metrics(panel.two_phase_target[tied], oof_prediction[tied]).items()
                },
                **{
                    f"independent_{key}": value
                    for key, value in metrics.scalar_metrics(panel.two_phase_target[tied], tied_prediction).items()
                },
            }
        )
        region_tables.append(leave_region_out(panel, selected, features[selected_index]))
        optimum, surface = optimum_and_surface(panel, selected)
        optimum_rows.append(optimum)
        surface.to_csv(args.output_dir / f"{panel.name}__surface.csv", index=False)
        render_surface(panel, surface, args.output_dir / f"{panel.name}__surface.html")
        first_order = grid.loc[grid["dynamics"].eq(momentum.Dynamics.FIRST_ORDER.value)].sort_values("rmse").iloc[0]
        selected_row = grid.iloc[0]
        ablation_rows.append(
            {
                "surface": panel.name,
                "selected_dynamics": selected_row["dynamics"],
                "selected_rmse": selected_row["rmse"],
                "best_first_order_rmse": first_order["rmse"],
                "relative_rmse_vs_first_order": selected_row["rmse"] / first_order["rmse"] - 1.0,
            }
        )
        grid_tables.append(grid)
        nested_tables.append(nested)
        tied_tables.append(tied_grid)

    metric_table = pd.DataFrame(metric_rows)
    nested_table = pd.concat(nested_tables, ignore_index=True)
    restrictions = pd.DataFrame(restriction_rows)
    regions = pd.concat(region_tables, ignore_index=True)
    optima = pd.DataFrame(optimum_rows)
    ablations = pd.DataFrame(ablation_rows)
    prior = pd.read_csv(OUTPUT_ROOT / "round1_starcoder_shape_refined107/surface_oof_metrics.csv")
    prior_best = prior.groupby("surface", as_index=False)["rmse"].min().rename(columns={"rmse": "prior_best_rmse"})
    comparison = metric_table[["surface", "nested_rmse"]].merge(prior_best, on="surface")
    comparison["relative_rmse"] = comparison["nested_rmse"] / comparison["prior_best_rmse"] - 1.0

    pd.concat(grid_tables, ignore_index=True).to_csv(args.output_dir / "hyperparameter_grid.csv", index=False)
    nested_table.to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    pd.concat(tied_tables, ignore_index=True).to_csv(args.output_dir / "independent_tied_grid.csv", index=False)
    metric_table.to_csv(args.output_dir / "surface_metrics.csv", index=False)
    restrictions.to_csv(args.output_dir / "single_phase_restriction.csv", index=False)
    regions.to_csv(args.output_dir / "leave_region_out.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    ablations.to_csv(args.output_dir / "first_order_ablation.csv", index=False)
    comparison.to_csv(args.output_dir / "prior_comparison.csv", index=False)
    (args.output_dir / "algebraic_audit.json").write_text(json.dumps(algebra, indent=2) + "\n")

    selected_rows = (
        pd.concat(grid_tables, ignore_index=True).sort_values("rmse").groupby("surface", as_index=False).first()
    )
    momentum_global = bool((selected_rows["dynamics"] == momentum.Dynamics.MOMENTUM.value).all())
    momentum_folds = bool(
        (
            nested_table.groupby("surface")["dynamics"].apply(
                lambda values: int((values == momentum.Dynamics.MOMENTUM.value).sum())
            )
            >= 3
        ).all()
    )
    damping = selected_rows["damping_ratio"].to_numpy(dtype=float)
    damping_transfer = bool(momentum_global and np.max(damping) / max(np.min(damping), 1e-12) <= 4.0)
    algebra_ok = algebra["maximum_tied_semigroup_error"] < 1e-10
    shape_ok = bool((comparison["relative_rmse"] <= 0.05).all())
    optimum_ok = bool((optima["distance_to_observed_best"] <= 0.15).all())
    passed = algebra_ok and momentum_global and momentum_folds and damping_transfer and shape_ok and optimum_ok
    status = "promoted_to_multi_swarm" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"algebra_ok={algebra_ok}; momentum_global={momentum_global}; momentum_folds={momentum_folds}; "
        f"damping_transfer={damping_transfer}; within_5pct_prior_shape={shape_ok}; optimum_distance_ok={optimum_ok}."
    )
    update_status(status, evidence, args.output_dir)
    report = [
        "# Round 17: optimizer-momentum gradient flow",
        "",
        r"The latent state is model position plus signed optimizer velocity. Each phase applies exact damped second-order gradient flow; first-order Hessian relaxation is the mandatory no-momentum ablation.",
        "",
        "## Algebra",
        "",
        f"Maximum tied-semigroup error: `{algebra['maximum_tied_semigroup_error']:.3e}`.",
        "",
        "## Nested surface metrics",
        "",
        metric_table.to_markdown(index=False),
        "",
        "## First-order ablation",
        "",
        ablations.to_markdown(index=False),
        "",
        "## Existing shape frontier",
        "",
        comparison.to_markdown(index=False),
        "",
        "## Nested selections",
        "",
        nested_table.to_markdown(index=False),
        "",
        "## Single-phase restriction",
        "",
        restrictions.to_markdown(index=False),
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
        "No historical or adversarial prediction was evaluated.",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(metric_table.to_string(index=False))
    print(ablations.to_string(index=False))
    print(comparison.to_string(index=False))
    print(selected_rows.to_string(index=False))
    print(optima.to_string(index=False))
    print(status, evidence)


if __name__ == "__main__":
    main()
