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
"""Falsify Jensen acquisition-rate response on both StarCoder surfaces."""

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
    jensen_acquisition_models as jensen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round16_jensen_acquisition_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
RATE_POWER_GRID = (0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0)
SHORTAGE_POWER_GRID = (0.25, 0.5, 1.0, 2.0)
SHORTAGE_OFFSET_GRID = (0.03, 0.1, 0.3, 1.0)
L2_GRID = (0.0, 0.1, 1.0, 10.0)
SEED = 20260719
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[jensen.JensenAcquisitionConfig]:
    return [
        jensen.JensenAcquisitionConfig(rate, shortage, offset, l2)
        for rate in RATE_POWER_GRID
        for shortage in SHORTAGE_POWER_GRID
        for offset in SHORTAGE_OFFSET_GRID
        for l2 in L2_GRID
    ]


def designs(
    panel: paired.PairedPanel,
    all_configs: list[jensen.JensenAcquisitionConfig],
) -> list[np.ndarray]:
    return [jensen.design_matrix(panel, panel.weights, config)[0] for config in all_configs]


def fit_predict(
    design: np.ndarray,
    target: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    l2: float,
) -> np.ndarray:
    names = tuple(f"feature_{index}" for index in range(design.shape[1]))
    head = paired.fit_linear_head(
        design[train],
        target[train],
        names,
        np.ones(design.shape[1], dtype=int),
        l2,
    )
    return head.predict(design[test])


def score_configs(
    panel: paired.PairedPanel,
    all_configs: list[jensen.JensenAcquisitionConfig],
    all_designs: list[np.ndarray],
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full((len(all_configs), panel.n), np.nan, dtype=float)
    for config_index, (config, design) in enumerate(zip(all_configs, all_designs, strict=True)):
        for train, test in folds:
            predictions[config_index, test] = fit_predict(design, panel.two_phase_target, train, test, config.l2)
    if not np.isfinite(predictions).all():
        raise RuntimeError(f"Incomplete predictions for {panel.name}")
    rmse = np.sqrt(np.mean((predictions - panel.two_phase_target[None, :]) ** 2, axis=1))
    return rmse, predictions


def select_surface(
    panel: paired.PairedPanel,
    all_configs: list[jensen.JensenAcquisitionConfig],
    all_designs: list[np.ndarray],
) -> tuple[int, np.ndarray, pd.DataFrame]:
    rmse, predictions = score_configs(panel, all_configs, all_designs, starcoder.surface_folds(panel))
    table = pd.DataFrame(
        [
            {"surface": panel.name, "config": config.key, **asdict(config), "rmse": float(rmse[index])}
            for index, config in enumerate(all_configs)
        ]
    ).sort_values("rmse")
    best = int(np.argmin(rmse))
    return best, predictions[best], table


def nested_selection(
    panel: paired.PairedPanel,
    all_configs: list[jensen.JensenAcquisitionConfig],
    all_designs: list[np.ndarray],
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    rows = []
    for fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner_folds = scalar_audit.stratified_folds(panel, outer_train, 4, SEED + fold)
        scores = []
        for config, design in zip(all_configs, all_designs, strict=True):
            local_prediction = np.full(panel.n, np.nan, dtype=float)
            for inner_train, inner_test in inner_folds:
                local_prediction[inner_test] = fit_predict(
                    design,
                    panel.two_phase_target,
                    inner_train,
                    inner_test,
                    config.l2,
                )
            test_indices = np.concatenate([test for _train, test in inner_folds])
            scores.append(
                float(np.sqrt(np.mean((local_prediction[test_indices] - panel.two_phase_target[test_indices]) ** 2)))
            )
        selected_index = int(np.argmin(scores))
        selected = all_configs[selected_index]
        prediction[outer_test] = fit_predict(
            all_designs[selected_index],
            panel.two_phase_target,
            outer_train,
            outer_test,
            selected.l2,
        )
        rows.append(
            {
                "surface": panel.name,
                "outer_fold": fold,
                "selected_config": selected.key,
                "inner_rmse": scores[selected_index],
                **asdict(selected),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested predictions for {panel.name}")
    return prediction, pd.DataFrame(rows)


def tied_selection(
    panel: paired.PairedPanel,
    all_configs: list[jensen.JensenAcquisitionConfig],
    all_designs: list[np.ndarray],
) -> tuple[int, np.ndarray, pd.DataFrame]:
    tied = np.flatnonzero(panel.paired_mask)
    folds = []
    for train, test in KFold(min(5, len(tied)), shuffle=True, random_state=SEED + 77).split(tied):
        folds.append((tied[train], tied[test]))
    predictions = np.full((len(all_configs), panel.n), np.nan, dtype=float)
    scores = []
    for index, (config, design) in enumerate(zip(all_configs, all_designs, strict=True)):
        for train, test in folds:
            predictions[index, test] = fit_predict(design, panel.two_phase_target, train, test, config.l2)
        scores.append(float(np.sqrt(np.mean((predictions[index, tied] - panel.two_phase_target[tied]) ** 2))))
    table = pd.DataFrame(
        [
            {"surface": panel.name, "config": config.key, **asdict(config), "rmse": scores[index]}
            for index, config in enumerate(all_configs)
        ]
    ).sort_values("rmse")
    best = int(np.argmin(scores))
    return best, predictions[best], table


def leave_region_out(
    panel: paired.PairedPanel,
    config: jensen.JensenAcquisitionConfig,
    design: np.ndarray,
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
        prediction = fit_predict(design, panel.two_phase_target, train, test, config.l2)
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


def algebraic_audit(panels: list[paired.PairedPanel]) -> dict[str, float]:
    maximum_linear_error = 0.0
    maximum_tied_boundary_error = 0.0
    rng = np.random.default_rng(SEED)
    for panel in panels:
        linear = jensen.acquisition_ratio(panel, panel.weights, 1.0)
        expected = panel.aggregate_weights / panel.proportional_weights[None, :]
        maximum_linear_error = max(maximum_linear_error, float(np.max(np.abs(linear - expected))))
        tied = rng.dirichlet(np.ones(panel.m), size=100)
        weights = np.stack([tied, tied], axis=1)
        for rate in RATE_POWER_GRID:
            ratio = jensen.acquisition_ratio(panel, weights, rate)
            expected_tied = tied**rate / panel.proportional_weights[None, :] ** rate
            maximum_tied_boundary_error = max(
                maximum_tied_boundary_error,
                float(np.max(np.abs(ratio - expected_tied))),
            )
    return {
        "maximum_rate_one_physical_exposure_error": maximum_linear_error,
        "maximum_tied_boundary_invariance_error": maximum_tied_boundary_error,
    }


def fit_full_model(
    panel: paired.PairedPanel,
    config: jensen.JensenAcquisitionConfig,
) -> jensen.JensenAcquisitionModel:
    return jensen.fit_model(panel, panel.two_phase_target, np.arange(panel.n), config)


def optimum_and_surface(
    panel: paired.PairedPanel,
    config: jensen.JensenAcquisitionConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model = fit_full_model(panel, config)
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
    return (
        {
            "surface": panel.name,
            "phase0_rare": float(p0.ravel()[best]),
            "phase1_rare": float(p1.ravel()[best]),
            "predicted_bpb": float(prediction[best]),
            "observed_best_phase0_rare": float(panel.weights[observed_best, 0, 1]),
            "observed_best_phase1_rare": float(panel.weights[observed_best, 1, 1]),
            "observed_best_bpb": float(panel.two_phase_target[observed_best]),
            "distance_to_observed_best": float(
                np.hypot(
                    p0.ravel()[best] - panel.weights[observed_best, 0, 1],
                    p1.ravel()[best] - panel.weights[observed_best, 1, 1],
                )
            ),
            "maximum_weight": float(max(p0.ravel()[best], p1.ravel()[best])),
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
        title=f"{panel.name}: Jensen acquisition-rate response",
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
    registry.loc[registry["id"].eq("JARA"), "status"] = status
    registry.loc[registry["id"].eq("JARA"), "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_16_starcoder_gate",
        "candidate_id": "JARA",
        "candidate_family": "Jensen acquisition-rate response",
        "hyperparameters": "Frozen preregistered grid; nested StarCoder selection",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_16_preregistration",
        "novelty_class": "Nonlinear instantaneous acquisition-rate law",
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
    algebra = algebraic_audit(panels)
    metric_rows = []
    grid_tables = []
    nested_tables = []
    tied_tables = []
    restriction_rows = []
    region_tables = []
    optimum_rows = []
    ablation_rows = []
    for panel in panels:
        all_designs = designs(panel, all_configs)
        selected_index, oof_prediction, grid = select_surface(panel, all_configs, all_designs)
        selected = all_configs[selected_index]
        nested_prediction, nested = nested_selection(panel, all_configs, all_designs)
        tied_index, tied_prediction, tied_grid = tied_selection(panel, all_configs, all_designs)
        tied_config = all_configs[tied_index]
        tied = panel.paired_mask
        metric_rows.append(
            {
                "surface": panel.name,
                "selected_config": selected.key,
                "nominal_parameter_count": 5,
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
                    for key, value in metrics.scalar_metrics(panel.two_phase_target[tied], tied_prediction[tied]).items()
                },
            }
        )
        region_tables.append(leave_region_out(panel, selected, all_designs[selected_index]))
        optimum, surface = optimum_and_surface(panel, selected)
        optimum_rows.append(optimum)
        surface.to_csv(args.output_dir / f"{panel.name}__surface.csv", index=False)
        render_surface(panel, surface, args.output_dir / f"{panel.name}__surface.html")
        linear = grid.loc[grid["rate_power"].eq(1.0)].sort_values("rmse").iloc[0]
        nonlinear = grid.iloc[0]
        ablation_rows.append(
            {
                "surface": panel.name,
                "selected_rate_power": nonlinear["rate_power"],
                "selected_rmse": nonlinear["rmse"],
                "best_linear_rmse": linear["rmse"],
                "relative_rmse_vs_linear": nonlinear["rmse"] / linear["rmse"] - 1.0,
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
    ablations.to_csv(args.output_dir / "linear_rate_ablation.csv", index=False)
    comparison.to_csv(args.output_dir / "prior_comparison.csv", index=False)
    (args.output_dir / "algebraic_audit.json").write_text(json.dumps(algebra, indent=2) + "\n")

    selected_rows = (
        pd.concat(grid_tables, ignore_index=True).sort_values("rmse").groupby("surface", as_index=False).first()
    )
    nonlinear_global = bool((~np.isclose(selected_rows["rate_power"], 1.0)).all())
    nonlinear_folds = bool(
        (
            nested_table.groupby("surface")["rate_power"].apply(lambda values: int((~np.isclose(values, 1.0)).sum()))
            >= 3
        ).all()
    )
    rate_sides = np.sign(selected_rows["rate_power"].to_numpy(dtype=float) - 1.0)
    direction_transfer = bool(np.all(rate_sides == rate_sides[0]))
    algebra_ok = max(algebra.values()) < 1e-10
    shape_ok = bool((comparison["relative_rmse"] <= 0.05).all())
    optimum_ok = bool((optima["distance_to_observed_best"] <= 0.15).all())
    passed = algebra_ok and nonlinear_global and nonlinear_folds and direction_transfer and shape_ok and optimum_ok
    status = "promoted_to_multi_swarm" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"algebra_ok={algebra_ok}; nonlinear_global={nonlinear_global}; nonlinear_folds={nonlinear_folds}; "
        f"direction_transfer={direction_transfer}; within_5pct_prior_shape={shape_ok}; optimum_distance_ok={optimum_ok}."
    )
    update_status(status, evidence, args.output_dir)
    report = [
        "# Round 16: Jensen acquisition-rate response",
        "",
        r"The useful-acquisition state obeys $\dot x_i=(w_i)^\zeta$. The exact $\zeta=1$ ablation is ordinary physical exposure; $\zeta\ne1$ creates a phase effect through Jensen's inequality before the response link.",
        "",
        "## Algebra",
        "",
        f"Maximum rate-one physical-exposure error: `{algebra['maximum_rate_one_physical_exposure_error']:.3e}`.",
        f"Maximum tied-boundary invariance error: `{algebra['maximum_tied_boundary_invariance_error']:.3e}`.",
        "",
        "## Nested surface metrics",
        "",
        metric_table.to_markdown(index=False),
        "",
        "## Linear-rate ablation",
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
