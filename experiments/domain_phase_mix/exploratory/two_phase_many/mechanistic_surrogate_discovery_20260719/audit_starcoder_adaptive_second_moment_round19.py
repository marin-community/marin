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
"""Falsify adaptive-second-moment gradient flow on both StarCoder surfaces."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
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
    adaptive_second_moment_models as adaptive,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_hessian_equilibrium_round11 as scalar_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round19_adaptive_second_moment_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
CURVATURE_GRID = (0.5, 1.0, 2.0)
ADAPTIVE_SPEED_GRID = (0.03, 0.1, 0.3, 1.0)
MEMORY_GRID = (0.25, 1.0, 4.0, 16.0)
EPSILON_GRID = (0.03, 0.1, 0.3, 1.0)
GRADIENT_FLOW_SPEED_GRID = (0.3, 1.0, 3.0, 10.0)
EVALUATION_GRID = (0.2, 0.5, 0.8)
L2_GRID = (0.0, 0.1, 1.0)
PRIOR_FRONTIER = {"starcoder_cosine_50_50": 0.065388405808633, "starcoder_wsd_80_20": 0.0457725108696099}
SEED = 20260719
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[adaptive.AdaptiveMomentConfig]:
    result = []
    for curvature in CURVATURE_GRID:
        for speed in GRADIENT_FLOW_SPEED_GRID:
            for evaluation in EVALUATION_GRID:
                for l2 in L2_GRID:
                    result.append(
                        adaptive.AdaptiveMomentConfig(
                            adaptive.Dynamics.GRADIENT_FLOW,
                            curvature,
                            speed,
                            0.0,
                            0.0,
                            evaluation,
                            l2,
                        )
                    )
        for speed in ADAPTIVE_SPEED_GRID:
            for memory in MEMORY_GRID:
                for epsilon in EPSILON_GRID:
                    for evaluation in EVALUATION_GRID:
                        for l2 in L2_GRID:
                            result.append(
                                adaptive.AdaptiveMomentConfig(
                                    adaptive.Dynamics.ADAPTIVE_SECOND_MOMENT,
                                    curvature,
                                    speed,
                                    memory,
                                    epsilon,
                                    evaluation,
                                    l2,
                                )
                            )
    return result


def feature_matrix(panel: paired.PairedPanel, all_configs: list[adaptive.AdaptiveMomentConfig]) -> np.ndarray:
    state_cache: dict[tuple[str, float, float, float, float], np.ndarray] = {}
    rows = []
    for config in all_configs:
        key = (
            config.dynamics.value,
            config.curvature_ratio,
            config.speed,
            config.memory_rate,
            config.epsilon,
        )
        if key not in state_cache:
            position, _moment = adaptive.terminal_state(panel.weights, panel.alpha0, config)
            state_cache[key] = position
        position = state_cache[key]
        broad_loss = 0.5 * (position + 0.5) ** 2
        rare_loss = 0.5 * config.curvature_ratio * (position - 0.5) ** 2
        rows.append((1.0 - config.evaluation_mix) * broad_loss + config.evaluation_mix * rare_loss)
    return np.asarray(rows, dtype=float)


def select_surface(
    panel: paired.PairedPanel,
    all_configs: list[adaptive.AdaptiveMomentConfig],
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
    all_configs: list[adaptive.AdaptiveMomentConfig],
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
        scores, _predictions = scalar_audit.score_configs(
            features[:, outer_train], panel.two_phase_target[outer_train], local_folds, l2
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
    all_configs: list[adaptive.AdaptiveMomentConfig],
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
    config: adaptive.AdaptiveMomentConfig,
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
            feature[None, :], panel.two_phase_target, train, test, np.asarray([config.l2])
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


def algebraic_audit(all_configs: list[adaptive.AdaptiveMomentConfig]) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    weights = rng.dirichlet(np.ones(2), size=64)
    errors = []
    for config in all_configs[:: max(1, len(all_configs) // 100)]:
        for split in (0.2, 0.5, 0.8):
            errors.append(adaptive.equivalent_tied_error(config, weights, split))
    return {"maximum_tied_semigroup_error": float(max(errors))}


def optimum_and_surface(
    panel: paired.PairedPanel,
    config: adaptive.AdaptiveMomentConfig,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model = adaptive.fit_model(panel.weights, panel.two_phase_target, np.arange(panel.n), panel.alpha0, config)
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
    position, moment = adaptive.terminal_state(weights[[best]], panel.alpha0, config)
    return (
        {
            "surface": panel.name,
            "phase0_rare": float(p0.ravel()[best]),
            "phase1_rare": float(p1.ravel()[best]),
            "predicted_bpb": float(prediction[best]),
            "terminal_position": float(position[0]),
            "terminal_second_moment": float(moment[0]),
            "observed_best_phase0_rare": float(panel.weights[observed_best, 0, 1]),
            "observed_best_phase1_rare": float(panel.weights[observed_best, 1, 1]),
            "observed_best_bpb": float(panel.two_phase_target[observed_best]),
            "distance_to_observed_best": float(
                np.hypot(
                    p0.ravel()[best] - panel.weights[observed_best, 0, 1],
                    p1.ravel()[best] - panel.weights[observed_best, 1, 1],
                )
            ),
            "response_amplitude": model.head.amplitude,
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
        title=f"{panel.name}: adaptive-second-moment gradient flow",
        template="plotly_white",
        scene={
            "xaxis_title": "Phase 0 StarCoder weight",
            "yaxis_title": "Phase 1 StarCoder weight",
            "zaxis_title": "BPB",
        },
        height=850,
        width=1000,
    )
    figure.write_html(output, include_plotlyjs="cdn", config=PLOT_CONFIG)


def result_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        key: value
        for key, value in metrics.scalar_metrics(observed, predicted).items()
        if isinstance(value, float | int)
    }


def update_registry_and_ledger(gates: dict[str, bool], output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    passed = all(gates.values())
    status = "promoted_to_multi_swarm" if passed else "blocked_before_multi_swarm"
    registry.loc[registry["id"].eq("ASMGF"), "status"] = status
    registry.loc[registry["id"].eq("ASMGF"), "status_evidence"] = "; ".join(
        f"{key}={value}" for key, value in gates.items()
    )
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_19_starcoder_gate",
        "candidate_id": "ASMGF",
        "candidate_family": "Adaptive-second-moment gradient flow",
        "hyperparameters": "Frozen preregistered grid; nested StarCoder selection",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_19_preregistration",
        "novelty_class": "Persistent adaptive optimizer preconditioner",
        "evaluation_status": status,
        "evidence_path": str(output_dir.relative_to(OUTPUT_ROOT)),
        "notes": "; ".join(f"{key}={value}" for key, value in gates.items()),
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
    rows = []
    selection_tables = []
    nested_tables = []
    tied_tables = []
    region_tables = []
    optimum_rows = []
    selected_configs: dict[str, adaptive.AdaptiveMomentConfig] = {}
    nested_rmse: dict[str, float] = {}
    for panel in panels:
        features = feature_matrix(panel, all_configs)
        selected_index, selected_prediction, selection = select_surface(panel, all_configs, features)
        nested_prediction, nested = nested_selection(panel, all_configs, features)
        tied_index, tied_prediction, tied = tied_selection(panel, all_configs, features)
        selected = all_configs[selected_index]
        selected_configs[panel.name] = selected
        nested_metrics = result_metrics(panel.two_phase_target, nested_prediction)
        nested_rmse[panel.name] = nested_metrics["rmse"]
        rows.append(
            {
                "surface": panel.name,
                "selected_config": selected.key,
                "nominal_parameter_count": 7 if selected.dynamics == adaptive.Dynamics.ADAPTIVE_SECOND_MOMENT else 5,
                **{
                    f"selection_{key}": value
                    for key, value in result_metrics(panel.two_phase_target, selected_prediction).items()
                },
                **{f"nested_{key}": value for key, value in nested_metrics.items()},
                "independent_tied_config": all_configs[tied_index].key,
                **{
                    f"independent_tied_{key}": value
                    for key, value in result_metrics(panel.two_phase_target[panel.paired_mask], tied_prediction).items()
                },
            }
        )
        selection_tables.append(selection)
        nested_tables.append(nested)
        tied_tables.append(tied)
        region_tables.append(leave_region_out(panel, selected, features[selected_index]))
        optimum, surface = optimum_and_surface(panel, selected)
        optimum_rows.append(optimum)
        surface.to_csv(args.output_dir / f"{panel.name}__surface.csv", index=False)
        render_surface(panel, surface, args.output_dir / f"{panel.name}__surface.html")

    result_table = pd.DataFrame(rows)
    selection_table = pd.concat(selection_tables, ignore_index=True)
    nested_table = pd.concat(nested_tables, ignore_index=True)
    tied_table = pd.concat(tied_tables, ignore_index=True)
    region_table = pd.concat(region_tables, ignore_index=True)
    optima = pd.DataFrame(optimum_rows)
    prior = pd.DataFrame(
        [
            {
                "surface": surface,
                "nested_rmse": value,
                "prior_best_rmse": PRIOR_FRONTIER[surface],
                "relative_rmse": value / PRIOR_FRONTIER[surface] - 1.0,
            }
            for surface, value in nested_rmse.items()
        ]
    )

    result_table.to_csv(args.output_dir / "surface_metrics.csv", index=False)
    selection_table.to_csv(args.output_dir / "hyperparameter_grid.csv", index=False)
    nested_table.to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    tied_table.to_csv(args.output_dir / "independent_tied_hyperparameter_grid.csv", index=False)
    region_table.to_csv(args.output_dir / "leave_region_out.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    prior.to_csv(args.output_dir / "prior_comparison.csv", index=False)
    Path(args.output_dir / "algebraic_audit.json").write_text(json.dumps(algebra, indent=2) + "\n")

    dynamics = [selected_configs[panel.name].dynamics for panel in panels]
    adaptive_fold_counts = (
        nested_table.assign(is_adaptive=nested_table["dynamics"].eq(adaptive.Dynamics.ADAPTIVE_SECOND_MOMENT.value))
        .groupby("surface")["is_adaptive"]
        .sum()
    )
    adaptive_configs = [
        config for config in selected_configs.values() if config.dynamics == adaptive.Dynamics.ADAPTIVE_SECOND_MOMENT
    ]
    if len(adaptive_configs) == 2:
        memory_ratio = max(config.memory_rate for config in adaptive_configs) / min(
            config.memory_rate for config in adaptive_configs
        )
        epsilon_ratio = max(config.epsilon for config in adaptive_configs) / min(
            config.epsilon for config in adaptive_configs
        )
    else:
        memory_ratio = np.inf
        epsilon_ratio = np.inf
    gates = {
        "algebra_ok": algebra["maximum_tied_semigroup_error"] < 1e-3,
        "adaptive_global": all(value == adaptive.Dynamics.ADAPTIVE_SECOND_MOMENT for value in dynamics),
        "adaptive_folds": all(adaptive_fold_counts.get(panel.name, 0) >= 3 for panel in panels),
        "memory_transfer": memory_ratio <= 4.0,
        "epsilon_transfer": epsilon_ratio <= 4.0,
        "within_5pct_prior_shape": bool((prior["relative_rmse"] <= 0.05).all()),
        "optimum_distance_ok": bool((optima["distance_to_observed_best"] <= 0.15).all()),
    }
    update_registry_and_ledger(gates, args.output_dir)

    report = [
        "# Adaptive-second-moment gradient flow: StarCoder falsification",
        "",
        "The form and full grid were preregistered before reading either surface in this round. Historical, adversarial, and sealed-confirmation targets were not read.",
        "",
        "## Gates",
        "",
        pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()]).to_markdown(index=False),
        "",
        "## Surface metrics",
        "",
        result_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Prior shape frontier",
        "",
        prior.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw optima",
        "",
        optima.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Leave-region-out",
        "",
        region_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "An adaptive optimizer state is supported only if it beats its exact ordinary-gradient-flow ablation on both schedules, selects compatible memory regimes, reaches the prior shape frontier, and keeps the raw optimum in the observed valley.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))


if __name__ == "__main__":
    main()
