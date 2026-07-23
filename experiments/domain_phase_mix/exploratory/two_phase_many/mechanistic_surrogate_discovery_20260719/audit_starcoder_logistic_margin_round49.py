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
"""Falsify logistic margin-competition flow on both StarCoder schedules."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_optimizer_time_fast_slow_round22 as audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    logistic_margin_models as logistic,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    starcoder_refined_data,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round49_logistic_margin_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
SHAPE_REFERENCE = {"starcoder_cosine_50_50": 0.06538840580863309, "starcoder_wsd_80_20": 0.04577251086960991}
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class FittedModel:
    panel: paired.PairedPanel
    config: logistic.Config
    head: paired.LinearHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.head.predict(design(self.panel, weights, self.config))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[logistic.Config]:
    return [
        logistic.Config(clock, rate, decay, angle, rare_ratio, l2)
        for clock in logistic.Clock
        for rate in (1.0, 4.0, 16.0)
        for decay in (0.0, 0.1, 1.0)
        for angle in (60.0, 90.0, 120.0, 150.0)
        for rare_ratio in (0.5, 1.0, 2.0)
        for l2 in (0.1, 1.0)
    ]


def transition_fraction(panel: paired.PairedPanel, config: logistic.Config) -> float:
    if config.clock is logistic.Clock.TOKEN:
        return panel.alpha0
    return audit.clock_audit.optimizer_clock_fraction(panel)[0]


def design(panel: paired.PairedPanel, weights: np.ndarray, config: logistic.Config) -> np.ndarray:
    return logistic.logistic_loss_design(weights[:, :, 1], transition_fraction(panel, config), config)


def fit_model(panel: paired.PairedPanel, indices: np.ndarray, config: logistic.Config) -> FittedModel:
    matrix = design(panel, panel.weights[indices], config)
    head = paired.fit_linear_head(
        matrix,
        panel.two_phase_target[indices],
        ("broad_log_loss", "rare_log_loss"),
        np.ones(2, dtype=int),
        config.l2,
    )
    return FittedModel(panel, config, head)


def all_designs(panel: paired.PairedPanel, candidates: list[logistic.Config]) -> list[np.ndarray]:
    return [design(panel, panel.weights, config) for config in candidates]


def mechanism_selection(
    panel: paired.PairedPanel,
    candidates: list[logistic.Config],
    designs: list[np.ndarray],
    nonorthogonal: bool,
) -> tuple[logistic.Config, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    indices = [index for index, config in enumerate(candidates) if (config.task_angle_degrees != 90.0) == nonorthogonal]
    subset_configs = [candidates[index] for index in indices]
    subset_designs = [designs[index] for index in indices]
    selected_index, global_prediction, selection = audit.select_global(panel, subset_configs, subset_designs)
    nested_prediction, nested = audit.nested_selection(panel, subset_configs, subset_designs)
    return subset_configs[selected_index], global_prediction, selection, nested_prediction, nested


def raw_optimum(panel: paired.PairedPanel, config: logistic.Config) -> tuple[dict[str, Any], pd.DataFrame]:
    model = fit_model(panel, np.arange(panel.n), config)
    axis = np.linspace(0.0, 1.0, 201)
    p0, p1 = np.meshgrid(axis, axis, indexing="ij")
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
    record = {
        "surface": panel.name,
        "mechanism": "nonorthogonal" if config.task_angle_degrees != 90.0 else "orthogonal",
        "config": config.key,
        "phase0_rare": float(p0.ravel()[best]),
        "phase1_rare": float(p1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "observed_phase0_rare": float(panel.weights[observed_best, 0, 1]),
        "observed_phase1_rare": float(panel.weights[observed_best, 1, 1]),
        "observed_bpb": float(panel.two_phase_target[observed_best]),
        "distance_to_observed_best": float(
            np.hypot(
                p0.ravel()[best] - panel.weights[observed_best, 0, 1],
                p1.ravel()[best] - panel.weights[observed_best, 1, 1],
            )
        ),
        "natural_coefficients": json.dumps(
            dict(zip(model.head.feature_names, model.head.coefficients_in_natural_units, strict=True)),
            sort_keys=True,
        ),
    }
    return record, pd.DataFrame({"phase0_rare": p0.ravel(), "phase1_rare": p1.ravel(), "predicted_bpb": prediction})


def render_surface(
    panel: paired.PairedPanel,
    surface: pd.DataFrame,
    output: Path,
    model_label: str = "logistic margin competition",
) -> None:
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
        title=f"{panel.name}: {model_label}",
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


def update_registry_and_ledger(gates: dict[str, bool], output_dir: Path) -> None:
    status = "promoted_to_multi_swarm" if all(gates.values()) else "blocked_before_multi_swarm"
    evidence = "; ".join(f"{key}={value}" for key, value in gates.items())
    registry = pd.read_csv(REGISTRY)
    registry.loc[registry["id"].eq("LMCF"), "status"] = status
    registry.loc[registry["id"].eq("LMCF"), "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_49_starcoder_gate",
        "candidate_id": "LMCF",
        "candidate_family": "Logistic margin-competition flow",
        "hyperparameters": "Frozen preregistered clock/rate/decay/angle/rare-scale/ridge grid",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_49_preregistration",
        "novelty_class": "Shared finite-margin log-loss gradient flow",
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
    candidates = configs()
    cosine = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine),
        starcoder.panel_from_dataset(starcoder_refined_data.load_refined_wsd80_starcoder(cosine)),
    ]
    metric_rows: list[dict[str, Any]] = []
    selection_tables: list[pd.DataFrame] = []
    nested_tables: list[pd.DataFrame] = []
    optimum_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selected_active: dict[str, logistic.Config] = {}
    active_nested_rmse: dict[str, float] = {}
    global_comparison: dict[str, bool] = {}
    fold_wins: dict[str, int] = {}
    integration_errors: dict[str, float] = {}
    for panel in panels:
        matrices = all_designs(panel, candidates)
        selected_by_mechanism: dict[bool, logistic.Config] = {}
        predictions_by_mechanism: dict[bool, np.ndarray] = {}
        for nonorthogonal in (False, True):
            selected, global_prediction, selection, nested_prediction, nested = mechanism_selection(
                panel, candidates, matrices, nonorthogonal
            )
            selected_by_mechanism[nonorthogonal] = selected
            predictions_by_mechanism[nonorthogonal] = nested_prediction
            label = "nonorthogonal" if nonorthogonal else "orthogonal"
            selection_tables.append(selection.assign(mechanism=label))
            nested_tables.append(nested.assign(mechanism=label))
            nested_metrics = audit.scalar_metrics(panel.two_phase_target, nested_prediction)
            metric_rows.append(
                {
                    "surface": panel.name,
                    "mechanism": label,
                    "selected_config": selected.key,
                    "nominal_parameter_count": 7,
                    "semigroup_error": logistic.semigroup_error(0.37, selected),
                    **{
                        f"global_{key}": value
                        for key, value in audit.scalar_metrics(panel.two_phase_target, global_prediction).items()
                    },
                    **{f"nested_{key}": value for key, value in nested_metrics.items()},
                }
            )
            optimum, surface = raw_optimum(panel, selected)
            optimum_rows.append(optimum)
            if nonorthogonal:
                surface.to_csv(args.output_dir / f"{panel.name}__surface.csv", index=False)
                render_surface(panel, surface, args.output_dir / f"{panel.name}__surface.html")

        active = selected_by_mechanism[True]
        selected_active[panel.name] = active
        integration_errors[panel.name] = logistic.integration_error(
            panel.weights[:16, :, 1], transition_fraction(panel, active), active
        )
        active_nested_rmse[panel.name] = float(
            np.sqrt(np.mean((predictions_by_mechanism[True] - panel.two_phase_target) ** 2))
        )
        ablation_rmse = float(np.sqrt(np.mean((predictions_by_mechanism[False] - panel.two_phase_target) ** 2)))
        global_comparison[panel.name] = active_nested_rmse[panel.name] < ablation_rmse
        wins = 0
        for fold, (_train, test) in enumerate(starcoder.surface_folds(panel)):
            ablation_fold = float(
                np.sqrt(np.mean((predictions_by_mechanism[False][test] - panel.two_phase_target[test]) ** 2))
            )
            active_fold = float(
                np.sqrt(np.mean((predictions_by_mechanism[True][test] - panel.two_phase_target[test]) ** 2))
            )
            wins += int(active_fold < ablation_fold)
            fold_rows.extend(
                [
                    {"surface": panel.name, "fold": fold, "mechanism": "orthogonal", "outer_rmse": ablation_fold},
                    {"surface": panel.name, "fold": fold, "mechanism": "nonorthogonal", "outer_rmse": active_fold},
                ]
            )
        fold_wins[panel.name] = wins

    metrics = pd.DataFrame(metric_rows)
    selections = pd.concat(selection_tables, ignore_index=True)
    nested = pd.concat(nested_tables, ignore_index=True)
    optima = pd.DataFrame(optimum_rows)
    folds = pd.DataFrame(fold_rows)
    active_optima = optima[optima["mechanism"].eq("nonorthogonal")]
    selected_angles = [config.task_angle_degrees for config in selected_active.values()]
    angle_regime_stable = all(angle < 90.0 for angle in selected_angles) or all(
        angle > 90.0 for angle in selected_angles
    )
    fold_angle_sides = nested[nested["mechanism"].eq("nonorthogonal")].assign(
        angle_side=lambda frame: np.sign(frame["task_angle_degrees"] - 90.0)
    )
    stable_fold_sides = all(
        group["angle_side"].value_counts().iloc[0] >= 3 for _surface, group in fold_angle_sides.groupby("surface")
    )
    gates = {
        "algebraic_semigroup": bool((metrics["semigroup_error"] < 1e-8).all()),
        "nonorthogonal_beats_ablation_global": all(global_comparison.values()),
        "nonorthogonal_beats_ablation_folds": all(wins >= 3 for wins in fold_wins.values()),
        "within_5pct_shape": all(
            active_nested_rmse[surface] <= 1.05 * SHAPE_REFERENCE[surface] for surface in SHAPE_REFERENCE
        ),
        "angle_regime_stable": angle_regime_stable and stable_fold_sides,
        "raw_optimum_distance_ok": bool((active_optima["distance_to_observed_best"] <= 0.15).all()),
        "schedule_signature_ok": bool(
            abs(
                float(active_optima.loc[active_optima["surface"].eq("starcoder_cosine_50_50"), "phase1_rare"].iloc[0])
                - float(active_optima.loc[active_optima["surface"].eq("starcoder_cosine_50_50"), "phase0_rare"].iloc[0])
            )
            <= 0.15
            and float(active_optima.loc[active_optima["surface"].eq("starcoder_wsd_80_20"), "phase1_rare"].iloc[0])
            > float(active_optima.loc[active_optima["surface"].eq("starcoder_wsd_80_20"), "phase0_rare"].iloc[0])
        ),
        "integration_stable": all(error < 1e-7 for error in integration_errors.values()),
    }
    metrics.to_csv(args.output_dir / "surface_metrics.csv", index=False)
    selections.to_csv(args.output_dir / "hyperparameter_grid.csv", index=False)
    nested.to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    folds.to_csv(args.output_dir / "foldwise_ablation.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    pd.DataFrame([{"surface": key, "integration_error": value} for key, value in integration_errors.items()]).to_csv(
        args.output_dir / "integration_audit.csv", index=False
    )
    pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()]).to_csv(
        args.output_dir / "gates.csv", index=False
    )
    update_registry_and_ledger(gates, args.output_dir)
    decision = "promoted_to_multi_swarm" if all(gates.values()) else "blocked_before_multi_swarm"
    report = [
        "# Round 49: logistic margin-competition flow",
        "",
        "The equations, orthogonal ablation, grid, and gates were frozen before evaluation. No new adversarial or sealed-confirmation outcome was read.",
        "",
        "## Decision",
        "",
        f"**{decision}.** " + "; ".join(f"{key}={value}" for key, value in gates.items()),
        "",
        "## Nested metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Foldwise nonorthogonal versus orthogonal",
        "",
        folds.to_markdown(index=False),
        "",
        "## Raw optima",
        "",
        optima.to_markdown(index=False),
        "",
        "## Selected nonorthogonal configurations",
        "",
        pd.DataFrame(
            [{"surface": surface, **asdict(config)} for surface, config in selected_active.items()]
        ).to_markdown(index=False),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
