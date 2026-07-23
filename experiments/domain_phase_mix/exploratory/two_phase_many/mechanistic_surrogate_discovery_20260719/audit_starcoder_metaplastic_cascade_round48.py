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
"""Falsify a directed metaplastic consolidation cascade on StarCoder."""

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
    metaplastic_cascade_models as cascade,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    starcoder_refined_data,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round48_metaplastic_cascade_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
SHAPE_REFERENCE = {"starcoder_cosine_50_50": 0.06538840580863309, "starcoder_wsd_80_20": 0.04577251086960991}
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class FittedModel:
    panel: paired.PairedPanel
    config: cascade.Config
    head: paired.LinearHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.head.predict(design(self.panel, weights, self.config))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[cascade.Config]:
    shallow = [
        cascade.Config(clock, 1, acquisition, forgetting, 1.0, 1.0, 0.0, rare_ratio, l2)
        for clock in cascade.Clock
        for acquisition in (1.0, 4.0, 16.0)
        for forgetting in (0.25, 1.0, 4.0)
        for rare_ratio in (0.5, 2.0)
        for l2 in (0.1, 1.0)
    ]
    deep = [
        cascade.Config(clock, 3, acquisition, forgetting, consolidation, depth_ratio, durable, rare_ratio, l2)
        for clock in cascade.Clock
        for acquisition in (1.0, 4.0, 16.0)
        for forgetting in (0.25, 1.0, 4.0)
        for consolidation in (0.25, 1.0, 4.0)
        for depth_ratio in (2.0, 8.0)
        for durable in (0.25, 0.75)
        for rare_ratio in (0.5, 2.0)
        for l2 in (0.1, 1.0)
    ]
    return [*shallow, *deep]


def transition_fraction(panel: paired.PairedPanel, config: cascade.Config) -> float:
    if config.clock is cascade.Clock.TOKEN:
        return panel.alpha0
    return audit.clock_audit.optimizer_clock_fraction(panel)[0]


def design(panel: paired.PairedPanel, weights: np.ndarray, config: cascade.Config) -> np.ndarray:
    rare = weights[:, :, 1]
    return cascade.unresolved_design(rare, transition_fraction(panel, config), config)


def fit_model(panel: paired.PairedPanel, indices: np.ndarray, config: cascade.Config) -> FittedModel:
    matrix = design(panel, panel.weights[indices], config)
    head = paired.fit_linear_head(
        matrix,
        panel.two_phase_target[indices],
        ("unresolved_broad", "unresolved_rare"),
        np.ones(2, dtype=int),
        config.l2,
    )
    return FittedModel(panel, config, head)


def all_designs(panel: paired.PairedPanel, candidates: list[cascade.Config]) -> list[np.ndarray]:
    return [design(panel, panel.weights, config) for config in candidates]


def level_selection(
    panel: paired.PairedPanel,
    candidates: list[cascade.Config],
    designs: list[np.ndarray],
    levels: int,
) -> tuple[cascade.Config, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    indices = [index for index, config in enumerate(candidates) if config.levels == levels]
    subset_configs = [candidates[index] for index in indices]
    subset_designs = [designs[index] for index in indices]
    selected_index, global_prediction, selection = audit.select_global(panel, subset_configs, subset_designs)
    nested_prediction, nested = audit.nested_selection(panel, subset_configs, subset_designs)
    return subset_configs[selected_index], global_prediction, selection, nested_prediction, nested


def raw_optimum(panel: paired.PairedPanel, config: cascade.Config) -> tuple[dict[str, Any], pd.DataFrame]:
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
        "levels": config.levels,
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


def render_surface(panel: paired.PairedPanel, surface: pd.DataFrame, levels: int, output: Path) -> None:
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
        title=f"{panel.name}: metaplastic cascade K={levels}",
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
    registry.loc[registry["id"].eq("MCCF"), "status"] = status
    registry.loc[registry["id"].eq("MCCF"), "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_48_starcoder_gate",
        "candidate_id": "MCCF",
        "candidate_family": "Metaplastic consolidation-cascade flow",
        "hyperparameters": "Frozen preregistered K=1/K=3 transition and ridge grids",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_48_preregistration",
        "novelty_class": "Directed triangular metaplastic cascade",
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
    selected_deep: dict[str, cascade.Config] = {}
    deep_nested_rmse: dict[str, float] = {}
    global_comparison: dict[str, bool] = {}
    fold_wins: dict[str, int] = {}
    for panel in panels:
        matrices = all_designs(panel, candidates)
        selected_by_level: dict[int, cascade.Config] = {}
        predictions_by_level: dict[int, np.ndarray] = {}
        for levels in (1, 3):
            selected, global_prediction, selection, nested_prediction, nested = level_selection(
                panel, candidates, matrices, levels
            )
            selected_by_level[levels] = selected
            predictions_by_level[levels] = nested_prediction
            selection_tables.append(selection.assign(levels=levels))
            nested_tables.append(nested.assign(levels=levels))
            nested_metrics = audit.scalar_metrics(panel.two_phase_target, nested_prediction)
            metric_rows.append(
                {
                    "surface": panel.name,
                    "levels": levels,
                    "selected_config": selected.key,
                    "nominal_parameter_count": 3 + (3 if levels == 1 else 7),
                    "semigroup_error": cascade.semigroup_error(0.37, selected),
                    **{
                        f"global_{key}": value
                        for key, value in audit.scalar_metrics(panel.two_phase_target, global_prediction).items()
                    },
                    **{f"nested_{key}": value for key, value in nested_metrics.items()},
                }
            )
            optimum, surface = raw_optimum(panel, selected)
            optimum_rows.append(optimum)
            surface.to_csv(args.output_dir / f"{panel.name}__K{levels}__surface.csv", index=False)
            render_surface(panel, surface, levels, args.output_dir / f"{panel.name}__K{levels}__surface.html")

        selected_deep[panel.name] = selected_by_level[3]
        deep_nested_rmse[panel.name] = float(np.sqrt(np.mean((predictions_by_level[3] - panel.two_phase_target) ** 2)))
        shallow_rmse = float(np.sqrt(np.mean((predictions_by_level[1] - panel.two_phase_target) ** 2)))
        global_comparison[panel.name] = deep_nested_rmse[panel.name] < shallow_rmse
        wins = 0
        for fold, (_train, test) in enumerate(starcoder.surface_folds(panel)):
            shallow_fold = float(np.sqrt(np.mean((predictions_by_level[1][test] - panel.two_phase_target[test]) ** 2)))
            deep_fold = float(np.sqrt(np.mean((predictions_by_level[3][test] - panel.two_phase_target[test]) ** 2)))
            wins += int(deep_fold < shallow_fold)
            fold_rows.extend(
                [
                    {"surface": panel.name, "fold": fold, "levels": 1, "outer_rmse": shallow_fold},
                    {"surface": panel.name, "fold": fold, "levels": 3, "outer_rmse": deep_fold},
                ]
            )
        fold_wins[panel.name] = wins

    metrics = pd.DataFrame(metric_rows)
    selections = pd.concat(selection_tables, ignore_index=True)
    nested = pd.concat(nested_tables, ignore_index=True)
    optima = pd.DataFrame(optimum_rows)
    folds = pd.DataFrame(fold_rows)
    deep_optima = optima[optima["levels"].eq(3)]
    selected_values = list(selected_deep.values())
    rates_interior = all(
        1.0 < config.acquisition_rate < 16.0
        and 0.25 < config.forgetting_ratio < 4.0
        and 0.25 < config.consolidation_rate < 4.0
        for config in selected_values
    )
    gates = {
        "algebraic_semigroup": bool((metrics["semigroup_error"] < 1e-10).all()),
        "cascade_beats_shallow_global": all(global_comparison.values()),
        "cascade_beats_shallow_folds": all(wins >= 3 for wins in fold_wins.values()),
        "within_5pct_shape": all(
            deep_nested_rmse[surface] <= 1.05 * SHAPE_REFERENCE[surface] for surface in SHAPE_REFERENCE
        ),
        "transition_rates_interior": rates_interior,
        "raw_optimum_distance_ok": bool((deep_optima["distance_to_observed_best"] <= 0.15).all()),
        "schedule_signature_ok": bool(
            abs(
                float(deep_optima.loc[deep_optima["surface"].eq("starcoder_cosine_50_50"), "phase1_rare"].iloc[0])
                - float(deep_optima.loc[deep_optima["surface"].eq("starcoder_cosine_50_50"), "phase0_rare"].iloc[0])
            )
            <= 0.15
            and float(deep_optima.loc[deep_optima["surface"].eq("starcoder_wsd_80_20"), "phase1_rare"].iloc[0])
            > float(deep_optima.loc[deep_optima["surface"].eq("starcoder_wsd_80_20"), "phase0_rare"].iloc[0])
        ),
    }
    metrics.to_csv(args.output_dir / "surface_metrics.csv", index=False)
    selections.to_csv(args.output_dir / "hyperparameter_grid.csv", index=False)
    nested.to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    folds.to_csv(args.output_dir / "foldwise_ablation.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()]).to_csv(
        args.output_dir / "gates.csv", index=False
    )
    update_registry_and_ledger(gates, args.output_dir)
    decision = "promoted_to_multi_swarm" if all(gates.values()) else "blocked_before_multi_swarm"
    report = [
        "# Round 48: metaplastic consolidation-cascade flow",
        "",
        "The equations, K=1 ablation, hyperparameter grid, and gates were frozen before evaluation. No new adversarial or sealed-confirmation outcome was read.",
        "",
        "## Decision",
        "",
        f"**{decision}.** " + "; ".join(f"{key}={value}" for key, value in gates.items()),
        "",
        "## Nested metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Foldwise K=3 versus K=1",
        "",
        folds.to_markdown(index=False),
        "",
        "## Raw optima",
        "",
        optima.to_markdown(index=False),
        "",
        "## Selected K=3 configurations",
        "",
        pd.DataFrame([{"surface": surface, **asdict(config)} for surface, config in selected_deep.items()]).to_markdown(
            index=False
        ),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
