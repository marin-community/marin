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
"""Falsify gradient-Gram coupled loss flow on both StarCoder schedules."""

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
    audit_starcoder_logistic_margin_round49 as plot_helpers,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    gradient_gram_loss_models as gradient_gram,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    starcoder_refined_data,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round51_gradient_gram_loss_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
SHAPE_REFERENCE = {"starcoder_cosine_50_50": 0.06538840580863309, "starcoder_wsd_80_20": 0.04577251086960991}


@dataclass(frozen=True)
class FittedModel:
    panel: paired.PairedPanel
    config: gradient_gram.Config
    head: paired.LinearHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return self.head.predict(design(self.panel, weights, self.config))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[gradient_gram.Config]:
    return [
        gradient_gram.Config(clock, rate, power, correlation, rare_ratio, l2)
        for clock in gradient_gram.Clock
        for rate in (1.0, 4.0, 16.0)
        for power in (1.0, 3.0, 7.0)
        for correlation in (0.0, 0.25, 0.5, 0.75)
        for rare_ratio in (0.5, 1.0, 2.0)
        for l2 in (0.1, 1.0)
    ]


def transition_fraction(panel: paired.PairedPanel, config: gradient_gram.Config) -> float:
    if config.clock is gradient_gram.Clock.TOKEN:
        return panel.alpha0
    return audit.clock_audit.optimizer_clock_fraction(panel)[0]


def design(panel: paired.PairedPanel, weights: np.ndarray, config: gradient_gram.Config) -> np.ndarray:
    return gradient_gram.terminal_unresolved(weights[:, :, 1], transition_fraction(panel, config), config)


def fit_model(panel: paired.PairedPanel, indices: np.ndarray, config: gradient_gram.Config) -> FittedModel:
    head = paired.fit_linear_head(
        design(panel, panel.weights[indices], config),
        panel.two_phase_target[indices],
        ("broad_excess_loss", "rare_excess_loss"),
        np.ones(2, dtype=int),
        config.l2,
    )
    return FittedModel(panel, config, head)


def select_mechanism(
    panel: paired.PairedPanel,
    candidates: list[gradient_gram.Config],
    designs: list[np.ndarray],
    coupled: bool,
) -> tuple[gradient_gram.Config, np.ndarray, pd.DataFrame, np.ndarray, pd.DataFrame]:
    indices = [index for index, config in enumerate(candidates) if (config.gradient_correlation > 0.0) == coupled]
    subset_configs = [candidates[index] for index in indices]
    subset_designs = [designs[index] for index in indices]
    selected_index, global_prediction, selection = audit.select_global(panel, subset_configs, subset_designs)
    nested_prediction, nested = audit.nested_selection(panel, subset_configs, subset_designs)
    return subset_configs[selected_index], global_prediction, selection, nested_prediction, nested


def raw_optimum(panel: paired.PairedPanel, config: gradient_gram.Config) -> tuple[dict[str, Any], pd.DataFrame]:
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
    return (
        {
            "surface": panel.name,
            "mechanism": "coupled" if config.gradient_correlation > 0.0 else "independent",
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
        },
        pd.DataFrame({"phase0_rare": p0.ravel(), "phase1_rare": p1.ravel(), "predicted_bpb": prediction}),
    )


def update_registry_and_ledger(gates: dict[str, bool], output_dir: Path) -> None:
    status = "promoted_to_multi_swarm" if all(gates.values()) else "blocked_before_multi_swarm"
    evidence = "; ".join(f"{key}={value}" for key, value in gates.items())
    registry = pd.read_csv(REGISTRY)
    registry.loc[registry["id"].eq("GGLLF"), "status"] = status
    registry.loc[registry["id"].eq("GGLLF"), "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_51_starcoder_gate",
        "candidate_id": "GGLLF",
        "candidate_family": "Gradient-Gram coupled loss flow",
        "hyperparameters": "Frozen preregistered clock/rate/power/correlation/rare-scale/ridge grid",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_51_preregistration",
        "novelty_class": "PSD task-gradient Gram coupled power-law loss state",
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
    selected_active: dict[str, gradient_gram.Config] = {}
    active_nested_rmse: dict[str, float] = {}
    global_comparison: dict[str, bool] = {}
    fold_wins: dict[str, int] = {}
    integration_errors: dict[str, float] = {}
    for panel in panels:
        matrices = [design(panel, panel.weights, config) for config in candidates]
        selected_by_mechanism: dict[bool, gradient_gram.Config] = {}
        predictions_by_mechanism: dict[bool, np.ndarray] = {}
        for coupled in (False, True):
            selected, global_prediction, selection, nested_prediction, nested = select_mechanism(
                panel, candidates, matrices, coupled
            )
            selected_by_mechanism[coupled] = selected
            predictions_by_mechanism[coupled] = nested_prediction
            label = "coupled" if coupled else "independent"
            selection_tables.append(selection.assign(surface=panel.name, mechanism=label))
            nested_tables.append(nested.assign(surface=panel.name, mechanism=label))
            metric_rows.append(
                {
                    "surface": panel.name,
                    "mechanism": label,
                    "selected_config": selected.key,
                    "nominal_parameter_count": 7,
                    "semigroup_error": gradient_gram.semigroup_error(0.37, selected),
                    **{
                        f"global_{key}": value
                        for key, value in audit.scalar_metrics(panel.two_phase_target, global_prediction).items()
                    },
                    **{
                        f"nested_{key}": value
                        for key, value in audit.scalar_metrics(panel.two_phase_target, nested_prediction).items()
                    },
                }
            )
            optimum, surface = raw_optimum(panel, selected)
            optimum_rows.append(optimum)
            if coupled:
                surface.to_csv(args.output_dir / f"{panel.name}__surface.csv", index=False)
                plot_helpers.render_surface(
                    panel,
                    surface,
                    args.output_dir / f"{panel.name}__surface.html",
                    model_label="gradient-Gram coupled loss flow",
                )

        active = selected_by_mechanism[True]
        selected_active[panel.name] = active
        integration_errors[panel.name] = gradient_gram.integration_error(
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
                    {"surface": panel.name, "fold": fold, "mechanism": "independent", "outer_rmse": ablation_fold},
                    {"surface": panel.name, "fold": fold, "mechanism": "coupled", "outer_rmse": active_fold},
                ]
            )
        fold_wins[panel.name] = wins

    metrics = pd.DataFrame(metric_rows)
    optima = pd.DataFrame(optimum_rows)
    active_optima = optima[optima["mechanism"].eq("coupled")]
    selected_correlations = [config.gradient_correlation for config in selected_active.values()]
    nested = pd.concat(nested_tables, ignore_index=True)
    fold_correlations = nested[nested["mechanism"].eq("coupled")]["gradient_correlation"]
    gates = {
        "algebraic_semigroup": bool((metrics["semigroup_error"] < 1e-7).all()),
        "coupled_beats_independent_global": all(global_comparison.values()),
        "coupled_beats_independent_folds": all(wins >= 3 for wins in fold_wins.values()),
        "within_5pct_shape": all(
            active_nested_rmse[surface] <= 1.05 * SHAPE_REFERENCE[surface] for surface in SHAPE_REFERENCE
        ),
        "correlation_interior": all(0.0 < value < 0.75 for value in selected_correlations),
        "correlation_stable": bool(fold_correlations.value_counts().iloc[0] >= 6),
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
    pd.concat(selection_tables, ignore_index=True).to_csv(args.output_dir / "hyperparameter_grid.csv", index=False)
    nested.to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(args.output_dir / "foldwise_ablation.csv", index=False)
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
        "# Round 51: gradient-Gram coupled loss flow",
        "",
        "The equations, independent-learning ablation, grid, and gates were frozen before evaluation. No new adversarial or sealed-confirmation outcome was read.",
        "",
        "## Decision",
        "",
        f"**{decision}.** " + "; ".join(f"{key}={value}" for key, value in gates.items()),
        "",
        "## Nested metrics",
        "",
        metrics.to_markdown(index=False),
        "",
        "## Foldwise coupled versus independent",
        "",
        pd.DataFrame(fold_rows).to_markdown(index=False),
        "",
        "## Raw optima",
        "",
        optima.to_markdown(index=False),
        "",
        "## Selected coupled configurations",
        "",
        pd.DataFrame(
            [{"surface": surface, **asdict(config)} for surface, config in selected_active.items()]
        ).to_markdown(index=False),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
