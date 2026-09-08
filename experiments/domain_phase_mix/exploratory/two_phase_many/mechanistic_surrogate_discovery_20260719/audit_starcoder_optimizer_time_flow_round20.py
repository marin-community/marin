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
"""Falsify optimizer-time task-potential flow on both StarCoder schedules."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import StrEnum
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
    audit_starcoder_nonlinear_task_potential_round14 as original_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    nonlinear_task_potential_models as nonlinear,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round20_optimizer_time_flow_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
PRIOR_FRONTIER = {"starcoder_cosine_50_50": 0.065388405808633, "starcoder_wsd_80_20": 0.0457725108696099}
SEED = 20260719
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Clock(StrEnum):
    TOKEN_TIME = "token_time"
    OPTIMIZER_TIME = "optimizer_time"


@dataclass(frozen=True)
class ClockConfig:
    clock: Clock
    potential: nonlinear.NonlinearPotentialConfig

    @property
    def key(self) -> str:
        return f"clock={self.clock.value},{self.potential.key}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[ClockConfig]:
    return [ClockConfig(clock, config) for clock in Clock for config in original_audit.configs()]


def optimizer_clock_fraction(panel: paired.PairedPanel) -> tuple[float, float]:
    boundary = panel.alpha0
    time = np.linspace(0.0, 1.0, 65537)
    if panel.name.startswith("starcoder_cosine"):
        learning_rate = 0.5 * (1.0 + np.cos(np.pi * time))
    else:
        decay_progress = np.maximum((time - boundary) / max(1.0 - boundary, 1e-12), 0.0)
        learning_rate = np.where(
            time <= boundary,
            1.0,
            0.5 * (1.0 + np.cos(np.pi * np.minimum(decay_progress, 1.0))),
        )
    early = float(np.trapezoid(learning_rate[time <= boundary], time[time <= boundary]))
    late = float(np.trapezoid(learning_rate[time >= boundary], time[time >= boundary]))
    return early / (early + late), early + late


def phase_fraction(panel: paired.PairedPanel, clock: Clock) -> float:
    if clock == Clock.TOKEN_TIME:
        return panel.alpha0
    return optimizer_clock_fraction(panel)[0]


def config_record(config: ClockConfig) -> dict[str, Any]:
    return {"clock": config.clock.value, **asdict(config.potential)}


def feature_matrix(panel: paired.PairedPanel, all_configs: list[ClockConfig]) -> np.ndarray:
    state_cache: dict[tuple[Any, ...], np.ndarray] = {}
    rows = []
    for config in all_configs:
        potential = config.potential
        key = (
            config.clock.value,
            potential.curvature_ratio,
            potential.quartic_strength,
            potential.quartic_ratio,
            potential.relaxation,
        )
        if key not in state_cache:
            state_cache[key] = nonlinear.terminal_state(panel.weights, phase_fraction(panel, config.clock), potential)
        broad, rare = nonlinear.task_potential(state_cache[key], potential)
        rows.append((1.0 - potential.evaluation_weight) * broad + potential.evaluation_weight * rare)
    return np.asarray(rows, dtype=float)


def select_surface(
    panel: paired.PairedPanel,
    all_configs: list[ClockConfig],
    features: np.ndarray,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    l2 = np.asarray([config.potential.l2 for config in all_configs], dtype=float)
    rmse, predictions = scalar_audit.score_configs(features, panel.two_phase_target, starcoder.surface_folds(panel), l2)
    best = int(np.argmin(rmse))
    table = pd.DataFrame(
        [
            {
                "surface": panel.name,
                "config": config.key,
                **config_record(config),
                "rmse": float(rmse[index]),
            }
            for index, config in enumerate(all_configs)
        ]
    ).sort_values("rmse")
    return best, predictions[best], table


def nested_selection(
    panel: paired.PairedPanel,
    all_configs: list[ClockConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    l2 = np.asarray([config.potential.l2 for config in all_configs], dtype=float)
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
            np.asarray([selected.potential.l2]),
        )[0]
        rows.append(
            {
                "surface": panel.name,
                "outer_fold": fold,
                "selected_config": selected.key,
                "inner_rmse": float(scores[selected_index]),
                **config_record(selected),
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested prediction for {panel.name}")
    return prediction, pd.DataFrame(rows)


def tied_selection(
    panel: paired.PairedPanel,
    all_configs: list[ClockConfig],
    features: np.ndarray,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    tied = np.flatnonzero(panel.paired_mask)
    folds = list(KFold(min(5, len(tied)), shuffle=True, random_state=SEED + 77).split(tied))
    l2 = np.asarray([config.potential.l2 for config in all_configs], dtype=float)
    rmse, predictions = scalar_audit.score_configs(features[:, tied], panel.two_phase_target[tied], folds, l2)
    best = int(np.argmin(rmse))
    table = pd.DataFrame(
        [
            {
                "surface": panel.name,
                "config": config.key,
                **config_record(config),
                "rmse": float(rmse[index]),
            }
            for index, config in enumerate(all_configs)
        ]
    ).sort_values("rmse")
    return best, predictions[best], table


def leave_region_out(panel: paired.PairedPanel, config: ClockConfig, feature: np.ndarray) -> pd.DataFrame:
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
            np.asarray([config.potential.l2]),
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


def optimum_and_surface(panel: paired.PairedPanel, config: ClockConfig) -> tuple[dict[str, Any], pd.DataFrame]:
    fraction = phase_fraction(panel, config.clock)
    model = nonlinear.fit_model(
        panel.weights,
        panel.two_phase_target,
        np.arange(panel.n),
        fraction,
        config.potential,
    )
    grid = np.linspace(0.0, 1.0, 201)
    p0, p1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - p0.ravel(), p0.ravel()]),
            np.column_stack([1.0 - p1.ravel(), p1.ravel()]),
        ],
        axis=1,
    )
    prediction = model.predict(weights, fraction)
    best = int(np.argmin(prediction))
    observed_best = int(np.argmin(panel.two_phase_target))
    state = nonlinear.terminal_state(weights[[best]], fraction, config.potential)
    return (
        {
            "surface": panel.name,
            "clock": config.clock.value,
            "token_phase0_fraction": panel.alpha0,
            "optimizer_phase0_fraction": optimizer_clock_fraction(panel)[0],
            "phase0_rare": float(p0.ravel()[best]),
            "phase1_rare": float(p1.ravel()[best]),
            "predicted_bpb": float(prediction[best]),
            "terminal_state": float(state[0]),
            "observed_best_phase0_rare": float(panel.weights[observed_best, 0, 1]),
            "observed_best_phase1_rare": float(panel.weights[observed_best, 1, 1]),
            "observed_best_bpb": float(panel.two_phase_target[observed_best]),
            "distance_to_observed_best": float(
                np.hypot(
                    p0.ravel()[best] - panel.weights[observed_best, 0, 1],
                    p1.ravel()[best] - panel.weights[observed_best, 1, 1],
                )
            ),
            "response_amplitude": model.natural_amplitude,
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
        title=f"{panel.name}: optimizer-time task-potential flow",
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
    status = "promoted_to_multi_swarm" if all(gates.values()) else "blocked_before_multi_swarm"
    registry.loc[registry["id"].eq("OTTPF"), "status"] = status
    registry.loc[registry["id"].eq("OTTPF"), "status_evidence"] = "; ".join(
        f"{key}={value}" for key, value in gates.items()
    )
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_20_starcoder_clock_gate",
        "candidate_id": "OTTPF",
        "candidate_family": "Optimizer-time task-potential flow",
        "hyperparameters": "Frozen preregistered token-clock/optimizer-clock comparison with fixed schedule integrals",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_20_preregistration",
        "novelty_class": "Separate physical-token and integrated-learning-rate clocks",
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
    rows = []
    selection_tables = []
    nested_tables = []
    tied_tables = []
    region_tables = []
    optimum_rows = []
    clock_rows = []
    nested_rmse: dict[str, float] = {}
    selected_configs: dict[str, ClockConfig] = {}
    for panel in panels:
        optimizer_fraction, total_lr_mass = optimizer_clock_fraction(panel)
        clock_rows.append(
            {
                "surface": panel.name,
                "token_phase0_fraction": panel.alpha0,
                "optimizer_phase0_fraction": optimizer_fraction,
                "normalized_total_lr_mass": total_lr_mass,
            }
        )
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
                "nominal_parameter_count": 7,
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
    clock_table = pd.DataFrame(clock_rows)
    clock_ablation = (
        selection_table.groupby(["surface", "clock"], as_index=False)["rmse"].min().sort_values(["surface", "rmse"])
    )
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
    clock_table.to_csv(args.output_dir / "clock_masses.csv", index=False)
    clock_ablation.to_csv(args.output_dir / "clock_ablation.csv", index=False)
    prior.to_csv(args.output_dir / "prior_comparison.csv", index=False)

    optimizer_fold_counts = (
        nested_table.assign(is_optimizer=nested_table["clock"].eq(Clock.OPTIMIZER_TIME.value))
        .groupby("surface")["is_optimizer"]
        .sum()
    )
    gates = {
        "optimizer_clock_global": all(config.clock == Clock.OPTIMIZER_TIME for config in selected_configs.values()),
        "optimizer_clock_folds": all(optimizer_fold_counts.get(panel.name, 0) >= 3 for panel in panels),
        "optimizer_clock_beats_token": bool(
            clock_ablation.groupby("surface")
            .apply(
                lambda frame: float(frame.loc[frame["clock"].eq(Clock.OPTIMIZER_TIME.value), "rmse"].iloc[0])
                < float(frame.loc[frame["clock"].eq(Clock.TOKEN_TIME.value), "rmse"].iloc[0]),
                include_groups=False,
            )
            .all()
        ),
        "within_5pct_prior_shape": bool((prior["relative_rmse"] <= 0.05).all()),
        "optimum_distance_ok": bool((optima["distance_to_observed_best"] <= 0.15).all()),
    }
    update_registry_and_ledger(gates, args.output_dir)

    report = [
        "# Optimizer-time task-potential flow: StarCoder falsification",
        "",
        "The form, clock ablation, and full grid were preregistered before this evaluation. The optimizer-time masses are fixed by the declared LR schedules. Historical, adversarial, and sealed-confirmation targets were not read.",
        "",
        "## Physical clocks",
        "",
        clock_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Gates",
        "",
        pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()]).to_markdown(index=False),
        "",
        "## Clock ablation",
        "",
        clock_ablation.to_markdown(index=False, floatfmt=".6f"),
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
        "The optimizer clock is supported only if it beats token time on both schedules, survives nested selection, reaches the prior shape frontier, and preserves the observed optimum geometry.",
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "candidate_count": len(all_configs),
                "historical_targets_read": False,
                "adversarial_targets_read": False,
                "sealed_confirmation_targets_read": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
