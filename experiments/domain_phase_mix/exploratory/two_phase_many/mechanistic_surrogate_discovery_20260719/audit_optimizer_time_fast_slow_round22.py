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
"""Falsify optimizer-time fast/slow consolidation on both StarCoder schedules."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
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
    audit_starcoder_hessian_equilibrium_round11 as scalar_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_optimizer_time_flow_round20 as clock_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metric_helpers,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    starcoder_refined_data,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round22_optimizer_time_fast_slow_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
PRIOR_GATE = {"starcoder_cosine_50_50": 0.06538840580863309, "starcoder_wsd_80_20": 0.04577251086960991}
SEED = 20260719
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


class Clock(StrEnum):
    TOKEN = "token_time"
    OPTIMIZER = "optimizer_time"


@dataclass(frozen=True)
class Config:
    clock: Clock
    learn_rate: float
    forget_rate: float
    consolidate_rate: float
    slow_weight: float
    l2: float

    @property
    def key(self) -> str:
        return (
            f"clock={self.clock.value},learn={self.learn_rate:g},forget={self.forget_rate:g},"
            f"consolidate={self.consolidate_rate:g},slow={self.slow_weight:g},l2={self.l2:g}"
        )


@dataclass(frozen=True)
class FittedModel:
    panel: paired.PairedPanel
    aggregate_config: paired.AggregateConfig
    config: Config
    head: paired.LinearHead

    def predict(self, weights: np.ndarray) -> np.ndarray:
        design, _names, _signs = combined_design(self.panel, weights, self.aggregate_config, self.config)
        return self.head.predict(design)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[Config]:
    return [
        Config(clock, learn, forget, consolidate, slow, l2)
        for clock in Clock
        for learn in (0.25, 1.0, 4.0, 16.0, 64.0)
        for forget in (0.125, 0.5, 2.0, 8.0, 32.0)
        for consolidate in (0.0625, 0.25, 1.0, 4.0, 16.0, 64.0)
        for slow in (0.25, 0.5, 0.75)
        for l2 in (0.1, 1.0)
    ]


def aggregate_config(panel: paired.PairedPanel) -> paired.AggregateConfig:
    if panel.name == "starcoder_cosine_50_50":
        return paired.AggregateConfig(shortage_power=1.0, shortage_offset=1.0, l2=1.0)
    if panel.name == "starcoder_wsd_80_20":
        return paired.AggregateConfig(shortage_power=1.0, shortage_offset=0.1, l2=1.0)
    raise ValueError(f"Unknown StarCoder surface {panel.name}")


def clock_fraction(panel: paired.PairedPanel, clock: Clock) -> float:
    if clock is Clock.TOKEN:
        return panel.alpha0
    return clock_audit.optimizer_clock_fraction(panel)[0]


def terminal_state(panel: paired.PairedPanel, weights: np.ndarray, config: Config) -> np.ndarray:
    phase0, phase1, _aggregate = paired.family_weight_mass(panel, weights)
    fast = np.zeros_like(phase0)
    slow = np.zeros_like(phase0)
    early = clock_fraction(panel, config.clock)
    update_config = paired.FastSlowConfig(
        learn_rate=config.learn_rate,
        forget_rate=config.forget_rate,
        consolidate_rate=config.consolidate_rate,
        slow_weight=config.slow_weight,
        l2=config.l2,
        state_level="family",
    )
    fast, slow = paired.update_fast_slow(fast, slow, phase0, early, update_config)
    fast, slow = paired.update_fast_slow(fast, slow, phase1, 1.0 - early, update_config)
    return (1.0 - config.slow_weight) * fast + config.slow_weight * slow


def phase_design(
    panel: paired.PairedPanel,
    weights: np.ndarray,
    config: Config,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    terminal = terminal_state(panel, weights, config)
    tied = terminal_state(panel, paired.tied_weights(panel, weights), config)
    design = -(terminal - tied)
    names = tuple(f"optimizer_clock_consolidated_capability::{family}" for family in panel.family_names)
    return design, names, np.ones(len(names), dtype=int)


def combined_design(
    panel: paired.PairedPanel,
    weights: np.ndarray,
    aggregate: paired.AggregateConfig,
    config: Config,
) -> tuple[np.ndarray, tuple[str, ...], np.ndarray]:
    aggregate_design, aggregate_names = paired.aggregate_design(
        panel,
        paired.tied_weights(panel, weights),
        aggregate,
    )
    temporal_design, temporal_names, temporal_signs = phase_design(panel, weights, config)
    names = (*aggregate_names, *temporal_names)
    signs = np.concatenate([np.ones(len(aggregate_names), dtype=int), temporal_signs])
    return np.column_stack([aggregate_design, temporal_design]), names, signs


def fit_model(
    panel: paired.PairedPanel,
    indices: np.ndarray,
    aggregate: paired.AggregateConfig,
    config: Config,
) -> FittedModel:
    design, names, signs = combined_design(panel, panel.weights[indices], aggregate, config)
    head = paired.fit_linear_head(design, panel.two_phase_target[indices], names, signs, config.l2)
    return FittedModel(panel, aggregate, config, head)


def all_designs(
    panel: paired.PairedPanel,
    aggregate: paired.AggregateConfig,
    candidates: list[Config],
) -> list[np.ndarray]:
    return [combined_design(panel, panel.weights, aggregate, config)[0] for config in candidates]


def fit_predict(
    panel: paired.PairedPanel,
    design: np.ndarray,
    config: Config,
    train: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
    aggregate_width = design.shape[1] - len(panel.family_names)
    names = tuple(f"aggregate::{index}" for index in range(aggregate_width)) + tuple(
        f"temporal::{family}" for family in panel.family_names
    )
    signs = np.ones(design.shape[1], dtype=int)
    head = paired.fit_linear_head(design[train], panel.two_phase_target[train], names, signs, config.l2)
    return head.predict(design[test])


def oof_score(
    panel: paired.PairedPanel,
    design: np.ndarray,
    config: Config,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, np.ndarray]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    for train, test in folds:
        prediction[test] = fit_predict(panel, design, config, train, test)
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete OOF prediction for {panel.name}/{config.key}")
    rmse = float(np.sqrt(np.mean((prediction - panel.two_phase_target) ** 2)))
    return rmse, prediction


def select_global(
    panel: paired.PairedPanel,
    candidates: list[Config],
    designs: list[np.ndarray],
) -> tuple[int, np.ndarray, pd.DataFrame]:
    folds = starcoder.surface_folds(panel)
    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    for index, (config, design) in enumerate(zip(candidates, designs, strict=True)):
        rmse, prediction = oof_score(panel, design, config, folds)
        predictions.append(prediction)
        rows.append(
            {"candidate_index": index, "surface": panel.name, "config": config.key, **config.__dict__, "rmse": rmse}
        )
    table = pd.DataFrame(rows).sort_values("rmse")
    selected = int(table.iloc[0]["candidate_index"])
    return selected, predictions[selected], table


def nested_selection(
    panel: paired.PairedPanel,
    candidates: list[Config],
    designs: list[np.ndarray],
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    for fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner_folds_global = scalar_audit.stratified_folds(panel, outer_train, 4, SEED + fold)
        inner_folds = [
            (np.flatnonzero(np.isin(outer_train, train)), np.flatnonzero(np.isin(outer_train, test)))
            for train, test in inner_folds_global
        ]
        scores: list[float] = []
        for config, design in zip(candidates, designs, strict=True):
            subset_panel = paired.PairedPanel(
                name=panel.name,
                target=panel.target,
                frame=panel.frame.iloc[outer_train].reset_index(drop=True),
                domain_names=panel.domain_names,
                family_names=panel.family_names,
                family_members=panel.family_members,
                weights=panel.weights[outer_train],
                c0=panel.c0,
                c1=panel.c1,
                two_phase_target=panel.two_phase_target[outer_train],
                one_phase_target=panel.one_phase_target[outer_train],
            )
            subset_design = design[outer_train]
            score, _inner_prediction = oof_score(subset_panel, subset_design, config, inner_folds)
            scores.append(score)
        selected_index = int(np.argmin(scores))
        selected = candidates[selected_index]
        prediction[outer_test] = fit_predict(
            panel,
            designs[selected_index],
            selected,
            outer_train,
            outer_test,
        )
        rows.append(
            {
                "surface": panel.name,
                "outer_fold": fold,
                "selected_index": selected_index,
                "selected_config": selected.key,
                "inner_rmse": scores[selected_index],
                **selected.__dict__,
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested prediction for {panel.name}")
    return prediction, pd.DataFrame(rows)


def scalar_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in metric_helpers.scalar_metrics(observed, predicted).items()
        if isinstance(value, float | int)
    }


def leave_region_out(
    panel: paired.PairedPanel,
    design: np.ndarray,
    config: Config,
) -> pd.DataFrame:
    contrast = panel.weights[:, 1, 1] - panel.weights[:, 0, 1]
    regions = {
        "late_rare_enriched": contrast > 0.1,
        "early_rare_enriched": contrast < -0.1,
        "near_phase_tied": np.abs(contrast) <= 0.1,
    }
    rows: list[dict[str, Any]] = []
    for region, mask in regions.items():
        train = np.flatnonzero(~mask)
        test = np.flatnonzero(mask)
        prediction = fit_predict(panel, design, config, train, test)
        rows.append(
            {
                "surface": panel.name,
                "region": region,
                "n_train": len(train),
                "n_test": len(test),
                **scalar_metrics(panel.two_phase_target[test], prediction),
            }
        )
    return pd.DataFrame(rows)


def optimum(
    panel: paired.PairedPanel,
    aggregate: paired.AggregateConfig,
    config: Config,
) -> tuple[dict[str, Any], pd.DataFrame]:
    model = fit_model(panel, np.arange(panel.n), aggregate, config)
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
            "config": config.key,
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
            "token_phase0_fraction": panel.alpha0,
            "transition_phase0_fraction": clock_fraction(panel, config.clock),
            "natural_coefficients": json.dumps(
                dict(zip(model.head.feature_names, model.head.coefficients_in_natural_units, strict=True)),
                sort_keys=True,
            ),
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
        title=f"{panel.name}: optimizer-time fast/slow consolidation",
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


def semigroup_audit(panel: paired.PairedPanel, config: Config) -> float:
    tied = paired.tied_weights(panel, panel.weights)
    terminal = terminal_state(panel, tied, config)
    phase0, _phase1, _aggregate = paired.family_weight_mass(panel, tied)
    update_config = paired.FastSlowConfig(
        config.learn_rate,
        config.forget_rate,
        config.consolidate_rate,
        config.slow_weight,
        config.l2,
        "family",
    )
    fast = np.zeros_like(phase0)
    slow = np.zeros_like(phase0)
    fast, slow = paired.update_fast_slow(fast, slow, phase0, 1.0, update_config)
    uninterrupted = (1.0 - config.slow_weight) * fast + config.slow_weight * slow
    return float(np.max(np.abs(terminal - uninterrupted)))


def update_registry_and_ledger(gates: dict[str, bool], output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    status = "promoted_to_multi_swarm" if all(gates.values()) else "blocked_before_multi_swarm"
    registry.loc[registry["id"].eq("OTFSC"), "status"] = status
    registry.loc[registry["id"].eq("OTFSC"), "status_evidence"] = "; ".join(
        f"{key}={value}" for key, value in gates.items()
    )
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_22_starcoder_gate",
        "candidate_id": "OTFSC",
        "candidate_family": "Optimizer-time fast/slow consolidation",
        "hyperparameters": "Frozen preregistered clock/rate/slow-weight/ridge grid with exact token-clock ablation",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_22_preregistration",
        "novelty_class": "Fixed optimizer-time transition for a two-timescale competence state",
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
    candidates = configs()
    cosine = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine),
        starcoder.panel_from_dataset(starcoder_refined_data.load_refined_wsd80_starcoder(cosine)),
    ]
    result_rows: list[dict[str, Any]] = []
    selection_tables: list[pd.DataFrame] = []
    nested_tables: list[pd.DataFrame] = []
    region_tables: list[pd.DataFrame] = []
    optimum_rows: list[dict[str, Any]] = []
    selected_configs: dict[str, Config] = {}
    nested_rmses: dict[str, float] = {}
    semigroup_errors: dict[str, float] = {}
    for panel in panels:
        aggregate = aggregate_config(panel)
        designs = all_designs(panel, aggregate, candidates)
        selected_index, selected_prediction, selection = select_global(panel, candidates, designs)
        nested_prediction, nested = nested_selection(panel, candidates, designs)
        selected = candidates[selected_index]
        selected_configs[panel.name] = selected
        nested_metrics = scalar_metrics(panel.two_phase_target, nested_prediction)
        nested_rmses[panel.name] = nested_metrics["rmse"]
        semigroup_errors[panel.name] = semigroup_audit(panel, selected)
        result_rows.append(
            {
                "surface": panel.name,
                "selected_config": selected.key,
                "nominal_parameter_count": 11,
                "semigroup_error": semigroup_errors[panel.name],
                **{
                    f"selection_{key}": value
                    for key, value in scalar_metrics(panel.two_phase_target, selected_prediction).items()
                },
                **{f"nested_{key}": value for key, value in nested_metrics.items()},
            }
        )
        selection_tables.append(selection)
        nested_tables.append(nested)
        region_tables.append(leave_region_out(panel, designs[selected_index], selected))
        optimum_row, surface = optimum(panel, aggregate, selected)
        optimum_rows.append(optimum_row)
        surface.to_csv(args.output_dir / f"{panel.name}__surface.csv", index=False)
        render_surface(panel, surface, args.output_dir / f"{panel.name}__surface.html")

    results = pd.DataFrame(result_rows)
    selection = pd.concat(selection_tables, ignore_index=True)
    nested = pd.concat(nested_tables, ignore_index=True)
    regions = pd.concat(region_tables, ignore_index=True)
    optima = pd.DataFrame(optimum_rows)
    clock_ablation = selection.groupby(["surface", "clock"], as_index=False)["rmse"].min()
    prior = pd.DataFrame(
        [
            {
                "surface": surface,
                "nested_rmse": rmse,
                "shape_gate_rmse": PRIOR_GATE[surface],
                "relative_rmse": rmse / PRIOR_GATE[surface] - 1.0,
            }
            for surface, rmse in nested_rmses.items()
        ]
    )
    selected_rate_ratio = max(
        selected_configs[panels[0].name].learn_rate, selected_configs[panels[1].name].learn_rate
    ) / min(
        selected_configs[panels[0].name].learn_rate,
        selected_configs[panels[1].name].learn_rate,
    )
    selected_consolidate_ratio = max(
        selected_configs[panels[0].name].consolidate_rate,
        selected_configs[panels[1].name].consolidate_rate,
    ) / min(
        selected_configs[panels[0].name].consolidate_rate,
        selected_configs[panels[1].name].consolidate_rate,
    )
    bounds = {
        "learn_rate": (0.25, 64.0),
        "forget_rate": (0.125, 32.0),
        "consolidate_rate": (0.0625, 64.0),
    }
    no_rate_boundaries = all(
        bounds[name][0] < getattr(config, name) < bounds[name][1]
        for config in selected_configs.values()
        for name in bounds
    )
    optimizer_counts = (
        nested.assign(is_optimizer=nested["clock"].eq(Clock.OPTIMIZER.value)).groupby("surface")["is_optimizer"].sum()
    )
    gates = {
        "semigroup_ok": all(error < 1e-10 for error in semigroup_errors.values()),
        "optimizer_clock_global": all(config.clock is Clock.OPTIMIZER for config in selected_configs.values()),
        "optimizer_clock_folds": all(int(optimizer_counts.get(panel.name, 0)) >= 3 for panel in panels),
        "optimizer_clock_beats_token": bool(
            clock_ablation.groupby("surface")
            .apply(
                lambda frame: float(frame.loc[frame["clock"].eq(Clock.OPTIMIZER.value), "rmse"].iloc[0])
                < float(frame.loc[frame["clock"].eq(Clock.TOKEN.value), "rmse"].iloc[0]),
                include_groups=False,
            )
            .all()
        ),
        "within_5pct_shape_gate": bool((prior["relative_rmse"] <= 0.05).all()),
        "optimum_distance_ok": bool((optima["distance_to_observed_best"] <= 0.15).all()),
        "rates_not_on_boundary": no_rate_boundaries,
        "rate_regime_transfer": selected_rate_ratio <= 4.0 and selected_consolidate_ratio <= 4.0,
    }
    results.to_csv(args.output_dir / "surface_metrics.csv", index=False)
    selection.to_csv(args.output_dir / "hyperparameter_grid.csv", index=False)
    nested.to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    regions.to_csv(args.output_dir / "leave_region_out.csv", index=False)
    optima.to_csv(args.output_dir / "predicted_optima.csv", index=False)
    clock_ablation.to_csv(args.output_dir / "clock_ablation.csv", index=False)
    prior.to_csv(args.output_dir / "prior_comparison.csv", index=False)
    update_registry_and_ledger(gates, args.output_dir)
    report = [
        "# Optimizer-time fast/slow consolidation: StarCoder falsification",
        "",
        "The clock, rate grid, aggregate spine, and stop criteria were frozen before fitting. No historical, adversarial, or sealed-confirmation target was evaluated.",
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
        results.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Corrected shape gate",
        "",
        prior.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Raw optima",
        "",
        optima.drop(columns=["natural_coefficients"]).to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Leave-region-out",
        "",
        regions.to_markdown(index=False, floatfmt=".6f"),
        "",
    ]
    (args.output_dir / "report.md").write_text("\n".join(report))
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(
            {
                "seed": SEED,
                "candidate_count": len(candidates),
                "historical_targets_read": False,
                "adversarial_targets_read": False,
                "sealed_confirmation_targets_read": False,
                "aggregate_spine_source": "Frozen prior IFSC schedule-specific selection",
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(results.to_string(index=False))
    print("\nGates")
    print(pd.DataFrame([{"gate": key, "passed": value} for key, value in gates.items()]).to_string(index=False))


if __name__ == "__main__":
    main()
