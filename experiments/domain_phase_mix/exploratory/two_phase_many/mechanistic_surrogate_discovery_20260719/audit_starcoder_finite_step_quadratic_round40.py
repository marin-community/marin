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
"""Falsify finite-step quadratic task flow on both StarCoder surfaces."""

from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    audit_starcoder_shared_private_round25 as shape_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    finite_step_quadratic_models as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    hessian_equilibrium_models as heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    starcoder_optimizer_schedule as schedules,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round40_finite_step_quadratic_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
CURVATURE_GRID = (0.5, 1.0, 2.0)
ANISOTROPY_GRID = (0.25, 1.0, 4.0)
ANGLE_GRID = (0.0, 45.0, 90.0)
RELAXATION_GRID = (1.0, 4.0, 16.0, 64.0)
EVALUATION_GRID = (-0.2, 0.0, 0.2)
ORDER_GRID = (1, 2, 4)
L2_GRID = (0.0, 0.1, 1.0)
SHAPE_REFERENCE = shape_audit.SHAPE_REFERENCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[candidate.FiniteStepQuadraticConfig]:
    return [
        candidate.FiniteStepQuadraticConfig(curvature, anisotropy, angle, relaxation, evaluation, order)
        for curvature in CURVATURE_GRID
        for anisotropy in ANISOTROPY_GRID
        for angle in ANGLE_GRID
        for relaxation in RELAXATION_GRID
        for evaluation in EVALUATION_GRID
        for order in ORDER_GRID
    ]


def schedule_for_panel(panel: paired.PairedPanel) -> schedules.OptimizerScheduleSpec:
    return schedules.schedule_for_name(panel.name)


def base_features(
    panel: paired.PairedPanel,
    all_configs: list[candidate.FiniteStepQuadraticConfig],
    weights: np.ndarray | None = None,
) -> np.ndarray:
    policies = panel.weights if weights is None else weights
    schedule = schedule_for_panel(panel)
    return np.asarray([candidate.response_feature(policies, schedule, config) for config in all_configs], dtype=float)


def expanded_variants(
    all_configs: list[candidate.FiniteStepQuadraticConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config_index = np.repeat(np.arange(len(all_configs)), len(L2_GRID))
    l2 = np.tile(np.asarray(L2_GRID, dtype=float), len(all_configs))
    return features[config_index], config_index, l2


def score_all(
    panel: paired.PairedPanel,
    all_configs: list[candidate.FiniteStepQuadraticConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    expanded, config_index, l2 = expanded_variants(all_configs, features)
    rmse, predictions = scalar_audit.score_configs(
        expanded,
        panel.two_phase_target,
        starcoder.surface_folds(panel),
        l2,
    )
    return rmse, predictions, config_index, l2


def best_by_order(
    panel: paired.PairedPanel,
    all_configs: list[candidate.FiniteStepQuadraticConfig],
    features: np.ndarray,
) -> tuple[pd.DataFrame, dict[int, tuple[candidate.FiniteStepQuadraticConfig, float, np.ndarray, np.ndarray]]]:
    rmse, predictions, config_index, l2 = score_all(panel, all_configs, features)
    rows = []
    selected: dict[int, tuple[candidate.FiniteStepQuadraticConfig, float, np.ndarray, np.ndarray]] = {}
    for order in ORDER_GRID:
        eligible = np.asarray([all_configs[int(index)].expansion_order == order for index in config_index], dtype=bool)
        local = np.flatnonzero(eligible)
        best = int(local[np.argmin(rmse[local])])
        config_position = int(config_index[best])
        config = all_configs[config_position]
        selected[order] = (config, float(l2[best]), predictions[best], features[config_position])
        rows.append(
            {
                "surface": panel.name,
                "expansion_order": order,
                "oof_rmse": float(rmse[best]),
                "l2": float(l2[best]),
                **asdict(config),
                **{
                    f"oof_{key}": value
                    for key, value in metrics.scalar_metrics(panel.two_phase_target, predictions[best]).items()
                },
            }
        )
    return pd.DataFrame(rows), selected


def fold_order_winners(
    panel: paired.PairedPanel,
    all_configs: list[candidate.FiniteStepQuadraticConfig],
    features: np.ndarray,
) -> pd.DataFrame:
    expanded, config_index, l2 = expanded_variants(all_configs, features)
    rows: list[dict[str, Any]] = []
    for outer_fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner_folds = scalar_audit.stratified_folds(panel, outer_train, 4, 20260719 + 100 * outer_fold)
        local_folds = [
            (
                np.flatnonzero(np.isin(outer_train, train)),
                np.flatnonzero(np.isin(outer_train, test)),
            )
            for train, test in inner_folds
        ]
        inner_rmse, _ = scalar_audit.score_configs(
            expanded[:, outer_train],
            panel.two_phase_target[outer_train],
            local_folds,
            l2,
        )
        for order in ORDER_GRID:
            eligible = np.asarray(
                [all_configs[int(index)].expansion_order == order for index in config_index], dtype=bool
            )
            local = np.flatnonzero(eligible)
            best = int(local[np.argmin(inner_rmse[local])])
            prediction = scalar_audit.fit_predict_all(
                expanded[[best]],
                panel.two_phase_target,
                outer_train,
                outer_test,
                l2[[best]],
            )[0]
            rows.append(
                {
                    "surface": panel.name,
                    "outer_fold": outer_fold,
                    "expansion_order": order,
                    "inner_rmse": float(inner_rmse[best]),
                    "outer_rmse": float(np.sqrt(np.mean((prediction - panel.two_phase_target[outer_test]) ** 2))),
                    "l2": float(l2[best]),
                    **asdict(all_configs[int(config_index[best])]),
                }
            )
    return pd.DataFrame(rows)


def raw_optimum(
    panel: paired.PairedPanel,
    config: candidate.FiniteStepQuadraticConfig,
    l2: float,
    fit_feature: np.ndarray,
) -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, 201)
    p0, p1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - p0.ravel(), p0.ravel()]),
            np.column_stack([1.0 - p1.ravel(), p1.ravel()]),
        ],
        axis=1,
    )
    feature = candidate.response_feature(weights, schedule_for_panel(panel), config)
    head = heads.fit_quadratic_head(fit_feature, panel.two_phase_target, np.arange(panel.n), l2)
    prediction = head.predict(feature)
    best = int(np.argmin(prediction))
    observed = int(np.argmin(panel.two_phase_target))
    return {
        "surface": panel.name,
        "expansion_order": config.expansion_order,
        "phase0_rare": float(p0.ravel()[best]),
        "phase1_rare": float(p1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "observed_phase0_rare": float(panel.weights[observed, 0, 1]),
        "observed_phase1_rare": float(panel.weights[observed, 1, 1]),
        "observed_bpb": float(panel.two_phase_target[observed]),
        "distance_to_observed_best": float(
            np.hypot(
                p0.ravel()[best] - panel.weights[observed, 0, 1],
                p1.ravel()[best] - panel.weights[observed, 1, 1],
            )
        ),
        "response_amplitude": float(head.coefficient / head.feature_scale),
    }


def update_status(status: str, evidence: str, output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    mask = registry["id"].eq("FSQF")
    registry.loc[mask, "status"] = status
    registry.loc[mask, "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_40_starcoder_decision",
        "candidate_id": "FSQF",
        "candidate_family": "Finite-step quadratic task flow",
        "hyperparameters": "Frozen round-40 grid with first-order continuous-flow ablation",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round-40 preregistration.",
        "novelty_class": "Higher optimizer-step moments in a discrete transition law",
        "evaluation_status": status,
        "evidence_path": str((output_dir / "report.md").relative_to(OUTPUT_ROOT)),
        "notes": evidence,
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    key = tuple(row[column] for column in identity)
    if key not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    all_configs = configs()
    cosine_data = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine_data),
        starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine_data)),
    ]

    summary_frames = []
    fold_frames = []
    optimum_rows = []
    exact_rows = []
    for panel in panels:
        features = base_features(panel, all_configs)
        summary, selected = best_by_order(panel, all_configs, features)
        summary_frames.append(summary)
        fold_frames.append(fold_order_winners(panel, all_configs, features))
        for order, (config, l2, _prediction, fit_feature) in selected.items():
            optimum_rows.append(raw_optimum(panel, config, l2, fit_feature))
            exact_rows.append(
                {
                    "surface": panel.name,
                    "expansion_order": order,
                    "approximation_error": candidate.exact_approximation_error(
                        panel.weights[:: max(1, panel.n // 32)], schedule_for_panel(panel), config
                    ),
                    "tied_boundary_error": candidate.tied_boundary_error(
                        np.linspace(0.05, 0.95, 19), schedule_for_panel(panel), config
                    ),
                }
            )

    summary_table = pd.concat(summary_frames, ignore_index=True)
    fold_table = pd.concat(fold_frames, ignore_index=True)
    optimum_table = pd.DataFrame(optimum_rows)
    exact_table = pd.DataFrame(exact_rows)
    summary_table.to_csv(args.output_dir / "global_oof_by_order.csv", index=False)
    fold_table.to_csv(args.output_dir / "foldwise_order_comparison.csv", index=False)
    optimum_table.to_csv(args.output_dir / "raw_optima.csv", index=False)
    exact_table.to_csv(args.output_dir / "exact_product_audit.csv", index=False)

    summary_index = summary_table.set_index(["surface", "expansion_order"])
    higher_beats_first = all(
        min(
            float(summary_index.loc[(panel.name, 2), "oof_rmse"]),
            float(summary_index.loc[(panel.name, 4), "oof_rmse"]),
        )
        < float(summary_index.loc[(panel.name, 1), "oof_rmse"])
        for panel in panels
    )
    selected_orders = {
        panel.name: int(
            summary_table.loc[summary_table["surface"].eq(panel.name)].sort_values("oof_rmse").iloc[0]["expansion_order"]
        )
        for panel in panels
    }
    higher_selected = all(order > 1 for order in selected_orders.values())
    fold_pivot = fold_table.pivot_table(index=["surface", "outer_fold"], columns="expansion_order", values="outer_rmse")
    higher_fold_wins = {
        surface: int((frame[[2, 4]].min(axis=1) < frame[1]).sum()) for surface, frame in fold_pivot.groupby(level=0)
    }
    fold_gate = all(value >= 3 for value in higher_fold_wins.values())
    selected_summary = summary_table.loc[
        summary_table.apply(lambda row: int(row["expansion_order"]) == selected_orders[row["surface"]], axis=1)
    ].set_index("surface")
    shape_gate = all(
        float(selected_summary.loc[panel.name, "oof_rmse"]) <= 1.05 * SHAPE_REFERENCE[panel.name] for panel in panels
    )
    selected_optima = optimum_table.loc[
        optimum_table.apply(lambda row: int(row["expansion_order"]) == selected_orders[row["surface"]], axis=1)
    ]
    optimum_gate = bool((selected_optima["distance_to_observed_best"] <= 0.15).all())
    exact_gate = bool((exact_table.loc[exact_table["expansion_order"].gt(1), "approximation_error"] < 2e-3).all())
    boundary_gate = bool((exact_table["tied_boundary_error"] < 1e-10).all())
    passed = (
        higher_beats_first
        and higher_selected
        and fold_gate
        and shape_gate
        and optimum_gate
        and exact_gate
        and boundary_gate
    )
    status = "promoted_after_starcoder" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"higher_order_beats_first={higher_beats_first}; selected_orders={selected_orders}; "
        f"higher_order_fold_wins={higher_fold_wins}; within_5pct_shape={shape_gate}; "
        f"raw_optimum_distance_ok={optimum_gate}; exact_product_ok={exact_gate}; tied_boundary_ok={boundary_gate}."
    )
    update_status(status, evidence, args.output_dir)

    report = [
        "# Round 40: finite-step quadratic task flow",
        "",
        "All schedule moments, task grids, and gates were frozen before this StarCoder evaluation. No historical, exposed-adversarial, or sealed-confirmation outcome was read.",
        "",
        "## Decision",
        "",
        f"**{status}.** {evidence}",
        "",
        "## Global OOF comparison",
        "",
        summary_table.to_markdown(index=False),
        "",
        "## Foldwise order comparison",
        "",
        fold_table.to_markdown(index=False),
        "",
        "## Raw optima",
        "",
        optimum_table.to_markdown(index=False),
        "",
        "## Exact product audit",
        "",
        exact_table.to_markdown(index=False),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
