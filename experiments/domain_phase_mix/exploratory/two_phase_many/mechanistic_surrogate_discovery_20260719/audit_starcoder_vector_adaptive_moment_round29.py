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
"""Falsify coordinatewise adaptive-moment gradient flow on StarCoder."""

from __future__ import annotations

import argparse
import json
import sys
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
    audit_starcoder_hessian_equilibrium_round11 as scalar_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as paired_screen,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    vector_adaptive_moment_models as adaptive,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round29_vector_adaptive_moment_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
SHAPE_REFERENCE = OUTPUT_ROOT / "round1_starcoder_shape_refined107/surface_oof_metrics.csv"
SEED = 20260719
CURVATURE_GRID = (0.25, 1.0, 4.0)
ANISOTROPY_GRID = (0.5, 2.0, 4.0)
ANGLE_GRID = (30.0, 75.0)
ADAPTIVE_SPEED_GRID = (1.0, 4.0)
MEMORY_GRID = (0.5, 2.0, 8.0)
EPSILON_GRID = (0.1, 0.3, 1.0)
GRADIENT_FLOW_SPEED_GRID = (1.0, 4.0, 16.0)
EVALUATION_GRID = (0.2, 0.5, 0.8)
L2_GRID = (0.1, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def geometries() -> list[adaptive.TaskGeometry]:
    return [
        adaptive.TaskGeometry(curvature, anisotropy, angle)
        for curvature in CURVATURE_GRID
        for anisotropy in ANISOTROPY_GRID
        for angle in ANGLE_GRID
    ]


def adaptive_configs() -> list[adaptive.AdaptiveMomentConfig]:
    return [
        adaptive.AdaptiveMomentConfig(geometry, speed, memory, epsilon, evaluation, l2)
        for geometry in geometries()
        for speed in ADAPTIVE_SPEED_GRID
        for memory in MEMORY_GRID
        for epsilon in EPSILON_GRID
        for evaluation in EVALUATION_GRID
        for l2 in L2_GRID
    ]


def ablation_configs() -> list[adaptive.GradientFlowConfig]:
    return [
        adaptive.GradientFlowConfig(geometry, speed, evaluation, l2)
        for geometry in geometries()
        for speed in GRADIENT_FLOW_SPEED_GRID
        for evaluation in EVALUATION_GRID
        for l2 in L2_GRID
    ]


def adaptive_feature_matrix(
    panel: paired.PairedPanel,
    configs: list[adaptive.AdaptiveMomentConfig],
) -> np.ndarray:
    cache: dict[tuple[Any, ...], np.ndarray] = {}
    rows = []
    for config in configs:
        key = (
            config.geometry,
            config.speed,
            config.memory_rate,
            config.epsilon,
            config.evaluation_rare_weight,
        )
        if key not in cache:
            cache[key] = adaptive.adaptive_response_feature(panel.weights, panel.alpha0, config)
        rows.append(cache[key])
    return np.asarray(rows, dtype=float)


def ablation_feature_matrix(
    panel: paired.PairedPanel,
    configs: list[adaptive.GradientFlowConfig],
) -> np.ndarray:
    cache: dict[tuple[Any, ...], np.ndarray] = {}
    rows = []
    for config in configs:
        key = (config.geometry, config.speed, config.evaluation_rare_weight)
        if key not in cache:
            cache[key] = adaptive.gradient_flow_response_feature(panel.weights, panel.alpha0, config)
        rows.append(cache[key])
    return np.asarray(rows, dtype=float)


def globally_selected_oof(
    panel: paired.PairedPanel,
    configs: list[adaptive.AdaptiveMomentConfig] | list[adaptive.GradientFlowConfig],
    features: np.ndarray,
) -> tuple[int, np.ndarray, pd.DataFrame]:
    l2_values = np.asarray([config.l2 for config in configs], dtype=float)
    rmse, predictions = scalar_audit.score_configs(
        features,
        panel.two_phase_target,
        starcoder.surface_folds(panel),
        l2_values,
    )
    best = int(np.argmin(rmse))
    table = pd.DataFrame(
        {
            "surface": panel.name,
            "config": [config.key for config in configs],
            "rmse": rmse,
        }
    ).sort_values("rmse")
    return best, predictions[best], table


def nested_selection(
    panel: paired.PairedPanel,
    configs: list[adaptive.AdaptiveMomentConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, pd.DataFrame]:
    prediction = np.full(panel.n, np.nan, dtype=float)
    rows: list[dict[str, Any]] = []
    l2_values = np.asarray([config.l2 for config in configs], dtype=float)
    for outer_fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner_folds = scalar_audit.stratified_folds(panel, outer_train, 4, SEED + 100 * outer_fold)
        local_folds = [
            (
                np.flatnonzero(np.isin(outer_train, train)),
                np.flatnonzero(np.isin(outer_train, test)),
            )
            for train, test in inner_folds
        ]
        inner_rmse, _ = scalar_audit.score_configs(
            features[:, outer_train],
            panel.two_phase_target[outer_train],
            local_folds,
            l2_values,
        )
        selected_index = int(np.argmin(inner_rmse))
        selected = configs[selected_index]
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
                "outer_fold": outer_fold,
                "selected_config": selected.key,
                "inner_rmse": float(inner_rmse[selected_index]),
                "curvature_ratio": selected.geometry.curvature_ratio,
                "anisotropy": selected.geometry.anisotropy,
                "angle_degrees": selected.geometry.angle_degrees,
                "speed": selected.speed,
                "memory_rate": selected.memory_rate,
                "epsilon": selected.epsilon,
                "evaluation_rare_weight": selected.evaluation_rare_weight,
                "l2": selected.l2,
            }
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError(f"Incomplete nested predictions for {panel.name}")
    return prediction, pd.DataFrame(rows)


def algebraic_audit(config: adaptive.AdaptiveMomentConfig) -> dict[str, float]:
    rng = np.random.default_rng(SEED)
    rare_weight = rng.uniform(size=64)
    state = rng.normal(scale=0.1, size=(64, 2))
    second_moment = np.square(rng.normal(scale=0.1, size=(64, 2)))
    first_state, first_moment = adaptive.adaptive_phase_update(
        state,
        second_moment,
        rare_weight,
        0.37,
        config,
    )
    split_state, split_moment = adaptive.adaptive_phase_update(
        first_state,
        first_moment,
        rare_weight,
        0.63,
        config,
    )
    whole_state, whole_moment = adaptive.adaptive_phase_update(
        state,
        second_moment,
        rare_weight,
        1.0,
        config,
    )
    return {
        "maximum_tied_state_semigroup_error": float(np.max(np.abs(split_state - whole_state))),
        "maximum_tied_moment_semigroup_error": float(np.max(np.abs(split_moment - whole_moment))),
        "minimum_second_moment": float(min(np.min(split_moment), np.min(whole_moment))),
    }


def raw_optimum(
    panel: paired.PairedPanel,
    config: adaptive.AdaptiveMomentConfig,
) -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, 101)
    phase0, phase1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - phase0.ravel(), phase0.ravel()]),
            np.column_stack([1.0 - phase1.ravel(), phase1.ravel()]),
        ],
        axis=1,
    )
    train_feature = adaptive.adaptive_response_feature(panel.weights, panel.alpha0, config)
    head = adaptive.fit_head(train_feature, panel.two_phase_target, np.arange(panel.n), config.l2)
    grid_feature = adaptive.adaptive_response_feature(weights, panel.alpha0, config)
    prediction = head.predict(grid_feature)
    best = int(np.argmin(prediction))
    observed_best = int(np.argmin(panel.two_phase_target))
    return {
        "surface": panel.name,
        "phase0_rare": float(phase0.ravel()[best]),
        "phase1_rare": float(phase1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "observed_best_phase0_rare": float(panel.weights[observed_best, 0, 1]),
        "observed_best_phase1_rare": float(panel.weights[observed_best, 1, 1]),
        "observed_best_bpb": float(panel.two_phase_target[observed_best]),
        "distance_to_observed_best": float(
            np.hypot(
                phase0.ravel()[best] - panel.weights[observed_best, 0, 1],
                phase1.ravel()[best] - panel.weights[observed_best, 1, 1],
            )
        ),
        "response_amplitude": head.natural_curvature,
    }


def update_status(status: str, evidence: str, output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    mask = registry["id"].eq("VAMGF")
    registry.loc[mask, "status"] = status
    registry.loc[mask, "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_29_starcoder_gate",
        "candidate_id": "VAMGF",
        "candidate_family": "Vector adaptive-moment gradient flow",
        "hyperparameters": "Frozen Round 29 grid; exact vector-gradient-flow ablation; nested selection only after stage-1 survival",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round_29_batch_preregistration",
        "novelty_class": "Coordinatewise adaptive preconditioner over non-collinear domain gradients",
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
    adaptive_grid = adaptive_configs()
    ablation_grid = ablation_configs()
    representative = adaptive.AdaptiveMomentConfig(geometries()[0], 1.0, 2.0, 0.3, 0.5, 0.1)
    algebra = algebraic_audit(representative)
    (args.output_dir / "algebraic_audit.json").write_text(json.dumps(algebra, indent=2) + "\n")

    cosine = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine),
        starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine)),
    ]
    stage1_rows = []
    adaptive_tables = []
    ablation_tables = []
    selected_adaptive: dict[str, adaptive.AdaptiveMomentConfig] = {}
    feature_cache: dict[str, np.ndarray] = {}
    for panel in panels:
        adaptive_features = adaptive_feature_matrix(panel, adaptive_grid)
        ablation_features = ablation_feature_matrix(panel, ablation_grid)
        adaptive_index, adaptive_prediction, adaptive_table = globally_selected_oof(
            panel,
            adaptive_grid,
            adaptive_features,
        )
        ablation_index, ablation_prediction, ablation_table = globally_selected_oof(
            panel,
            ablation_grid,
            ablation_features,
        )
        selected_adaptive[panel.name] = adaptive_grid[adaptive_index]
        feature_cache[panel.name] = adaptive_features
        adaptive_metrics = paired_screen.scalar_metrics(panel.two_phase_target, adaptive_prediction)
        ablation_metrics = paired_screen.scalar_metrics(panel.two_phase_target, ablation_prediction)
        stage1_rows.append(
            {
                "surface": panel.name,
                "adaptive_config": adaptive_grid[adaptive_index].key,
                "ablation_config": ablation_grid[ablation_index].key,
                **{f"adaptive_{key}": value for key, value in adaptive_metrics.items()},
                **{f"ablation_{key}": value for key, value in ablation_metrics.items()},
                "rmse_delta_adaptive_minus_ablation": adaptive_metrics["rmse"] - ablation_metrics["rmse"],
            }
        )
        adaptive_tables.append(adaptive_table)
        ablation_tables.append(ablation_table)

    stage1 = pd.DataFrame(stage1_rows)
    stage1.to_csv(args.output_dir / "stage1_global_oof_comparison.csv", index=False)
    pd.concat(adaptive_tables, ignore_index=True).to_csv(args.output_dir / "adaptive_grid.csv", index=False)
    pd.concat(ablation_tables, ignore_index=True).to_csv(args.output_dir / "ablation_grid.csv", index=False)
    stage1_passed = bool((stage1["rmse_delta_adaptive_minus_ablation"] < 0.0).all())
    report = [
        "# Round 29: vector adaptive-moment gradient flow",
        "",
        "## Frozen mechanism",
        "",
        r"The state follows coordinatewise RMSProp flow $\dot z=-k g/(\sqrt v+\epsilon)$ and $\dot v=\kappa(g\odot g-v)$ under non-collinear broad/rare quadratic gradients. The ablation uses exact vector gradient flow with matched task geometry and evaluation potential.",
        "",
        "## Algebraic audit",
        "",
        f"- Maximum tied state semigroup error: `{algebra['maximum_tied_state_semigroup_error']:.3e}`.",
        f"- Maximum tied moment semigroup error: `{algebra['maximum_tied_moment_semigroup_error']:.3e}`.",
        f"- Minimum second moment: `{algebra['minimum_second_moment']:.3e}`.",
        "",
        "## Stage 1: global OOF against exact no-memory flow",
        "",
        stage1.to_markdown(index=False),
        "",
    ]
    if not stage1_passed:
        evidence = (
            "Exact vector gradient flow matched or beat VAMGF on at least one StarCoder schedule; stage 2 was not run."
        )
        update_status("blocked_before_nested_starcoder", evidence, args.output_dir)
        report.extend(
            ["## Decision", "", f"**Blocked.** {evidence}", "", "No Delphi or adversarial outcomes were evaluated."]
        )
        (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
        print(stage1.to_string(index=False))
        print(evidence)
        return

    shape_reference = pd.read_csv(SHAPE_REFERENCE).groupby("surface", as_index=False)["rmse"].min()
    nested_rows = []
    nested_tables = []
    optimum_rows = []
    for panel in panels:
        nested_prediction, nested_table = nested_selection(panel, adaptive_grid, feature_cache[panel.name])
        metrics = paired_screen.scalar_metrics(panel.two_phase_target, nested_prediction)
        nested_rows.append({"surface": panel.name, **metrics})
        nested_tables.append(nested_table)
        optimum_rows.append(raw_optimum(panel, selected_adaptive[panel.name]))

    nested_metrics = pd.DataFrame(nested_rows).merge(shape_reference, on="surface", suffixes=("", "_reference"))
    nested_metrics["relative_to_reference"] = nested_metrics["rmse"] / nested_metrics["rmse_reference"] - 1.0
    nested_folds = pd.concat(nested_tables, ignore_index=True)
    optima = pd.DataFrame(optimum_rows)
    nested_metrics.to_csv(args.output_dir / "nested_oof_metrics.csv", index=False)
    nested_folds.to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    optima.to_csv(args.output_dir / "raw_optima.csv", index=False)

    compatible_memory = nested_folds.groupby("surface")["memory_rate"].agg(["min", "max"])
    compatible_memory["ratio"] = compatible_memory["max"] / compatible_memory["min"]
    compatible_memory.to_csv(args.output_dir / "memory_stability.csv")
    shape_ok = bool((nested_metrics["relative_to_reference"] <= 0.05).all())
    optimum_ok = bool((optima["distance_to_observed_best"] <= 0.15).all())
    memory_ok = bool((compatible_memory["ratio"] <= 4.0).all())
    passed = shape_ok and optimum_ok and memory_ok
    status = "promoted_to_multi_swarm" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"adaptive_beats_ablation={stage1_passed}; within_5pct_shape={shape_ok}; "
        f"raw_optimum_distance_ok={optimum_ok}; fold_memory_ratio_ok={memory_ok}."
    )
    update_status(status, evidence, args.output_dir)
    report.extend(
        [
            "## Nested StarCoder audit",
            "",
            nested_metrics.to_markdown(index=False),
            "",
            "## Foldwise memory stability",
            "",
            compatible_memory.reset_index().to_markdown(index=False),
            "",
            "## Raw optima",
            "",
            optima.to_markdown(index=False),
            "",
            "## Decision",
            "",
            f"Status: **{status}**. {evidence}",
            "",
            "No Delphi or adversarial outcomes were evaluated.",
        ]
    )
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(stage1.to_string(index=False))
    print(nested_metrics.to_string(index=False))
    print(optima.to_string(index=False))
    print(status, evidence)


if __name__ == "__main__":
    main()
