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
"""Falsify architecture-depth and adaptive-kernel mechanisms on StarCoder."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import export_mixture_fit_observatory as observatory
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    adaptive_ntk_models as antk,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    audit_starcoder_clipped_task_flow_round35 as clock,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    audit_starcoder_hessian_equilibrium_round11 as scalar_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    audit_starcoder_shared_private_round25 as shape_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    balanced_depth_models as balanced,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    hessian_equilibrium_models as heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    screen_paired_dynamics_round1 as metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round45_architecture_kernel_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
L2_GRID = (0.0, 0.1, 1.0)
SHAPE_REFERENCE = shape_audit.SHAPE_REFERENCE


@dataclass(frozen=True)
class Mechanism:
    candidate_id: str
    family: str
    active_transition: str
    transitions: tuple[str, ...]
    configs: tuple[Any, ...]
    feature: Callable[[np.ndarray, float, Any, int], np.ndarray]
    integration_error: Callable[[np.ndarray, float, Any], float]
    stability_parameter: str
    stability_grid: tuple[float, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def balanced_configs() -> tuple[balanced.BalancedDepthConfig, ...]:
    rows = []
    for relaxation in (0.5, 2.0, 8.0, 32.0):
        for ratio in (0.125, 0.5, 2.0, 8.0):
            for evaluation in (0.25, 0.5, 0.75):
                rows.extend(
                    [
                        balanced.BalancedDepthConfig(10, relaxation, ratio, evaluation, "declared_depth"),
                        balanced.BalancedDepthConfig(1, relaxation, ratio, evaluation, "depth_one"),
                        balanced.BalancedDepthConfig(10, relaxation, ratio, evaluation, "frozen_trunk"),
                    ]
                )
    return tuple(rows)


def antk_configs() -> tuple[antk.AdaptiveKernelConfig, ...]:
    rows = []
    for angle in (30.0, 60.0, 90.0):
        for anisotropy in (2.0, 8.0):
            for residual in (0.5, 2.0, 8.0):
                for curvature in (0.5, 2.0):
                    for evaluation in (0.25, 0.5, 0.75):
                        for adaptation in (0.5, 2.0, 8.0):
                            rows.append(
                                antk.AdaptiveKernelConfig(
                                    angle, anisotropy, adaptation, residual, curvature, evaluation, "adaptive"
                                )
                            )
                        rows.extend(
                            [
                                antk.AdaptiveKernelConfig(
                                    angle, anisotropy, 0.0, residual, curvature, evaluation, "frozen"
                                ),
                                antk.AdaptiveKernelConfig(
                                    angle, anisotropy, 0.0, residual, curvature, evaluation, "instantaneous"
                                ),
                            ]
                        )
    return tuple(rows)


def mechanisms() -> tuple[Mechanism, ...]:
    return (
        Mechanism(
            "BDLMTF",
            "Balanced-depth linear multitask flow",
            "declared_depth",
            ("declared_depth", "depth_one", "frozen_trunk"),
            balanced_configs(),
            lambda weights, alpha, config, steps: balanced.response_feature(
                weights, alpha, config, steps_per_unit=steps
            ),
            balanced.integration_error,
            "head_rate_ratio",
            (0.125, 0.5, 2.0, 8.0),
        ),
        Mechanism(
            "ANTKF",
            "Adaptive neural-tangent-kernel flow",
            "adaptive",
            ("adaptive", "frozen", "instantaneous"),
            antk_configs(),
            lambda weights, alpha, config, steps: antk.response_feature(weights, alpha, config, steps_per_unit=steps),
            antk.integration_error,
            "kernel_adaptation",
            (0.5, 2.0, 8.0),
        ),
    )


def features(panel, mechanism: Mechanism, weights: np.ndarray | None = None, *, steps: int = 192) -> np.ndarray:
    policies = panel.weights if weights is None else weights
    alpha = clock.optimizer_phase0_fraction(panel)
    return np.asarray([mechanism.feature(policies, alpha, config, steps) for config in mechanism.configs])


def expanded(mechanism: Mechanism, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.repeat(np.arange(len(mechanism.configs)), len(L2_GRID))
    l2 = np.tile(np.asarray(L2_GRID), len(mechanism.configs))
    return feature[indices], indices, l2


def best_by_transition(panel, mechanism: Mechanism, feature: np.ndarray):
    design, indices, l2 = expanded(mechanism, feature)
    rmse, predictions = scalar_audit.score_configs(design, panel.two_phase_target, starcoder.surface_folds(panel), l2)
    rows = []
    selected = {}
    for transition in mechanism.transitions:
        positions = np.flatnonzero([mechanism.configs[int(index)].transition == transition for index in indices])
        best = int(positions[np.argmin(rmse[positions])])
        config_index = int(indices[best])
        config = mechanism.configs[config_index]
        selected[transition] = (config, float(l2[best]), feature[config_index])
        rows.append(
            {
                "candidate_id": mechanism.candidate_id,
                "surface": panel.name,
                "transition": transition,
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


def fold_comparison(panel, mechanism: Mechanism, feature: np.ndarray) -> pd.DataFrame:
    design, indices, l2 = expanded(mechanism, feature)
    rows = []
    for outer_fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner = scalar_audit.stratified_folds(panel, outer_train, 4, 20260719 + 100 * outer_fold)
        local = [
            (np.flatnonzero(np.isin(outer_train, train)), np.flatnonzero(np.isin(outer_train, test)))
            for train, test in inner
        ]
        inner_rmse, _ = scalar_audit.score_configs(
            design[:, outer_train], panel.two_phase_target[outer_train], local, l2
        )
        for transition in mechanism.transitions:
            positions = np.flatnonzero([mechanism.configs[int(index)].transition == transition for index in indices])
            best = int(positions[np.argmin(inner_rmse[positions])])
            prediction = scalar_audit.fit_predict_all(
                design[[best]], panel.two_phase_target, outer_train, outer_test, l2[[best]]
            )[0]
            rows.append(
                {
                    "candidate_id": mechanism.candidate_id,
                    "surface": panel.name,
                    "outer_fold": outer_fold,
                    "transition": transition,
                    "outer_rmse": float(np.sqrt(np.mean((prediction - panel.two_phase_target[outer_test]) ** 2))),
                    "inner_rmse": float(inner_rmse[best]),
                    "l2": float(l2[best]),
                    **asdict(mechanism.configs[int(indices[best])]),
                }
            )
    return pd.DataFrame(rows)


def raw_optimum(panel, mechanism: Mechanism, config, l2: float, fit_feature: np.ndarray) -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, 151)
    p0, p1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - p0.ravel(), p0.ravel()]),
            np.column_stack([1.0 - p1.ravel(), p1.ravel()]),
        ],
        axis=1,
    )
    feature = mechanism.feature(weights, clock.optimizer_phase0_fraction(panel), config, 384)
    head = heads.fit_quadratic_head(fit_feature, panel.two_phase_target, np.arange(panel.n), l2)
    prediction = head.predict(feature)
    best = int(np.argmin(prediction))
    observed = int(np.argmin(panel.two_phase_target))
    return {
        "candidate_id": mechanism.candidate_id,
        "surface": panel.name,
        "transition": config.transition,
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
    }


def update_status(mechanism: Mechanism, status: str, evidence: str, output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    registry.loc[registry["id"].eq(mechanism.candidate_id), ["status", "status_evidence"]] = [
        status,
        evidence,
    ]
    registry.to_csv(REGISTRY, index=False)
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": f"round_45_{mechanism.candidate_id.lower()}_decision",
        "candidate_id": mechanism.candidate_id,
        "candidate_family": mechanism.family,
        "hyperparameters": "Frozen source/architecture constants, finite mechanistic grids, mandatory nested ablations, and ridge grid",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round-45 portfolio preregistration.",
        "novelty_class": mechanism.family,
        "evaluation_status": status,
        "evidence_path": str((output_dir / f"{mechanism.candidate_id.lower()}_report.md").relative_to(OUTPUT_ROOT)),
        "notes": evidence,
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[key] for key in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


def evaluate(mechanism: Mechanism, panels: list[Any], output_dir: Path) -> None:
    summary_frames = []
    fold_frames = []
    optimum_rows = []
    integration_rows = []
    for panel in panels:
        feature = features(panel, mechanism)
        summary, selected = best_by_transition(panel, mechanism, feature)
        summary_frames.append(summary)
        fold_frames.append(fold_comparison(panel, mechanism, feature))
        for transition, (config, l2, fit_feature) in selected.items():
            optimum_rows.append(raw_optimum(panel, mechanism, config, l2, fit_feature))
            integration_rows.append(
                {
                    "candidate_id": mechanism.candidate_id,
                    "surface": panel.name,
                    "transition": transition,
                    "integration_error": mechanism.integration_error(
                        panel.weights[:: max(1, panel.n // 24)],
                        clock.optimizer_phase0_fraction(panel),
                        config,
                    ),
                }
            )
    summary = pd.concat(summary_frames, ignore_index=True)
    folds = pd.concat(fold_frames, ignore_index=True)
    optima = pd.DataFrame(optimum_rows)
    integration = pd.DataFrame(integration_rows)
    summary.to_csv(output_dir / f"{mechanism.candidate_id.lower()}_global_oof.csv", index=False)
    folds.to_csv(output_dir / f"{mechanism.candidate_id.lower()}_foldwise.csv", index=False)
    optima.to_csv(output_dir / f"{mechanism.candidate_id.lower()}_raw_optima.csv", index=False)
    integration.to_csv(output_dir / f"{mechanism.candidate_id.lower()}_integration.csv", index=False)

    active = summary.loc[summary["transition"].eq(mechanism.active_transition)].set_index("surface")
    ablation = summary.loc[~summary["transition"].eq(mechanism.active_transition)].groupby("surface")["oof_rmse"].min()
    global_gate = bool((active["oof_rmse"] < ablation).all())
    pivot = folds.pivot_table(index=["surface", "outer_fold"], columns="transition", values="outer_rmse")
    wins = {
        surface: int(
            (
                frame[mechanism.active_transition]
                < frame[[name for name in mechanism.transitions if name != mechanism.active_transition]].min(axis=1)
            ).sum()
        )
        for surface, frame in pivot.groupby(level=0)
    }
    fold_gate = all(value >= 3 for value in wins.values())
    shape_gate = all(float(active.loc[panel.name, "oof_rmse"]) <= 1.05 * SHAPE_REFERENCE[panel.name] for panel in panels)
    active_folds = folds.loc[folds["transition"].eq(mechanism.active_transition)]
    values = active_folds[mechanism.stability_parameter].to_numpy(dtype=float)
    interior_gate = bool(
        active[mechanism.stability_parameter]
        .between(mechanism.stability_grid[0], mechanism.stability_grid[-1], inclusive="neither")
        .all()
    )
    log_iqr = float(np.quantile(np.log2(values), 0.75) - np.quantile(np.log2(values), 0.25))
    stability_gate = log_iqr <= 2.0
    active_optima = optima.loc[optima["transition"].eq(mechanism.active_transition)]
    optimum_gate = bool((active_optima["distance_to_observed_best"] <= 0.15).all())
    integration_gate = bool((integration["integration_error"] < 2e-3).all())
    passed = (
        global_gate
        and fold_gate
        and shape_gate
        and interior_gate
        and stability_gate
        and optimum_gate
        and integration_gate
    )
    status = "promoted_after_starcoder" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"active_beats_ablations={global_gate}; fold_wins={wins}; within_5pct_shape={shape_gate}; "
        f"interior_clock={interior_gate}; log2_clock_iqr={log_iqr:.4g}; clock_stable={stability_gate}; "
        f"raw_optimum_distance_ok={optimum_gate}; integration_stable={integration_gate}."
    )
    update_status(mechanism, status, evidence, output_dir)
    report = [
        f"# Round 45: {mechanism.family}",
        "",
        "The equations, grids, ablations, and gates were frozen before evaluation. No historical-development, adversarial-development, or sealed-confirmation outcome was read during selection.",
        "",
        "## Decision",
        "",
        f"**{status}.** {evidence}",
        "",
        "## Global OOF comparison",
        "",
        summary.to_markdown(index=False),
        "",
        "## Foldwise comparison",
        "",
        folds.to_markdown(index=False),
        "",
        "## Raw optima",
        "",
        optima.to_markdown(index=False),
        "",
        "## Numerical audit",
        "",
        integration.to_markdown(index=False),
    ]
    (output_dir / f"{mechanism.candidate_id.lower()}_report.md").write_text("\n".join(report) + "\n")
    print(f"{mechanism.candidate_id}: {evidence}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosine = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine),
        starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine)),
    ]
    for mechanism in mechanisms():
        evaluate(mechanism, panels, args.output_dir)


if __name__ == "__main__":
    main()
