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
"""Falsify deep matrix-factorization spectral bias on StarCoder."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import export_mixture_fit_observatory as observatory
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
    deep_matrix_factorization_models as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    hessian_equilibrium_models as heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    screen_paired_dynamics_round1 as metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round47_matrix_factorization_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ANGLE_GRID = (30.0, 60.0, 90.0)
RELAXATION_GRID = (0.5, 2.0, 8.0, 32.0)
RARE_CURVATURE_GRID = (0.25, 1.0, 4.0)
EVALUATION_GRID = (0.25, 0.5, 0.75)
L2_GRID = (0.0, 0.1, 1.0)
TRANSITIONS = ("factorized", "direct")
SHAPE_REFERENCE = shape_audit.SHAPE_REFERENCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[candidate.MatrixFactorizationConfig]:
    return [
        candidate.MatrixFactorizationConfig(angle, relaxation, curvature, evaluation, transition)
        for angle in ANGLE_GRID
        for relaxation in RELAXATION_GRID
        for curvature in RARE_CURVATURE_GRID
        for evaluation in EVALUATION_GRID
        for transition in TRANSITIONS
    ]


def features(panel, all_configs, weights: np.ndarray | None = None, *, steps: int = 256) -> np.ndarray:
    policies = panel.weights if weights is None else weights
    alpha = clock.optimizer_phase0_fraction(panel)
    return np.asarray(
        [candidate.response_feature(policies, alpha, config, steps_per_unit=steps) for config in all_configs]
    )


def expanded(all_configs, feature: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    indices = np.repeat(np.arange(len(all_configs)), len(L2_GRID))
    l2 = np.tile(np.asarray(L2_GRID), len(all_configs))
    return feature[indices], indices, l2


def best_by_transition(panel, all_configs, feature: np.ndarray):
    design, indices, l2 = expanded(all_configs, feature)
    rmse, predictions = scalar_audit.score_configs(design, panel.two_phase_target, starcoder.surface_folds(panel), l2)
    rows = []
    selected = {}
    for transition in TRANSITIONS:
        positions = np.flatnonzero([all_configs[int(index)].transition == transition for index in indices])
        best = int(positions[np.argmin(rmse[positions])])
        config_index = int(indices[best])
        config = all_configs[config_index]
        selected[transition] = (config, float(l2[best]), feature[config_index])
        rows.append(
            {
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


def fold_comparison(panel, all_configs, feature: np.ndarray) -> pd.DataFrame:
    design, indices, l2 = expanded(all_configs, feature)
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
        for transition in TRANSITIONS:
            positions = np.flatnonzero([all_configs[int(index)].transition == transition for index in indices])
            best = int(positions[np.argmin(inner_rmse[positions])])
            prediction = scalar_audit.fit_predict_all(
                design[[best]], panel.two_phase_target, outer_train, outer_test, l2[[best]]
            )[0]
            rows.append(
                {
                    "surface": panel.name,
                    "outer_fold": outer_fold,
                    "transition": transition,
                    "outer_rmse": float(np.sqrt(np.mean((prediction - panel.two_phase_target[outer_test]) ** 2))),
                    "inner_rmse": float(inner_rmse[best]),
                    "l2": float(l2[best]),
                    **asdict(all_configs[int(indices[best])]),
                }
            )
    return pd.DataFrame(rows)


def raw_optimum(panel, config, l2: float, fit_feature: np.ndarray) -> dict[str, Any]:
    grid = np.linspace(0.0, 1.0, 151)
    p0, p1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - p0.ravel(), p0.ravel()]),
            np.column_stack([1.0 - p1.ravel(), p1.ravel()]),
        ],
        axis=1,
    )
    feature = candidate.response_feature(weights, clock.optimizer_phase0_fraction(panel), config, steps_per_unit=512)
    head = heads.fit_quadratic_head(fit_feature, panel.two_phase_target, np.arange(panel.n), l2)
    prediction = head.predict(feature)
    best = int(np.argmin(prediction))
    observed = int(np.argmin(panel.two_phase_target))
    return {
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


def update_status(status: str, evidence: str, output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    registry.loc[registry["id"].eq("DMFSB"), ["status", "status_evidence"]] = [status, evidence]
    registry.to_csv(REGISTRY, index=False)
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_47_matrix_factorization_decision",
        "candidate_id": "DMFSB",
        "candidate_family": "Deep matrix-factorization spectral-bias flow",
        "hyperparameters": "Frozen rank-2 factorization, task-geometry/clock/ridge grids, and direct-W ablation",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round-47 preregistration.",
        "novelty_class": "Bilinear matrix-factorization latent state",
        "evaluation_status": status,
        "evidence_path": str((output_dir / "report.md").relative_to(OUTPUT_ROOT)),
        "notes": evidence,
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[key] for key in identity) not in existing:
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
    summary_frames = []
    fold_frames = []
    optimum_rows = []
    integration_rows = []
    for panel in panels:
        feature = features(panel, all_configs)
        summary, selected = best_by_transition(panel, all_configs, feature)
        summary_frames.append(summary)
        fold_frames.append(fold_comparison(panel, all_configs, feature))
        for transition, (config, l2, fit_feature) in selected.items():
            optimum_rows.append(raw_optimum(panel, config, l2, fit_feature))
            integration_rows.append(
                {
                    "surface": panel.name,
                    "transition": transition,
                    "integration_error": candidate.integration_error(
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
    summary.to_csv(args.output_dir / "global_oof_by_transition.csv", index=False)
    folds.to_csv(args.output_dir / "foldwise_transition_comparison.csv", index=False)
    optima.to_csv(args.output_dir / "raw_optima.csv", index=False)
    integration.to_csv(args.output_dir / "integration_audit.csv", index=False)

    active = summary.loc[summary["transition"].eq("factorized")].set_index("surface")
    ablation = summary.loc[summary["transition"].eq("direct")].set_index("surface")
    global_gate = bool((active["oof_rmse"] < ablation["oof_rmse"]).all())
    pivot = folds.pivot_table(index=["surface", "outer_fold"], columns="transition", values="outer_rmse")
    wins = {surface: int((frame["factorized"] < frame["direct"]).sum()) for surface, frame in pivot.groupby(level=0)}
    fold_gate = all(value >= 3 for value in wins.values())
    shape_gate = all(float(active.loc[panel.name, "oof_rmse"]) <= 1.05 * SHAPE_REFERENCE[panel.name] for panel in panels)
    active_folds = folds.loc[folds["transition"].eq("factorized")]
    relaxation_interior = bool(
        active["relaxation"].between(RELAXATION_GRID[0], RELAXATION_GRID[-1], inclusive="neither").all()
    )
    relaxation_iqr = float(
        np.quantile(np.log2(active_folds["relaxation"]), 0.75) - np.quantile(np.log2(active_folds["relaxation"]), 0.25)
    )
    stability_gate = relaxation_iqr <= 2.0
    active_optima = optima.loc[optima["transition"].eq("factorized")]
    optimum_gate = bool((active_optima["distance_to_observed_best"] <= 0.15).all())
    integration_gate = bool((integration["integration_error"] < 2e-3).all())
    passed = (
        global_gate
        and fold_gate
        and shape_gate
        and relaxation_interior
        and stability_gate
        and optimum_gate
        and integration_gate
    )
    status = "promoted_after_starcoder" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"factorized_beats_direct={global_gate}; fold_wins={wins}; within_5pct_shape={shape_gate}; "
        f"relaxation_interior={relaxation_interior}; log2_relaxation_iqr={relaxation_iqr:.4g}; "
        f"stable={stability_gate}; raw_optimum_distance_ok={optimum_gate}; integration_stable={integration_gate}."
    )
    update_status(status, evidence, args.output_dir)
    report = [
        "# Round 47: deep matrix-factorization spectral-bias flow",
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
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(evidence)


if __name__ == "__main__":
    main()
