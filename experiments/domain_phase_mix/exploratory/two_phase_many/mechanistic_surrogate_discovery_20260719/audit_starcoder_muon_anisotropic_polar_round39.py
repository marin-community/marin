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
"""Falsify anisotropic Muon polar flow on both StarCoder surfaces."""

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
    audit_starcoder_clipped_task_flow_round35 as clock,
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
    hessian_equilibrium_models as heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    muon_anisotropic_polar_models as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round39_muon_anisotropic_polar_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ANGLE_GRID = (30.0, 60.0, 90.0)
RARE_CURVATURE_GRID = (0.5, 1.0, 2.0)
ANISOTROPY_GRID = (0.25, 0.5, 1.0, 2.0, 4.0)
RELAXATION_GRID = (1.0, 2.0, 4.0, 8.0)
EVALUATION_GRID = (0.2, 0.5, 0.8)
UPDATE_RULES = ("euclidean", "normalized", "polar")
L2_GRID = (0.0, 0.1, 1.0)
SHAPE_REFERENCE = shape_audit.SHAPE_REFERENCE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[candidate.MuonAnisotropicPolarConfig]:
    return [
        candidate.MuonAnisotropicPolarConfig(angle, rare, anisotropy, relaxation, evaluation, rule)
        for angle in ANGLE_GRID
        for rare in RARE_CURVATURE_GRID
        for anisotropy in ANISOTROPY_GRID
        for relaxation in RELAXATION_GRID
        for evaluation in EVALUATION_GRID
        for rule in UPDATE_RULES
    ]


def base_features(
    panel: paired.PairedPanel,
    all_configs: list[candidate.MuonAnisotropicPolarConfig],
    weights: np.ndarray | None = None,
    *,
    steps_per_unit: int = candidate.INTEGRATION_STEPS_PER_UNIT,
) -> np.ndarray:
    policies = panel.weights if weights is None else weights
    phase0 = clock.optimizer_phase0_fraction(panel)
    return np.asarray(
        [candidate.response_feature(policies, phase0, config, steps_per_unit=steps_per_unit) for config in all_configs],
        dtype=float,
    )


def expanded_variants(
    all_configs: list[candidate.MuonAnisotropicPolarConfig],
    features: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    config_index = np.repeat(np.arange(len(all_configs)), len(L2_GRID))
    l2 = np.tile(np.asarray(L2_GRID, dtype=float), len(all_configs))
    return features[config_index], config_index, l2


def score_all(
    panel: paired.PairedPanel,
    all_configs: list[candidate.MuonAnisotropicPolarConfig],
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


def best_by_rule(
    panel: paired.PairedPanel,
    all_configs: list[candidate.MuonAnisotropicPolarConfig],
    features: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, tuple[candidate.MuonAnisotropicPolarConfig, float, np.ndarray, np.ndarray]]]:
    rmse, predictions, config_index, l2 = score_all(panel, all_configs, features)
    rows = []
    selected: dict[str, tuple[candidate.MuonAnisotropicPolarConfig, float, np.ndarray, np.ndarray]] = {}
    for rule in UPDATE_RULES:
        eligible = np.asarray([all_configs[int(index)].update_rule == rule for index in config_index], dtype=bool)
        local = np.flatnonzero(eligible)
        best = int(local[np.argmin(rmse[local])])
        config_position = int(config_index[best])
        config = all_configs[config_position]
        selected[rule] = (config, float(l2[best]), predictions[best], features[config_position])
        trajectory_separation = candidate.trajectory_rule_separation(
            panel.weights, clock.optimizer_phase0_fraction(panel), config
        )
        rows.append(
            {
                "surface": panel.name,
                "update_rule": rule,
                "oof_rmse": float(rmse[best]),
                "l2": float(l2[best]),
                "trajectory_polar_vs_normalized": trajectory_separation,
                **asdict(config),
                **{
                    f"oof_{key}": value
                    for key, value in metrics.scalar_metrics(panel.two_phase_target, predictions[best]).items()
                },
            }
        )
    return pd.DataFrame(rows), selected


def fold_rule_winners(
    panel: paired.PairedPanel,
    all_configs: list[candidate.MuonAnisotropicPolarConfig],
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
        for rule in UPDATE_RULES:
            eligible = np.asarray([all_configs[int(index)].update_rule == rule for index in config_index], dtype=bool)
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
                    "update_rule": rule,
                    "inner_rmse": float(inner_rmse[best]),
                    "outer_rmse": float(np.sqrt(np.mean((prediction - panel.two_phase_target[outer_test]) ** 2))),
                    "l2": float(l2[best]),
                    **asdict(all_configs[int(config_index[best])]),
                }
            )
    return pd.DataFrame(rows)


def raw_optimum(
    panel: paired.PairedPanel,
    config: candidate.MuonAnisotropicPolarConfig,
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
    feature = base_features(panel, [config], weights, steps_per_unit=128)[0]
    head = heads.fit_quadratic_head(fit_feature, panel.two_phase_target, np.arange(panel.n), l2)
    prediction = head.predict(feature)
    best = int(np.argmin(prediction))
    observed = int(np.argmin(panel.two_phase_target))
    return {
        "surface": panel.name,
        "update_rule": config.update_rule,
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
    mask = registry["id"].eq("MAPTF")
    registry.loc[mask, "status"] = status
    registry.loc[mask, "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_39_starcoder_decision",
        "candidate_id": "MAPTF",
        "candidate_family": "Muon anisotropic polar task flow",
        "hyperparameters": "Frozen round-39 grid with isotropic, Euclidean, and normalized ablations",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round-39 preregistration.",
        "novelty_class": "Domain-specific anisotropic input covariance under Muon polar flow",
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
    integration_rows = []
    for panel in panels:
        features = base_features(panel, all_configs)
        summary, selected = best_by_rule(panel, all_configs, features)
        summary_frames.append(summary)
        fold_frames.append(fold_rule_winners(panel, all_configs, features))
        for rule, (config, l2, _prediction, fit_feature) in selected.items():
            optimum_rows.append(raw_optimum(panel, config, l2, fit_feature))
            integration_rows.append(
                {
                    "surface": panel.name,
                    "update_rule": rule,
                    "integration_error_64_vs_192": candidate.integration_error(
                        panel.weights[:: max(1, panel.n // 24)],
                        clock.optimizer_phase0_fraction(panel),
                        config,
                    ),
                }
            )

    summary_table = pd.concat(summary_frames, ignore_index=True)
    fold_table = pd.concat(fold_frames, ignore_index=True)
    optimum_table = pd.DataFrame(optimum_rows)
    integration_table = pd.DataFrame(integration_rows)
    summary_table.to_csv(args.output_dir / "global_oof_by_rule.csv", index=False)
    fold_table.to_csv(args.output_dir / "foldwise_rule_comparison.csv", index=False)
    optimum_table.to_csv(args.output_dir / "raw_optima.csv", index=False)
    integration_table.to_csv(args.output_dir / "integration_audit.csv", index=False)

    polar_global = summary_table.loc[summary_table["update_rule"].eq("polar")].set_index("surface")
    ablation_best = summary_table.loc[~summary_table["update_rule"].eq("polar")].groupby("surface")["oof_rmse"].min()
    polar_beats_ablation = bool((polar_global["oof_rmse"] < ablation_best).all())
    anisotropy_active = bool((np.abs(np.log(polar_global["input_anisotropy"])) > 1e-9).all())
    trajectory_active = bool((polar_global["trajectory_polar_vs_normalized"] > 1e-3).all())
    fold_pivot = fold_table.pivot_table(index=["surface", "outer_fold"], columns="update_rule", values="outer_rmse")
    polar_fold_wins = {
        surface: int((frame["polar"] < frame[["euclidean", "normalized"]].min(axis=1)).sum())
        for surface, frame in fold_pivot.groupby(level=0)
    }
    fold_gate = all(value >= 3 for value in polar_fold_wins.values())
    shape_gate = all(
        float(polar_global.loc[panel.name, "oof_rmse"]) <= 1.05 * SHAPE_REFERENCE[panel.name] for panel in panels
    )
    polar_optima = optimum_table.loc[optimum_table["update_rule"].eq("polar")]
    optimum_gate = bool((polar_optima["distance_to_observed_best"] <= 0.15).all())
    integration_gate = bool((integration_table["integration_error_64_vs_192"] < 2e-3).all())
    passed = (
        polar_beats_ablation
        and anisotropy_active
        and trajectory_active
        and fold_gate
        and shape_gate
        and optimum_gate
        and integration_gate
    )
    status = "promoted_after_starcoder" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"polar_beats_ablation={polar_beats_ablation}; anisotropy_active={anisotropy_active}; "
        f"trajectory_active={trajectory_active}; fold_wins={polar_fold_wins}; within_5pct_shape={shape_gate}; "
        f"raw_optimum_distance_ok={optimum_gate}; integration_stable={integration_gate}."
    )
    update_status(status, evidence, args.output_dir)

    report = [
        "# Round 39: anisotropic Muon polar task flow",
        "",
        "The covariance mechanism and all grids were frozen before this StarCoder evaluation. No historical, exposed-adversarial, or sealed-confirmation outcome was read.",
        "",
        "## Decision",
        "",
        f"**{status}.** {evidence}",
        "",
        "## Global OOF comparison",
        "",
        summary_table.to_markdown(index=False),
        "",
        "## Foldwise rule comparison",
        "",
        fold_table.to_markdown(index=False),
        "",
        "## Raw optima",
        "",
        optimum_table.to_markdown(index=False),
        "",
        "## Numerical audit",
        "",
        integration_table.to_markdown(index=False),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")


if __name__ == "__main__":
    main()
