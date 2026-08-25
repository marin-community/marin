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
"""Falsify exact batch-composition averaging at finite-NS geometries."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

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
    batch_composition_muon_models as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    hessian_equilibrium_models as heads,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    screen_paired_dynamics_round1 as metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round42_batch_composition_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
L2_GRID = (0.0, 0.1, 1.0)
RULES = ("mean", "hypergeometric")
SHAPE_REFERENCE = shape_audit.SHAPE_REFERENCE
OPTIMUM_GRID_SIZE = 31
GEOMETRY = {
    "starcoder_cosine_50_50": (30.0, 2.0, 4.0, 8.0, 0.2),
    "starcoder_wsd_80_20": (60.0, 0.5, 0.5, 4.0, 0.5),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs(panel_name: str) -> list[candidate.BatchCompositionMuonConfig]:
    geometry = GEOMETRY[panel_name]
    return [candidate.BatchCompositionMuonConfig(*geometry, rule) for rule in RULES]


def features(panel, all_configs, weights: np.ndarray | None = None, *, steps_per_unit: int = 192) -> np.ndarray:
    policies = panel.weights if weights is None else weights
    phase0 = clock.optimizer_phase0_fraction(panel)
    return np.asarray(
        [candidate.response_feature(policies, phase0, config, steps_per_unit=steps_per_unit) for config in all_configs]
    )


def score(panel, feature: np.ndarray, folds) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    config_index = np.repeat(np.arange(len(RULES)), len(L2_GRID))
    l2 = np.tile(np.asarray(L2_GRID), len(RULES))
    design = feature[config_index]
    rmse, predictions = scalar_audit.score_configs(design, panel.two_phase_target, folds, l2)
    return rmse, predictions, config_index, l2


def global_comparison(
    panel, all_configs, feature: np.ndarray
) -> tuple[pd.DataFrame, dict[str, tuple[object, float, np.ndarray]]]:
    rmse, predictions, config_index, l2 = score(panel, feature, starcoder.surface_folds(panel))
    rows = []
    selected = {}
    for rule_index, rule in enumerate(RULES):
        positions = np.flatnonzero(config_index == rule_index)
        best = int(positions[np.argmin(rmse[positions])])
        selected[rule] = (all_configs[rule_index], float(l2[best]), feature[rule_index])
        rows.append(
            {
                "surface": panel.name,
                "composition_rule": rule,
                "oof_rmse": float(rmse[best]),
                "l2": float(l2[best]),
                **asdict(all_configs[rule_index]),
                **{
                    f"oof_{key}": value
                    for key, value in metrics.scalar_metrics(panel.two_phase_target, predictions[best]).items()
                },
            }
        )
    return pd.DataFrame(rows), selected


def fold_comparison(panel, feature: np.ndarray) -> pd.DataFrame:
    rows = []
    for outer_fold, (outer_train, outer_test) in enumerate(starcoder.surface_folds(panel)):
        inner = scalar_audit.stratified_folds(panel, outer_train, 4, 20260719 + 100 * outer_fold)
        local = [
            (np.flatnonzero(np.isin(outer_train, train)), np.flatnonzero(np.isin(outer_train, test)))
            for train, test in inner
        ]
        inner_rmse, _predictions, config_index, l2 = score(
            type("LocalPanel", (), {"two_phase_target": panel.two_phase_target[outer_train]})(),
            feature[:, outer_train],
            local,
        )
        design = feature[config_index]
        for rule_index, rule in enumerate(RULES):
            positions = np.flatnonzero(config_index == rule_index)
            best = int(positions[np.argmin(inner_rmse[positions])])
            prediction = scalar_audit.fit_predict_all(
                design[[best]], panel.two_phase_target, outer_train, outer_test, l2[[best]]
            )[0]
            rows.append(
                {
                    "surface": panel.name,
                    "outer_fold": outer_fold,
                    "composition_rule": rule,
                    "inner_rmse": float(inner_rmse[best]),
                    "outer_rmse": float(np.sqrt(np.mean((prediction - panel.two_phase_target[outer_test]) ** 2))),
                    "l2": float(l2[best]),
                }
            )
    return pd.DataFrame(rows)


def raw_optimum(panel, config, l2: float, fit_feature: np.ndarray) -> dict[str, float | str]:
    grid = np.linspace(0.0, 1.0, OPTIMUM_GRID_SIZE)
    p0, p1 = np.meshgrid(grid, grid, indexing="ij")
    weights = np.stack(
        [
            np.column_stack([1.0 - p0.ravel(), p0.ravel()]),
            np.column_stack([1.0 - p1.ravel(), p1.ravel()]),
        ],
        axis=1,
    )
    feature = features(panel, [config], weights, steps_per_unit=192)[0]
    head = heads.fit_quadratic_head(fit_feature, panel.two_phase_target, np.arange(panel.n), l2)
    prediction = head.predict(feature)
    best = int(np.argmin(prediction))
    observed = int(np.argmin(panel.two_phase_target))
    return {
        "surface": panel.name,
        "composition_rule": config.composition_rule,
        "phase0_rare": float(p0.ravel()[best]),
        "phase1_rare": float(p1.ravel()[best]),
        "predicted_bpb": float(prediction[best]),
        "distance_to_observed_best": float(
            np.hypot(p0.ravel()[best] - panel.weights[observed, 0, 1], p1.ravel()[best] - panel.weights[observed, 1, 1])
        ),
    }


def update_status(status: str, evidence: str, output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    registry.loc[registry["id"].eq("BCNSF"), ["status", "status_evidence"]] = [status, evidence]
    registry.to_csv(REGISTRY, index=False)
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_42_batch_composition_starcoder_decision",
        "candidate_id": "BCNSF",
        "candidate_family": "Batch-composition Newton-Schulz flow",
        "hyperparameters": "Finite-NS geometries independently selected in round 41; exact B=128/N=2048 law; l2 {0,0.1,1}; 31x31 raw-optimum grid",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round-41 portfolio preregistration.",
        "novelty_class": "Jensen drift through exact stochastic batch composition",
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
    cosine = observatory.load_cosine_starcoder()
    panels = [starcoder.panel_from_dataset(cosine), starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine))]
    summaries = []
    folds = []
    optima = []
    numerics = []
    for panel in panels:
        all_configs = configs(panel.name)
        feature = features(panel, all_configs)
        summary, selected = global_comparison(panel, all_configs, feature)
        summaries.append(summary)
        folds.append(fold_comparison(panel, feature))
        for rule, (config, l2, fit_feature) in selected.items():
            optima.append(raw_optimum(panel, config, l2, fit_feature))
            coarse = candidate.response_feature(
                panel.weights[:: max(1, panel.n // 16)],
                clock.optimizer_phase0_fraction(panel),
                config,
                steps_per_unit=64,
            )
            fine = candidate.response_feature(
                panel.weights[:: max(1, panel.n // 16)],
                clock.optimizer_phase0_fraction(panel),
                config,
                steps_per_unit=192,
            )
            numerics.append(
                {
                    "surface": panel.name,
                    "composition_rule": rule,
                    "integration_error_64_vs_192": float(np.max(np.abs(coarse - fine))),
                    "trajectory_separation_from_mean": float(
                        np.mean(
                            np.linalg.norm(
                                candidate.terminal_state(
                                    panel.weights,
                                    clock.optimizer_phase0_fraction(panel),
                                    all_configs[1],
                                    steps_per_unit=192,
                                )
                                - candidate.terminal_state(
                                    panel.weights,
                                    clock.optimizer_phase0_fraction(panel),
                                    all_configs[0],
                                    steps_per_unit=192,
                                ),
                                axis=(1, 2),
                            )
                        )
                    )
                    if rule == "hypergeometric"
                    else np.nan,
                }
            )
    summary = pd.concat(summaries, ignore_index=True)
    fold_table = pd.concat(folds, ignore_index=True)
    optimum = pd.DataFrame(optima)
    numerical = pd.DataFrame(numerics)
    summary.to_csv(args.output_dir / "global_oof_by_composition.csv", index=False)
    fold_table.to_csv(args.output_dir / "foldwise_composition_comparison.csv", index=False)
    optimum.to_csv(args.output_dir / "raw_optima.csv", index=False)
    numerical.to_csv(args.output_dir / "numerical_audit.csv", index=False)

    stochastic = summary.loc[summary["composition_rule"].eq("hypergeometric")].set_index("surface")
    mean = summary.loc[summary["composition_rule"].eq("mean")].set_index("surface")
    global_gate = bool((stochastic["oof_rmse"] < mean["oof_rmse"]).all())
    fold_pivot = fold_table.pivot_table(index=["surface", "outer_fold"], columns="composition_rule", values="outer_rmse")
    wins = {
        surface: int((frame["hypergeometric"] < frame["mean"]).sum()) for surface, frame in fold_pivot.groupby(level=0)
    }
    fold_gate = all(value >= 3 for value in wins.values())
    shape_gate = all(
        float(stochastic.loc[panel.name, "oof_rmse"]) <= 1.05 * SHAPE_REFERENCE[panel.name] for panel in panels
    )
    stochastic_optima = optimum.loc[optimum["composition_rule"].eq("hypergeometric")]
    optimum_gate = bool((stochastic_optima["distance_to_observed_best"] <= 0.15).all())
    stochastic_numerics = numerical.loc[numerical["composition_rule"].eq("hypergeometric")]
    active_gate = bool((stochastic_numerics["trajectory_separation_from_mean"] > 1e-3).all())
    integration_gate = bool((stochastic_numerics["integration_error_64_vs_192"] < 2e-3).all())
    passed = global_gate and fold_gate and shape_gate and optimum_gate and active_gate and integration_gate
    status = "promoted_after_starcoder" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"stochastic_beats_mean={global_gate}; fold_wins={wins}; within_5pct_shape={shape_gate}; "
        f"raw_optimum_distance_ok={optimum_gate}; stochastic_transition_active={active_gate}; integration_stable={integration_gate}."
    )
    update_status(status, evidence, args.output_dir)
    report = [
        "# Round 42: batch-composition Newton-Schulz flow",
        "",
        "The loader law and gates were frozen before the algebraic audit. This StarCoder audit uses the finite-NS geometries independently selected in Round 41 and does not inspect any historical, adversarial-development, or sealed-confirmation outcome.",
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
        fold_table.to_markdown(index=False),
        "",
        "## Raw optima",
        "",
        optimum.to_markdown(index=False),
        "",
        "## Numerical audit",
        "",
        numerical.to_markdown(index=False),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(evidence)


if __name__ == "__main__":
    main()
