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
"""Falsify the source-exact finite Newton-Schulz Muon map on StarCoder."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many import export_mixture_fit_observatory as observatory
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    audit_starcoder_clipped_task_flow_round35 as clock,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    audit_starcoder_muon_anisotropic_polar_round39 as prior_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    muon_anisotropic_polar_models as candidate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round41_finite_newton_schulz_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
UPDATE_RULES = ("euclidean", "normalized", "polar", "newton_schulz")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def map_separation(
    panel,
    config: candidate.MuonAnisotropicPolarConfig,
) -> float:
    finite = candidate.MuonAnisotropicPolarConfig(
        config.task_angle_degrees,
        config.rare_curvature,
        config.input_anisotropy,
        config.relaxation,
        config.evaluation_rare_weight,
        "newton_schulz",
    )
    polar = candidate.MuonAnisotropicPolarConfig(
        config.task_angle_degrees,
        config.rare_curvature,
        config.input_anisotropy,
        config.relaxation,
        config.evaluation_rare_weight,
        "polar",
    )
    left = candidate.terminal_state(panel.weights, clock.optimizer_phase0_fraction(panel), finite)
    right = candidate.terminal_state(panel.weights, clock.optimizer_phase0_fraction(panel), polar)
    return float(np.mean(np.linalg.norm(left - right, axis=(1, 2))))


def update_status(status: str, evidence: str, output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    mask = registry["id"].eq("FNSMF")
    registry.loc[mask, "status"] = status
    registry.loc[mask, "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_41_finite_newton_schulz_decision",
        "candidate_id": "FNSMF",
        "candidate_family": "Finite Newton-Schulz Muon flow",
        "hyperparameters": "Frozen round-41 grid; source-fixed quintic coefficients, five iterations, and epsilon; exact polar, normalized, and Euclidean ablations",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round-41 portfolio preregistration.",
        "novelty_class": "Finite polynomial singular-value dynamics in the declared Muon update",
        "evaluation_status": status,
        "evidence_path": str((output_dir / "report.md").relative_to(OUTPUT_ROOT)),
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
    prior_audit.UPDATE_RULES = UPDATE_RULES
    all_configs = prior_audit.configs()
    cosine_data = observatory.load_cosine_starcoder()
    panels = [
        starcoder.panel_from_dataset(cosine_data),
        starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine_data)),
    ]

    summaries = []
    folds = []
    optima = []
    numerical = []
    for panel in panels:
        features = prior_audit.base_features(panel, all_configs)
        summary, selected = prior_audit.best_by_rule(panel, all_configs, features)
        summaries.append(summary)
        folds.append(prior_audit.fold_rule_winners(panel, all_configs, features))
        for rule, (config, l2, _prediction, fit_feature) in selected.items():
            optima.append(prior_audit.raw_optimum(panel, config, l2, fit_feature))
            numerical.append(
                {
                    "surface": panel.name,
                    "update_rule": rule,
                    "integration_error_64_vs_192": candidate.integration_error(
                        panel.weights[:: max(1, panel.n // 24)],
                        clock.optimizer_phase0_fraction(panel),
                        config,
                    ),
                    "finite_vs_polar_trajectory": map_separation(panel, config) if rule == "newton_schulz" else np.nan,
                }
            )

    summary_table = pd.concat(summaries, ignore_index=True)
    fold_table = pd.concat(folds, ignore_index=True)
    optimum_table = pd.DataFrame(optima)
    numerical_table = pd.DataFrame(numerical)
    summary_table.to_csv(args.output_dir / "global_oof_by_rule.csv", index=False)
    fold_table.to_csv(args.output_dir / "foldwise_rule_comparison.csv", index=False)
    optimum_table.to_csv(args.output_dir / "raw_optima.csv", index=False)
    numerical_table.to_csv(args.output_dir / "numerical_audit.csv", index=False)

    finite = summary_table.loc[summary_table["update_rule"].eq("newton_schulz")].set_index("surface")
    ablation = summary_table.loc[~summary_table["update_rule"].eq("newton_schulz")].groupby("surface")["oof_rmse"].min()
    global_gate = bool((finite["oof_rmse"] < ablation).all())
    fold_pivot = fold_table.pivot_table(index=["surface", "outer_fold"], columns="update_rule", values="outer_rmse")
    fold_wins = {
        surface: int((frame["newton_schulz"] < frame[["euclidean", "normalized", "polar"]].min(axis=1)).sum())
        for surface, frame in fold_pivot.groupby(level=0)
    }
    fold_gate = all(value >= 3 for value in fold_wins.values())
    shape_gate = all(
        float(finite.loc[panel.name, "oof_rmse"]) <= 1.05 * prior_audit.SHAPE_REFERENCE[panel.name] for panel in panels
    )
    finite_optima = optimum_table.loc[optimum_table["update_rule"].eq("newton_schulz")]
    optimum_gate = bool((finite_optima["distance_to_observed_best"] <= 0.15).all())
    finite_numerical = numerical_table.loc[numerical_table["update_rule"].eq("newton_schulz")]
    active_gate = bool((finite_numerical["finite_vs_polar_trajectory"] > 1e-3).all())
    integration_gate = bool((finite_numerical["integration_error_64_vs_192"] < 2e-3).all())
    passed = global_gate and fold_gate and shape_gate and optimum_gate and active_gate and integration_gate
    status = "promoted_after_starcoder" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"finite_beats_all_ablations={global_gate}; fold_wins={fold_wins}; within_5pct_shape={shape_gate}; "
        f"raw_optimum_distance_ok={optimum_gate}; finite_map_active={active_gate}; integration_stable={integration_gate}."
    )
    update_status(status, evidence, args.output_dir)

    report = [
        "# Round 41: finite Newton-Schulz Muon flow",
        "",
        "The exact finite map, grids, and gates were frozen before this evaluation. No historical, exposed-adversarial, or sealed-confirmation outcome was read.",
        "",
        "## Decision",
        "",
        f"**{status}.** {evidence}",
        "",
        "## Global OOF comparison",
        "",
        summary_table.to_markdown(index=False),
        "",
        "## Foldwise comparison",
        "",
        fold_table.to_markdown(index=False),
        "",
        "## Raw optima",
        "",
        optimum_table.to_markdown(index=False),
        "",
        "## Numerical audit",
        "",
        numerical_table.to_markdown(index=False),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(evidence)


if __name__ == "__main__":
    main()
