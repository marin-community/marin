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
"""Falsify the frozen replenishable-plasticity model on StarCoder."""

from __future__ import annotations

import argparse
import csv
import json
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
    audit_starcoder_potential_round2 as starcoder,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    audit_starcoder_shared_private_round25 as audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    paired_dynamics_models as paired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    plasticity_reserve_models as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (  # noqa: E402
    screen_paired_dynamics_round1 as metrics,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round34_plasticity_reserve_starcoder"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
SHAPE_REFERENCE = audit.SHAPE_REFERENCE
FOUNDATION_RATE_GRID = (0.5, 2.0, 8.0)
RESERVE_RECOVERY_GRID = (0.5, 2.0, 8.0)
SPECIALIST_RATE_GRID = (0.5, 2.0, 8.0)
RARE_FOUNDATION_GRID = (0.0, 0.3, 1.0)
DEPLETION_GRID = (0.0, 0.5, 2.0, 8.0)
L2_GRID = (0.1, 1.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs() -> list[candidate.PlasticityReserveConfig]:
    return [
        candidate.PlasticityReserveConfig(foundation, recovery, specialist, rare_efficiency, depletion, l2)
        for foundation in FOUNDATION_RATE_GRID
        for recovery in RESERVE_RECOVERY_GRID
        for specialist in SPECIALIST_RATE_GRID
        for rare_efficiency in RARE_FOUNDATION_GRID
        for depletion in DEPLETION_GRID
        for l2 in L2_GRID
    ]


def designs(
    panel: paired.PairedPanel,
    all_configs: list[candidate.PlasticityReserveConfig],
    weights: np.ndarray | None = None,
) -> list[np.ndarray]:
    policies = panel.weights if weights is None else weights
    return [candidate.design(policies, panel.alpha0, panel.c0, panel.c1, config)[0] for config in all_configs]


def raw_optimum(
    panel: paired.PairedPanel,
    config: candidate.PlasticityReserveConfig,
    fit_design: np.ndarray,
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
    candidate_design = designs(panel, [config], weights)[0]
    head = audit.fit_head(fit_design, panel.two_phase_target, np.arange(panel.n), config.l2)
    prediction = head.predict(candidate_design)
    best = int(np.argmin(prediction))
    observed = int(np.argmin(panel.two_phase_target))
    return {
        "surface": panel.name,
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
        "late_specialization": bool(p1.ravel()[best] > p0.ravel()[best]),
    }


def update_registry(status: str, evidence: str) -> None:
    with REGISTRY.open(newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fields = reader.fieldnames
    if fields is None:
        raise RuntimeError("Registry has no header")
    for row in rows:
        if row["id"] == "RSPR":
            row["status"] = status
            row["status_evidence"] = evidence
    temporary = REGISTRY.with_suffix(".tmp")
    with temporary.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(REGISTRY)


def record_ledger(status: str, evidence: str) -> None:
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_34_starcoder_decision",
        "candidate_id": "RSPR",
        "candidate_family": "Replenishable specialist-plasticity reserve",
        "hyperparameters": "Frozen round-34 grid; nested StarCoder selection",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round-34 preregistration.",
        "novelty_class": "Consumable specialist-plasticity state",
        "evaluation_status": status,
        "evidence_path": "round34_plasticity_reserve_starcoder/report.md",
        "notes": evidence,
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[column] for column in identity) not in existing:
        pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True).to_csv(LEDGER, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cosine = observatory.load_cosine_starcoder()
    panels = [starcoder.panel_from_dataset(cosine), starcoder.panel_from_dataset(starcoder.load_refined_wsd80(cosine))]
    all_configs = configs()
    active_indices = np.asarray(
        [index for index, config in enumerate(all_configs) if config.reserve_depletion_rate > 0.0]
    )
    ablation_indices = np.asarray(
        [index for index, config in enumerate(all_configs) if config.reserve_depletion_rate == 0.0]
    )
    stage1_rows: list[dict[str, Any]] = []
    selected_active: dict[str, candidate.PlasticityReserveConfig] = {}
    design_cache: dict[str, list[np.ndarray]] = {}
    stage1_passed = True
    for panel in panels:
        panel_designs = designs(panel, all_configs)
        design_cache[panel.name] = panel_designs
        rmse, predictions = audit.score_configs(panel, all_configs, panel_designs, starcoder.surface_folds(panel))
        active_index = int(active_indices[np.argmin(rmse[active_indices])])
        ablation_index = int(ablation_indices[np.argmin(rmse[ablation_indices])])
        selected_active[panel.name] = all_configs[active_index]
        active_metrics = metrics.scalar_metrics(panel.two_phase_target, predictions[active_index])
        ablation_metrics = metrics.scalar_metrics(panel.two_phase_target, predictions[ablation_index])
        stage1_rows.append(
            {
                "surface": panel.name,
                "active_config": json.dumps(asdict(all_configs[active_index]), sort_keys=True),
                "ablation_config": json.dumps(asdict(all_configs[ablation_index]), sort_keys=True),
                **{f"active_{key}": value for key, value in active_metrics.items()},
                **{f"ablation_{key}": value for key, value in ablation_metrics.items()},
                "rmse_delta_active_minus_ablation": float(rmse[active_index] - rmse[ablation_index]),
            }
        )
        stage1_passed &= bool(rmse[active_index] < rmse[ablation_index])

    stage1 = pd.DataFrame(stage1_rows)
    stage1.to_csv(args.output_dir / "stage1_global_oof_comparison.csv", index=False)
    algebraic = max(candidate.tied_policy_error(config) for config in all_configs)
    (args.output_dir / "algebraic_audit.json").write_text(
        json.dumps({"max_tied_semigroup_error": algebraic}, indent=2) + "\n"
    )
    report = [
        "# Round 34: replenishable specialist-plasticity reserve",
        "",
        "The equation, grid, and exact depletion-free ablation were frozen before fitting. No Delphi historical or adversarial outcomes were read.",
        "",
        "## Algebraic audit",
        "",
        f"Maximum tied semigroup error: `{algebraic:.3e}`.",
        "",
        "## Stage 1: depletion versus nondepleting reserve",
        "",
        stage1.to_markdown(index=False),
        "",
    ]
    if not stage1_passed:
        evidence = (
            "The nondepleting-reserve ablation matched or beat active depletion on at least one StarCoder schedule."
        )
        status = "blocked_before_nested_starcoder"
        update_registry(status, evidence)
        record_ledger(status, evidence)
        report.extend(
            ["## Decision", "", f"**Blocked.** {evidence}", "", "No Delphi or adversarial outcomes were evaluated."]
        )
        (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
        print(stage1.to_string(index=False))
        print(evidence)
        return

    nested_rows: list[dict[str, Any]] = []
    nested_tables: list[pd.DataFrame] = []
    optimum_rows = []
    for panel in panels:
        nested_prediction, nested_table = audit.nested_prediction(panel, all_configs, design_cache[panel.name])
        nested_rows.append({"surface": panel.name, **metrics.scalar_metrics(panel.two_phase_target, nested_prediction)})
        nested_tables.append(nested_table)
        config = selected_active[panel.name]
        config_index = all_configs.index(config)
        optimum_rows.append(raw_optimum(panel, config, design_cache[panel.name][config_index]))

    nested_metrics = pd.DataFrame(nested_rows)
    nested_folds = pd.concat(nested_tables, ignore_index=True)
    optima = pd.DataFrame(optimum_rows)
    nested_metrics.to_csv(args.output_dir / "nested_oof_metrics.csv", index=False)
    nested_folds.to_csv(args.output_dir / "nested_fold_selections.csv", index=False)
    optima.to_csv(args.output_dir / "raw_optima.csv", index=False)
    active_fraction = nested_folds.groupby("surface")["reserve_depletion_rate"].apply(
        lambda values: float((values > 0).mean())
    )
    ratio = nested_folds.loc[nested_folds["reserve_depletion_rate"] > 0].copy()
    ratio["recovery_to_depletion"] = ratio["reserve_recovery_rate"] / ratio["reserve_depletion_rate"]
    ratio_span = ratio.groupby("surface")["recovery_to_depletion"].agg(["min", "max"])
    stability = pd.concat([active_fraction.rename("active_fraction"), ratio_span], axis=1).reset_index()
    stability["ratio_span"] = stability["max"] / stability["min"]
    stability.to_csv(args.output_dir / "mechanism_stability.csv", index=False)
    shape_ok = all(
        float(nested_metrics.loc[nested_metrics["surface"].eq(surface), "rmse"].iloc[0]) <= 1.05 * reference
        for surface, reference in SHAPE_REFERENCE.items()
    )
    optimum_ok = bool((optima["distance_to_observed_best"] <= 0.15).all())
    fold_active = bool((stability["active_fraction"] >= 0.6).all())
    ratio_ok = bool((stability["ratio_span"] <= 4.0).all())
    global_interior = all(selected_active[panel.name].reserve_depletion_rate < max(DEPLETION_GRID) for panel in panels)
    passed = algebraic < 1e-8 and shape_ok and optimum_ok and fold_active and ratio_ok and global_interior
    status = "promoted_to_multi_swarm" if passed else "blocked_before_multi_swarm"
    evidence = (
        f"depletion_beats_ablation={stage1_passed}; within_5pct_shape={shape_ok}; optimum_distance_ok={optimum_ok}; "
        f"fold_depletion_active={fold_active}; recovery_depletion_ratio_stable={ratio_ok}; depletion_interior={global_interior}."
    )
    update_registry(status, evidence)
    record_ledger(status, evidence)
    report.extend(
        [
            "## Nested StarCoder audit",
            "",
            nested_metrics.to_markdown(index=False),
            "",
            "## Mechanism stability",
            "",
            stability.to_markdown(index=False),
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
