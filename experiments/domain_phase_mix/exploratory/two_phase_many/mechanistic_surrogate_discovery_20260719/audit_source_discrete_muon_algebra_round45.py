# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "tabulate>=0.9"]
# ///
"""Target-free materiality audit for source-discrete Muon momentum."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    source_discrete_muon_models as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    starcoder_optimizer_schedule as schedules,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round45_source_discrete_muon_algebra"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ANGLE_GRID = (30.0, 60.0, 90.0)
RARE_CURVATURE_GRID = (0.5, 1.0, 2.0)
ANISOTROPY_GRID = (0.25, 1.0, 4.0)
MEDIAN_GATE = 1e-3
P95_GATE = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def update_status(status: str, evidence: str, output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    registry.loc[registry["id"].eq("SDMMF"), ["status", "status_evidence"]] = [status, evidence]
    registry.to_csv(REGISTRY, index=False)
    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_45_source_discrete_muon_algebra",
        "candidate_id": "SDMMF",
        "candidate_family": "Source-discrete momentum Muon flow",
        "hyperparameters": "Source beta=0.95, Nesterov, NS5, epsilon=1e-5, peak LR=0.02, exact schedules; target-free 3x3x3 task geometry",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round-45 portfolio preregistration.",
        "novelty_class": "Matrix momentum inside finite NS before exact projection",
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
    rng = np.random.default_rng(20260719)
    p0 = rng.uniform(size=64)
    p1 = rng.uniform(size=64)
    weights = np.stack(
        [np.column_stack([1.0 - p0, p0]), np.column_stack([1.0 - p1, p1])],
        axis=1,
    )
    rows = []
    differences = []
    for schedule in (schedules.COSINE_50_50, schedules.WSD_80_20):
        for angle in ANGLE_GRID:
            for rare_curvature in RARE_CURVATURE_GRID:
                for anisotropy in ANISOTROPY_GRID:
                    active = candidate.SourceDiscreteMuonConfig(
                        angle, rare_curvature, anisotropy, 0.5, candidate.SOURCE_MOMENTUM
                    )
                    ablation = candidate.SourceDiscreteMuonConfig(angle, rare_curvature, anisotropy, 0.5, 0.0)
                    active_state = candidate.terminal_state(weights, schedule, active)
                    ablation_state = candidate.terminal_state(weights, schedule, ablation)
                    distance = np.linalg.norm(active_state - ablation_state, axis=(1, 2))
                    differences.extend(distance.tolist())
                    rows.append(
                        {
                            "surface": schedule.name.value,
                            "angle_degrees": angle,
                            "rare_curvature": rare_curvature,
                            "input_anisotropy": anisotropy,
                            "median_terminal_state_difference": float(np.median(distance)),
                            "p95_terminal_state_difference": float(np.quantile(distance, 0.95)),
                            "max_terminal_state_difference": float(np.max(distance)),
                        }
                    )
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "momentum_materiality_by_geometry.csv", index=False)
    values = np.asarray(differences)
    median = float(np.median(values))
    p95 = float(np.quantile(values, 0.95))
    maximum = float(np.max(values))
    material = median >= MEDIAN_GATE and p95 >= P95_GATE
    status = "active_after_algebra" if material else "blocked_before_starcoder"
    evidence = (
        f"median_terminal_state_difference={median:.8g}; p95={p95:.8g}; max={maximum:.8g}; "
        f"median_gate={MEDIAN_GATE:g}; p95_gate={P95_GATE:g}; algebraically_material={material}."
    )
    update_status(status, evidence, args.output_dir)
    report = [
        "# Round 45: source-discrete Muon momentum algebraic audit",
        "",
        "All optimizer constants, schedules, geometry grid, and materiality thresholds were frozen before this target-free audit. No target values were read.",
        "",
        "## Decision",
        "",
        f"**{status}.** {evidence}",
        "",
        "## Geometry strata",
        "",
        table.to_markdown(index=False),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(evidence)


if __name__ == "__main__":
    main()
