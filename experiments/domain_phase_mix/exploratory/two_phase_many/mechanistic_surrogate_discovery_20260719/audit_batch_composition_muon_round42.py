# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "scipy>=1.15", "tabulate>=0.9"]
# ///
"""Algebraically screen stochastic batch composition through finite Muon."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    batch_composition_muon_models as candidate,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    muon_polar_matrix_models as muon,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round42_batch_composition_algebra"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ANGLE_GRID = (30.0, 60.0, 90.0)
RARE_CURVATURE_GRID = (0.5, 1.0, 2.0)
ANISOTROPY_GRID = (0.25, 1.0, 4.0)
RARE_WEIGHT_GRID = tuple(np.linspace(0.0, 1.0, 21))
MEDIAN_CORRECTION_GATE = 1e-3
P95_CORRECTION_GATE = 1e-2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def configs(rule: str) -> list[candidate.BatchCompositionMuonConfig]:
    return [
        candidate.BatchCompositionMuonConfig(angle, rare, anisotropy, 4.0, 0.5, rule)
        for angle in ANGLE_GRID
        for rare in RARE_CURVATURE_GRID
        for anisotropy in ANISOTROPY_GRID
    ]


def update_status(status: str, evidence: str, output_dir: Path) -> None:
    registry = pd.read_csv(REGISTRY)
    mask = registry["id"].eq("BCNSF")
    registry.loc[mask, "status"] = status
    registry.loc[mask, "status_evidence"] = evidence
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
        "round_id": "round_42_batch_composition_algebra",
        "candidate_id": "BCNSF",
        "candidate_family": "Batch-composition Newton-Schulz flow",
        "hyperparameters": (
            "Source-fixed N=2048 and B=128; angles {30,60,90}; rare curvature {0.5,1,2}; "
            "anisotropy {0.25,1,4}; 21 mixture weights; median correction gate 0.001 and p95 gate 0.01"
        ),
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "See round-41 portfolio preregistration.",
        "novelty_class": "Jensen drift from stochastic batch composition before Muon's nonlinear map",
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
    for rare_weight in RARE_WEIGHT_GRID:
        fractions, probabilities = candidate.composition_distribution(rare_weight)
        realized_weight = candidate.rare_count_per_block(rare_weight) / candidate.MIXTURE_BLOCK_SIZE
        assert abs(float(probabilities.sum()) - 1.0) < 1e-12
        assert abs(float(np.dot(fractions, probabilities)) - realized_weight) < 1e-12
    rng = np.random.default_rng(20260719)
    base_states = muon.normalize_state(rng.normal(size=(64, 2, 2)))
    rows = []
    corrections = []
    for mean_config, stochastic_config in zip(configs("mean"), configs("hypergeometric"), strict=True):
        state = np.repeat(base_states, len(RARE_WEIGHT_GRID), axis=0)
        rare_weight = np.tile(np.asarray(RARE_WEIGHT_GRID, dtype=float), len(base_states))
        mean = candidate.expected_update_direction(state, rare_weight, mean_config)
        stochastic = candidate.expected_update_direction(state, rare_weight, stochastic_config)
        difference = np.linalg.norm(stochastic - mean, axis=(1, 2))
        mean_norm = np.linalg.norm(mean, axis=(1, 2))
        relative = difference / np.maximum(mean_norm, 1e-12)
        attenuation = 1.0 - np.linalg.norm(stochastic, axis=(1, 2))
        corrections.extend(relative.tolist())
        rows.append(
            {
                "angle_degrees": mean_config.task_angle_degrees,
                "rare_curvature": mean_config.rare_curvature,
                "input_anisotropy": mean_config.input_anisotropy,
                "median_relative_direction_correction": float(np.median(relative)),
                "p95_relative_direction_correction": float(np.quantile(relative, 0.95)),
                "max_relative_direction_correction": float(np.max(relative)),
                "median_update_attenuation": float(np.median(attenuation)),
                "p95_update_attenuation": float(np.quantile(attenuation, 0.95)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(args.output_dir / "geometry_correction_summary.csv", index=False)
    corrections_array = np.asarray(corrections, dtype=float)
    median = float(np.median(corrections_array))
    p95 = float(np.quantile(corrections_array, 0.95))
    maximum = float(np.max(corrections_array))
    active = median >= MEDIAN_CORRECTION_GATE and p95 >= P95_CORRECTION_GATE
    status = "active_after_algebra" if active else "blocked_before_starcoder"
    evidence = (
        f"median_relative_direction_correction={median:.8g}; p95={p95:.8g}; max={maximum:.8g}; "
        f"median_gate={MEDIAN_CORRECTION_GATE:g}; p95_gate={P95_CORRECTION_GATE:g}; algebraically_material={active}."
    )
    update_status(status, evidence, args.output_dir)
    report = [
        "# Round 42: batch-composition Muon algebraic audit",
        "",
        "The exact loader counts, global batch size, finite Newton-Schulz map, geometry grid, and materiality thresholds were frozen before this audit. No target values were read.",
        "",
        "## Decision",
        "",
        f"**{status}.** {evidence}",
        "",
        "The relative correction is the Frobenius distance between the expected per-batch finite-Muon direction and finite Muon applied once to the mean gradient, divided by the mean-map direction norm.",
        "",
        "## Geometry strata",
        "",
        table.to_markdown(index=False),
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(evidence)


if __name__ == "__main__":
    main()
