# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "tabulate>=0.9",
# ]
# ///
"""Record and quantify the StarCoder optimizer-schedule provenance correction."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260719 import (
    starcoder_optimizer_schedule as schedules,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
DEFAULT_OUTPUT = OUTPUT_ROOT / "round37_schedule_provenance_erratum"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
AFFECTED = ("OTTPF", "OTFSC", "SGDDD", "AAGF", "CTPF", "SPTF")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def legacy_learning_rate(name: str, time: np.ndarray, boundary: float) -> np.ndarray:
    if name.startswith("starcoder_cosine"):
        return 0.5 * (1.0 + np.cos(np.pi * time))
    progress = np.clip((time - boundary) / (1.0 - boundary), 0.0, 1.0)
    return np.where(time <= boundary, 1.0, 0.5 * (1.0 + np.cos(np.pi * progress)))


def legacy_optimizer_fraction(name: str, boundary: float) -> float:
    time = np.linspace(0.0, 1.0, 65537)
    learning_rate = legacy_learning_rate(name, time, boundary)
    early = float(np.trapezoid(learning_rate[time <= boundary], time[time <= boundary]))
    late = float(np.trapezoid(learning_rate[time >= boundary], time[time >= boundary]))
    return early / (early + late)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    curves = []
    for spec in (schedules.COSINE_50_50, schedules.WSD_80_20):
        early, late = spec.phase_learning_rate_masses()
        old = legacy_optimizer_fraction(spec.name.value, spec.phase0_fraction)
        rows.append(
            {
                "surface": spec.name.value,
                "total_steps": spec.total_steps,
                "phase_boundary_step": spec.phase_boundary_step,
                "realized_phase0_fraction": spec.phase0_fraction,
                "warmup_steps": spec.warmup_steps,
                "stable_steps": spec.stable_steps,
                "decay_steps": spec.decay_steps,
                "phase0_lr_mass": early,
                "phase1_lr_mass": late,
                "corrected_optimizer_phase0_fraction": spec.optimizer_phase0_fraction(),
                "legacy_optimizer_phase0_fraction": old,
                "absolute_fraction_correction": spec.optimizer_phase0_fraction() - old,
                "provenance": spec.provenance,
            }
        )
        steps = np.arange(spec.total_steps, dtype=float)
        corrected = spec.learning_rate_at_steps(steps)
        normalized = steps / spec.total_steps
        legacy = legacy_learning_rate(spec.name.value, normalized, spec.phase0_fraction)
        curves.extend(
            {
                "surface": spec.name.value,
                "step": int(step),
                "normalized_time": float(time),
                "phase": 0 if step < spec.phase_boundary_step else 1,
                "corrected_learning_rate": float(corrected_lr),
                "legacy_learning_rate": float(legacy_lr),
            }
            for step, time, corrected_lr, legacy_lr in zip(steps, normalized, corrected, legacy, strict=True)
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(args.output_dir / "schedule_masses.csv", index=False)
    pd.DataFrame(curves).to_csv(args.output_dir / "schedule_curves.csv", index=False)

    registry = pd.read_csv(REGISTRY)
    missing = set(AFFECTED) - set(registry["id"])
    if missing:
        raise ValueError(f"Affected registry rows are missing: {sorted(missing)}")
    for candidate_id in AFFECTED:
        prior = registry.loc[registry["id"].eq(candidate_id), "status"].iloc[0]
        evidence = registry.loc[registry["id"].eq(candidate_id), "status_evidence"].iloc[0]
        registry.loc[registry["id"].eq(candidate_id), "status"] = "reopened_schedule_provenance_erratum"
        registry.loc[registry["id"].eq(candidate_id), "status_evidence"] = (
            f"Prior status {prior} is provisional: the StarCoder LR clock omitted warmup. Original evidence: {evidence}"
        )
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_37_schedule_provenance_erratum",
        "candidate_id": "SCHEDULE_PROVENANCE_ERRATUM",
        "candidate_family": "StarCoder optimizer-time provenance correction",
        "hyperparameters": "No model hyperparameters changed; exact persisted warmup/stable/decay schedule only",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Source and W&B audit found a 1000-step warmup omitted from the historical cosine clock",
        "novelty_class": "Identification/provenance correction, not a new model mechanism",
        "evaluation_status": "affected_routes_reopened_without_adversarial_evaluation",
        "evidence_path": str(args.output_dir.relative_to(OUTPUT_ROOT)),
        "notes": f"Affected routes: {','.join(AFFECTED)}. Running sealed confirmation panel was not inspected.",
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(row[column] for column in identity) not in existing:
        pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True).to_csv(LEDGER, index=False)

    report = f"""# Round 37: StarCoder optimizer-schedule provenance erratum

Several optimizer-time screens used a peak-LR schedule from step zero. The historical 50/50 cosine runs instead used a 1,000-step linear warmup, while the WSD runs used a 38-step warmup followed by a stable phase and boundary-aligned cosine decay. This is an input-provenance error, not evidence for a new mechanism.

## Corrected schedules

{summary.to_markdown(index=False, floatfmt=".6f")}

## Scope

The affected routes are {", ".join(AFFECTED)}. Their old artifacts remain unchanged, but their registry decisions are provisional until their exact frozen grids are rerun with this corrected clock. Token-time and schedule-independent negative results are unaffected. No historical 3e18, exposed adversarial, or sealed-confirmation target was read during this correction.
"""
    (args.output_dir / "report.md").write_text(report)
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(
            {
                "affected_routes": AFFECTED,
                "adversarial_targets_read": False,
                "sealed_confirmation_targets_read": False,
                "model_hyperparameters_changed": False,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
