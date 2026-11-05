# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Freeze the first feasibility protocol using only a completed native control."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np

from experiments.grug.tied_drafter.acceptance import fit_progress_forecast


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    raw = args.calibration.read_bytes()
    calibration = json.loads(raw)
    if calibration["status"] != "complete":
        raise ValueError("Freeze only from a completed control calibration")
    rows = {row["step"]: row for row in calibration["checkpoints"]}
    steps = sorted(rows)
    confidence = 1 - 0.025 / (2 * (len(steps) - 1))
    curve = []
    checks = []
    for first, last in pairwise(steps):
        pair = np.array([first, last])
        fit = fit_progress_forecast(
            pair, np.array([rows[s]["groups"]["calibration"] for s in pair]), confidence=confidence
        )
        check = fit_progress_forecast(pair, np.array([rows[s]["groups"]["check"] for s in pair]), confidence=confidence)
        curve.append(replace(fit, upper_loss_per_step=max(fit.upper_loss_per_step, check.upper_loss_per_step)))
        checks.append(check)
    floor = max(row.get("repeatability_max_loss_difference", 0.0) for row in rows.values())
    source = Path(__file__).parent
    protocol = {
        "frozen_utc": datetime.now(UTC).isoformat(),
        "calibration_sha256": hashlib.sha256(raw).hexdigest(),
        "implementation_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(source.glob("*.py"))
        },
        "seed": calibration["seed"],
        "recipe": "width_quarter",
        "anchor_step": 256,
        "native_step_seconds": (rows[320]["training_seconds"] - rows[192]["training_seconds"]) / 128,
        "native_timing_policy": "Mean control update plus data-wait time from steps 193 through 320",
        "horizons": [16, 8, 32],
        "draft_learning_rate": 0.001,
        "scales": [1.0, 0.5],
        "maximum_staleness_steps": 64,
        "verification_batches": 1,
        "sequences_per_group": 8,
        "assumed_acceptance": 0.8,
        "candidate_confidence": 1 - 0.025 / 3,
        "repeatability_floor": max(2 * floor, 1e-6),
        "forecast_curve": [asdict(part) for part in curve],
        "forecast_lookahead": 64,
        "forecast_policy": (
            "Measured piecewise control curve, with the largest local upper progress slope over the next 64 updates"
        ),
        "independent_check_slopes": [asdict(f) for f in checks],
        "extrapolation_equivalent_steps": 4,
        "loss_targets": [
            {
                "control_step": s,
                "loss": float(np.mean(rows[s]["groups"]["target"])),
                "native_control_training_seconds": rows[s]["training_seconds"],
            }
            for s in (288, 320)
        ],
        "optimizer_policy": (
            "Reset draft AdamW each snapshot; preserve hero moments, EMA, pending router state, "
            "ordinary step and data cursor on parameter jumps"
        ),
        "rejection_policy": "Retain original complete hero state; advance native continuation in the recorded outcome",
        "persistence_policy": (
            "On identical next 64 batches, compare native, selected draft scale, and extrapolation "
            "at 0/1/4/16/64 updates, even when the proposed jump is rejected"
        ),
        "data_policy": (
            "Train from index zero; draft consumes the next horizon batches starting at snapshot_step. "
            "Verification 30000+8*horizon; persistence 40000+8*horizon; final target indices 20000..20003. "
            "These are reserved mixture indices, not proven document-disjoint sets."
        ),
        "statistics": (
            "Nominal grouped t bounds with a 2.5% forecast budget and 2.5% split over three horizons "
            "and their predefined scale trials. Adjacent packed-sequence groups can be correlated; "
            "bounds are exploratory, not a guarantee of population coverage."
        ),
        "sample_size_policy": (
            "One full native batch (128 sequences) per candidate evaluation; uncertain results are rejected. "
            "Four reserved batches measure persistence. No sample-size tuning after treatment outcomes."
        ),
        "expansion_gate": (
            "Repeated positive 64-update persistence and projected speedup >=1.2 after measured overhead; "
            "then at least three independent paired seeds. No superior resource-efficiency claim "
            "without equal-total-hardware control."
        ),
        "initial_gpu_hour_cap": 10,
        "notes": (
            "A diagnostic trial is not an end-to-end accelerated benchmark. Diagnostic branch time is "
            "accounted as experiment cost but not represented as deployment overhead."
        ),
    }
    args.output.write_text(json.dumps(protocol, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
