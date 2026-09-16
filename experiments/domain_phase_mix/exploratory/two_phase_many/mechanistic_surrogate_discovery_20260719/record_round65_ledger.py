# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the frozen worst-residual exposure-pattern diagnostic to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ROUND_ID = "round_65_worst_exposure_patterns"
CANDIDATE_ID = "frozen_baseline_worst_exposure_pattern_diagnostic"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-65 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "frozen_baseline_residual_diagnostic",
        "hyperparameters": "No fit or selection; top-k fixed at 10 before diagnostic aggregation",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "The negative synthesis required direct identification of which policy geometries concentrate "
            "the frozen baselines' largest archive optimism errors."
        ),
        "novelty_class": "descriptive diagnostic only; no candidate mechanism proposed",
        "evaluation_status": "completed_worst_exposure_pattern_audit_no_promotion",
        "evidence_path": "round65_worst_exposure_patterns/report.md",
        "notes": (
            "Support distance and phase TV correlate positively with optimism for every audited model and target; "
            "max epoch and aggregate tilt do not. No residual feature or calibrator was introduced."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-65 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
