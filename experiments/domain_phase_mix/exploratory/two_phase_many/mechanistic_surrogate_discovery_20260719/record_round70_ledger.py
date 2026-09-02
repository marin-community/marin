# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the cross-scale measurement-error diagnostic to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ROUND_ID = "round_70_cross_scale_measurement_error"
CANDIDATE_ID = "phase_transfer_measurement_error_bound"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-70 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "cross_scale_identification_diagnostic",
        "hyperparameters": "Independent-run phase-difference noise=2*sigma^2; point and upper-95pct nuisance bounds",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Round-61 found weak and attenuated cross-scale phase-effect transfer; repeat data can falsify a measurement-error explanation.",
        "novelty_class": "errors-in-variables diagnostic; no candidate mechanism",
        "evaluation_status": "completed_measurement_error_bound_no_promotion",
        "evidence_path": "round70_cross_scale_measurement_error/report.md",
        "notes": "Deattenuation does not repair cross-scale phase transfer; nuisance noise is too small relative to the observed phase-effect variance.",
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-70 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
