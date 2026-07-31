# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the future-confirmation specification audit to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ROUND_ID = "round_72_future_confirmation_design"
CANDIDATE_ID = "inactive_confirmation_panel_specification_audit"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-72 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "future_confirmation_design_diagnostic",
        "hyperparameters": "gamma=(0.8,0.2); simplex safety=0.90; direction seed=20260719; six decisive repeats",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "The inactive confirmation plan required an exact contrast-ray and family-direction algorithm to be independently reproducible.",
        "novelty_class": "preregistration specification audit; no candidate mechanism",
        "evaluation_status": "completed_inactive_design_validation_no_promotion",
        "evidence_path": "round72_future_confirmation_design/report.md",
        "notes": "Validated deterministic direction generation and 86-policy/116-run arithmetic; panel remains inactive.",
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-72 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
