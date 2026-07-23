# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the untouched-confirmation multiplicity audit to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ROUND_ID = "round_74_confirmation_multiplicity"
CANDIDATE_ID = "inactive_confirmation_multiplicity_audit"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-74 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "future_confirmation_statistical_design",
        "hyperparameters": "FWER=0.05; two superiority targets; Holm first threshold=0.025; effect=0.005 BPB",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "The inactive confirmation acceptance rule allowed a superiority claim on either of two targets without explicit multiplicity control.",
        "novelty_class": "preregistration multiplicity and power audit; no candidate mechanism",
        "evaluation_status": "completed_inactive_design_correction_no_promotion",
        "evidence_path": "round74_confirmation_multiplicity/report.md",
        "notes": "Holm correction requires 7 Table-9 repeats at point noise or 15 at its upper-95% nuisance bound; inactive design uses 15.",
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-74 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
