# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the phase-reversal observability diagnostic to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ROUND_ID = "round_73_phase_reversal_observability"
CANDIDATE_ID = "odd_even_phase_reversal_identification_diagnostic"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-73 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "phase_reversal_identification_diagnostic",
        "hyperparameters": "exact coordinate tolerance=1e-10; no fitted response; phase fractions 0.5/0.5 and 0.8/0.2",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Rejected phase models confound odd order effects with even phase-variation costs when aggregate and contrast are correlated.",
        "novelty_class": "identification invariant; no candidate mechanism and no reopening of rejected phase models",
        "evaluation_status": "completed_design_observability_audit_no_promotion",
        "evidence_path": "round73_phase_reversal_observability/report.md",
        "notes": "Exact reversal identifies odd/even components on supported StarCoder triples; the 39-bucket random swarm lacks exact reflected contrasts.",
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-73 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
