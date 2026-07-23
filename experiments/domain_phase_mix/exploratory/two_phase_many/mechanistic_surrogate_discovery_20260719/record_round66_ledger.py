# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the frozen one-phase-frontier phase-benefit diagnostic to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ROUND_ID = "round_66_frontier_phase_benefit"
CANDIDATE_ID = "matched_policy_frontier_phase_benefit_diagnostic"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-66 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "matched_policy_design_diagnostic",
        "hyperparameters": "Fixed top-k={10,25,50}; 20000 paired bootstrap resamples; seed=20260719",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": False,
        "observations_inspiring_mechanism": (
            "The matched 300M/Delphi decomposition showed that phase corrections contract with scale and are "
            "anticorrelated with aggregate quality, motivating a direct frontier-slice design audit."
        ),
        "novelty_class": "descriptive paired-policy diagnostic only; no candidate mechanism proposed",
        "evaluation_status": "completed_design_diagnostic_no_promotion",
        "evidence_path": "round66_frontier_phase_benefit/report.md",
        "notes": (
            "Same-scale frontier slices are labeled mechanically coupled; cross-scale ranking is the primary "
            "safeguard. The result supports signed aggregate-preserving phase fibers, not a surrogate promotion."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-66 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
