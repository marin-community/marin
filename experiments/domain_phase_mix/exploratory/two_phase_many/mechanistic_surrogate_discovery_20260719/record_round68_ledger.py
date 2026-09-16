# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the frozen support-abstention diagnostic to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ROUND_ID = "round_68_support_abstention"
CANDIDATE_ID = "frozen_baseline_support_abstention_diagnostic"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-68 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "frozen_baseline_deployment_diagnostic",
        "hyperparameters": "Fixed nearest-support coverage={0.10,0.25,0.50,0.75,1.00}",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Support distance predicts baseline error, requiring a direct distinction between admissible abstention and "
            "inadmissible residual calibration."
        ),
        "novelty_class": "deployment diagnostic only; no candidate mechanism or tuned threshold proposed",
        "evaluation_status": "completed_abstention_tradeoff_no_promotion",
        "evidence_path": "round68_support_abstention/report.md",
        "notes": (
            "Abstention reduces catastrophic optimism but can increase full-archive regret; it remains a deployment "
            "constraint and cannot validate the response law."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-68 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
