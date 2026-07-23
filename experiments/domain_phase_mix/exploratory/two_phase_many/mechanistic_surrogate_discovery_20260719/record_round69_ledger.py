# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the frozen repeat-noise influence diagnostic to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ROUND_ID = "round_69_repeat_noise_influence"
CANDIDATE_ID = "future_confirmation_repeat_noise_influence_diagnostic"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-69 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "future_confirmation_design_diagnostic",
        "hyperparameters": "Leave one of 10 exact-policy repeat groups out; effect=0.005 BPB; alpha=0.05; power=0.80",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Round-67 pooled repeat variance had only 16 residual degrees of freedom and a wide Table-9 interval.",
        "novelty_class": "nuisance-variance influence diagnostic; no candidate mechanism",
        "evaluation_status": "completed_confirmation_design_robustness_no_promotion",
        "evidence_path": "round69_repeat_noise_influence/report.md",
        "notes": "Tests whether the future repeat allocation is driven by one exact-policy group; no response model or threshold changed.",
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-69 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
