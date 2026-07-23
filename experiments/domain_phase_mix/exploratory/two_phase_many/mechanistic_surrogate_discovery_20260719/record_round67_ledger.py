# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the future-confirmation power audit to the data-use ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ROUND_ID = "round_67_confirmation_power"
CANDIDATE_ID = "future_confirmation_repeat_power_diagnostic"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-67 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "future_confirmation_design_diagnostic",
        "hyperparameters": "One-sided alpha=0.05; target power=0.80; effects={0.002,0.005,0.010}",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": False,
        "observations_inspiring_mechanism": (
            "The inactive confirmation plan used three repeats per arm without an explicit power audit against its "
            "already frozen 0.005-BPB acceptance threshold."
        ),
        "novelty_class": "experimental-design audit only; no candidate mechanism proposed",
        "evaluation_status": "completed_power_audit_confirmation_design_revised",
        "evidence_path": "round67_confirmation_power/report.md",
        "notes": (
            "Independent exact-policy repeats imply three Table-9 repeats per arm are underpowered; the inactive "
            "confirmation preregistration now uses six as the minimum."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-67 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
