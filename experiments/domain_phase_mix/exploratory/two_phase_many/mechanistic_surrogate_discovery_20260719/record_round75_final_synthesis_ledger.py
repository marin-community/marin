# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the terminal negative synthesis and supersession note to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
FINAL_REPORT = OUTPUT_ROOT / "final_synthesis/final_report.md"
ROUND_ID = "round_75_final_negative_synthesis"
CANDIDATE_ID = "terminal_negative_verdict_and_design_supersession"
FROZEN_GATE_DIGEST = "c4f711312423f038ef8610950d1ae6be30ffba588648177fbf5077e6931f93be"


def main() -> None:
    if not FINAL_REPORT.is_file():
        raise ValueError("Build the final synthesis before recording its terminal ledger event")
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-75 terminal synthesis already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "terminal_scientific_synthesis",
        "hyperparameters": f"frozen gate SHA-256={FROZEN_GATE_DIGEST}; 99 registered routes; 15 decisive repeats",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "All 58 new routes and 41 inherited routes reached explicit terminal gates; no candidate survived to "
            "promotion, independent review, or untouched confirmation."
        ),
        "novelty_class": "terminal negative synthesis and preregistration supersession; no candidate mechanism",
        "evaluation_status": "completed_negative_result_no_promotion_no_confirmation",
        "evidence_path": "final_synthesis/final_report.md",
        "notes": (
            "No headline surrogate is recommended. Round 74 supersedes the historical Round 67/72 six-repeat, "
            "116-run planning entries: the inactive untouched panel now fixes 15 decisive repeats and at most 170 "
            "runs with Holm control. No sealed outcome was read and no training job was submitted."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-75 terminal synthesis to {LEDGER}")


if __name__ == "__main__":
    main()
