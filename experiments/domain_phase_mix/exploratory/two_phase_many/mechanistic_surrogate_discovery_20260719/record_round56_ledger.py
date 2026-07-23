# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the coordinate-only round-56 identification audit to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ROUND_ID = "round_56_two_stage_design_identifiability"
CANDIDATE_ID = "same_budget_aggregate_phase_fiber_design"


def main() -> None:
    ledger = pd.read_csv(LEDGER)
    already_recorded = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if already_recorded.any():
        print("round-56 identification audit already recorded")
        return

    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "experimental_design_diagnostic",
        "hyperparameters": "280 rows: 140 phase-tied anchors plus 70 signed phase-fiber pairs",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Random two-phase designs have full contrast rank but must identify aggregate utility and phase "
            "transport from the same outcomes; diagnostic uses coordinates only and reads no BPB values."
        ),
        "novelty_class": "coordinate-only identification diagnostic; no response model",
        "evaluation_status": "completed_design_diagnostic_no_promotion",
        "evidence_path": "round56_two_stage_design_identifiability/report.md",
        "notes": (
            "At equal row count, signed phase-fiber pairs reduce aggregate/contrast canonical correlation "
            "from max 0.833 to numerical zero while retaining full 76-dimensional joint rank. This supports "
            "a future intervention design and does not repair or promote a current surrogate."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-56 row to {LEDGER}")


if __name__ == "__main__":
    main()
