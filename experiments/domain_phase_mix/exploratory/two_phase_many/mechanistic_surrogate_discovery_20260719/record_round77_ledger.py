# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///

"""Append the terminal row-level prediction export to the data-use ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
REPORT = OUTPUT_ROOT / "round77_final_row_predictions/report.md"
ROUND_ID = "round_77_final_row_prediction_export"
CANDIDATE_ID = "terminal_row_prediction_export_no_model_choice"


def main() -> None:
    if not REPORT.is_file():
        raise ValueError("Run the Round-77 row-level export before recording it")
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-77 row-level export already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "terminal_reproducibility_export",
        "hyperparameters": "none; exact dashboard prediction arrays and frozen provenance joins",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "The terminal bundle exposed aggregate metrics but required reopening the Observatory JSON to inspect "
            "every residual, optimism value, and proposal stratum."
        ),
        "novelty_class": "reproducibility export; no model, mechanism, or hyperparameter proposal",
        "evaluation_status": "completed_terminal_row_export_no_promotion",
        "evidence_path": "round77_final_row_predictions/report.md",
        "notes": (
            "Exported 12,780 uniquely keyed predictions for 710 heldout runs, two targets, and nine baseline models; "
            "no sealed outcome was read."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-77 export to {LEDGER}")


if __name__ == "__main__":
    main()
