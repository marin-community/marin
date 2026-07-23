# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///

"""Append the terminal deliverable-traceability audit to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
REPORT = OUTPUT_ROOT / "round78_deliverable_traceability/report.md"
ROUND_ID = "round_78_terminal_deliverable_traceability"
CANDIDATE_ID = "terminal_deliverable_traceability_no_model_choice"


def main() -> None:
    if not REPORT.is_file():
        raise ValueError("Run the Round-78 traceability audit before recording it")
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-78 traceability already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "terminal_requirements_traceability",
        "hyperparameters": "none; 21 immutable deliverable checks",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "The final drive needed a direct map from each requested deliverable and scientific boundary to its "
            "checksummed source rather than relying on narrative completeness."
        ),
        "novelty_class": "terminal QA diagnostic; no model, mechanism, or hyperparameter proposal",
        "evaluation_status": "completed_all_21_deliverable_checks_passed",
        "evidence_path": "round78_deliverable_traceability/report.md",
        "notes": "All 21 terminal requirements passed; no sealed outcome was read.",
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-78 traceability to {LEDGER}")


if __name__ == "__main__":
    main()
