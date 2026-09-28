# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the registry-coverage audit to the append-only data-use ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ROUND_ID = "round_57_registry_mechanism_coverage"
CANDIDATE_ID = "complete_registry_duplication_audit"


def main() -> None:
    ledger = pd.read_csv(LEDGER)
    recorded = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if recorded.any():
        print("round-57 registry audit already recorded")
        return

    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "registry_integrity_diagnostic",
        "hyperparameters": "Six primary mechanism classes; token-Jaccard duplicate-screen threshold 0.35",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Final completeness audit required evidence that the search did not repeatedly rename one exposure model."
        ),
        "novelty_class": "registry integrity diagnostic; no new mechanism",
        "evaluation_status": "completed_registry_audit_no_promotion",
        "evidence_path": "round57_registry_mechanism_coverage/report.md",
        "notes": (
            "All 58 new routes map exactly once to six primary mechanism classes and a terminal gate. One high-overlap "
            "description pair, PWD/ESR, has distinct transition laws and independent blocking evidence."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-57 row to {LEDGER}")


if __name__ == "__main__":
    main()
