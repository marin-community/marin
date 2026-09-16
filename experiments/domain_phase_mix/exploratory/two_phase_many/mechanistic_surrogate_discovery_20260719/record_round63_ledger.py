# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the provenance-checked prior-stability audit to the data-use ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    round_id = "round_63_prior_stability_carryforward"
    candidate_id = "prior_complexity_identifiability_and_optimum_stability"
    exists = ledger["round_id"].eq(round_id) & ledger["candidate_id"].eq(candidate_id)
    if exists.any():
        print("round-63 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": round_id,
        "candidate_id": candidate_id,
        "candidate_family": "prior_evidence_provenance_audit",
        "hyperparameters": "No fit or selection; exact prior source artifacts carried by SHA-256",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "The final bundle required explicit parameter, effective-complexity, and raw-optimum "
            "stability evidence rather than a prose pointer to the prior drive."
        ),
        "novelty_class": "provenance-only inherited diagnostic; no model mechanism proposed",
        "evaluation_status": "completed_prior_stability_carryforward_no_promotion",
        "evidence_path": "round63_prior_stability_carryforward/report.md",
        "notes": (
            "All 25 refits per target/policy predict unsupported raw optima below the observed frontier; "
            "60/89 nonlinear parameter pairs are not cross-panel stable."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-63 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
