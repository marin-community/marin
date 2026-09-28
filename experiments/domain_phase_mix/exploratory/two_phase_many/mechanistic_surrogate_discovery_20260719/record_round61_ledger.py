# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the matched-policy variance-decomposition diagnostic to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    round_id = "round_61_cross_scale_variance_decomposition"
    candidate_id = "matched_aggregate_phase_effect_variance_audit"
    exists = ledger["round_id"].eq(round_id) & ledger["candidate_id"].eq(candidate_id)
    if exists.any():
        print("round-61 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": round_id,
        "candidate_id": candidate_id,
        "candidate_family": "cross_scale_identification_diagnostic",
        "hyperparameters": "Exact matched-policy identity on 238 coordinates; no fit or tuning",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Observed one-phase policies transfer better across 300M and Delphi than two-phase policies; "
            "decompose this gap into aggregate response, paired phase correction, and their covariance."
        ),
        "novelty_class": "cross-scale variance diagnostic; no model mechanism proposed",
        "evaluation_status": "completed_cross_scale_diagnostic_no_promotion",
        "evidence_path": "round61_cross_scale_variance_decomposition/report.md",
        "notes": (
            "At 300M phase-delta SD is about equal to aggregate SD, but at Delphi it is about half; "
            "aggregate and phase delta are strongly anticorrelated at both scales."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-61 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
