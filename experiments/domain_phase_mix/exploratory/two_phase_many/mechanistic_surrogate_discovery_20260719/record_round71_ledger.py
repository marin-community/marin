# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the cross-target phase-state diagnostic to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
ROUND_ID = "round_71_cross_target_phase_state"
CANDIDATE_ID = "cross_target_scalar_phase_state_diagnostic"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-71 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "cross_target_identification_diagnostic",
        "hyperparameters": "20,000 paired policy bootstraps; seed=20260719; no fitted response",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": "Substantial phase-effect correlation across targets could support a shared latent transition, subject to sign and scale tests.",
        "novelty_class": "shared-state identification diagnostic; does not reopen rejected joint latent transport",
        "evaluation_status": "completed_scalar_shared_state_falsification_no_promotion",
        "evidence_path": "round71_cross_target_phase_state/report.md",
        "notes": "Targets share phase signal but differ in sign for about one fifth of policies and have significantly different scale attenuation ratios.",
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-71 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
