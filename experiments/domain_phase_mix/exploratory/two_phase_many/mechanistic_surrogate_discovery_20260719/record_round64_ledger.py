# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the frozen policy-class robustness diagnostic to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    round_id = "round_64_policy_class_robustness"
    candidate_id = "frozen_baseline_policy_class_split"
    exists = ledger["round_id"].eq(round_id) & ledger["candidate_id"].eq(candidate_id)
    if exists.any():
        print("round-64 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": round_id,
        "candidate_id": candidate_id,
        "candidate_family": "policy_class_stratification_diagnostic",
        "hyperparameters": "Frozen baseline predictions; no fitting, tuning, or pooling across policy class",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Pooled 710-run metrics can hide whether a two-phase fit transfers differently to tied and "
            "phase-varying policies."
        ),
        "novelty_class": "frozen policy-class diagnostic; no model mechanism proposed",
        "evaluation_status": "completed_policy_class_robustness_diagnostic_no_promotion",
        "evidence_path": "round64_policy_class_robustness/report.md",
        "notes": (
            "Model rankings and diagnostic winners differ between one-phase and two-phase heldouts; "
            "pooled metrics do not identify a policy-class-robust incumbent."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-64 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
