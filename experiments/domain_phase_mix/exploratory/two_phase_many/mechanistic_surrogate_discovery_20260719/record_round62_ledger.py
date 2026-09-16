# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append the adversarial stratum-robustness diagnostic to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def main() -> None:
    ledger = pd.read_csv(LEDGER).fillna("")
    round_id = "round_62_adversarial_strata_robustness"
    candidate_id = "exposed_baseline_worst_stratum_audit"
    exists = ledger["round_id"].eq(round_id) & ledger["candidate_id"].eq(candidate_id)
    if exists.any():
        print("round-62 diagnostic already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": round_id,
        "candidate_id": candidate_id,
        "candidate_family": "adversarial_stratification_diagnostic",
        "hyperparameters": "Frozen baseline predictions; no pooling, fitting, or tuning",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Pooled adversarial RMSE can hide response compression and rank reversal within baseline-ranked, "
            "challenger-ranked, and high-disagreement candidate strata."
        ),
        "novelty_class": "exposed-panel stratification diagnostic; no model mechanism proposed",
        "evaluation_status": "completed_adversarial_strata_diagnostic_no_promotion",
        "evidence_path": "round62_adversarial_strata_robustness/report.md",
        "notes": (
            "Every baseline has minimum selection-stratum calibration slope below 0.5 on both targets; "
            "selection-stratum Spearman is negative for 6/11 Uncheatable and 8/11 Table-9 baselines."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-62 diagnostic to {LEDGER}")


if __name__ == "__main__":
    main()
