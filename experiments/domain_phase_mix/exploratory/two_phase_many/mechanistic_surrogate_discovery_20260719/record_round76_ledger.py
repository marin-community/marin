# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///

"""Append the independent final-metric reproduction audit to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
REPORT = OUTPUT_ROOT / "round76_final_metric_reproduction/report.md"
ROUND_ID = "round_76_final_metric_reproduction"
CANDIDATE_ID = "independent_metric_reproduction_no_model_choice"


def main() -> None:
    if not REPORT.is_file():
        raise ValueError("Run the Round-76 metric reproduction audit before recording it")
    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-76 metric reproduction already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "metric_implementation_integrity_diagnostic",
        "hyperparameters": "independent formulas; float tolerance=5e-12; integer counts exact",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "The terminal negative result depends on calibration, optimism, lower-tail, and Regret@k signs and masks; "
            "a self-consistent implementation error had to be excluded independently."
        ),
        "novelty_class": "reproducibility diagnostic; no model, mechanism, or hyperparameter proposal",
        "evaluation_status": "completed_exact_metric_reproduction_no_promotion",
        "evidence_path": "round76_final_metric_reproduction/report.md",
        "notes": (
            "Independently reproduced 680 archive and target-matched adversarial scalar metrics from row-level "
            "predictions. Maximum absolute difference was 6.7e-16; no sealed outcome was read."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-76 metric reproduction to {LEDGER}")


if __name__ == "__main__":
    main()
