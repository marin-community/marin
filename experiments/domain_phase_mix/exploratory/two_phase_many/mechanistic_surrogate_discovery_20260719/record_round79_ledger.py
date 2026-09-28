# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///

"""Record reconciliation of exposed-panel-only row predictions."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
PREDICTIONS = OUTPUT_ROOT / "round77_final_row_predictions/adversarial_row_predictions.csv"
ROUND_ID = "round_79_exposed_prediction_reconciliation"
CANDIDATE_ID = "exposed_only_row_prediction_traceability_no_model_choice"


def main() -> None:
    predictions = pd.read_csv(PREDICTIONS)
    target_matched = predictions.loc[predictions["target_relation"].eq("target_matched")]
    cross_target = predictions.loc[predictions["target_relation"].eq("cross_target")]
    if len(predictions) != 2_400 or target_matched["model"].nunique() != 11 or cross_target["model"].nunique() != 9:
        raise ValueError("Run the corrected row-prediction export before recording reconciliation")

    ledger = pd.read_csv(LEDGER).fillna("")
    exists = ledger["round_id"].eq(ROUND_ID) & ledger["candidate_id"].eq(CANDIDATE_ID)
    if exists.any():
        print("round-79 reconciliation already recorded")
        return
    row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": CANDIDATE_ID,
        "candidate_family": "terminal_prediction_traceability",
        "hyperparameters": "none; row-level export only",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Terminal QA found that two exposed-panel-only baselines had aggregate metrics but no row-level "
            "residual export."
        ),
        "novelty_class": "terminal provenance correction; no model, mechanism, or hyperparameter proposal",
        "evaluation_status": "completed_2400_adversarial_predictions_reconciled",
        "evidence_path": "round77_final_row_predictions/report.md",
        "notes": (
            "All 11 exposed models now have target-matched residuals; the nine archive-wide models also have "
            "cross-target residuals. No sealed outcome was read and no metric or decision changed."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([row], columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended round-79 reconciliation to {LEDGER}")


if __name__ == "__main__":
    main()
