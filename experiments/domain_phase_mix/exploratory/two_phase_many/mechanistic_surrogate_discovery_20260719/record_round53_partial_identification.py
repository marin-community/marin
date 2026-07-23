# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
# ]
# ///
"""Record the Round 53 diagnostic in the append-only data-use ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
LEDGER = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/data_use_ledger.csv"
ROUND_ID = "round_53_partial_identification"


def main() -> None:
    ledger = pd.read_csv(LEDGER)
    if ledger["round_id"].eq(ROUND_ID).any():
        print(f"{ROUND_ID} is already recorded")
        return
    record = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "candidate_id": "finite_design_partial_identification_audit",
        "candidate_family": "diagnostic_not_a_surrogate",
        "hyperparameters": "OOF-only 5% gate set and preregistered descriptive 15% Rashomon set",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "All physical transitions through Round 52 failed the StarCoder shape or raw-optimum gates; exposed "
            "development predictions show no single Pareto-dominant incumbent."
        ),
        "novelty_class": "formal identifiability diagnostic; no model proposed or tuned",
        "evaluation_status": "completed_no_candidate_promotion",
        "evidence_path": "round53_partial_identification/report.md",
        "notes": (
            "No surrogate was fit. The running sealed phase-fiber panel was not read. Fit-near-equivalent models "
            "select materially different policies; finite-design smoothness cannot identify a raw optimum."
        ),
    }
    updated = pd.concat([ledger, pd.DataFrame([record])], ignore_index=True)
    updated.to_csv(LEDGER, index=False)
    print(f"Recorded {ROUND_ID}")


if __name__ == "__main__":
    main()
