# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append heldout provenance and coordinate-balance diagnostics to the ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def main() -> None:
    ledger = pd.read_csv(LEDGER)
    timestamp = datetime.now(UTC).isoformat()
    definitions = (
        {
            "round_id": "round_58_heldout_provenance",
            "candidate_id": "append_only_archive_provenance_audit",
            "candidate_family": "data_provenance_diagnostic",
            "hyperparameters": "Exact policy SHA-256 identity; no tolerance-based deduplication",
            "observations_inspiring_mechanism": "Final archive integrity check; no response mechanism proposed.",
            "novelty_class": "data provenance diagnostic; no new mechanism",
            "evaluation_status": "completed_provenance_audit_no_promotion",
            "evidence_path": "round58_heldout_provenance/report.md",
            "notes": (
                "710 fit-disjoint run rows comprise 690 policy hashes; 12 fit-coordinate aliases are excluded. "
                "All 120 adversarial rows join one-to-one to frozen target, policy, stratum, and proposal metadata."
            ),
        },
        {
            "round_id": "round_59_coordinate_balanced_metrics",
            "candidate_id": "coordinate_balanced_archive_sensitivity",
            "candidate_family": "metric_weighting_diagnostic",
            "hyperparameters": "Average observed and predicted BPB by exact mixture SHA-256 before metric computation",
            "observations_inspiring_mechanism": (
                "Round-58 audit found 710 run rows but 690 unique policy hashes; diagnostic prevents repeat rows from "
                "receiving unintended statistical weight."
            ),
            "novelty_class": "metric sensitivity diagnostic; no new mechanism",
            "evaluation_status": "completed_weighting_sensitivity_no_promotion",
            "evidence_path": "round59_coordinate_balanced_metrics/report.md",
            "notes": (
                "Coordinate balancing changes no Regret@1 value, shifts RMSE by at most 0.000632 BPB, and leaves "
                "both target-specific RMSE winners unchanged."
            ),
        },
    )
    rows = []
    for definition in definitions:
        exists = ledger["round_id"].eq(definition["round_id"]) & ledger["candidate_id"].eq(definition["candidate_id"])
        if exists.any():
            continue
        rows.append(
            {
                "timestamp": timestamp,
                **definition,
                "adversarial_outcomes_available_before_proposal": True,
                "adversarial_outcomes_inspected_before_proposal": True,
            }
        )
    if not rows:
        print("round-58/59 diagnostics already recorded")
        return
    updated = pd.concat([ledger, pd.DataFrame(rows, columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended {len(rows)} round-58/59 rows to {LEDGER}")


if __name__ == "__main__":
    main()
