# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Append explicit reconciliation rows for incomplete historical ledger edges."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"

RECONCILIATION = {
    "FCF": "round1_paired_dynamics/paired_screen_metrics.csv; round1_starcoder_shape/surface_oof_metrics.csv",
    "IFSC": "round1_starcoder_shape_refined107/surface_oof_metrics.csv",
    "PWD": "round2_potential_phase/report.md; round2_potential_starcoder/report.md",
    "ESR": "round2_potential_phase/report.md; round2_potential_starcoder/report.md",
    "FSC": "round25_shared_private_starcoder/report.md",
    "FPCGF": "round25_shared_private_starcoder/report.md",
    "FSCR": "round26_cascade_replay_starcoder/report.md",
}


def main() -> None:
    registry = pd.read_csv(REGISTRY).fillna("")
    ledger = pd.read_csv(LEDGER).fillna("")
    timestamp = datetime.now(UTC).isoformat()
    rows = []
    for route_id, evidence_path in RECONCILIATION.items():
        key = f"{route_id}:historical_ledger_edge"
        exists = ledger["round_id"].eq("round_60_ledger_reconciliation") & ledger["candidate_id"].eq(key)
        if exists.any():
            continue
        route = registry.loc[registry["id"].eq(route_id)]
        if len(route) != 1:
            raise ValueError(f"Expected one registry row for {route_id}, found {len(route)}")
        record = route.iloc[0]
        rows.append(
            {
                "timestamp": timestamp,
                "round_id": "round_60_ledger_reconciliation",
                "candidate_id": key,
                "candidate_family": record["family"],
                "hyperparameters": "No model or hyperparameter change; registry-to-ledger provenance repair only",
                "adversarial_outcomes_available_before_proposal": True,
                "adversarial_outcomes_inspected_before_proposal": True,
                "observations_inspiring_mechanism": (
                    "A final append-only audit found that this historical route had registry evidence but only one "
                    "explicit ledger edge. This row records the missing provenance without changing historical chronology."
                ),
                "novelty_class": "ledger provenance repair; no mechanism or evaluation change",
                "evaluation_status": "historical_ledger_edge_reconciled_no_model_evaluation",
                "evidence_path": evidence_path,
                "notes": f"Terminal status remains {record['status']}: {record['status_evidence']}",
            }
        )
    if not rows:
        print("round-60 reconciliation already recorded")
        return
    updated = pd.concat([ledger, pd.DataFrame(rows, columns=ledger.columns)], ignore_index=True)
    candidate_rows = updated["candidate_id"].astype(str).str.strip().ne("")
    if updated.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger reconciliation would introduce a duplicate round/candidate key")
    updated.to_csv(LEDGER, index=False)
    print(f"appended {len(rows)} round-60 reconciliation rows to {LEDGER}")


if __name__ == "__main__":
    main()
