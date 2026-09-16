# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Record the frozen round-one rejection without evaluating adversarial outcomes."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY_PATH = OUTPUT / "approach_registry.csv"
LEDGER_PATH = OUTPUT / "data_use_ledger.csv"
ROUND_ID = "round_1_historical_rejection"


EVIDENCE = {
    "PMVT": (
        "blocked_before_adversarial",
        "Frozen PMVT was rejected without adversarial evaluation. Historical 3e18 RMSE is 0.01658/0.02825 "
        "for Uncheatable/Table-9, calibration slopes are 0.836/0.727, and Regret@1 is 0.00626/0.02467. "
        "It leaves 2/6 errors above 0.05 BPB. StarCoder leave-region RMSE is 0.412/0.567. Cross-scale "
        "phase transfer compresses Delphi response (slope 0.466 Uncheatable, 0.339 Table-9). The model "
        "reduces the worst historical error but fails shape, calibration, and decision gates.",
    ),
    "TEA": (
        "blocked_before_adversarial",
        "Frozen terminal-equilibrium adaptation was rejected without adversarial evaluation. Historical 3e18 RMSE is "
        "0.01659/0.03233 for Uncheatable/Table-9, calibration slopes are 0.852/0.696, and it leaves 3/24 errors "
        "above 0.05 BPB. StarCoder leave-region RMSE is 0.130/0.217. Cross-scale phase transfer compresses "
        "Delphi response (slope 0.510 Uncheatable, 0.296 Table-9). Its fast-equilibrium limit is identifiable, "
        "but the transition law does not transfer.",
    ),
}


def main() -> None:
    registry = pd.read_csv(REGISTRY_PATH).fillna("")
    for candidate, (status, evidence) in EVIDENCE.items():
        mask = registry["id"].eq(candidate)
        if int(mask.sum()) != 1:
            raise ValueError(f"Expected one registry row for {candidate}")
        registry.loc[mask, "status"] = status
        registry.loc[mask, "status_evidence"] = evidence
    registry.to_csv(REGISTRY_PATH, index=False)

    ledger = pd.read_csv(LEDGER_PATH).fillna("")
    timestamp = datetime.now(UTC).isoformat()
    rows = []
    for candidate in EVIDENCE:
        if ((ledger["round_id"] == ROUND_ID) & (ledger["candidate_id"] == candidate)).any():
            continue
        rows.append(
            {
                "timestamp": timestamp,
                "round_id": ROUND_ID,
                "candidate_id": candidate,
                "candidate_family": registry.loc[registry["id"].eq(candidate), "family"].iloc[0],
                "hyperparameters": "Frozen in round1_candidate_freeze/candidate_freeze.json",
                "adversarial_outcomes_available_before_proposal": True,
                "adversarial_outcomes_inspected_before_proposal": True,
                "observations_inspiring_mechanism": (
                    "Matched 300M/Delphi phase contrasts and non-adversarial StarCoder/production evidence."
                ),
                "novelty_class": "round-one frozen candidate",
                "evaluation_status": "rejected at historical and cross-scale gates; adversarial panel not evaluated",
                "evidence_path": "round1_historical_heldouts/report.md; round1_cross_scale_matched_policy/report.md",
                "notes": (
                    "Historical failures were dominated by joint multi-bucket underexposure, high repetition, and "
                    "distance from support. These exposed observations may inform a materially new next batch, but "
                    "neither candidate may be retuned and re-presented as new."
                ),
            }
        )
    if rows:
        ledger = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
        ledger.to_csv(LEDGER_PATH, index=False)


if __name__ == "__main__":
    main()
