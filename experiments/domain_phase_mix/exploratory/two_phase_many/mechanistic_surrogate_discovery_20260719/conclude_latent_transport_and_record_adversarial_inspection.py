# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Reject joint latent phase transport and record exposed-panel inspection."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def main() -> None:
    registry = pd.read_csv(REGISTRY)
    ledger = pd.read_csv(LEDGER)

    registry.loc[registry["id"].eq("JLPT"), ["status", "status_evidence"]] = [
        "blocked_before_adversarial",
        "Rejected at the matched-coordinate identification gate without StarCoder, historical-heldout, or "
        "candidate-level adversarial evaluation. Joint nested OOF improves over a zero phase correction by "
        "14.1-30.9% across four outputs, but full rank is selected in three of five folds and rank two is "
        "statistically tied with rank three. More importantly, independently fitted PMVT/fast-slow phase laws "
        "have lower phase-delta RMSE on every output: 0.00887/0.01853 versus 0.01242/0.02573 at 300M and "
        "0.00661/0.02117 versus 0.00836/0.02315 at Delphi. The shared latent transport direction therefore "
        "does not provide a stronger or lower-dimensional identification argument.",
    ]

    now = datetime.now(UTC).isoformat()
    rows = [
        {
            "timestamp": now,
            "round_id": "round_8_identification_rejection",
            "candidate_id": "JLPT",
            "candidate_family": "Joint latent phase transport",
            "hyperparameters": "Frozen rank={1,2,3}, remaining-offset, ridge, and nonnegative contrast-cost grid",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "Cross-target and cross-scale phase-delta correlations on matched fit coordinates.",
            "novelty_class": "Joint multi-output identification of a shared phase-displacement subspace",
            "evaluation_status": "rejected at matched-coordinate identification gate; no candidate-level adversarial evaluation",
            "evidence_path": "round8_joint_latent_phase_transport/report.md",
            "notes": "The low-rank restriction is not selected over full rank and all four outputs lose to existing independently fitted phase laws.",
        },
        {
            "timestamp": now,
            "round_id": "adversarial_mechanism_inspection_batch_1",
            "candidate_id": "diagnostic_only",
            "candidate_family": "Exposed adversarial mechanism inspection",
            "hyperparameters": "No candidate fit; descriptive rank correlations and physical-feature diagnostics only",
            "adversarial_outcomes_available_before_proposal": True,
            "adversarial_outcomes_inspected_before_proposal": True,
            "observations_inspiring_mechanism": "Target-matched outcomes, policy class, selection stratum, proposer origin, aggregate/phase concentration, support distance, simulated epochs, and existing baseline residuals were inspected.",
            "novelty_class": "Development-panel diagnosis, not a surrogate",
            "evaluation_status": "inspection complete; no new candidate predictions evaluated",
            "evidence_path": "data_use_ledger.csv",
            "notes": "High-disagreement strata are harder. Coarse physical summaries explain much of Uncheatable variation but less Table-9 variation; domain correlations are proposal-series confounded. This inspection does not justify a universal concentration penalty or any output calibrator.",
        },
    ]
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    additions = pd.DataFrame(rows, columns=ledger.columns)
    additions = additions.loc[
        [tuple(value) not in existing for value in additions[identity].itertuples(index=False, name=None)]
    ]
    ledger = pd.concat([ledger, additions], ignore_index=True)

    registry.to_csv(REGISTRY, index=False)
    ledger.to_csv(LEDGER, index=False)
    print(registry.loc[registry["id"].eq("JLPT"), ["id", "family", "status"]].to_string(index=False))
    print(ledger.tail(len(additions))[["round_id", "candidate_id", "evaluation_status"]].to_string(index=False))


if __name__ == "__main__":
    main()
