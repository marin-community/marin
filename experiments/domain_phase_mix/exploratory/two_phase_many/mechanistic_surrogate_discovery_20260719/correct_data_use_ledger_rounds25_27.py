# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Append schema-corrected ledger rows for frozen rounds 25--27."""

from __future__ import annotations

import csv
from datetime import UTC, datetime
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LEDGER = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/data_use_ledger.csv"


def corrected_rows(timestamp: str) -> list[dict[str, str]]:
    return [
        {
            "timestamp": timestamp,
            "round_id": "round_25_batch_preregistration_corrected",
            "candidate_id": "FSC,FPCGF",
            "candidate_family": "Shared/private capability batch",
            "hyperparameters": "Frozen grids recorded in register_shared_private_batch_round25.py",
            "adversarial_outcomes_available_before_proposal": "True",
            "adversarial_outcomes_inspected_before_proposal": "True",
            "observations_inspiring_mechanism": (
                "The refined WSD optimum is broad-early and rare-late; prior scalar retained-exposure and optimizer-memory routes miss that geometry."
            ),
            "novelty_class": "Separate foundation and specialist latent states",
            "evaluation_status": "batch frozen before StarCoder evaluation",
            "evidence_path": "round25_shared_private_starcoder/report.md",
            "notes": "Corrects the sparse schema-mismatched row; no outcome or decision changed.",
        },
        {
            "timestamp": timestamp,
            "round_id": "round_26_batch_preregistration_corrected",
            "candidate_id": "FSCR",
            "candidate_family": "Foundation-specialization cascade with literal replay",
            "hyperparameters": "Frozen FSC grid plus exact physical replay features",
            "adversarial_outcomes_available_before_proposal": "True",
            "adversarial_outcomes_inspected_before_proposal": "True",
            "observations_inspiring_mechanism": (
                "FSC selected maximum specialist acquisition yet could not price StarCoder-only BPB above 2.5; exact finite-subset replay was omitted."
            ),
            "novelty_class": "Exact cumulative traversal beyond first exposure",
            "evaluation_status": "batch frozen before StarCoder evaluation",
            "evidence_path": "round26_cascade_replay_starcoder/report.md",
            "notes": "Corrects the sparse schema-mismatched row; no outcome or decision changed.",
        },
        {
            "timestamp": timestamp,
            "round_id": "round_27_batch_preregistration_corrected",
            "candidate_id": "PLSC,PLAFK",
            "candidate_family": "Power-law error-kinetics batch",
            "hyperparameters": "Frozen grids recorded in register_power_law_error_batch_round27.py",
            "adversarial_outcomes_available_before_proposal": "True",
            "adversarial_outcomes_inspected_before_proposal": "True",
            "observations_inspiring_mechanism": (
                "Round-25/26 exponential competence fits selected maximum specialist rates; scaling-law evidence motivates power-law learning-error kinetics."
            ),
            "novelty_class": "Power-law latent-error transition with exact exponential ablation",
            "evaluation_status": "batch frozen before StarCoder evaluation",
            "evidence_path": "round27_power_law_error_starcoder/report.md",
            "notes": "Corrects the sparse schema-mismatched row; no outcome or decision changed.",
        },
        {
            "timestamp": timestamp,
            "round_id": "round_27_starcoder_gate",
            "candidate_id": "PLSC",
            "candidate_family": "Power-law foundation-specialization kinetics",
            "hyperparameters": "Frozen Round-27 grid with nested StarCoder selection",
            "adversarial_outcomes_available_before_proposal": "True",
            "adversarial_outcomes_inspected_before_proposal": "True",
            "observations_inspiring_mechanism": "See round_27_batch_preregistration_corrected.",
            "novelty_class": "Power-law foundation and prerequisite-gated specialist errors",
            "evaluation_status": "blocked_before_multi_swarm",
            "evidence_path": "round27_power_law_error_starcoder/report.md",
            "notes": (
                "Both schedules select zeta=0; WSD selects nu=0; specialist rates hit the upper boundary; "
                "nested RMSE=0.173559/0.100946 and cosine raw-optimum distance=0.228020."
            ),
        },
        {
            "timestamp": timestamp,
            "round_id": "round_27_starcoder_gate",
            "candidate_id": "PLAFK",
            "candidate_family": "Power-law acquisition-forgetting kinetics",
            "hyperparameters": "Frozen Round-27 grid with nested StarCoder selection",
            "adversarial_outcomes_available_before_proposal": "True",
            "adversarial_outcomes_inspected_before_proposal": "True",
            "observations_inspiring_mechanism": "See round_27_batch_preregistration_corrected.",
            "novelty_class": "Power-law acquisition with broad-induced specialist forgetting",
            "evaluation_status": "blocked_before_multi_swarm",
            "evidence_path": "round27_power_law_error_starcoder/report.md",
            "notes": (
                "Both schedules select zeta=0; specialist rates hit the upper boundary; nested "
                "RMSE=0.173302/0.104080 and cosine raw-optimum distance=0.229157."
            ),
        },
    ]


def main() -> None:
    with LEDGER.open(newline="") as source:
        reader = csv.DictReader(source)
        existing = list(reader)
        fields = reader.fieldnames
    if fields is None:
        raise RuntimeError("Ledger has no header")
    known = {(row["round_id"], row["candidate_id"]) for row in existing}
    rows = [
        row
        for row in corrected_rows(datetime.now(UTC).isoformat())
        if (row["round_id"], row["candidate_id"]) not in known
    ]
    if not rows:
        return
    with LEDGER.open("a", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writerows(rows)


if __name__ == "__main__":
    main()
