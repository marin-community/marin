# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = []
# ///
"""Preregister the replay-complete foundation cascade before fitting it."""

from __future__ import annotations

import csv
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"


def append_row(path: Path, row: dict[str, str], id_column: str, row_id: str) -> None:
    with path.open(newline="") as source:
        reader = csv.DictReader(source)
        rows = list(reader)
        fields = reader.fieldnames
    if fields is None:
        raise RuntimeError(f"{path} has no header")
    if any(existing[id_column] == row_id for existing in rows):
        return
    with path.open("a", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fields)
        writer.writerow({name: row.get(name, "") for name in fields})


def main() -> None:
    append_row(
        REGISTRY,
        {
            "id": "FSCR",
            "family": "Foundation-specialization cascade with literal replay",
            "relationship_to_prior": (
                "A nested completion of frozen FSC, not a retuning. It adds exact finite-subset replay, a physical invariant "
                "that FSC omitted. It differs from retained-state route A because specialist acquisition is gated by a "
                "separate cross-domain foundation state."
            ),
            "materially_new_mechanism": (
                "Repeated subset traversals are an independently measured harm state rather than being forced into bounded "
                "competence. The completed response has both bounded unresolved error and unbounded physical replay."
            ),
            "mechanistic_premise": (
                "Foundation competence gates specialist acquisition, while examples beyond one materialized subset traversal "
                "can add repetition harm even after useful competence saturates."
            ),
            "governing_equations": (
                "dg/dt=k_g[(1-p)+rho p](1-g); ds/dt=k_s p g^nu(1-s); R_i=(E_i-1)_+; "
                "Y=b+A_g(1-g_T)+A_s(1-s_T)+sum_i H_i R_i, all amplitudes nonnegative. H=0 is exact FSC."
            ),
            "latent_state": (
                "Bounded foundation competence g, bounded specialist competence s, cumulative physical exposure E_i, and "
                "literal repeated traversal R_i."
            ),
            "state_transition": (
                "Autonomous monotone competence acquisition plus deterministic exposure accounting from simulated epochs."
            ),
            "response_link": (
                "Nonnegative BPB amplitudes on terminal unresolved errors and exact replay counts; no output calibrator."
            ),
            "additional_degrees_of_freedom": (
                "FSC's four nonlinear state parameters plus one nonnegative harm amplitude per observed domain, intercept, "
                "and ridge. The previous FSC fit is the zero-replay nested ablation."
            ),
            "units_and_symmetries": (
                "g and s are dimensionless; E and R are simulated epochs; transition rates are inverse normalized time; "
                "response amplitudes carry BPB per error unit or BPB per replayed epoch. Fixed state normalization removes "
                "continuous scale symmetry."
            ),
            "single_phase_restriction": (
                "Tie the phase policies. The autonomous competence transition composes exactly and total physical exposure "
                "is unchanged by the artificial boundary. Independently refit the same restricted form on tied policies."
            ),
            "starcoder_signature": (
                "The prerequisite rotates WSD toward broad-early/rare-late; literal code replay raises both high-code arms. "
                "The same interior prerequisite/rate regime should fit cosine and WSD without boundary rates."
            ),
            "catastrophic_optimism_resolution": (
                "A rare-heavy policy must pay both missing-foundation error and measured repeat harm, preventing bounded "
                "specialist saturation from making all-code corners look favorable."
            ),
            "response_compression_resolution": (
                "The unbounded replay state expands the high-loss range through sampler physics while the two competence "
                "states retain resolution near the optimum."
            ),
            "scale_transfer_expectation": (
                "Replay depends on simulated epochs and should transfer exactly under the same materialization semantics; "
                "dimensionless competence rates may require a tokens-per-parameter clock law."
            ),
            "cheapest_falsification": (
                "Replay amplitudes collapse, prerequisite nu=0 wins, rates hit boundaries or disagree by over 4x, nested "
                "RMSE misses either corrected StarCoder reference by over 5%, or raw optima miss by over 0.15."
            ),
            "status": "active_frozen_round26",
            "status_evidence": "Frozen after round-25 StarCoder diagnostics and before this completed form was fit.",
        },
        "id",
        "FSCR",
    )
    append_row(
        LEDGER,
        {
            "round_id": "round26_cascade_literal_replay_frozen",
            "timestamp_utc": "2026-07-19T00:00:00Z",
            "model_ids": "FSCR",
            "hyperparameters_frozen_before_adversarial": "true",
            "adversarial_outcomes_inspected": "none for this candidate",
            "observations_inspiring_mechanism": (
                "Round-25 FSC selected maximum specialist acquisition yet could not represent StarCoder-only BPB above 2.5; "
                "the omitted state is exact finite-subset replay already implied by the sampler."
            ),
            "genuinely_new_or_retuning": (
                "new measured response state nested on frozen FSC; not a coefficient retuning and not inspired by "
                "adversarial target values"
            ),
            "data_used_for_selection": "round-25 StarCoder diagnostics and exact epoch coefficients",
            "data_explicitly_not_used": (
                "historical Delphi heldouts, all 120 exposed adversarial policies, and sealed frontier phase-fiber outcomes"
            ),
            "decision": "frozen before first fit",
            "evidence_path": "round26_cascade_replay_starcoder/report.md",
        },
        "round_id",
        "round26_cascade_literal_replay_frozen",
    )


if __name__ == "__main__":
    main()
