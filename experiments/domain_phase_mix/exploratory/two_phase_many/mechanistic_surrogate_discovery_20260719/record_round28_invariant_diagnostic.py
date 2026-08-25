# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Record the descriptive aggregate/contrast basis audit in the append-only ledger."""

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
    row = {
        "id": "ACPD",
        "family": "Aggregate-contrast polynomial diagnostic",
        "relationship_to_prior": (
            "Descriptive invariant basis related to the rejected PMVT aggregate-plus-contrast decomposition; "
            "not an admissible mechanistic surrogate."
        ),
        "materially_new_mechanism": "None; this is a shape diagnostic over aggregate exposure and phase contrast.",
        "mechanistic_premise": (
            "If StarCoder geometry is simple in aggregate exposure a and phase contrast d, low-order coefficients can "
            "suggest a response law worth deriving independently."
        ),
        "governing_equations": "Y=sum_{r+s<=k} beta_rs a^r d^s with ridge-selected degree k.",
        "latent_state": "None.",
        "state_transition": "None; direct descriptive regression.",
        "response_link": "Unconstrained polynomial; therefore inadmissible as a headline model.",
        "additional_degrees_of_freedom": "All polynomial terms through cross-validated degree 1-6.",
        "units_and_symmetries": "a and d are dimensionless; coefficients have BPB units and degree-dependent scaling.",
        "single_phase_restriction": "d=0 leaves a univariate polynomial in aggregate exposure.",
        "starcoder_signature": (
            "Alternating coefficient signs should reveal whether a reciprocal or saturating learning-curve expansion is plausible."
        ),
        "catastrophic_optimism_resolution": "None; unconstrained polynomials can extrapolate catastrophically.",
        "response_compression_resolution": "None; diagnostic only.",
        "scale_transfer_expectation": "No transfer claim.",
        "cheapest_falsification": "A low OOF error paired with a remote raw optimum demonstrates nonmechanistic extrapolation.",
        "status": "descriptive_only_not_admissible",
        "status_evidence": (
            "Nested RMSE was 0.03895 on cosine and 0.04977 on WSD, but raw optimum distance was 0.384 and 0.214; "
            "the basis diagnoses shape without supporting optimization."
        ),
    }
    registry = registry.loc[~registry["id"].eq("ACPD")]
    registry = pd.concat([registry, pd.DataFrame([row], columns=registry.columns)], ignore_index=True)
    registry.to_csv(REGISTRY, index=False)

    ledger = pd.read_csv(LEDGER)
    ledger_row = {
        "timestamp": datetime.now(UTC).isoformat(),
        "round_id": "round_28_descriptive_diagnostic",
        "candidate_id": "ACPD",
        "candidate_family": "Aggregate-contrast polynomial diagnostic",
        "hyperparameters": "Nested degree={1,...,6}, ridge={1e-6,...,1}; descriptive only",
        "adversarial_outcomes_available_before_proposal": True,
        "adversarial_outcomes_inspected_before_proposal": True,
        "observations_inspiring_mechanism": (
            "Repeated mechanistic failures motivated a coordinate audit in aggregate exposure and phase contrast."
        ),
        "novelty_class": "Descriptive invariant basis; no mechanistic claim",
        "evaluation_status": "descriptive_only_not_admissible",
        "evidence_path": "round28_starcoder_invariant_basis/report.md",
        "notes": (
            "No Delphi or adversarial predictions were evaluated. Low nested RMSE did not prevent remote raw optima."
        ),
    }
    identity = ["round_id", "candidate_id", "evaluation_status"]
    existing = set(map(tuple, ledger[identity].itertuples(index=False, name=None)))
    if tuple(ledger_row[column] for column in identity) not in existing:
        ledger = pd.concat([ledger, pd.DataFrame([ledger_row], columns=ledger.columns)], ignore_index=True)
        ledger.to_csv(LEDGER, index=False)


if __name__ == "__main__":
    main()
