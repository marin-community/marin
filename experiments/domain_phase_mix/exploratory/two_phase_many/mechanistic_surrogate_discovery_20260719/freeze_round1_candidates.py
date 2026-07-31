# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2"]
# ///
"""Freeze round-one candidates before historical or adversarial evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND1 = DEFAULT_OUTPUT / "round1_paired_dynamics"
STARCODER = DEFAULT_OUTPUT / "round1_starcoder_shape_refined"
PRODUCTION = DEFAULT_OUTPUT / "round1_production_transfer"
REGISTRY_PATH = DEFAULT_OUTPUT / "approach_registry.csv"
LEDGER_PATH = DEFAULT_OUTPUT / "data_use_ledger.csv"
ROUND_ID = "round_1_nonadversarial_freeze"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT / "round1_candidate_freeze")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def input_manifest() -> dict[str, str]:
    paths = (
        ROUND1 / "selected_configs.csv",
        ROUND1 / "paired_screen_metrics.csv",
        ROUND1 / "phase_hyperparameter_grid.csv",
        STARCODER / "surface_oof_metrics.csv",
        STARCODER / "leave_region_out_metrics.csv",
        STARCODER / "predicted_optima.csv",
        PRODUCTION / "selected_configs_and_metrics.csv",
    )
    return {str(path.relative_to(RESEARCH_DIR)): sha256(path) for path in paths}


def candidate_definitions() -> dict[str, dict[str, Any]]:
    aggregate = {"shortage_power": 0.25, "shortage_offset": 1.0, "l2": 1.0}
    return {
        "paired_marginal_value_transport": {
            "status": "frozen_for_historical_gate",
            "equation": ("Y(w0,w1)=F(a)+sum_i theta_i m_i(a)d_i+sum_f chi_f sum_{i in f} p(i|f)[m_i(a)d_i]^2"),
            "definitions": {
                "a": "alpha0*w0+alpha1*w1",
                "d": "alpha0*alpha1*(w1-w0)/p",
                "m": "1/(2+E(a)/E_proportional)",
            },
            "mechanisms": (
                "signed transport of a fixed aggregate allocation toward the terminal phase; "
                "nonnegative family curvature prices phase-contrast magnitude"
            ),
            "aggregate_config": aggregate,
            "target_configs": {
                "uncheatable": {
                    "remaining_offset": 2.0,
                    "l2": 0.1,
                    "transport_level": "bucket",
                    "mismatch_level": "family",
                    "include_signed_transport": True,
                    "include_quadratic_mismatch": True,
                },
                "table9": {
                    "remaining_offset": 2.0,
                    "l2": 1.0,
                    "transport_level": "bucket",
                    "mismatch_level": "family",
                    "include_signed_transport": True,
                    "include_quadratic_mismatch": True,
                },
            },
            "nominal_parameter_count": "1 intercept + 42 aggregate + 39 signed transport + 3 family curvature",
            "selection_evidence": (
                "The bucket-transport/family-curvature form is the cross-scale stable PMVT variant; "
                "both terms improve paired-delta CV on all four 300M/Delphi target panels."
            ),
            "known_blocker": (
                "StarCoder leave-region RMSE is 0.41/0.57 on cosine/WSD, so the local transport expansion "
                "does not extrapolate across the full two-domain phase surface."
            ),
        },
        "terminal_equilibrium_adaptation": {
            "status": "frozen_for_historical_gate",
            "equation": ("Y(w0,w1)=F(a)-sum_i b_i[phi_r(w1_i)-phi_r(a_i)], phi_r(w)=w/(w+r(1-w)), b_i>=0"),
            "definitions": {"a": "alpha0*w0+alpha1*w1", "phi": "terminal equilibrium competence"},
            "mechanisms": (
                "terminal competence relaxes rapidly to the equilibrium induced by the final mixture; "
                "the aggregate spine separately prices total exposure and literal replay"
            ),
            "aggregate_config": aggregate,
            "target_configs": {
                "uncheatable": {"saturation_ratio": 0.0625, "l2": 1.0},
                "table9": {"saturation_ratio": 0.25, "l2": 1.0},
            },
            "nominal_parameter_count": "1 intercept + 42 aggregate + 39 nonnegative terminal-capability amplitudes",
            "selection_evidence": (
                "The quasi-steady consolidation model selected complete terminal consolidation on every paired panel; "
                "this is its identifiable limiting form and matches or improves paired CV with two fewer nonlinear parameters."
            ),
            "known_blocker": (
                "StarCoder OOF remains above existing surface baselines and leave-region RMSE is 0.13/0.22; "
                "the rapid-terminal-equilibrium assumption may be too strong away from the sampled phase fibers."
            ),
        },
    }


def registry_row_terminal() -> dict[str, str]:
    return {
        "id": "TEA",
        "family": "Terminal-equilibrium adaptation",
        "relationship_to_prior": (
            "Identifiable boundary limit of round-one quasi-steady consolidation, itself a paired-identification "
            "reopening of prior route AL."
        ),
        "materially_new_mechanism": (
            "The phase correction is the change in a normalized terminal equilibrium state rather than a free "
            "phase head; paired counterfactuals identify its response amplitudes independently of aggregate exposure."
        ),
        "mechanistic_premise": (
            "When fast adaptation is much quicker than normalized training progress, terminal capability reflects "
            "the equilibrium induced by the final mixture, while total exposure remains an independent state."
        ),
        "governing_equations": (
            "a=alpha0*w0+alpha1*w1; phi_r(w)=w/[w+r(1-w)]; Y=F_1p(a)-sum_i b_i[phi_r(w1_i)-phi_r(a_i)], b_i>=0."
        ),
        "latent_state": "Per-bucket dimensionless terminal equilibrium competence phi_r(w).",
        "state_transition": (
            "Fast competence relaxes to phi_r(w) within each constant-mixture phase; the retained limit uses the "
            "terminal phase equilibrium after paired CV sends the consolidation rate to its fast boundary."
        ),
        "response_link": "Nonnegative capability amplitudes reduce BPB; aggregate deficit and replay remain in F_1p.",
        "additional_degrees_of_freedom": (
            "One target-specific saturation ratio selected on grouped paired CV and one nonnegative amplitude per bucket."
        ),
        "units_and_symmetries": (
            "w, r, and phi are dimensionless; b has BPB units. The correction is exactly zero for tied phases. "
            "The fixed normalization removes state/amplitude scale symmetry."
        ),
        "single_phase_restriction": (
            "w0=w1=a makes the terminal correction zero and leaves the independently fitted aggregate spine exactly."
        ),
        "starcoder_signature": (
            "A recency-tilted Nike swoosh whose final-phase arm saturates; the terminal equilibrium shifts between "
            "cosine 50/50 and WSD 80/20 only through the fitted saturation ratio."
        ),
        "catastrophic_optimism_resolution": (
            "Aggregate underexposure cannot be canceled by terminal adaptation because the two mechanisms are "
            "estimated from orthogonal matched-policy contrasts."
        ),
        "response_compression_resolution": (
            "The independently fitted aggregate spine preserves one-phase range and the saturating terminal state "
            "adds measured phase variation without a free output calibrator."
        ),
        "scale_transfer_expectation": (
            "The equilibrium ratio is dimensionless and should transfer if fast adaptation is schedule-normalized; "
            "failure is expected if adaptation time is determined by absolute tokens."
        ),
        "cheapest_falsification": (
            "Historical 3e18 calibration remains compressed, StarCoder leave-region error stays above the Pareto "
            "baseline, or the selected saturation ratio is bootstrap-unstable."
        ),
        "status": "active_frozen_historical_gate",
        "status_evidence": (
            "Paired-delta RMSE 0.00882/0.01787 at 300M and 0.00658/0.02117 at Delphi; production OOF RMSE "
            "0.00818. StarCoder remains a known transfer failure. Frozen before historical/adversarial evaluation."
        ),
    }


def update_registry() -> None:
    registry = pd.read_csv(REGISTRY_PATH).fillna("")
    updates = {
        "PMVT": (
            "active_frozen_historical_gate",
            "Paired-delta RMSE improves over zero on all four paired panels and production OOF RMSE is 0.00877, "
            "but StarCoder leave-region RMSE is 0.41/0.57. Frozen before historical/adversarial evaluation.",
        ),
        "FCF": (
            "blocked_round1",
            "Paired-delta RMSE is 0.01430/0.02588 at 300M and 0.01117/0.02624 at Delphi; the commutator barely "
            "beats zero on Delphi and does not resolve phase response.",
        ),
        "IFSC": (
            "blocked_boundary_round1",
            "Bucket-state fits improve paired deltas, but acquisition hits the screened maximum on three of four "
            "panels and the quasi-steady consolidation rate hits its maximum on all four; separate rates are not identified.",
        ),
    }
    for candidate_id, (status, evidence) in updates.items():
        mask = registry["id"].eq(candidate_id)
        if int(mask.sum()) != 1:
            raise ValueError(f"Expected exactly one registry row for {candidate_id}")
        registry.loc[mask, "status"] = status
        registry.loc[mask, "status_evidence"] = evidence
    terminal = registry_row_terminal()
    if registry["id"].eq("TEA").any():
        for column, value in terminal.items():
            registry.loc[registry["id"].eq("TEA"), column] = value
    else:
        registry = pd.concat([registry, pd.DataFrame([terminal])], ignore_index=True)
    registry.to_csv(REGISTRY_PATH, index=False)


def update_ledger(freeze_path: Path) -> None:
    ledger = pd.read_csv(LEDGER_PATH).fillna("")
    ledger = ledger.loc[~ledger["round_id"].eq(ROUND_ID)].copy()
    timestamp = datetime.now(UTC).isoformat()
    rows = []
    for candidate_id, family in (
        ("PMVT", "Paired marginal-value transport"),
        ("TEA", "Terminal-equilibrium adaptation"),
    ):
        rows.append(
            {
                "timestamp": timestamp,
                "round_id": ROUND_ID,
                "candidate_id": candidate_id,
                "candidate_family": family,
                "hyperparameters": "Frozen in round1_candidate_freeze/candidate_freeze.json",
                "adversarial_outcomes_available_before_proposal": True,
                "adversarial_outcomes_inspected_before_proposal": True,
                "observations_inspiring_mechanism": (
                    "Matched 300M/Delphi one-vs-two-phase contrasts, StarCoder surfaces, production OOF, and prior "
                    "aggregate response-compression diagnosis; no row-level adversarial value selected this form."
                ),
                "novelty_class": "New paired identification law and mechanistic phase state",
                "evaluation_status": "Frozen before historical and adversarial round-one evaluation",
                "evidence_path": str(freeze_path.relative_to(DEFAULT_OUTPUT)),
                "notes": (
                    "All adversarial outcomes are exposed development evidence. This freeze prevents historical or "
                    "adversarial retuning; any eventual recommendation remains provisional."
                ),
            }
        )
    ledger = pd.concat([ledger, pd.DataFrame(rows)], ignore_index=True)
    ledger.to_csv(LEDGER_PATH, index=False)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = input_manifest()
    freeze = {
        "created_at": datetime.now(UTC).isoformat(),
        "round_id": ROUND_ID,
        "historical_outcomes_used_for_selection": False,
        "adversarial_outcomes_used_for_selection": False,
        "confirmatory_phase_fiber_outcomes_used": False,
        "candidate_definitions": candidate_definitions(),
        "input_sha256": manifest,
    }
    canonical = json.dumps(freeze, sort_keys=True, separators=(",", ":"))
    freeze["freeze_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    freeze_path = args.output_dir / "candidate_freeze.json"
    freeze_path.write_text(json.dumps(freeze, indent=2, sort_keys=True) + "\n")
    update_registry()
    update_ledger(freeze_path)
    report = f"""# Round-one candidate freeze

Freeze SHA-256: `{freeze["freeze_sha256"]}`

This artifact was written before loading the 352 historical heldout targets or evaluating either candidate on the 120-row adversarial development panel. The untouched frontier phase-fiber panel remains forbidden.

## Frozen candidates

- **Paired marginal-value transport (PMVT):** bucket-level signed transport plus family-level nonnegative contrast curvature. It improves matched-policy phase deltas across both scales, but fails StarCoder leave-region transfer.
- **Terminal-equilibrium adaptation (TEA):** the identifiable limiting case selected when the more general quasi-steady consolidation model drives its consolidation rate to the upper boundary. It is simpler and at least as accurate as the boundary model, but also remains weaker than the StarCoder Pareto baselines.

## Blocked routes

- **Family commutator flow:** too weak on paired-difference CV, especially at Delphi.
- **Full identified fast/slow consolidation:** nonlinear rates remain boundary-selected; the separate timescales are not identified.
- **Quasi-steady consolidation:** complete consolidation is selected on every paired panel, so its extra consolidation-rate and slow-share parameters are unsupported.
"""
    (args.output_dir / "report.md").write_text(report)
    print(json.dumps({"freeze_path": str(freeze_path), "freeze_sha256": freeze["freeze_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
