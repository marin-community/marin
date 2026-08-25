# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas"]
# ///
"""Append the frozen round-54/55 diagnostics to the data-use ledger."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
MODEL_IDS = (
    "linear",
    "olmix_loglinear",
    "canonical",
    "effective_exposure",
    "effective_exposure_geometry",
    "separate_heads",
    "grp",
    "compact_retained_state",
    "bucket_family_grp",
    "hierarchical_phase_bucket_replay",
    "bucket_family_power_separate_heads",
)
TARGETS = ("uncheatable", "table9")
FIT_MODES = ("independent_one_phase_refit", "algebraic_restriction_of_two_phase_fit")


def row_key(row: pd.Series | dict[str, object]) -> tuple[str, str]:
    return str(row["round_id"]), str(row["candidate_id"])


def main() -> None:
    ledger = pd.read_csv(LEDGER)
    existing = {row_key(row) for _, row in ledger.iterrows()}
    timestamp = datetime.now(UTC).isoformat()
    rows: list[dict[str, object]] = []

    for target in TARGETS:
        for model in MODEL_IDS:
            for fit_mode in FIT_MODES:
                candidate_id = f"{target}:{model}:{fit_mode}"
                if ("round_54_one_phase_restriction", candidate_id) in existing:
                    continue
                rows.append(
                    {
                        "timestamp": timestamp,
                        "round_id": "round_54_one_phase_restriction",
                        "candidate_id": candidate_id,
                        "candidate_family": model,
                        "hyperparameters": (
                            "Existing Observatory selection procedure refit on 238 one-phase rows"
                            if fit_mode == "independent_one_phase_refit"
                            else "Existing frozen two-phase fit evaluated algebraically on tied inputs"
                        ),
                        "adversarial_outcomes_available_before_proposal": True,
                        "adversarial_outcomes_inspected_before_proposal": True,
                        "observations_inspiring_mechanism": (
                            "Required audit distinguishing an algebraically tied restriction from an independently "
                            "fitted one-phase model; no new mechanism proposed."
                        ),
                        "novelty_class": "diagnostic refit of an exposed existing family",
                        "evaluation_status": "completed_diagnostic_no_promotion",
                        "evidence_path": "round54_single_phase_refit/report.md",
                        "notes": (
                            f"Target={target}; mode={fit_mode}. Fit protocol and hyperparameter procedure were frozen "
                            "before this batch evaluation. Results are development evidence only."
                        ),
                    }
                )

    for target in TARGETS:
        for model in ("separate_heads", "hierarchical_phase_bucket_replay"):
            candidate_id = f"{target}:{model}:low_tail_influence"
            if ("round_55_low_tail_influence", candidate_id) in existing:
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "round_id": "round_55_low_tail_influence",
                    "candidate_id": candidate_id,
                    "candidate_family": model,
                    "hyperparameters": "Frozen full-panel structural setting; delete k in {0,1,3,7,14}; refit response head only",
                    "adversarial_outcomes_available_before_proposal": True,
                    "adversarial_outcomes_inspected_before_proposal": True,
                    "observations_inspiring_mechanism": (
                        "Collaborator concern that a small noisy low-loss tail could dominate the fitted surface; "
                        "diagnostic only, not a trimming proposal."
                    ),
                    "novelty_class": "influence diagnostic; no new mechanism",
                    "evaluation_status": "completed_structural_sensitivity_no_promotion",
                    "evidence_path": "round55_low_tail_influence/report.md",
                    "notes": (
                        f"Target={target}. Tail deletion did not remove archive optimism or selected regret. Raw-optimum "
                        "coordinates remain qualified by the separately preregistered convergence follow-up."
                    ),
                }
            )

    convergence_path = OUTPUT_ROOT / "round55_low_tail_influence/numerical_followup/convergence_summary.csv"
    if convergence_path.exists():
        convergence = pd.read_csv(convergence_path)
        for result in convergence.itertuples(index=False):
            candidate_id = f"{result.target}:{result.model}:excluded_{int(result.excluded_count)}:convergence"
            if ("round_55b_low_tail_optimum_numerics", candidate_id) in existing:
                continue
            rows.append(
                {
                    "timestamp": timestamp,
                    "round_id": "round_55b_low_tail_optimum_numerics",
                    "candidate_id": candidate_id,
                    "candidate_family": result.model,
                    "hyperparameters": (
                        "Frozen round-55 fit and deletion; 15 preregistered L-BFGS-B starts; maxiter=2000; maxfun=250000"
                    ),
                    "adversarial_outcomes_available_before_proposal": True,
                    "adversarial_outcomes_inspected_before_proposal": True,
                    "observations_inspiring_mechanism": (
                        "Numerical convergence warnings in the frozen round-55 raw-optimum audit; "
                        "no mechanism or scientific threshold changed."
                    ),
                    "novelty_class": "convergence-only diagnostic; no new mechanism",
                    "evaluation_status": "completed_numerical_followup_no_promotion",
                    "evidence_path": "round55_low_tail_influence/numerical_followup/report.md",
                    "notes": (
                        f"Target={result.target}; excluded_count={int(result.excluded_count)}; "
                        f"successful_starts={int(result.successful_starts)}/{int(result.total_starts)}; "
                        f"objective_delta={float(result.objective_delta):.6g}; "
                        f"policy_l1_shift={float(result.l1_from_original_optimum):.6g}."
                    ),
                }
            )

    if not rows:
        print("ledger already contains all round-54/55 rows")
        return
    updated = pd.concat([ledger, pd.DataFrame(rows, columns=ledger.columns)], ignore_index=True)
    if updated[["round_id", "candidate_id"]].duplicated().any():
        raise ValueError("Ledger update would introduce duplicate round/candidate keys")
    updated.to_csv(LEDGER, index=False)
    print(f"appended {len(rows)} rows to {LEDGER}")


if __name__ == "__main__":
    main()
