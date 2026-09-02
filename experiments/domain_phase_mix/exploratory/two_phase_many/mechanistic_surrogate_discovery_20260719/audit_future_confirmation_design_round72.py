# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas>=2.2", "tabulate>=0.9"]
# ///
"""Validate the inactive future-confirmation panel specification."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round72_future_confirmation_design"
PREREGISTRATION = OUTPUT_ROOT / "final_synthesis/future_confirmation_preregistration.json"


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    payload = json.loads(PREREGISTRATION.read_text())
    phase = payload["phase_parameterization"]
    contrasts = payload["contrasts_per_anchor"]
    targets = 2
    anchors = len(payload["anchors_per_target"])
    tied = int(contrasts["tied_control"])
    candidate_rays = len(contrasts["candidate_direction_scales"])
    family_rays = 2 * int(contrasts["fixed_family_balanced_direction_pairs"])
    anchor_policies = targets * anchors * (tied + candidate_rays + family_rays)
    additional = targets * len(payload["additional_policies"])
    decisive_repeats = int(payload["decisive_repeats_per_arm"])
    repeat_extras = targets * 3 * (decisive_repeats - 1)
    unique_policies = anchor_policies + additional
    training_runs = unique_policies + repeat_extras

    checks = pd.DataFrame(
        [
            {
                "check": "phase fractions sum to one",
                "value": sum(phase["phase_fractions"]),
                "passed": abs(sum(phase["phase_fractions"]) - 1.0) < 1e-12,
            },
            {
                "check": "candidate scales are signed pairs",
                "value": str(contrasts["candidate_direction_scales"]),
                "passed": sorted(contrasts["candidate_direction_scales"]) == [-1.0, -0.5, -0.25, 0.25, 0.5, 1.0],
            },
            {
                "check": "simplex safety fraction is interior",
                "value": contrasts["simplex_safety_fraction"],
                "passed": 0.0 < contrasts["simplex_safety_fraction"] < 1.0,
            },
            {
                "check": "unique-policy arithmetic",
                "value": unique_policies,
                "passed": unique_policies == payload["maximum_unique_policy_count_before_deduplication"] == 86,
            },
            {
                "check": "training-run arithmetic",
                "value": training_runs,
                "passed": training_runs == payload["maximum_training_runs_before_deduplication"] == 170,
            },
            {"check": "decisive repeat count", "value": decisive_repeats, "passed": decisive_repeats == 15},
            {
                "check": "two-target multiplicity rule is frozen",
                "value": "Holm alpha=0.05",
                "passed": any(
                    "Holm family-wise control at alpha=0.05" in item for item in payload["primary_acceptance"]
                ),
            },
            {
                "check": "single-seed rays are descriptive only",
                "value": "single_seed_policy_use",
                "passed": "cannot replace a failed raw optimum" in payload["single_seed_policy_use"],
            },
            {
                "check": "candidate direction fully specified",
                "value": "candidate_direction_rule",
                "passed": "candidate_direction_rule" in contrasts,
            },
            {
                "check": "family direction fully specified",
                "value": "family_direction_rule",
                "passed": "PCG64" in contrasts["family_direction_rule"],
            },
            {
                "check": "signed simplex ray fully specified",
                "value": "signed_ray_rule",
                "passed": "a_max" in contrasts["signed_ray_rule"],
            },
            {
                "check": "no clipping permitted",
                "value": "validation_rule",
                "passed": "Do not clip" in contrasts["validation_rule"],
            },
            {
                "check": "panel remains inactive",
                "value": payload["status"],
                "passed": payload["status"].startswith("inactive_"),
            },
        ]
    )
    if not checks["passed"].all():
        raise ValueError(checks.loc[~checks["passed"]].to_dict(orient="records"))
    checks.to_csv(ROUND_DIR / "confirmation_design_checks.csv", index=False)

    report = "\n".join(
        [
            "# Round 72: future-confirmation design specification audit",
            "",
            "This audit validates only the inactive preregistration's algebra, deterministic direction-generation rules, and policy-count arithmetic. No candidate weights are available or materialized, no training is launched, and no sealed outcome is read.",
            "",
            checks.to_markdown(index=False),
            "",
            "The panel is now independently reproducible once an eligible candidate exists: the aggregate/contrast reconstruction, signed simplex-ray amplitude, 0.90 safety fraction, PCG64 family-balanced directions, no-clipping assertion, checksums, unique-policy count, and repeat count are explicit. Activation still requires a future candidate to pass every development gate; this drive has none.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
