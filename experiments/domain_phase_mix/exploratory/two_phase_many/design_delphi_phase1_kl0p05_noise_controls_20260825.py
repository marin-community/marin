# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["pandas"]
# ///
"""Freeze exact-prefix branch-noise controls for the Delphi KL0.05 prefix.

Run from the repository root with::

    uv run experiments/domain_phase_mix/exploratory/two_phase_many/\
design_delphi_phase1_kl0p05_noise_controls_20260825.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONTINUATION_SUMMARY = (
    SCRIPT_DIR / "reference_outputs" / "delphi_phase1_common_branches_20260824" / "continuation_summary.csv"
)
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_phase1_kl0p05_noise_controls_20260825"
EXPECTED_CONTINUATION_SUMMARY_SHA256 = "0b5f31b587d3af4a6adfb6763d4896f9fd9f8fb7a27598c00bcc700d85be5ee4"
PREFIX_CANDIDATE_ID = "shared_bounded_ensemble_kl0p05"
PROPORTIONAL_CONTINUATION_ID = "control_proportional"
EXPECTED_HIGH_EXPOSURE_CONTINUATION_ID = "fit_maximin_26"
REPEATS_PER_ACTION = 4
DATA_SEED_BASE = 962_000
EXISTING_SEED_BRANCH_CODE_COMMIT = "d016caa0fbd0f1f50e29ffa0c9dea5d40f5438e2"
NOISE_CONTROL_BASE_COMMIT = "983f450523"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--continuation-summary", type=Path, default=DEFAULT_CONTINUATION_SUMMARY)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_design(continuation_summary_path: Path) -> tuple[pd.DataFrame, dict[str, object]]:
    actual_sha256 = file_sha256(continuation_summary_path)
    if actual_sha256 != EXPECTED_CONTINUATION_SUMMARY_SHA256:
        raise ValueError(f"Continuation summary changed: {actual_sha256} != {EXPECTED_CONTINUATION_SUMMARY_SHA256}")
    summary = pd.read_csv(continuation_summary_path)
    fit = summary[summary.fit_budget.astype(str).str.lower().eq("true")]
    high_exposure = fit.nlargest(1, "max_phase_1_materialized_epoch").iloc[0]
    high_exposure_id = str(high_exposure.continuation_id)
    if high_exposure_id != EXPECTED_HIGH_EXPOSURE_CONTINUATION_ID:
        raise ValueError(
            f"Highest-exposure continuation changed: {high_exposure_id} != {EXPECTED_HIGH_EXPOSURE_CONTINUATION_ID}"
        )

    rows = []
    actions = (PROPORTIONAL_CONTINUATION_ID, high_exposure_id)
    for action_position, continuation_id in enumerate(actions):
        for repeat_index in range(1, REPEATS_PER_ACTION + 1):
            rows.append(
                {
                    "prefix_candidate_id": PREFIX_CANDIDATE_ID,
                    "continuation_id": continuation_id,
                    "repeat_index": repeat_index,
                    "data_seed": DATA_SEED_BASE + action_position * REPEATS_PER_ACTION + repeat_index - 1,
                }
            )
    design = pd.DataFrame(rows)
    manifest: dict[str, object] = {
        "design_stage": "kl0p05_exact_prefix_branch_noise",
        "endpoint_metric_values_used_for_selection": False,
        "prefix_candidate_id": PREFIX_CANDIDATE_ID,
        "prefix_repeat_seed": 0,
        "trainer_seed": 0,
        "actions": list(actions),
        "repeats_per_action": REPEATS_PER_ACTION,
        "control_rows": len(design),
        "fit_budget_rows": 0,
        "data_seed_base": DATA_SEED_BASE,
        "continuation_summary_sha256": actual_sha256,
        "existing_seed_branch_code_commit": EXISTING_SEED_BRANCH_CODE_COMMIT,
        "noise_control_base_commit": NOISE_CONTROL_BASE_COMMIT,
        "exchangeability_audit": (
            "The d016caa0..983f450 diff changes branch run-ID/manifest identity and result materialization only; "
            "the branch training function, optimizer restoration, data construction, and evaluation path are unchanged."
        ),
        "high_exposure_action_selection": (
            "the runtime-exact fit continuation with maximum phase-1 materialized exposure in the frozen "
            "outcome-blind Wave 1A design"
        ),
        "high_exposure_action_max_phase_1_materialized_epoch": float(high_exposure.max_phase_1_materialized_epoch),
        "interpretation": (
            "Together with each action's existing data-seed-930000 Wave 1 endpoint, these four fresh data seeds "
            "provide five observations per action at one exact KL0.05 boundary checkpoint. They screen for gross "
            "branch sampling variation and heteroskedasticity and remain outside the 180-row fit budget."
        ),
    }
    return design, manifest


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    design, manifest = build_design(args.continuation_summary)
    design_path = args.output_dir / "noise_controls.csv"
    design.to_csv(design_path, index=False)
    manifest["noise_controls_sha256"] = file_sha256(design_path)
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(design.to_string(index=False))
    print("\n", json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
