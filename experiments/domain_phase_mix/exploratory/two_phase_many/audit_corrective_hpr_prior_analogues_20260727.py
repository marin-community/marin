# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "tabulate",
# ]
# ///
"""Audit the corrective HPR panel against already observed nearby policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "corrective_hpr_280_decomposed_panel_20260727"
DEFAULT_HELDOUT_PATH = (
    SCRIPT_DIR / "reference_outputs" / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
)
REFERENCE_PREFIX = "3e18_heldout:"
TARGET_METRICS = {
    "uncheatable": "uncheatable_bpb",
    "table9": "table9_macro_bpb",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--heldout-path", type=Path, default=DEFAULT_HELDOUT_PATH)
    return parser.parse_args()


def heldout_row(heldout: pd.DataFrame, reference: str) -> pd.Series:
    if not reference.startswith(REFERENCE_PREFIX):
        raise ValueError(f"Nearest reference is not a 3e18 heldout: {reference}")
    heldout_id = reference.removeprefix(REFERENCE_PREFIX)
    matched = heldout.loc[heldout["heldout_id"] == heldout_id]
    if len(matched) != 1:
        raise ValueError(f"Expected one heldout row for {heldout_id}, found {len(matched)}")
    return matched.iloc[0]


def build_audit(manifest: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
    controls = manifest.set_index("candidate_id")
    rows: list[dict[str, object]] = []
    for _, candidate in manifest.loc[manifest["policy_class"] == "two_phase"].iterrows():
        control = controls.loc[candidate["aggregate_control_id"]]
        candidate_reference = str(candidate["nearest_existing_reference"])
        control_reference = str(control["nearest_existing_reference"])
        candidate_observation = heldout_row(heldout, candidate_reference)
        control_observation = heldout_row(heldout, control_reference)
        metric = TARGET_METRICS[str(candidate["target"])]
        candidate_value = float(candidate_observation[metric])
        control_value = float(control_observation[metric])
        rows.append(
            {
                "candidate_id": candidate["candidate_id"],
                "target": candidate["target"],
                "aggregate_kl_budget": candidate["aggregate_kl_budget"],
                "phase_information_budget": candidate["phase_information_budget"],
                "nearest_existing_policy_tv": candidate["nearest_existing_policy_tv"],
                "within_two_phase_predicted_gain": candidate["within_two_phase_predicted_gain"],
                "observed_near_gain": control_value - candidate_value,
                "nearest_existing_reference": candidate_reference,
                "control_near_reference": control_reference,
                "candidate_observed_bpb": candidate_value,
                "control_observed_bpb": control_value,
                "same_reference_as_control": candidate_reference == control_reference,
            }
        )
    return pd.DataFrame(rows)


def build_summary(audit: pd.DataFrame) -> pd.DataFrame:
    return (
        audit.groupby(["target", "aggregate_kl_budget"], as_index=False)
        .agg(
            candidates=("candidate_id", "size"),
            mean_nearest_policy_tv=("nearest_existing_policy_tv", "mean"),
            max_nearest_policy_tv=("nearest_existing_policy_tv", "max"),
            mean_predicted_gain=("within_two_phase_predicted_gain", "mean"),
            mean_observed_near_gain=("observed_near_gain", "mean"),
            observed_gain_share=("observed_near_gain", lambda values: float((values > 0).mean())),
            same_reference_share=("same_reference_as_control", "mean"),
        )
        .sort_values(["target", "aggregate_kl_budget"])
    )


def write_report(panel_dir: Path, audit: pd.DataFrame, summary: pd.DataFrame) -> None:
    uncheatable = audit.loc[audit["target"] == "uncheatable"]
    table9 = audit.loc[audit["target"] == "table9"]
    report = "\n".join(
        [
            "# Corrective HPR launch decision",
            "",
            "## Decision",
            "",
            "- **Retain:** six independently fitted, exact-280 tied controls.",
            "- **Block:** 24 aggregate-matched two-phase HPR paths.",
            "",
            "The two-phase paths are not a clean new validation. Their nearest already observed policies are "
            f"only {audit['nearest_existing_policy_tv'].min():.4f} to "
            f"{audit['nearest_existing_policy_tv'].max():.4f} weighted policy TV away. "
            "Nearby July-20 HPR validations already contradict the predicted phase-gain magnitude.",
            "",
            "## Prior-analogue summary",
            "",
            summary.to_markdown(index=False, floatfmt=".6f"),
            "",
            "Across the Uncheatable paths, HPR predicts a mean gain of "
            f"{uncheatable['within_two_phase_predicted_gain'].mean():.6f} BPB, while the nearby observed "
            f"policies average {uncheatable['observed_near_gain'].mean():+.6f} BPB relative to their nearby "
            "tied controls. Across Table-9, the corresponding values are "
            f"{table9['within_two_phase_predicted_gain'].mean():.6f} and "
            f"{table9['observed_near_gain'].mean():+.6f} BPB.",
            "",
            "The exact-280 tied controls answer a narrower provenance question not resolved by those phase "
            "paths: whether independently fitting the one-phase restriction on exactly 280 canonical 300M "
            "rows changes the constrained tied endpoints. They do not validate HPR's phase-order mechanism.",
            "",
            "## Interpretation",
            "",
            "The original exact-duplicate gate used a numerical tolerance of `1e-9`, so it correctly rejected "
            "bit-identical coordinates but did not detect scientific near-replication. This audit is therefore "
            "a pre-submit evidence gate, not a correction to the policy materialization.",
            "",
            "Predicted gain is tied-prediction minus phase-policy prediction, so positive values favor the "
            "two-phase policy. Observed near gain uses the same sign convention on the nearest previously "
            "validated policy and its nearest tied control.",
            "",
            "## Candidate-level evidence",
            "",
            audit.to_markdown(index=False, floatfmt=".6f"),
            "",
        ]
    )
    (panel_dir / "launch_blocker.md").write_text(report)


def main() -> None:
    args = parse_args()
    manifest = pd.read_csv(args.panel_dir / "candidate_manifest.csv")
    heldout = pd.read_csv(args.heldout_path)
    audit = build_audit(manifest, heldout)
    summary = build_summary(audit)
    audit.to_csv(args.panel_dir / "prior_analogue_audit.csv", index=False)
    summary.to_csv(args.panel_dir / "prior_analogue_summary.csv", index=False)
    write_report(args.panel_dir, audit, summary)


if __name__ == "__main__":
    main()
