# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas>=2.2",
#   "tabulate>=0.9",
# ]
# ///

"""Trace every requested terminal deliverable to machine-checked evidence."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_ROOT = SCRIPT_DIR.parent
OUTPUT_ROOT = TWO_PHASE_ROOT / "reference_outputs/mechanistic_surrogate_discovery_20260719"
FINAL_DIR = OUTPUT_ROOT / "final_synthesis"
ROUND_DIR = OUTPUT_ROOT / "round78_deliverable_traceability"


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    registry = pd.read_csv(OUTPUT_ROOT / "approach_registry.csv").fillna("")
    ledger = pd.read_csv(OUTPUT_ROOT / "data_use_ledger.csv").fillna("")
    manifest = json.loads((FINAL_DIR / "manifest.json").read_text())
    confirmation = json.loads((FINAL_DIR / "future_confirmation_preregistration.json").read_text())
    evidence = pd.read_csv(FINAL_DIR / "evidence_index.csv")
    source_inventory = pd.read_csv(FINAL_DIR / "source_inventory.csv")
    report = (FINAL_DIR / "final_report.md").read_text()

    evidence_names = set(evidence["artifact"])
    terminal_statuses = ~registry["status"].isin({"active", "promoted"})
    complete_registry = registry.astype(str).apply(lambda column: column.str.strip().ne("")).all().all()
    candidate_rows = ledger["candidate_id"].astype(str).str.strip().ne("")
    checks = [
        (
            "complete approach registry",
            len(registry) == 99 and bool(terminal_statuses.all()) and bool(complete_registry),
            "approach_registry.csv",
            f"{len(registry)} complete terminal routes",
        ),
        (
            "prior-route linkage",
            int(registry["id"].str.startswith("prior_").sum()) == 41,
            "approach_registry.csv",
            "41 inherited routes explicitly linked",
        ),
        (
            "append-only data-use ledger",
            not ledger.loc[candidate_rows, ["round_id", "candidate_id"]].duplicated().any(),
            "../data_use_ledger.csv",
            f"{len(ledger)} rows with unique candidate keys",
        ),
        (
            "frozen Pareto baseline and gate",
            manifest["frozen_gate_digest"] == "c4f711312423f038ef8610950d1ae6be30ffba588648177fbf5077e6931f93be",
            "manifest.json",
            "aggregate frozen-gate digest matches",
        ),
        (
            "comparable archive metrics",
            len(pd.read_csv(FINAL_DIR / "heldout_pareto_baseline.csv")) == 18,
            "heldout_pareto_baseline.csv",
            "nine models by two targets",
        ),
        (
            "complete row-level residuals",
            len(pd.read_csv(FINAL_DIR / "all_3e18_row_predictions.csv")) == 12_780
            and len(pd.read_csv(FINAL_DIR / "adversarial_row_predictions.csv")) == 2_400,
            "all_3e18_row_predictions.csv; adversarial_row_predictions.csv",
            "710 runs by two targets by nine models; all 11 exposed models target matched",
        ),
        (
            "target-matched and cross-target adversarial tables",
            (FINAL_DIR / "adversarial_target_matched_metrics.csv").is_file()
            and (FINAL_DIR / "adversarial_cross_target_metrics.csv").is_file(),
            "adversarial_target_matched_metrics.csv; adversarial_cross_target_metrics.csv",
            "both target relations persisted separately",
        ),
        (
            "proposal-stratified adversarial diagnostics",
            len(pd.read_csv(FINAL_DIR / "adversarial_proposal_strata_metrics.csv")) > 0,
            "adversarial_proposal_strata_metrics.csv",
            "candidate target, policy class, proposer, origin, and selection strata",
        ),
        (
            "single-phase restriction and independent refit",
            set(pd.read_csv(FINAL_DIR / "one_phase_restriction_comparison.csv")["fit_mode"])
            == {"algebraic_restriction_of_two_phase_fit", "independent_one_phase_refit"},
            "one_phase_restriction_comparison.csv",
            "both required one-phase semantics present",
        ),
        (
            "StarCoder surface diagnostics",
            {"StarCoder cosine surface", "StarCoder WSD surface"}.issubset(evidence_names),
            "evidence_index.csv",
            "both training schedules indexed",
        ),
        (
            "cross-scale diagnostics",
            "scale transfer" in evidence_names and "cross-scale component transfer" in evidence_names,
            "evidence_index.csv",
            "matched-policy and component transfer indexed",
        ),
        (
            "calibration and residual visualizations",
            "support calibration" in evidence_names and "worst-residual exposure visualization" in evidence_names,
            "evidence_index.csv",
            "support-calibration and failure-atlas plots indexed",
        ),
        (
            "raw optimum diagnostics",
            "low-tail optimum" in evidence_names and (FINAL_DIR / "prior_raw_optimum_crossfit_summary.csv").is_file(),
            "evidence_index.csv; prior_raw_optimum_crossfit_summary.csv",
            "sensitivity, support, and crossfit evidence present",
        ),
        (
            "parameter count and identifiability",
            (FINAL_DIR / "baseline_complexity.csv").is_file()
            and (FINAL_DIR / "prior_parameter_identifiability.csv").is_file(),
            "baseline_complexity.csv; prior_parameter_identifiability.csv",
            "nominal complexity and inherited stability persisted",
        ),
        (
            "rerunnable source inventory",
            len(source_inventory) == manifest["source_file_count"]
            and source_inventory.loc[source_inventory["has_main_entry_point"].astype(bool), "has_pep723_metadata"]
            .astype(bool)
            .all(),
            "source_inventory.csv",
            f"{len(source_inventory)} checksummed sources; all entry points have PEP 723",
        ),
        (
            "exact rejection evidence",
            registry["status_evidence"].astype(str).str.strip().ne("").all(),
            "approach_registry.csv",
            "every route has terminal evidence",
        ),
        (
            "negative verdict",
            manifest["verdict"] == "no_candidate_passed" and "No investigated family passed" in report,
            "final_report.md; manifest.json",
            "0 of 99 routes pass",
        ),
        (
            "promotion and adversarial-review boundary",
            manifest["promoted_candidate_count"] == 0 and manifest["claude_reviews_run"] == 0,
            "manifest.json",
            "no ineligible route was reviewed as promoted",
        ),
        (
            "sealed-evidence boundary",
            manifest["sealed_confirmation_outcomes_inspected"] is False,
            "manifest.json",
            "no sealed confirmation outcome inspected",
        ),
        (
            "inactive future confirmation design",
            confirmation["status"].startswith("inactive_")
            and confirmation["maximum_unique_policy_count_before_deduplication"] == 86
            and confirmation["decisive_repeats_per_arm"] == 15
            and "training_configuration" in confirmation,
            "future_confirmation_preregistration.json",
            "86 policies, 15 repeats, exact training contract",
        ),
        (
            "independent metric reproduction",
            len(pd.read_csv(FINAL_DIR / "metric_reproduction.csv")) == 680,
            "metric_reproduction.csv",
            "680 independently reconstructed scalar comparisons",
        ),
    ]
    frame = pd.DataFrame(checks, columns=["requirement", "passed", "artifact", "evidence"])
    if not frame["passed"].all():
        failed = frame.loc[~frame["passed"], "requirement"].tolist()
        raise AssertionError(f"Terminal deliverable traceability failed: {failed}")
    frame.to_csv(ROUND_DIR / "deliverable_traceability.csv", index=False)
    report_text = "\n".join(
        [
            "# Round 78: terminal deliverable traceability",
            "",
            f"All {len(frame)} requested terminal requirements map to machine-checked artifacts.",
            "",
            frame.to_markdown(index=False),
            "",
            "This audit introduces no model or hyperparameter choice and reads no sealed confirmation outcome.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report_text + "\n")
    print(frame.to_json(orient="records", indent=2))


if __name__ == "__main__":
    main()
