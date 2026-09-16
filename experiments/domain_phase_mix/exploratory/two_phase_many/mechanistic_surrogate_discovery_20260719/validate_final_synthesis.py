# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2"]
# ///
"""Validate the final negative-result bundle and its append-only provenance."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
TWO_PHASE_ROOT = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[4]
OUTPUT_ROOT = TWO_PHASE_ROOT / "reference_outputs/mechanistic_surrogate_discovery_20260719"
FINAL_DIR = OUTPUT_ROOT / "final_synthesis"
FROZEN_DIR = OUTPUT_ROOT / "frozen_gate"
REGISTRY = OUTPUT_ROOT / "approach_registry.csv"
LEDGER = OUTPUT_ROOT / "data_use_ledger.csv"
REQUIRED_FINAL_FILES = (
    "final_report.md",
    "executive_summary.md",
    "data_dictionary.md",
    "metric_reproduction_summary.csv",
    "metric_reproduction.csv",
    "all_3e18_row_predictions.csv",
    "adversarial_row_predictions.csv",
    "row_prediction_summary.csv",
    "deliverable_traceability.csv",
    "acceptance_gate_evaluation.csv",
    "heldout_pareto_baseline.csv",
    "adversarial_target_matched_metrics.csv",
    "adversarial_cross_target_metrics.csv",
    "adversarial_proposal_strata_metrics.csv",
    "adversarial_stratum_winners.csv",
    "one_phase_restriction_comparison.csv",
    "low_tail_influence_metrics.csv",
    "low_tail_optimum_convergence.csv",
    "two_stage_design_identifiability.csv",
    "mechanism_coverage.csv",
    "route_ledger_coverage.csv",
    "superseded_batch_markers.csv",
    "cross_scale_variance_decomposition.csv",
    "cross_scale_component_transfer.csv",
    "cross_scale_noise_inputs.csv",
    "phase_transfer_deattenuation.csv",
    "cross_target_component_transfer.csv",
    "phase_attenuation_bootstrap.csv",
    "adversarial_stratum_robustness.csv",
    "heldout_provenance_index.csv",
    "excluded_coordinate_aliases.csv",
    "coordinate_repeat_groups.csv",
    "archive_split_summary.csv",
    "adversarial_provenance.csv",
    "coordinate_balanced_metrics.csv",
    "coordinate_balanced_comparison.csv",
    "diagnostic_winners.csv",
    "baseline_complexity.csv",
    "paired_bootstrap_comparisons.csv",
    "calibration_bins.csv",
    "support_stratified_metrics.csv",
    "support_degradation_summary.csv",
    "support_abstention_metrics.csv",
    "maximum_safe_coverage.csv",
    "worst_exposure_feature_correlations.csv",
    "worst_exposure_feature_summary.csv",
    "worst_exposure_top10_enrichment.csv",
    "worst_optimism_rows.csv",
    "frontier_phase_delta_summaries.csv",
    "one_phase_rank_phase_delta_correlations.csv",
    "phase_benefit_sign_transitions.csv",
    "repeat_noise_estimates.csv",
    "confirmation_power.csv",
    "required_repeats.csv",
    "repeat_noise_leave_one_group_out.csv",
    "repeat_noise_influence_summary.csv",
    "prior_parameter_identifiability.csv",
    "prior_hyperparameter_cross_panel_stability.csv",
    "prior_candidate_active_set_complexity.csv",
    "prior_raw_optimum_convex_support.csv",
    "prior_raw_optimum_crossfit_summary.csv",
    "prior_heldout_calibration_bootstrap.csv",
    "prior_stability_source_manifest.csv",
    "policy_class_metrics.csv",
    "policy_class_winners.csv",
    "policy_class_rank_transfer.csv",
    "evidence_index.csv",
    "future_confirmation_preregistration.md",
    "future_confirmation_preregistration.json",
    "confirmation_design_checks.csv",
    "phase_reversal_observability_summary.csv",
    "multiplicity_adjusted_required_repeats.csv",
    "multiplicity_adjusted_power.csv",
    "source_inventory.csv",
    "heldout_pareto_tradeoffs.html",
    "adversarial_compression_tradeoffs.html",
    "one_phase_restriction_transfer.html",
    "manifest.json",
)
REQUIRED_REPORT_SECTIONS = (
    "## Verdict",
    "## Frozen boundary",
    "## Search outcome",
    "### Mechanism coverage",
    "### Data-use ledger integrity",
    "### Independent metric reproduction",
    "### Terminal row-level predictions",
    "## Why no incumbent can be declared the answer",
    "### Proposal-stratum robustness",
    "### Exposed cross-target transfer",
    "### Adversarial policy/proposal strata",
    "## Heldout provenance and weighting",
    "## Identification results",
    "### Support-stratified degradation",
    "### Support-abstention audit",
    "### Worst-residual exposure patterns",
    "### Same-budget design audit",
    "### Phase-reversal observability",
    "### Cross-scale variance decomposition",
    "### Cross-scale measurement-error bound",
    "### Cross-target phase-state audit",
    "### Frontier phase-benefit audit",
    "## Single-phase restriction audit",
    "### Policy-class heldout robustness",
    "## Low-tail influence audit",
    "## Complexity, identifiability, and optimum stability",
    "## Optimization conclusion",
    "## Scientific recommendation",
    "### Confirmation power audit",
    "### Repeat-noise influence audit",
    "### Confirmation multiplicity audit",
    "## Deliverables",
)
ALLOWED_BATCH_MARKERS_WITHOUT_CANDIDATE_ID = {
    "round25_shared_private_batch_frozen",
    "round26_cascade_literal_replay_frozen",
    "round27_power_law_error_batch_frozen",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    registry = pd.read_csv(REGISTRY).fillna("")
    ledger = pd.read_csv(LEDGER).fillna("")
    acceptance = pd.read_csv(FINAL_DIR / "acceptance_gate_evaluation.csv")
    evidence = pd.read_csv(FINAL_DIR / "evidence_index.csv")
    manifest = json.loads((FINAL_DIR / "manifest.json").read_text())
    confirmation = json.loads((FINAL_DIR / "future_confirmation_preregistration.json").read_text())
    report = (FINAL_DIR / "final_report.md").read_text()
    executive_summary = (FINAL_DIR / "executive_summary.md").read_text()
    data_dictionary = (FINAL_DIR / "data_dictionary.md").read_text()

    missing_files = [name for name in REQUIRED_FINAL_FILES if not (FINAL_DIR / name).is_file()]
    assert not missing_files, missing_files
    assert len(registry) == 99
    assert int(registry["id"].str.startswith("prior_").sum()) == 41
    assert not registry.isna().any().any()
    assert not registry.astype(str).apply(lambda column: column.str.strip().eq("")).any().any()
    assert not registry["id"].duplicated().any()
    assert not registry["family"].duplicated().any()
    assert not registry["status"].isin({"active", "promoted"}).any()

    assert len(ledger) == manifest["ledger_rows"]
    candidate_rows = ledger[ledger["candidate_id"].astype(str).str.strip().ne("")]
    assert not candidate_rows[["round_id", "candidate_id"]].duplicated().any()
    assert ledger["round_id"].astype(str).str.strip().ne("").all()
    batch_markers = set(ledger.loc[ledger["candidate_id"].astype(str).str.strip().eq(""), "round_id"])
    assert batch_markers == ALLOWED_BATCH_MARKERS_WITHOUT_CANDIDATE_ID

    assert len(acceptance) == len(registry)
    assert set(acceptance["route_id"]) == set(registry["id"])
    assert not acceptance["all_required_gates_pass"].astype(bool).any()
    assert not acceptance["full_adversarial_gate_reached"].astype(bool).any()
    assert manifest["verdict"] == "no_candidate_passed"
    assert manifest["promoted_candidate_count"] == 0
    assert manifest["claude_reviews_run"] == 0
    assert manifest["sealed_confirmation_outcomes_inspected"] is False
    assert manifest["round56_complete"] is True
    assert manifest["round57_complete"] is True
    assert manifest["round58_complete"] is True
    assert manifest["round59_complete"] is True
    assert manifest["round60_complete"] is True
    assert manifest["round61_complete"] is True
    assert manifest["round62_complete"] is True
    assert manifest["round63_complete"] is True
    assert manifest["round64_complete"] is True
    assert manifest["round65_complete"] is True
    assert manifest["round66_complete"] is True
    assert manifest["round67_complete"] is True
    assert manifest["round68_complete"] is True
    assert manifest["round69_complete"] is True
    assert manifest["round70_complete"] is True
    assert manifest["round71_complete"] is True
    assert manifest["round72_complete"] is True
    assert manifest["round73_complete"] is True
    assert manifest["round74_complete"] is True
    assert manifest["round75_complete"] is True
    assert manifest["round76_complete"] is True
    assert manifest["round77_complete"] is True
    assert manifest["round78_complete"] is True
    assert confirmation["status"].startswith("inactive_")

    source_inventory = pd.read_csv(FINAL_DIR / "source_inventory.csv")
    source_paths = sorted(SCRIPT_DIR.glob("*.py"))
    assert len(source_inventory) == len(source_paths) == manifest["source_file_count"]
    assert source_inventory["path"].nunique() == len(source_inventory)
    for source in source_inventory.itertuples(index=False):
        source_path = TWO_PHASE_ROOT / str(source.path)
        assert source_path.is_file(), source.path
        assert sha256(source_path) == source.sha256, source.path
    entry_points = source_inventory.loc[source_inventory["has_main_entry_point"].astype(bool)]
    assert entry_points["has_pep723_metadata"].astype(bool).all()
    assert source_inventory["module_purpose"].astype(str).str.strip().ne("").all()

    metric_reproduction = pd.read_csv(FINAL_DIR / "metric_reproduction.csv")
    metric_reproduction_summary = pd.read_csv(FINAL_DIR / "metric_reproduction_summary.csv")
    assert len(metric_reproduction) == 680
    assert len(metric_reproduction_summary) == 4
    assert metric_reproduction["passed"].astype(bool).all()
    assert metric_reproduction_summary["passed"].astype(bool).all()
    assert float(metric_reproduction["absolute_difference"].max()) < 1e-12

    row_predictions = pd.read_csv(FINAL_DIR / "all_3e18_row_predictions.csv")
    adversarial_predictions = pd.read_csv(FINAL_DIR / "adversarial_row_predictions.csv")
    row_prediction_summary = pd.read_csv(FINAL_DIR / "row_prediction_summary.csv")
    prediction_key = ["heldout_id", "target", "model"]
    assert len(row_predictions) == 12_780
    assert not row_predictions.duplicated(prediction_key).any()
    assert row_predictions["heldout_id"].nunique() == 710
    assert row_predictions["mixture_sha256"].nunique() == 690
    assert set(row_predictions["target"]) == {"table9", "uncheatable"}
    assert row_predictions["model"].nunique() == 9
    assert set(row_predictions["target_relation"]) == {"non_adversarial", "target_matched", "cross_target"}
    assert len(adversarial_predictions) == 2_400
    assert adversarial_predictions["heldout_id"].nunique() == 120
    target_matched = adversarial_predictions.loc[adversarial_predictions["target_relation"].eq("target_matched")]
    cross_target = adversarial_predictions.loc[adversarial_predictions["target_relation"].eq("cross_target")]
    assert target_matched["model"].nunique() == 11
    assert cross_target["model"].nunique() == 9
    assert len(row_prediction_summary) == 58
    frozen_adversarial = pd.read_csv(FROZEN_DIR / "adversarial_target_matched_predictions.csv")
    adversarial_provenance = pd.read_csv(FINAL_DIR / "adversarial_provenance.csv")
    frozen_adversarial = frozen_adversarial.merge(
        adversarial_provenance[["candidate_id", "heldout_id"]],
        on="candidate_id",
        how="left",
        validate="many_to_one",
    )
    exposed_reconciliation = frozen_adversarial.merge(
        target_matched[["heldout_id", "target", "model", "observed", "predicted"]],
        on=["heldout_id", "target", "model"],
        how="outer",
        validate="one_to_one",
        suffixes=("_frozen", "_terminal"),
        indicator=True,
    )
    assert len(exposed_reconciliation) == 1_320
    assert exposed_reconciliation["_merge"].eq("both").all()
    observed_difference = exposed_reconciliation["observed_frozen"] - exposed_reconciliation["observed_terminal"]
    predicted_difference = exposed_reconciliation["predicted_frozen"] - exposed_reconciliation["predicted_terminal"]
    assert float(observed_difference.abs().max()) < 1e-12
    assert float(predicted_difference.abs().max()) < 1e-12
    residual_identity = (
        row_predictions["predicted"] - row_predictions["observed"] - row_predictions["residual_predicted_minus_observed"]
    )
    optimism_identity = (
        row_predictions["observed"] - row_predictions["predicted"] - row_predictions["optimism_observed_minus_predicted"]
    )
    assert float(residual_identity.abs().max()) < 1e-12
    assert float(optimism_identity.abs().max()) < 1e-12
    adversarial_residual_identity = (
        adversarial_predictions["predicted"]
        - adversarial_predictions["observed"]
        - adversarial_predictions["residual_predicted_minus_observed"]
    )
    assert float(adversarial_residual_identity.abs().max()) < 1e-12
    deliverable_traceability = pd.read_csv(FINAL_DIR / "deliverable_traceability.csv")
    assert len(deliverable_traceability) == 21
    assert deliverable_traceability["passed"].astype(bool).all()

    design = pd.read_csv(FINAL_DIR / "two_stage_design_identifiability.csv")
    random_joint = design.loc[design["design"].eq("random_two_phase_280") & design["block"].eq("joint")].iloc[0]
    paired_joint = design.loc[design["design"].eq("two_stage_140_tied_70_pairs") & design["block"].eq("joint")].iloc[0]
    assert int(random_joint["numerical_rank"]) == 76
    assert int(paired_joint["numerical_rank"]) == 76
    assert float(random_joint["max_canonical_correlation"]) > 0.8
    assert float(paired_joint["max_canonical_correlation"]) < 1e-12

    reversal = pd.read_csv(FINAL_DIR / "phase_reversal_observability_summary.csv")
    delphi_reversal = reversal.loc[reversal["surface"].eq("delphi_3e18_39_bucket_fit_swarm")].iloc[0]
    assert int(delphi_reversal["non_tied_coordinate_count"]) == 238
    assert int(delphi_reversal["feasible_reflection_count"]) == 0
    assert int(delphi_reversal["exact_reflection_count_at_1e_8"]) == 0

    mechanism = pd.read_csv(FINAL_DIR / "mechanism_coverage.csv")
    assert int(mechanism["route_count"].sum()) == 58
    assert mechanism["mechanism_group"].nunique() == 6

    ledger_coverage = pd.read_csv(FINAL_DIR / "route_ledger_coverage.csv")
    marker_coverage = pd.read_csv(FINAL_DIR / "superseded_batch_markers.csv")
    assert len(ledger_coverage) == 58
    empirical_coverage = ledger_coverage.loc[~ledger_coverage["single_edge_theoretical_or_descriptive"]]
    assert len(empirical_coverage) == 55
    assert empirical_coverage["complete_freeze_edge_after_reconciliation"].all()
    assert empirical_coverage["complete_terminal_edge_after_reconciliation"].all()
    assert int(ledger_coverage["reconciled_historical_edge"].sum()) == 7
    assert len(marker_coverage) == 3
    assert marker_coverage["status"].eq("preserved_append_only_and_superseded").all()

    variance_decomposition = pd.read_csv(FINAL_DIR / "cross_scale_variance_decomposition.csv")
    component_transfer = pd.read_csv(FINAL_DIR / "cross_scale_component_transfer.csv")
    assert len(variance_decomposition) == 4
    assert float(variance_decomposition["variance_identity_error"].max()) < 1e-12
    assert set(component_transfer["component"]) == {"aggregate", "phase_delta", "two_phase"}
    delphi_phase_ratio = variance_decomposition.loc[
        variance_decomposition["scale"].eq("delphi"), "phase_to_aggregate_sd_ratio"
    ]
    assert delphi_phase_ratio.between(0.45, 0.60).all()

    deattenuation = pd.read_csv(FINAL_DIR / "phase_transfer_deattenuation.csv")
    assert len(deattenuation) == 4
    upper = deattenuation.loc[deattenuation["noise_bound"].eq("upper_95pct")]
    assert upper["deattenuated_phase_transfer_pearson"].lt(0.70).all()
    assert upper["errors_in_variables_slope"].lt(0.45).all()

    cross_target = pd.read_csv(FINAL_DIR / "cross_target_component_transfer.csv")
    phase_rows = cross_target.loc[cross_target["component"].eq("phase_delta")]
    assert len(phase_rows) == 2
    assert phase_rows["pearson"].between(0.69, 0.79).all()
    assert phase_rows["raw_sign_agreement"].between(0.78, 0.80).all()
    attenuation = pd.read_csv(FINAL_DIR / "phase_attenuation_bootstrap.csv")
    difference = attenuation.loc[attenuation["quantity"].eq("table9_minus_uncheatable")].iloc[0]
    assert float(difference["low_95pct"]) > 0.0

    stratum_robustness = pd.read_csv(FINAL_DIR / "adversarial_stratum_robustness.csv")
    selection_robustness = stratum_robustness.loc[stratum_robustness["stratum_type"].eq("selection_stratum")]
    assert len(selection_robustness) == 22
    assert selection_robustness["minimum_calibration_slope"].lt(0.5).all()
    adversarial_stratum_winners = pd.read_csv(FINAL_DIR / "adversarial_stratum_winners.csv")
    assert len(adversarial_stratum_winners) == 18
    assert set(adversarial_stratum_winners["candidate_target"]) == {"table9", "uncheatable"}
    assert set(adversarial_stratum_winners["stratum_type"]) == {
        "origin",
        "policy_class",
        "proposal_models",
        "selection_stratum",
    }
    assert (
        not adversarial_stratum_winners[["best_rmse_model", "best_regret_model", "best_calibrated_model"]]
        .isna()
        .any()
        .any()
    )

    prior_complexity = pd.read_csv(FINAL_DIR / "prior_candidate_active_set_complexity.csv")
    prior_hyperparameters = pd.read_csv(FINAL_DIR / "prior_hyperparameter_cross_panel_stability.csv")
    prior_support = pd.read_csv(FINAL_DIR / "prior_raw_optimum_convex_support.csv")
    prior_crossfit = pd.read_csv(FINAL_DIR / "prior_raw_optimum_crossfit_summary.csv")
    prior_sources = pd.read_csv(FINAL_DIR / "prior_stability_source_manifest.csv")
    assert len(prior_complexity) == 4
    assert int((~prior_hyperparameters["cross_panel_stable"].astype(bool)).sum()) == 60
    assert prior_support["distance_over_fit_p95"].gt(2.0).all()
    assert prior_crossfit["fraction_below_observed_frontier"].eq(1.0).all()
    assert len(prior_sources) == 6
    for source in prior_sources.itertuples(index=False):
        source_path = TWO_PHASE_ROOT / str(source.source_path)
        assert source_path.is_file(), source.source_path
        assert sha256(source_path) == source.source_sha256

    policy_class_metrics = pd.read_csv(FINAL_DIR / "policy_class_metrics.csv")
    policy_class_transfer = pd.read_csv(FINAL_DIR / "policy_class_rank_transfer.csv")
    assert set(policy_class_metrics["heldout_policy_class"]) == {"one_phase", "two_phase"}
    assert set(policy_class_transfer["source_panel"]) == {
        "historical",
        "adversarial_target_matched",
    }
    rmse_transfer = policy_class_transfer.loc[policy_class_transfer["metric"].eq("rmse")]
    assert len(rmse_transfer) == 4
    assert rmse_transfer["one_vs_two_model_rank_spearman"].lt(0.75).all()

    provenance = pd.read_csv(FINAL_DIR / "heldout_provenance_index.csv")
    aliases = pd.read_csv(FINAL_DIR / "excluded_coordinate_aliases.csv")
    adversarial_provenance = pd.read_csv(FINAL_DIR / "adversarial_provenance.csv")
    coordinate_comparison = pd.read_csv(FINAL_DIR / "coordinate_balanced_comparison.csv")
    assert len(provenance) == 710
    assert provenance["mixture_sha256"].nunique() == 690
    assert len(aliases) == 12
    assert len(adversarial_provenance) == 120
    assert float(coordinate_comparison["delta_regret_at_1"].abs().max()) < 1e-12
    assert float(coordinate_comparison["delta_rmse"].abs().max()) < 0.001

    support_metrics = pd.read_csv(FINAL_DIR / "support_stratified_metrics.csv")
    support_degradation = pd.read_csv(FINAL_DIR / "support_degradation_summary.csv")
    assert set(support_metrics["support_quartile"]) == {"Q1 nearest", "Q2", "Q3", "Q4 farthest"}
    assert len(support_degradation) == 7
    assert support_degradation["rmse_ratio_farthest_over_nearest"].gt(1.5).all()

    abstention = pd.read_csv(FINAL_DIR / "support_abstention_metrics.csv")
    safe_coverage = pd.read_csv(FINAL_DIR / "maximum_safe_coverage.csv")
    assert set(abstention["coverage_fraction"]) == {0.10, 0.25, 0.50, 0.75, 1.00}
    table9_safe = safe_coverage.loc[safe_coverage["target"].eq("table9")]
    uncheatable_safe = safe_coverage.loc[safe_coverage["target"].eq("uncheatable")]
    assert table9_safe["maximum_tested_coverage_without_severe_optimism"].eq(0.10).all()
    assert uncheatable_safe["maximum_tested_coverage_without_severe_optimism"].between(0.50, 0.75).all()
    assert safe_coverage["worst_optimism"].le(0.05).all()

    exposure_summary = pd.read_csv(FINAL_DIR / "worst_exposure_feature_summary.csv")
    optimism_rows = pd.read_csv(FINAL_DIR / "worst_optimism_rows.csv")
    phase_support = exposure_summary.loc[exposure_summary["feature"].isin({"support_distance", "phase_tv"})]
    assert len(phase_support) == 4
    assert phase_support["minimum_optimism_correlation"].gt(0.0).all()
    assert len(optimism_rows) == 180

    frontier_phase = pd.read_csv(FINAL_DIR / "frontier_phase_delta_summaries.csv")
    cross_scale_frontier = frontier_phase.loc[
        ~frontier_phase["same_scale_selection"].astype(bool)
        & frontier_phase["slice"].isin({"top_10", "top_25", "top_50"})
    ]
    assert len(cross_scale_frontier) == 12
    assert cross_scale_frontier["mean_phase_delta"].gt(0.0).all()
    assert cross_scale_frontier["fraction_two_phase_better"].between(0.20, 0.40).all()

    confirmation_noise = pd.read_csv(FINAL_DIR / "repeat_noise_estimates.csv")
    confirmation_power = pd.read_csv(FINAL_DIR / "confirmation_power.csv")
    required_repeats = pd.read_csv(FINAL_DIR / "required_repeats.csv")
    assert len(confirmation_noise) == 2
    assert set(required_repeats["minimum_repeats_per_arm_for_80pct_power"]) == {2, 6}
    table9_three = confirmation_power.loc[
        confirmation_power["target"].eq("table9")
        & confirmation_power["effect_bpb"].eq(0.005)
        & confirmation_power["repeats_per_arm"].eq(3),
        "power",
    ].item()
    table9_six = confirmation_power.loc[
        confirmation_power["target"].eq("table9")
        & confirmation_power["effect_bpb"].eq(0.005)
        & confirmation_power["repeats_per_arm"].eq(6),
        "power",
    ].item()
    assert table9_three < 0.60
    assert table9_six > 0.80
    assert confirmation["repeat_plan"].startswith("Use 15 independent")
    assert confirmation["decisive_repeats_per_arm"] == 15
    assert confirmation["maximum_unique_policy_count_before_deduplication"] == 86
    assert confirmation["maximum_training_runs_before_deduplication"] == 170
    training = confirmation["training_configuration"]
    assert training["target_flops"] == 3e18
    assert training["model"]["total_trainable_parameters"] == 358_304_128
    assert training["model"]["non_embedding_parameters"] == 128_469_376
    assert training["tokens_and_batching"]["train_steps"] == 3007
    assert training["tokens_and_batching"]["realized_train_tokens"] == 1_576_534_016
    assert training["tokens_and_batching"]["sequence_length"] == 4096
    assert training["tokens_and_batching"]["global_batch_size"] == 128
    assert training["schedule"]["phase_fractions"] == [0.8, 0.2]
    assert training["schedule"]["phase_boundary_fraction"] == 0.8
    assert training["data"]["bucket_count"] == 39
    assert training["data"]["simulated_epoch_target_budget"] == 6_325_183_647_689
    assert training["data"]["tokenizer"] == "meta-llama/Meta-Llama-3.1-8B"
    assert training["optimizer"]["name"] == "AdamH"
    assert training["optimizer"]["learning_rate"] == 0.01
    assert training["optimizer"]["warmup_fraction"] == 0.1
    assert training["optimizer"]["decay_fraction"] == 0.2
    assert "vary data_seed only" in training["seed_rule"]
    provenance_contract = training["provenance"]
    for path_key, digest_key in (
        ("launcher_path", "launcher_sha256"),
        ("resolved_manifest_path", "resolved_manifest_sha256"),
        ("domain_config_path", "domain_config_sha256"),
        ("token_count_config_path", "token_count_config_sha256"),
        ("tokenizer_config_path", "tokenizer_config_sha256"),
    ):
        source_path = REPO_ROOT / provenance_contract[path_key]
        assert source_path.is_file(), source_path
        assert sha256(source_path) == provenance_contract[digest_key], source_path
    assert provenance_contract["scaling_fit_sha256"] == (
        "097328aada40b0beb8b38c765ae0b30bf1767623a2b2eacd6c5c02a77af49f2b"
    )
    assert provenance_contract["source_fit_panel_sha256"] == (
        "4f283bacb4ef269c396277cbd518ef74212a51741c909a1e1e9ace040751d507"
    )
    confirmation_report = (FINAL_DIR / "future_confirmation_preregistration.md").read_text()
    assert "## Frozen training configuration" in confirmation_report
    assert "358,304,128-parameter" in confirmation_report
    design_checks = pd.read_csv(FINAL_DIR / "confirmation_design_checks.csv")
    assert len(design_checks) == 13
    assert design_checks["passed"].astype(bool).all()

    multiplicity_required = pd.read_csv(FINAL_DIR / "multiplicity_adjusted_required_repeats.csv")
    required_by_target_and_bound = {
        (row.target, row.noise_bound): int(row.minimum_repeats_per_arm_for_80pct_power)
        for row in multiplicity_required.itertuples(index=False)
    }
    assert required_by_target_and_bound == {
        ("table9", "pooled_point"): 7,
        ("table9", "upper_95pct"): 15,
        ("uncheatable", "pooled_point"): 2,
        ("uncheatable", "upper_95pct"): 3,
    }
    assert "cannot replace a failed raw optimum" in confirmation["single_seed_policy_use"]
    assert any("Holm family-wise control at alpha=0.05" in item for item in confirmation["primary_acceptance"])

    repeat_influence = pd.read_csv(FINAL_DIR / "repeat_noise_influence_summary.csv")
    assert len(repeat_influence) == 2
    assert repeat_influence["maximum_absolute_relative_sd_change"].lt(0.11).all()
    table9_influence = repeat_influence.loc[repeat_influence["target"].eq("table9")].iloc[0]
    assert float(table9_influence["minimum_power_at_six_repeats"]) >= 0.80
    assert int(table9_influence["maximum_required_repeats_per_arm"]) == 6

    missing_sections = [section for section in REQUIRED_REPORT_SECTIONS if section not in report]
    assert not missing_sections, missing_sections
    assert "No investigated family passed the frozen acceptance gate" in report
    assert "/Users/" not in report
    assert "No headline surrogate is recommended" in executive_summary
    assert "86 unique policies" in executive_summary
    assert "170 runs" in executive_summary
    assert "15 decisive repeats" in executive_summary
    assert "/Users/" not in executive_summary
    assert "BPB is lower-is-better throughout" in data_dictionary
    assert "algebraic_restriction_of_two_phase_fit" in data_dictionary
    assert "independent_one_phase_refit" in data_dictionary
    assert "No sealed outcome was read" in data_dictionary
    assert "/Users/" not in data_dictionary
    terminal_synthesis = ledger.loc[
        ledger["round_id"].eq("round_75_final_negative_synthesis")
        & ledger["candidate_id"].eq("terminal_negative_verdict_and_design_supersession")
    ]
    assert len(terminal_synthesis) == 1
    assert terminal_synthesis["notes"].str.contains("15 decisive repeats").all()
    assert terminal_synthesis["notes"].str.contains("No sealed outcome was read").all()

    missing_evidence = []
    for relative in evidence["path"]:
        path = TWO_PHASE_ROOT / str(relative)
        if not path.is_file():
            missing_evidence.append(str(relative))
    assert not missing_evidence, missing_evidence

    for csv_name in REQUIRED_FINAL_FILES:
        if csv_name.endswith(".csv"):
            frame = pd.read_csv(FINAL_DIR / csv_name)
            assert len(frame) > 0, csv_name
            assert not frame.duplicated().any(), csv_name
            numeric = frame.select_dtypes(include=[np.number])
            assert not np.isinf(numeric.to_numpy(dtype=float)).any(), csv_name

    for artifact_name in REQUIRED_FINAL_FILES:
        if artifact_name.endswith((".csv", ".html", ".json", ".md")):
            assert "/Users/" not in (FINAL_DIR / artifact_name).read_text(), artifact_name

    checksums = {name: sha256(FINAL_DIR / name) for name in REQUIRED_FINAL_FILES}
    validation = {
        "status": "passed",
        "route_count": len(registry),
        "prior_route_count": int(registry["id"].str.startswith("prior_").sum()),
        "new_route_count": int((~registry["id"].str.startswith("prior_")).sum()),
        "promoted_route_count": 0,
        "ledger_row_count": len(ledger),
        "acceptance_rows": len(acceptance),
        "all_required_gates_pass_count": 0,
        "sealed_confirmation_outcomes_inspected": False,
        "file_sha256": checksums,
    }
    (FINAL_DIR / "validation.json").write_text(json.dumps(validation, indent=2, sort_keys=True) + "\n")
    print(json.dumps(validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
