# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Synthesize the frozen mechanistic-surrogate discovery artifacts.

This script reads only persisted development evidence. It never reads the
sealed adversarial stress panel, and it verifies the frozen gate digest before
creating any scorecard.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from experiments.domain_phase_mix.exploratory.two_phase_many.mechanistic_surrogate_discovery_20260717 import (
    freeze_baseline_gate as gate,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
ARTIFACT_ROOT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260717"
DEFAULT_OUTPUT = ARTIFACT_ROOT / "final_synthesis"
REGISTRY_PATH = SCRIPT_DIR / "approach_registry.md"
FROZEN_GATE = ARTIFACT_ROOT / "frozen_gate/acceptance_gate.json"
FROZEN_MANIFEST = ARTIFACT_ROOT / "frozen_gate/frozen_manifest.json"
BASELINE_METRICS = ARTIFACT_ROOT / "frozen_gate/baseline_metrics.csv"
COLLISION_METRICS = ARTIFACT_ROOT / "round12_kish_collision/metrics.csv"
PHASE_METRICS = ARTIFACT_ROOT / "round15_phase_boundary_adaptation/metrics.csv"
PHASE_TRANSFER = ARTIFACT_ROOT / "phase_information_transfer_audit/metrics.csv"
STABILITY_METRICS = ARTIFACT_ROOT / "closest_candidate_stability/stability_summary.csv"
STABILITY_RECORDS = ARTIFACT_ROOT / "closest_candidate_stability/selection_records.csv"
FAILURE_ATLAS = ARTIFACT_ROOT / "failure_atlas/heldout_failure_atlas.csv"
SUPPORT_BINS = ARTIFACT_ROOT / "deployment_support_audit/support_binned_calibration.csv"
RAW_OPTIMA = ARTIFACT_ROOT / "raw_optimum_audit/raw_optima.csv"
RAW_OPTIMUM_STABILITY = ARTIFACT_ROOT / "raw_optimum_audit/raw_optimum_stability_summary.csv"
COLLISION_BOOTSTRAP = ARTIFACT_ROOT / "collision_paired_bootstrap/bootstrap_summary.csv"
COLLISION_TRANSFER = ARTIFACT_ROOT / "collision_transfer_audit/selected_transfer.csv"
ALGEBRAIC_AUDIT = ARTIFACT_ROOT / "algebraic_audit.json"
SHAPE_TRANSFER = ARTIFACT_ROOT / "shape_transfer_audit/transfer_summary.csv"
SCALAR_SUFFICIENCY = ARTIFACT_ROOT / "scalar_invariant_sufficiency/scalar_invariant_diagnostics.csv"
POLICY_NOISE_FLOOR = ARTIFACT_ROOT / "policy_determinacy_audit/policy_only_noise_floor.csv"
POLICY_DUPLICATES = ARTIFACT_ROOT / "policy_determinacy_audit/exact_policy_groups.csv"
INTERVENTION_TRANSFER = ARTIFACT_ROOT / "intervention_source_transfer/metrics.csv"
BASELINE_TRANSFER_SUMMARY = ARTIFACT_ROOT / "baseline_family_transfer_audit/fit_to_heldout_rank_summary.csv"
BASELINE_TRANSFER_RANKS = ARTIFACT_ROOT / "baseline_family_transfer_audit/fit_to_heldout_rank_transfer.csv"
BASELINE_TRANSFER_DASHBOARD = ARTIFACT_ROOT / "baseline_family_transfer_audit/baseline_family_transfer.html"
PHASE_RESTRICTION_SUMMARY = ARTIFACT_ROOT / "phase_tied_restriction_audit/restriction_summary.csv"
PHASE_RESTRICTION_DASHBOARD = ARTIFACT_ROOT / "phase_tied_restriction_audit/phase_tied_restriction.html"
CANDIDATE_COMPLEXITY = ARTIFACT_ROOT / "candidate_complexity_audit/active_set_complexity.csv"
CALIBRATION_BOOTSTRAP = ARTIFACT_ROOT / "heldout_calibration_bootstrap/bootstrap_summary.csv"
CALIBRATION_BOOTSTRAP_DASHBOARD = ARTIFACT_ROOT / "heldout_calibration_bootstrap/heldout_calibration_bootstrap.html"
CALIBRATION_PARETO = ARTIFACT_ROOT / "calibration_pareto_audit/calibration_pareto_metrics.csv"
CALIBRATION_PARETO_DASHBOARD = ARTIFACT_ROOT / "calibration_pareto_audit/calibration_pareto.html"
MULTIVARIATE_UPPER_BOUND = ARTIFACT_ROOT / "multivariate_invariant_upper_bound/metrics.csv"
MULTIVARIATE_UPPER_BOUND_STABILITY = ARTIFACT_ROOT / "multivariate_invariant_upper_bound/feature_stability.csv"
MULTIVARIATE_UPPER_BOUND_DASHBOARD = (
    ARTIFACT_ROOT / "multivariate_invariant_upper_bound/multivariate_invariant_upper_bound.html"
)
SERIES_STRUCTURE = ARTIFACT_ROOT / "series_residual_structure/series_structure_summary.csv"
SERIES_STRUCTURE_METRICS = ARTIFACT_ROOT / "series_residual_structure/series_metrics.csv"
SERIES_STRUCTURE_DASHBOARD = ARTIFACT_ROOT / "series_residual_structure/series_residual_structure.html"
WORST_POLICY_SUMMARY = ARTIFACT_ROOT / "worst_heldout_policy_visualizations/worst_policy_summary.csv"
WORST_POLICY_EXPOSURES = ARTIFACT_ROOT / "worst_heldout_policy_visualizations/worst_policy_exposures.csv"
WORST_POLICY_DASHBOARDS = (
    ARTIFACT_ROOT / "worst_heldout_policy_visualizations/delphi_3e18_uncheatable_worst_policy_exposures.html",
    ARTIFACT_ROOT / "worst_heldout_policy_visualizations/delphi_3e18_table9_worst_policy_exposures.html",
)
HYPERPARAMETER_EQUIFINALITY = ARTIFACT_ROOT / "hyperparameter_equifinality_audit/family_equifinality_summary.csv"
HYPERPARAMETER_CROSS_PANEL = ARTIFACT_ROOT / "hyperparameter_equifinality_audit/cross_panel_selection_stability.csv"
HYPERPARAMETER_EQUIFINALITY_DASHBOARD = (
    ARTIFACT_ROOT / "hyperparameter_equifinality_audit/hyperparameter_equifinality.html"
)
MODEL_DISAGREEMENT_SUMMARY = ARTIFACT_ROOT / "model_disagreement_warning_audit/disagreement_warning_summary.csv"
MODEL_DISAGREEMENT_BINS = ARTIFACT_ROOT / "model_disagreement_warning_audit/disagreement_bins.csv"
MODEL_DISAGREEMENT_DASHBOARD = ARTIFACT_ROOT / "model_disagreement_warning_audit/model_disagreement_warning.html"
RIDGE_PATH_METRICS = ARTIFACT_ROOT / "ridge_calibration_path_audit/ridge_path_metrics.csv"
RIDGE_PATH_DASHBOARD = ARTIFACT_ROOT / "ridge_calibration_path_audit/ridge_calibration_path.html"
UNCERTAINTY_TRANSFER = ARTIFACT_ROOT / "oof_uncertainty_transfer_audit/uncertainty_transfer_summary.csv"
UNCERTAINTY_TRANSFER_DASHBOARD = ARTIFACT_ROOT / "oof_uncertainty_transfer_audit/oof_uncertainty_transfer.html"
DECOMPOSITION_SUMMARY = ARTIFACT_ROOT / "worst_policy_feature_decomposition/worst_policy_decomposition_summary.csv"
DECOMPOSITION_DETAILS = ARTIFACT_ROOT / "worst_policy_feature_decomposition/worst_policy_feature_contributions.csv"
DECOMPOSITION_DASHBOARD = ARTIFACT_ROOT / "worst_policy_feature_decomposition/worst_policy_feature_decomposition.html"
CANCELLATION_METRICS = ARTIFACT_ROOT / "additive_cancellation_audit/cancellation_metrics.csv"
CANCELLATION_QUARTILES = ARTIFACT_ROOT / "additive_cancellation_audit/cancellation_quartiles.csv"
CANCELLATION_DASHBOARD = ARTIFACT_ROOT / "additive_cancellation_audit/additive_cancellation_diagnostic.html"
CONVEX_SUPPORT_METRICS = ARTIFACT_ROOT / "convex_support_audit/convex_support_metrics.csv"
CONVEX_SUPPORT_CALIBRATION = ARTIFACT_ROOT / "convex_support_audit/support_stratified_calibration.csv"
CONVEX_SUPPORT_OPTIMA = ARTIFACT_ROOT / "convex_support_audit/raw_optimum_convex_support.csv"
CONVEX_SUPPORT_DASHBOARD = ARTIFACT_ROOT / "convex_support_audit/convex_support_audit.html"
SUPPORT_STRATIFIED_BASELINES = ARTIFACT_ROOT / "support_stratified_baselines/support_stratified_baseline_metrics.csv"
RAW_OPTIMUM_CROSSFIT = ARTIFACT_ROOT / "raw_optimum_crossfit_audit/crossfit_optimum_summary.csv"
RAW_OPTIMUM_CROSSFIT_ROWS = ARTIFACT_ROOT / "raw_optimum_crossfit_audit/crossfit_optimum_predictions.csv"
RAW_OPTIMUM_CROSSFIT_DASHBOARD = ARTIFACT_ROOT / "raw_optimum_crossfit_audit/crossfit_optimum_predictions.html"
SUPPORT_DIRECTION_SUMMARY = ARTIFACT_ROOT / "convex_support_direction_audit/support_direction_summary.csv"
SUPPORT_DIRECTION_ROWS = ARTIFACT_ROOT / "convex_support_direction_audit/heldout_support_directions.csv"
SUPPORT_DIRECTION_SOURCES = ARTIFACT_ROOT / "convex_support_direction_audit/heldout_projection_source_mass.csv"
SUPPORT_DIRECTION_DASHBOARD = ARTIFACT_ROOT / "convex_support_direction_audit/convex_support_directions.html"
DESIGN_RANK = ARTIFACT_ROOT / "design_identifiability_audit/design_numerical_rank.csv"
DESIGN_POLICY_ENERGY = ARTIFACT_ROOT / "design_identifiability_audit/policy_weak_direction_energy.csv"
DESIGN_POLICY_SUMMARY = ARTIFACT_ROOT / "design_identifiability_audit/policy_weak_direction_summary.csv"
DESIGN_SPECTRUM = ARTIFACT_ROOT / "design_identifiability_audit/design_singular_spectrum.csv"
DESIGN_LOADINGS = ARTIFACT_ROOT / "design_identifiability_audit/weak_direction_loadings.csv"
DESIGN_DASHBOARD = ARTIFACT_ROOT / "design_identifiability_audit/design_identifiability.html"
OPTIMUM_PATH_SUMMARY = ARTIFACT_ROOT / "raw_optimum_support_path_audit/raw_optimum_support_path_summary.csv"
OPTIMUM_PATH_ROWS = ARTIFACT_ROOT / "raw_optimum_support_path_audit/raw_optimum_support_paths.csv"
OPTIMUM_PATH_DASHBOARD = ARTIFACT_ROOT / "raw_optimum_support_path_audit/raw_optimum_support_paths.html"
TRIMMED_CALIBRATION = ARTIFACT_ROOT / "trimmed_calibration_audit/trimmed_calibration_metrics.csv"
TRIMMED_CALIBRATION_DASHBOARD = ARTIFACT_ROOT / "trimmed_calibration_audit/trimmed_calibration.html"
PROVENANCE_SUMMARY = ARTIFACT_ROOT / "provenance_audit/swarm_provenance_summary.csv"
PROVENANCE_METRIC_RECOMPUTATION = ARTIFACT_ROOT / "provenance_audit/frozen_metric_recomputation.csv"
EXPECTED_MANIFEST_DIGEST = "e9226286eadc5bfd16b747c9d681b8ffb69dd170ee9138141392da24acf7d8c4"
PROTOCOL_INPUTS = (REGISTRY_PATH, FROZEN_GATE, FROZEN_MANIFEST)
EVIDENCE_INPUTS = (
    BASELINE_METRICS,
    COLLISION_METRICS,
    PHASE_METRICS,
    PHASE_TRANSFER,
    STABILITY_METRICS,
    STABILITY_RECORDS,
    FAILURE_ATLAS,
    SUPPORT_BINS,
    RAW_OPTIMA,
    RAW_OPTIMUM_STABILITY,
    COLLISION_BOOTSTRAP,
    COLLISION_TRANSFER,
    ALGEBRAIC_AUDIT,
    SHAPE_TRANSFER,
    SCALAR_SUFFICIENCY,
    POLICY_NOISE_FLOOR,
    POLICY_DUPLICATES,
    INTERVENTION_TRANSFER,
    BASELINE_TRANSFER_SUMMARY,
    BASELINE_TRANSFER_RANKS,
    BASELINE_TRANSFER_DASHBOARD,
    PHASE_RESTRICTION_SUMMARY,
    PHASE_RESTRICTION_DASHBOARD,
    CANDIDATE_COMPLEXITY,
    CALIBRATION_BOOTSTRAP,
    CALIBRATION_BOOTSTRAP_DASHBOARD,
    CALIBRATION_PARETO,
    CALIBRATION_PARETO_DASHBOARD,
    MULTIVARIATE_UPPER_BOUND,
    MULTIVARIATE_UPPER_BOUND_STABILITY,
    MULTIVARIATE_UPPER_BOUND_DASHBOARD,
    SERIES_STRUCTURE,
    SERIES_STRUCTURE_METRICS,
    SERIES_STRUCTURE_DASHBOARD,
    WORST_POLICY_SUMMARY,
    WORST_POLICY_EXPOSURES,
    *WORST_POLICY_DASHBOARDS,
    HYPERPARAMETER_EQUIFINALITY,
    HYPERPARAMETER_CROSS_PANEL,
    HYPERPARAMETER_EQUIFINALITY_DASHBOARD,
    MODEL_DISAGREEMENT_SUMMARY,
    MODEL_DISAGREEMENT_BINS,
    MODEL_DISAGREEMENT_DASHBOARD,
    RIDGE_PATH_METRICS,
    RIDGE_PATH_DASHBOARD,
    UNCERTAINTY_TRANSFER,
    UNCERTAINTY_TRANSFER_DASHBOARD,
    DECOMPOSITION_SUMMARY,
    DECOMPOSITION_DETAILS,
    DECOMPOSITION_DASHBOARD,
    CANCELLATION_METRICS,
    CANCELLATION_QUARTILES,
    CANCELLATION_DASHBOARD,
    CONVEX_SUPPORT_METRICS,
    CONVEX_SUPPORT_CALIBRATION,
    CONVEX_SUPPORT_OPTIMA,
    CONVEX_SUPPORT_DASHBOARD,
    SUPPORT_STRATIFIED_BASELINES,
    RAW_OPTIMUM_CROSSFIT,
    RAW_OPTIMUM_CROSSFIT_ROWS,
    RAW_OPTIMUM_CROSSFIT_DASHBOARD,
    SUPPORT_DIRECTION_SUMMARY,
    SUPPORT_DIRECTION_ROWS,
    SUPPORT_DIRECTION_SOURCES,
    SUPPORT_DIRECTION_DASHBOARD,
    DESIGN_RANK,
    DESIGN_POLICY_ENERGY,
    DESIGN_POLICY_SUMMARY,
    DESIGN_SPECTRUM,
    DESIGN_LOADINGS,
    DESIGN_DASHBOARD,
    OPTIMUM_PATH_SUMMARY,
    OPTIMUM_PATH_ROWS,
    OPTIMUM_PATH_DASHBOARD,
    TRIMMED_CALIBRATION,
    TRIMMED_CALIBRATION_DASHBOARD,
    PROVENANCE_SUMMARY,
    PROVENANCE_METRIC_RECOMPUTATION,
)
METRIC_COLUMNS = (
    "n",
    "rmse",
    "mae",
    "spearman",
    "bias_predicted_minus_observed",
    "calibration_slope_observed_on_predicted",
    "regret_at_1",
    "regret_at_3",
    "regret_at_5",
    "lower_tail_optimism",
    "low_tail_rmse",
    "optimism_gt_0p05_count",
    "worst_optimism",
    "selected_optimism",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def verify_inputs() -> None:
    for path in (*PROTOCOL_INPUTS, *EVIDENCE_INPUTS):
        if not path.exists():
            raise FileNotFoundError(path)
    for path in EVIDENCE_INPUTS:
        gate.assert_sealed_absent(path)
    if sha256(FROZEN_MANIFEST) != EXPECTED_MANIFEST_DIGEST:
        raise ValueError("The immutable acceptance gate changed after candidate generation")


def field(section: str, names: tuple[str, ...]) -> str:
    for name in names:
        match = re.search(rf"^- \*\*{re.escape(name)}:\*\* (.+)$", section, re.MULTILINE)
        if match:
            return match.group(1).strip()
    return ""


def approach_registry() -> pd.DataFrame:
    text = REGISTRY_PATH.read_text()
    matches = list(re.finditer(r"^## ([A-Z]{1,2})\. (.+)$", text, re.MULTILINE))
    rows: list[dict[str, str]] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        section = text[match.end() : end]
        status_evidence = field(section, ("Status",))
        if not status_evidence:
            continue
        status_match = re.match(r"([A-Za-z -]+?)(?:\.| after| as| because)", status_evidence)
        status = status_match.group(1).strip().lower() if status_match else status_evidence.split()[0].lower()
        units = field(section, ("Units and symmetries", "Units and limits"))
        rows.append(
            {
                "id": match.group(1),
                "family": match.group(2).strip(),
                "premise": field(section, ("Premise",)),
                "latent_state": field(section, ("Latent state", "State/invariant", "Invariant", "State")),
                "state_transition": field(section, ("State transition", "Transition")),
                "response": field(section, ("Response", "Response and units")),
                "additional_degrees_of_freedom": field(section, ("Additional degrees of freedom",)),
                "units_and_symmetries": (
                    units
                    or "Registry-wide convention: exposures and latent states are dimensionless; response amplitudes "
                    "and intercepts have BPB units. Fixed state normalization removes continuous scale symmetry; "
                    "factor permutations/signs/rotations are handled through invariant implied states."
                ),
                "single_phase_restriction": field(section, ("Single-phase restriction",)),
                "starcoder_signature": field(section, ("Expected StarCoder signature",)),
                "optimism_resolution": field(section, ("Expected optimism fix",)),
                "cheapest_falsification": field(section, ("Cheapest falsification",)),
                "status": status,
                "status_evidence": status_evidence,
            }
        )
    return pd.DataFrame(rows)


def validate_registry(registry: pd.DataFrame) -> None:
    required = (
        "premise",
        "response",
        "additional_degrees_of_freedom",
        "units_and_symmetries",
        "single_phase_restriction",
        "starcoder_signature",
        "optimism_resolution",
        "cheapest_falsification",
        "status_evidence",
    )
    for column in required:
        missing = registry.loc[registry[column].eq(""), "id"].tolist()
        if missing:
            raise ValueError(f"Registry entries missing {column}: {missing}")
    missing_state = registry.loc[registry["latent_state"].eq("") & registry["state_transition"].eq(""), "id"].tolist()
    if missing_state:
        raise ValueError(f"Registry entries missing both state and transition: {missing_state}")


def candidate_metrics() -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for artifact, path in (
        ("finite_collision", COLLISION_METRICS),
        ("phase_boundary", PHASE_METRICS),
        ("phase_transfer", PHASE_TRANSFER),
    ):
        frame = pd.read_csv(path)
        frame.insert(0, "artifact", artifact)
        frames.append(frame)
    columns = [
        "artifact",
        "dataset",
        "swarm",
        "target",
        "split",
        "mechanism",
        "config",
        "model",
        "baseline_model",
        "parameter_count",
        "effective_degrees_of_freedom",
        *METRIC_COLUMNS,
    ]
    normalized = pd.concat(frames, ignore_index=True, sort=False)
    return normalized.reindex(columns=columns)


def all_screen_metric_paths() -> list[Path]:
    paths = [Path(path) for path in glob.glob(str(ARTIFACT_ROOT / "**/*metrics.csv"), recursive=True)]
    paths.extend(ARTIFACT_ROOT.glob("initial_screen/*/selected_metrics.csv"))
    return sorted({path for path in paths if "final_synthesis" not in path.parts})


def all_screen_metrics(paths: list[Path]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for path in paths:
        gate.assert_sealed_absent(path)
        frame = pd.read_csv(path)
        if "rmse" not in frame or "split" not in frame:
            continue
        normalized = pd.DataFrame(
            {
                "artifact": str(path.parent.relative_to(ARTIFACT_ROOT)),
                "panel": frame.get("panel", frame.get("dataset", frame.get("swarm", ""))),
                "target": frame.get("target", ""),
                "model_family": frame.get(
                    "family", frame.get("mechanism", frame.get("model", frame.get("variant", "")))
                ),
                "config": frame.get("config", ""),
                "split": frame["split"],
            }
        )
        for column in ("parameter_count", "effective_degrees_of_freedom", *METRIC_COLUMNS):
            normalized[column] = frame[column] if column in frame else np.nan
        rows.append(normalized)
    if not rows:
        raise ValueError("No comparable screen metric artifacts found")
    return pd.concat(rows, ignore_index=True)


def baseline_for(metrics: pd.DataFrame, dataset: str, split: str) -> pd.Series:
    rows = metrics.loc[metrics["dataset"].eq(dataset) & metrics["split"].eq(split)]
    rows = rows.loc[rows["config"].eq("baseline")]
    if len(rows) != 1:
        raise ValueError(f"Expected one baseline for {dataset}/{split}, got {len(rows)}")
    return rows.iloc[0]


def selected_candidate_rows() -> pd.DataFrame:
    collision = pd.read_csv(COLLISION_METRICS)
    phase = pd.read_csv(PHASE_METRICS)
    collision_candidates = {
        "aggregate_collision__beta-0": "finite_collision_aggregate",
        "within_phase_collision__beta-0": "finite_collision_within_phase",
    }
    collision_rows: list[pd.DataFrame] = []
    for config, candidate in collision_candidates.items():
        selected = collision.loc[
            collision["dataset"].isin(("delphi_3e18_uncheatable", "delphi_3e18_table9"))
            & collision["split"].isin(("fit_oof", "heldout_policy_matched"))
            & collision["config"].eq(config)
        ].copy()
        selected["candidate"] = candidate
        selected["mechanism_active"] = selected["collision_coefficient_active"].astype(bool)
        collision_rows.append(selected)
    collision_selected = pd.concat(collision_rows, ignore_index=True)

    phase_rows: list[pd.DataFrame] = []
    selected_config = {
        "delphi_3e18_uncheatable": "phase_information__eps-0.01",
        "delphi_3e18_table9": "phase_information__eps-0.1",
    }
    for dataset, config in selected_config.items():
        selected = phase.loc[phase["dataset"].eq(dataset) & phase["config"].eq(config)].copy()
        selected["candidate"] = "phase_information"
        selected["mechanism_active"] = selected["amplitude"].gt(1e-10)
        phase_rows.append(selected)
    phase_selected = pd.concat(phase_rows, ignore_index=True)
    keep = ["dataset", "split", "config", "parameter_count", *METRIC_COLUMNS, "candidate", "mechanism_active"]
    return pd.concat(
        [collision_selected.reindex(columns=keep), phase_selected.reindex(columns=keep)],
        ignore_index=True,
    )


def gate_scorecard() -> pd.DataFrame:
    gate_config = json.loads(FROZEN_GATE.read_text())["acceptance_gate"]
    collision = pd.read_csv(COLLISION_METRICS)
    candidates = selected_candidate_rows()
    stability = pd.read_csv(STABILITY_METRICS)
    stability_records = pd.read_csv(STABILITY_RECORDS)
    raw_stability = pd.read_csv(RAW_OPTIMUM_STABILITY)
    collision_transfer = pd.read_csv(COLLISION_TRANSFER)
    phase_transfer = pd.read_csv(PHASE_TRANSFER)
    rows: list[dict[str, object]] = []
    for candidate in candidates["candidate"].unique():
        for dataset in ("delphi_3e18_uncheatable", "delphi_3e18_table9"):
            fit = candidates.loc[
                candidates["candidate"].eq(candidate)
                & candidates["dataset"].eq(dataset)
                & candidates["split"].eq("fit_oof")
            ].iloc[0]
            held = candidates.loc[
                candidates["candidate"].eq(candidate)
                & candidates["dataset"].eq(dataset)
                & candidates["split"].eq("heldout_policy_matched")
            ].iloc[0]
            base_fit = baseline_for(collision, dataset, "fit_oof")
            base_held = baseline_for(collision, dataset, "heldout_policy_matched")
            oof_pass = fit["rmse"] <= (1.0 + gate_config["core_oof_rmse_relative_tolerance"]) * base_fit["rmse"]
            regret_pass = held["regret_at_1"] <= (
                base_held["regret_at_1"] + gate_config["policy_matched_regret_at_1_absolute_tolerance"]
            )
            optimism_pass = held["optimism_gt_0p05_count"] <= base_held["optimism_gt_0p05_count"]
            slope_error = abs(held["calibration_slope_observed_on_predicted"] - 1.0)
            base_slope_error = abs(base_held["calibration_slope_observed_on_predicted"] - 1.0)
            calibration_pass = slope_error <= base_slope_error
            nested_ablation_pass = bool(fit["mechanism_active"] and fit["rmse"] < base_fit["rmse"])
            material_pass = any(
                (
                    held["rmse"] <= 0.9 * base_held["rmse"],
                    held["optimism_gt_0p05_count"] <= base_held["optimism_gt_0p05_count"] - 2,
                    held["worst_optimism"] <= base_held["worst_optimism"] - 0.02,
                    slope_error <= 0.8 * base_slope_error,
                )
            )
            is_collision = candidate.startswith("finite_collision_")
            family = "collision" if is_collision else "phase_boundary"
            stable = stability.loc[stability["dataset"].eq(dataset) & stability["family"].eq(family)].iloc[0]
            collision_config = {
                "finite_collision_aggregate": "aggregate_collision__beta-0",
                "finite_collision_within_phase": "within_phase_collision__beta-0",
            }
            candidate_config = collision_config.get(
                candidate,
                {
                    "delphi_3e18_uncheatable": "phase_information__eps-0.01",
                    "delphi_3e18_table9": "phase_information__eps-0.1",
                }[dataset],
            )
            family_records = stability_records.loc[
                stability_records["dataset"].eq(dataset) & stability_records["family"].eq(family)
            ]
            candidate_records = family_records.loc[family_records["config"].eq(candidate_config)]
            candidate_selection_frequency = len(candidate_records) / len(family_records)
            candidate_amplitudes = candidate_records["extra_amplitude"].to_numpy(dtype=float)
            candidate_amplitude_median = float(np.median(candidate_amplitudes)) if len(candidate_amplitudes) else 0.0
            candidate_amplitude_mad = (
                float(np.median(np.abs(candidate_amplitudes - candidate_amplitude_median)))
                if len(candidate_amplitudes)
                else np.inf
            )
            candidate_mad_ratio = candidate_amplitude_mad / max(abs(candidate_amplitude_median), 1e-12)
            stability_pass = bool(candidate_selection_frequency >= 0.8 and candidate_mad_ratio <= 0.5)
            raw_model = (
                candidate.removeprefix("finite_collision_") + "_collision" if is_collision else "phase_information"
            )
            raw = raw_stability.loc[raw_stability["dataset"].eq(dataset) & raw_stability["model"].eq(raw_model)].iloc[0]
            raw_optimum_pass = bool(
                raw["convergence_rate"] >= 0.8 and raw["median_tv_from_full"] <= 0.15 and raw["p90_max_epoch"] <= 25.0
            )
            if is_collision:
                transfer_mechanism = candidate.removeprefix("finite_collision_") + "_collision"
                related = collision_transfer.loc[
                    ~collision_transfer["swarm"].eq("delphi_3e18")
                    & collision_transfer["mechanism"].eq(transfer_mechanism)
                ]
                independent_improvements = int((related["relative_rmse"] <= 0.995).sum())
            else:
                related = phase_transfer.loc[
                    ~phase_transfer["swarm"].eq("delphi_3e18") & phase_transfer["split"].eq("fit_secondary_oof")
                ]
                baseline_by_panel = related.loc[related["model"].eq("baseline")].set_index(["swarm", "target"])["rmse"]
                candidate_by_panel = related.loc[related["model"].eq("phase_information")].set_index(
                    ["swarm", "target"]
                )["rmse"]
                independent_improvements = int((candidate_by_panel / baseline_by_panel <= 0.995).sum())
            independent_panel_support_pass = independent_improvements >= 2
            required = (
                oof_pass,
                regret_pass,
                optimism_pass,
                calibration_pass,
                nested_ablation_pass,
                material_pass,
                stability_pass,
                independent_panel_support_pass,
                raw_optimum_pass,
            )
            rows.append(
                {
                    "candidate": candidate,
                    "dataset": dataset,
                    "oof_rmse_pass": oof_pass,
                    "regret_at_1_pass": regret_pass,
                    "optimism_count_pass": optimism_pass,
                    "calibration_pass": calibration_pass,
                    "nested_ablation_pass": nested_ablation_pass,
                    "material_improvement_pass": material_pass,
                    "parameter_stability_pass": stability_pass,
                    "candidate_config_selection_frequency": candidate_selection_frequency,
                    "candidate_amplitude_mad_over_abs_median": candidate_mad_ratio,
                    "family_modal_config": stable["modal_config"],
                    "independent_panel_support_pass": independent_panel_support_pass,
                    "independent_panel_improvement_count": independent_improvements,
                    "raw_optimum_pass": raw_optimum_pass,
                    "all_required_gates_pass": all(required),
                    "candidate_oof_rmse": fit["rmse"],
                    "baseline_oof_rmse": base_fit["rmse"],
                    "candidate_heldout_rmse": held["rmse"],
                    "baseline_heldout_rmse": base_held["rmse"],
                    "candidate_regret_at_1": held["regret_at_1"],
                    "baseline_regret_at_1": base_held["regret_at_1"],
                    "candidate_optimism_count": held["optimism_gt_0p05_count"],
                    "baseline_optimism_count": base_held["optimism_gt_0p05_count"],
                    "candidate_worst_optimism": held["worst_optimism"],
                    "baseline_worst_optimism": base_held["worst_optimism"],
                    "candidate_calibration_slope": held["calibration_slope_observed_on_predicted"],
                    "baseline_calibration_slope": base_held["calibration_slope_observed_on_predicted"],
                }
            )
    return pd.DataFrame(rows)


def write_baseline_comparison(output_dir: Path, baselines: pd.DataFrame) -> None:
    delphi = baselines.loc[
        baselines["swarm"].eq("delphi_3e18")
        & baselines["policy"].eq("two_phase")
        & baselines["split"].isin(["fit_oof", "heldout_policy_matched"])
    ].copy()
    delphi["parameter_count"] = pd.to_numeric(delphi["parameter_count"], errors="coerce")
    delphi = delphi.loc[delphi["parameter_count"].notna()]
    pivot = delphi.pivot_table(
        index=["target", "model", "parameter_count"], columns="split", values="rmse"
    ).reset_index()
    figure = px.scatter(
        pivot,
        x="fit_oof",
        y="heldout_policy_matched",
        color="target",
        symbol="target",
        size="parameter_count",
        hover_name="model",
        text="model",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Good grouped-OOF fit does not identify 3e18 deployment error",
        labels={"fit_oof": "Grouped OOF RMSE", "heldout_policy_matched": "Policy-matched heldout RMSE"},
    )
    figure.update_traces(textposition="top center")
    figure.update_layout(template="plotly_white")
    figure.write_html(output_dir / "baseline_oof_vs_heldout.html", include_plotlyjs="cdn")


def write_calibration_figure(output_dir: Path) -> None:
    atlas = pd.read_csv(FAILURE_ATLAS)
    baseline = atlas.loc[atlas["mechanism"].eq("baseline")].copy()
    baseline["target"] = baseline["dataset"].str.replace("delphi_3e18_", "", regex=False)
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=("Uncheatable calibration", "Table-9 calibration", "Uncheatable residual", "Table-9 residual"),
    )
    colors = {"uncheatable": "#18796f", "table9": "#d85b31"}
    for col, target in enumerate(("uncheatable", "table9"), start=1):
        rows = baseline.loc[baseline["target"].eq(target)]
        figure.add_trace(
            go.Scatter(
                x=rows["predicted"],
                y=rows["observed"],
                mode="markers",
                marker={"color": colors[target], "size": 6, "opacity": 0.65},
                customdata=np.stack([rows["row_id"], rows["max_epoch"], rows["phase_tv"]], axis=1),
                hovertemplate="%{customdata[0]}<br>pred=%{x:.4f}<br>obs=%{y:.4f}<br>max epochs=%{customdata[1]:.2f}<br>phase TV=%{customdata[2]:.3f}<extra></extra>",
                showlegend=False,
            ),
            row=1,
            col=col,
        )
        lo = float(min(rows["observed"].min(), rows["predicted"].min()))
        hi = float(max(rows["observed"].max(), rows["predicted"].max()))
        figure.add_trace(
            go.Scatter(
                x=[lo, hi], y=[lo, hi], mode="lines", line={"dash": "dash", "color": "#5d6d78"}, showlegend=False
            ),
            row=1,
            col=col,
        )
        figure.add_trace(
            go.Scatter(
                x=rows["observed"],
                y=rows["optimism"],
                mode="markers",
                marker={"color": colors[target], "size": 6, "opacity": 0.65},
                customdata=rows[["row_id", "support_distance", "top_epoch_buckets"]].to_numpy(),
                hovertemplate="%{customdata[0]}<br>obs=%{x:.4f}<br>optimism=%{y:.4f}<br>support distance=%{customdata[1]:.2f}<br>%{customdata[2]}<extra></extra>",
                showlegend=False,
            ),
            row=2,
            col=col,
        )
        figure.add_hline(y=0.0, line_dash="dash", line_color="#5d6d78", row=2, col=col)
        figure.add_hline(y=0.05, line_dash="dot", line_color="#b3272d", row=2, col=col)
    figure.update_xaxes(title_text="Predicted BPB", row=1)
    figure.update_yaxes(title_text="Observed BPB", row=1)
    figure.update_xaxes(title_text="Observed BPB", row=2)
    figure.update_yaxes(title_text="Optimism = observed - predicted", row=2)
    figure.update_layout(height=900, template="plotly_white", title="Frozen 3e18 heldout calibration and optimism")
    figure.write_html(output_dir / "heldout_calibration_and_residuals.html", include_plotlyjs="cdn")


def write_support_figure(output_dir: Path) -> None:
    frame = pd.read_csv(SUPPORT_BINS)
    figure = px.line(
        frame,
        x="mean_distance_ratio",
        y="mean_optimism",
        color="target",
        markers=True,
        text="optimism_gt_0p05_count",
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Optimism grows outside fit support (numbers label >0.05 errors)",
        labels={
            "mean_distance_ratio": "Mean distance / fit nearest-neighbor median",
            "mean_optimism": "Mean observed - predicted BPB",
        },
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color="#5d6d78")
    figure.update_traces(textposition="top center")
    figure.update_layout(template="plotly_white")
    figure.write_html(output_dir / "support_binned_calibration.html", include_plotlyjs="cdn")


def write_starcoder_figure(output_dir: Path) -> None:
    frames: list[pd.DataFrame] = []
    for schedule in ("starcoder_cosine_starcoder_bpb", "starcoder_wsd80_starcoder_bpb"):
        path = ARTIFACT_ROOT / f"initial_screen/{schedule}/selected_metrics.csv"
        frame = pd.read_csv(path)
        frame = frame.loc[frame["split"].eq("leave_region_out")].copy()
        frame["schedule"] = "50/50 cosine" if "cosine" in schedule else "80/20 WSD"
        frames.append(frame)
    metrics = pd.concat(frames, ignore_index=True)
    figure = px.bar(
        metrics,
        x="family",
        y="rmse",
        color="schedule",
        barmode="group",
        hover_data=["config", "spearman", "regret_at_1", "optimism_gt_0p05_count", "worst_optimism"],
        color_discrete_sequence=px.colors.diverging.RdYlGn_r,
        title="Two-domain falsification: leave-region-out StarCoder RMSE",
        labels={"family": "Mechanistic family", "rmse": "Leave-region-out RMSE"},
    )
    figure.update_xaxes(tickangle=-30)
    figure.update_layout(template="plotly_white")
    figure.write_html(output_dir / "starcoder_leave_region_out.html", include_plotlyjs="cdn")


def write_gate_figure(output_dir: Path, scorecard: pd.DataFrame) -> None:
    gate_columns = [column for column in scorecard if column.endswith("_pass") and column != "all_required_gates_pass"]
    labels = scorecard["candidate"] + " / " + scorecard["dataset"].str.replace("delphi_3e18_", "", regex=False)
    matrix = scorecard[gate_columns].astype(int).to_numpy()
    figure = go.Figure(
        data=go.Heatmap(
            z=matrix,
            x=[column.removesuffix("_pass").replace("_", " ") for column in gate_columns],
            y=labels,
            zmin=0,
            zmax=1,
            colorscale=[[0, "#b3272d"], [0.499, "#b3272d"], [0.5, "#e7b23b"], [1, "#18796f"]],
            text=np.where(matrix == 1, "PASS", "FAIL"),
            texttemplate="%{text}",
            showscale=False,
        )
    )
    figure.update_layout(template="plotly_white", title="Frozen acceptance-gate scorecard")
    figure.write_html(output_dir / "acceptance_gate_scorecard.html", include_plotlyjs="cdn")


def write_report(
    output_dir: Path,
    registry: pd.DataFrame,
    scorecard: pd.DataFrame,
    baselines: pd.DataFrame,
) -> None:
    delphi = baselines.loc[
        baselines["swarm"].eq("delphi_3e18")
        & baselines["policy"].eq("two_phase")
        & baselines["split"].eq("heldout_policy_matched")
    ]
    best_rows = delphi.sort_values(["target", "rmse"]).groupby("target", as_index=False).first()
    status_counts = registry["status"].value_counts().rename_axis("status").reset_index(name="count")
    failures = scorecard.loc[~scorecard["all_required_gates_pass"]]
    raw = pd.read_csv(RAW_OPTIMA)
    raw_stability = pd.read_csv(RAW_OPTIMUM_STABILITY)
    bootstrap = pd.read_csv(COLLISION_BOOTSTRAP)
    collision_transfer = pd.read_csv(COLLISION_TRANSFER)
    shape_transfer = pd.read_csv(SHAPE_TRANSFER)
    scalar_sufficiency = pd.read_csv(SCALAR_SUFFICIENCY)
    policy_noise_floor = pd.read_csv(POLICY_NOISE_FLOOR)
    intervention_transfer = pd.read_csv(INTERVENTION_TRANSFER)
    baseline_transfer = pd.read_csv(BASELINE_TRANSFER_SUMMARY)
    phase_restriction = pd.read_csv(PHASE_RESTRICTION_SUMMARY)
    phase_restriction_panel = phase_restriction.groupby("panel", as_index=False).agg(
        model_count=("model", "nunique"),
        median_restricted_over_refit_rmse=("rmse_ratio_restricted_over_refit", "median"),
        worst_restricted_over_refit_rmse=("rmse_ratio_restricted_over_refit", "max"),
        median_prediction_disagreement=("restriction_prediction_rmse", "median"),
        worst_pointwise_disagreement=("restriction_prediction_max_abs", "max"),
    )
    candidate_complexity = pd.read_csv(CANDIDATE_COMPLEXITY)
    calibration_bootstrap = pd.read_csv(CALIBRATION_BOOTSTRAP)
    calibration_pareto = pd.read_csv(CALIBRATION_PARETO)
    calibration_frontier = calibration_pareto.loc[calibration_pareto["pareto_optimal"]]
    multivariate_upper_bound = pd.read_csv(MULTIVARIATE_UPPER_BOUND)
    multivariate_stability = pd.read_csv(MULTIVARIATE_UPPER_BOUND_STABILITY)
    series_structure = pd.read_csv(SERIES_STRUCTURE)
    worst_policy_summary = pd.read_csv(WORST_POLICY_SUMMARY)
    hyperparameter_equifinality = pd.read_csv(HYPERPARAMETER_EQUIFINALITY)
    hyperparameter_cross_panel = pd.read_csv(HYPERPARAMETER_CROSS_PANEL)
    model_disagreement = pd.read_csv(MODEL_DISAGREEMENT_SUMMARY)
    ridge_path = pd.read_csv(RIDGE_PATH_METRICS)
    uncertainty_transfer = pd.read_csv(UNCERTAINTY_TRANSFER)
    decomposition = pd.read_csv(DECOMPOSITION_SUMMARY)
    cancellation = pd.read_csv(CANCELLATION_METRICS)
    convex_support = pd.read_csv(CONVEX_SUPPORT_METRICS)
    convex_calibration = pd.read_csv(CONVEX_SUPPORT_CALIBRATION)
    convex_optima = pd.read_csv(CONVEX_SUPPORT_OPTIMA)
    support_stratified_baselines = pd.read_csv(SUPPORT_STRATIFIED_BASELINES)
    raw_optimum_crossfit = pd.read_csv(RAW_OPTIMUM_CROSSFIT)
    support_directions = pd.read_csv(SUPPORT_DIRECTION_SUMMARY)
    support_direction_sources = pd.read_csv(SUPPORT_DIRECTION_SOURCES)
    design_rank = pd.read_csv(DESIGN_RANK)
    design_policy_summary = pd.read_csv(DESIGN_POLICY_SUMMARY)
    optimum_path_summary = pd.read_csv(OPTIMUM_PATH_SUMMARY)
    trimmed_calibration = pd.read_csv(TRIMMED_CALIBRATION)
    provenance_summary = pd.read_csv(PROVENANCE_SUMMARY)
    provenance_metric_recomputation = pd.read_csv(PROVENANCE_METRIC_RECOMPUTATION)
    strongest_outside_support = (
        support_stratified_baselines.loc[support_stratified_baselines["support_region"].eq("outside")]
        .sort_values(["dataset", "rmse"])
        .groupby("dataset", as_index=False)
        .first()
    )
    ridge_selected = ridge_path.loc[ridge_path["is_fit_selected"] & ridge_path["split"].eq("heldout_policy_matched")]
    ridge_oracle = (
        ridge_path.loc[ridge_path["split"].eq("heldout_policy_matched")]
        .sort_values(["dataset", "rmse"])
        .groupby("dataset", as_index=False)
        .first()
    )
    nonlinear_families = hyperparameter_equifinality.loc[
        hyperparameter_equifinality["identifiability_status"].ne("not_applicable")
    ]
    weak_hyperparameter_families = int(nonlinear_families["weakly_identified"].sum())
    tested_cross_panel_parameters = hyperparameter_cross_panel.loc[hyperparameter_cross_panel["n_panels"].ge(2)]
    unstable_cross_panel_parameters = int((~tested_cross_panel_parameters["cross_panel_stable"]).sum())
    scalar_best = (
        scalar_sufficiency.sort_values(["dataset", "oof_corrected_residual_rmse"])
        .groupby("dataset", as_index=False)
        .first()
    )
    report = f"""# Mechanistic surrogate discovery: final report

## Verdict

**No investigated model passed the frozen acceptance gate.** No new headline surrogate is recommended, and no candidate was promoted to independent Claude review. Invoking a reviewer before promotion would have violated the preregistered search protocol and encouraged post-hoc rescue of a failed family.

The strongest development-heldout predictor remains the pre-search early-family asymmetric baseline, but that is a *reference model*, not a validated universal training law. It reaches policy-matched heldout RMSE {best_rows.loc[best_rows["target"].eq("uncheatable"), "rmse"].iloc[0]:.5f} on Uncheatable and {best_rows.loc[best_rows["target"].eq("table9"), "rmse"].iloc[0]:.5f} on Table-9. Its calibration slopes are 1.132 and 1.235, and it still makes four >0.05-BPB optimism errors on each target.

## What was tested

The registry contains {len(registry)} distinct mechanistic routes. Status counts are:

{status_counts.to_markdown(index=False)}

These routes span retained-state ODEs, finite-corpus coverage/replay, bottlenecks, survival/hazard models, phase-specific heads, recency kernels, family competition, support divergence, positive unresolved-error states, directional foundation transfer, replay/collision invariants, effective-sample-size laws, phase-boundary debt, learning-rate and gradient-noise mechanisms, reliability/precision systems, finite capacity, exact subset traversal, gradient-flow bowls, consolidation, and diversity-gated competition.

## Decisive evidence

1. **Fit-panel equivalence does not identify deployment behavior.** Four Table-9 models inside the frozen 5% OOF-RMSE equivalence set disagree by more than 0.05 BPB on 51 frozen heldouts and by as much as 0.200 BPB.
2. **Ordinary grouped OOF does not reliably choose the structural family.** On Delphi, fit-OOF versus policy-matched-heldout RMSE rank correlation is only 0.176 for Uncheatable and 0.382 for Table-9. Effective-exposure ranks first in fit-OOF on both targets but falls to fifth and eighth on heldouts.
3. **The fitted single-phase restriction is not stable.** On the paired 300M designs, restricting two-phase coefficients to phase-tied inputs has median RMSE ratios of 1.46 (Uncheatable) and 1.41 (Table-9) relative to fitting the same one-phase form. Algebraic tying does not remove phase-design confounding.
4. **The deployment points are genuinely out of support.** The median heldout distance is 4.32 times the median fit-to-fit nearest-neighbor distance; all 259 policy-matched heldouts leave the fit panel's 1--99% interval on at least one invariant. Distance is diagnostic only and is never added to a model.
5. **The closest nested mechanism does not pass as an exact model.** Aggregate collision at beta zero is the fit-fold modal collision form (46/50 Uncheatable folds and 39/50 Table-9 folds), while the within-phase counterpart is unstable. The aggregate form worsens policy-matched heldout RMSE on both targets and raises Uncheatable Regret@1 from 0.00215 to 0.01081. Its raw two-phase optima remain 12.1--15.2 standardized support units away, reach roughly 42 simulated epochs, and converge in at most 10% of fit-fold refits. The within-phase form slightly improves Uncheatable heldout RMSE but has the same regret failure. Neither exact variant passes the immutable gate or transfers materially to two independent panels.
6. **The phase-boundary route is not stable.** Fit-fold selection usually chooses an early-abandonment term rather than the provisionally favorable phase-information term, and the latter's coefficient collapses to zero on most related swarms. This is not transferable phase dynamics.
7. **Two-domain falsification is severe.** Families that look plausible on one StarCoder schedule fail leave-region-out prediction on the other; the best first-round phase-head model still has leave-region-out RMSE 0.105 on cosine and 0.151 on WSD.
8. **Raw optimization remains unsafe.** The unregularized optimum table and bootstrap audit below show whether each near-miss finds unsupported concentration or an unstable policy. Deployment KL/trust regions were deliberately excluded from this judgment.
9. **Training and evaluation randomness is too small to explain the failures.** Exact duplicate policies imply the policy-only repeat floors below, well below the 0.05--0.16 BPB optimism failures.

{policy_noise_floor.to_markdown(index=False, floatfmt=".6f")}

10. **Panel-stratified CV leakage is not the explanation.** Refitting on the 241 qsplit rows and testing all 39 domain deletions gives the strict intervention-transfer results below. Neither target has a >0.05-BPB error, and both RMSE values remain below the deployment-heldout failures.

{intervention_transfer.loc[intervention_transfer["evaluation"].eq("qsplit_to_domain_deletion")].to_markdown(index=False, floatfmt=".6f")}

11. **Residuals cluster by candidate-generation series.** Series identity explains 25.3% of Table-9 and 27.4% of Uncheatable residual sum of squares (permutation (p=0.0121/0.0046)). This label is not an admissible input. It shows that candidate generators bundle policy changes the fit panel never varied independently.

{series_structure.to_markdown(index=False, floatfmt=".6f")}

12. **Most nonlinear transition parameters are weakly identified.** {weak_hyperparameter_families} of {len(nonlinear_families)} screened nonlinear families have broad near-optimal profiles, boundary selections, or both. Across panels, {unstable_cross_panel_parameters} of {len(tested_cross_panel_parameters)} family-parameter pairs tested on at least two panels change across more than 25% of their grid or lack a 50% modal setting. A fitted shape coefficient is therefore not automatically evidence for a transferable transition law.

13. **Coefficient shrinkage is not the omitted mechanism.** Holding the strongest deficit geometry and output link fixed, sweeping ridge over seven orders of magnitude leaves all four severe Uncheatable errors at the oracle-RMSE setting and cannot improve Table-9 heldout RMSE at all. Stronger shrinkage eventually worsens rank, regret, and calibration.

14. **Model disagreement is a useful warning, not a surrogate.** Frozen-model disagreement detects any-model >0.05-BPB optimism with AUC 0.971 on Table-9 and 0.973 on Uncheatable; no consensus failure occurs in the lowest disagreement quartile. This supports abstention or targeted data acquisition, but it is deliberately excluded from model fitting because an ensemble is not a mechanistic law.

15. **Fit-OOF uncertainty does not certify the extreme tail.** Central conformal coverage is often conservative, but the nominal 99% Uncheatable upper bound misses 8/259 policies (binomial excess-miss (p=0.005)) and the worst observation lies another 0.060 BPB beyond an already 0.0305-BPB radius. Table-9 requires a 0.091-BPB radius at 99% and still has two misses. Ordinary CV error bars are too weak or too wide to make raw optimum extrapolation safe.

16. **The severe failures are extrapolation failures in the fitted state, not hidden interpolation failures.** Projecting every heldout state onto the convex hull of the 280 fit states and calibrating distance against leave-one-fit-row-out projections puts all eight >0.05-BPB errors outside the fit 95th-percentile radius. Inside that radius, the strongest baseline has RMSE 0.0072/0.0107, slopes 0.931/1.058, and zero severe errors on Uncheatable/Table-9. Outside it, RMSE rises to 0.0128/0.0211 and all severe errors appear. The raw optima lie 2.04--2.14 times beyond the fit radius.

17. **No existing mechanistic family solves extrapolation after support is held fixed.** The pre-search early-family asymmetric baseline is still the lowest-RMSE model outside empirical support on both targets. Thus the negative result is not an artifact of averaging easy interpolation rows with hard extrapolation rows.

18. **Additive cancellation is not the missing law.** At the ten worst policies, shortage charges and surplus/replay credits nearly cancel while observed harm remains. Across all 259 heldouts, however, opposing contribution mass is negatively rank-correlated with optimism on both targets, and cancellation-fraction confidence intervals are not stable on Table-9. The pattern is an explanation of selected failures, not a transferable criterion for adding a gate or interaction.

19. **The raw-optimum fantasy is stable across refits, not ordinary parameter noise.** Every one of 25 independently refit grouped-fold models predicts both frozen raw optima to beat the best observed fit policy. The mean predicted advantage is 0.053--0.055 BPB on Uncheatable and 0.081--0.082 BPB on Table-9, while prediction SD is only 0.002--0.003 BPB. Combined with unstable optimum weights and convex-support distances above 2 times the fit radius, this is shared structural extrapolation rather than a noisy coefficient estimate.

20. **The unsupported state directions identify missing interventions, not a missing scalar surcharge.** Severe failures' convex projections place 95%--99% of their mass on qsplit rows and almost none on domain deletions, compared with about one-third deletion mass for ordinary heldouts. The dominant unsupported coordinates are pooled broad-text state, phase-0 broad-text state, and Stack-Edu/tech replay. The next informative data should vary early broad exposure and late specialization while holding aggregate exposure fixed.

21. **Linear null-space confounding does not explain the failures.** The 56-feature design has numerical rank 53 at relative singular-value threshold 0.001. Yet severe policies and raw optima place effectively zero state energy in those three null directions and less than 0.1% in directions below 0.01. Their state norms are much larger than ordinary heldouts. The failure is radial range extrapolation along estimable coordinates, not an omitted linear combination recoverable by more ridge tuning.

22. **Neither convex-hull nor sparse local-support labels certify the optimizer path.** Along proportional-to-raw-optimum interpolation, the model claims 70%--75% of its total gain before crossing the median fit-to-fit nearest-neighbor radius and 77%--83% before crossing the 95th-percentile radius. The nearest observed policy at those boundaries remains proportional or a worse deletion. Sparse high-dimensional support therefore cannot convert the predicted gain into evidence; targeted observations along the path are required.

23. **The miss is concentrated in the tail that optimization targets.** Removing the worst 2% of absolute heldout residuals as a descriptive diagnostic moves the Uncheatable observed-on-predicted slope from 1.132 to 1.021 and Table-9 from 1.235 to 1.129, while eliminating all >0.05-BPB errors. This is not a trimming rule: it uses heldout outcomes. It explains how high ordinary Spearman can coexist with an unsafe optimum surface.

24. **Frozen-export or metric-code drift does not explain the result.** The dashboard SHA-256 still matches the preregistered gate, all source files are present, phase weights and aggregate policies satisfy their declared invariants, and all 196 dashboard-derived frozen metric rows recompute to within (10^{-12}). The 300M and Delphi exporters use different metadata conventions for whether `heldoutCount` includes shared aliases, but scoring excludes aliases consistently and every Delphi fit/heldout coordinate overlap is explicitly marked.

### Frozen baseline-family transfer

The fit winner and heldout winner disagree on both Delphi targets. This is a structural-selection failure, not merely a small hyperparameter tie.

{baseline_transfer.to_markdown(index=False, floatfmt=".5f")}

### Empirical phase-tied restriction

The same functional form is refitted in the one-phase class, rather than averaging a two-phase fit. Values above one mean that the two-phase coefficients transfer worse to phase-tied policies than the one-phase refit.

{phase_restriction_panel.to_markdown(index=False, floatfmt=".5f")}

### Candidate complexity and local identifiability

Effective degrees of freedom are the active-set ridge hat trace in the fitted link space after nonlinear hyperparameters are frozen. They are a lower bound because hyperparameter selection is excluded.

{candidate_complexity.drop(columns="active_parameters").to_markdown(index=False, floatfmt=".5f")}

### Nonlinear hyperparameter equifinality

The table reports the fit-only nonlinear profile audit. A family is marked weak when at least half of its panels have broad near-optimal support or at least 75% of selections occur on a grid boundary.

{nonlinear_families[["family", "n_panels", "median_nonlinear_parameters", "median_near_1pct_fraction", "median_parameter_span", "boundary_selection_fraction", "identifiability_status"]].to_markdown(index=False, floatfmt=".5f")}

### Heldout calibration uncertainty

The 259 heldouts come from 28 correlated training series. These intervals use a training-series block bootstrap rather than an IID row bootstrap.

{calibration_bootstrap.to_markdown(index=False, floatfmt=".6f")}

### Tail concentration (descriptive only)

Rows are removed by heldout absolute residual, so this table cannot select a model, fit a calibrator, or define a deployment rule. It quantifies why global fit metrics understate optimizer risk.

{trimmed_calibration.to_markdown(index=False, floatfmt=".6f")}

### Data provenance and metric reproducibility

{provenance_summary.to_markdown(index=False, floatfmt=".3e")}

{provenance_metric_recomputation.to_markdown(index=False, floatfmt=".3e")}

### Calibration is not sufficient

The nondominated baseline set below minimizes heldout RMSE, absolute observed-on-predicted slope error, >0.05-BPB optimism count, worst optimism, and Regret@1 jointly. No model solves all five: a nearly unit calibration slope can retain the same extreme optimism failures, while a lower worst error can coexist with poor global calibration or regret.

{calibration_frontier[["target", "model", "rmse", "calibration_slope_error", "optimism_gt_0p05_count", "worst_optimism", "regret_at_1"]].to_markdown(index=False, floatfmt=".6f")}

### Ridge path does not repair the raw surface

The fit-selected settings exactly reproduce the frozen reference. The oracle rows are shown only as a diagnostic upper bound; heldouts never select the coefficient.

{ridge_selected[["dataset", "l2", "rmse", "spearman", "calibration_slope_observed_on_predicted", "regret_at_1", "optimism_gt_0p05_count", "worst_optimism"]].to_markdown(index=False, floatfmt=".6f")}

{ridge_oracle[["dataset", "l2", "rmse", "spearman", "calibration_slope_observed_on_predicted", "regret_at_1", "optimism_gt_0p05_count", "worst_optimism"]].to_markdown(index=False, floatfmt=".6f")}

### Disagreement as an epistemic warning

{model_disagreement.to_markdown(index=False, floatfmt=".6f")}

### Fit-OOF uncertainty transfer

The bounds use only fit-panel OOF residuals. Heldouts neither select the radius nor alter predictions.

{uncertainty_transfer.loc[uncertainty_transfer["nominal_coverage"].isin([0.95, 0.99])].to_markdown(index=False, floatfmt=".6f")}

### Convex-support separation

The convex distance is used only to audit identifiability. It is not a model feature or deployment penalty.

{convex_support.to_markdown(index=False, floatfmt=".6f")}

{convex_calibration.to_markdown(index=False, floatfmt=".6f")}

{convex_optima.to_markdown(index=False, floatfmt=".6f")}

### Strongest extrapolator by target

{strongest_outside_support[["dataset", "model", "count", "rmse", "observed_on_predicted_slope", "optimism_gt_0p05_count", "worst_optimism"]].to_markdown(index=False, floatfmt=".6f")}

### Existing-channel decomposition

The exact fitted BPB delta from the nearest fit design is allocated across the frozen model's response channels. Positive unrepresented harm means the observed heldout degradation exceeds the net predicted degradation.

{decomposition.loc[decomposition["displayed_worst"]].groupby("dataset", as_index=False).agg(median_design_distance=("mechanistic_design_distance", "median"), median_observed_delta=("observed_delta_from_nearest", "median"), median_predicted_delta=("predicted_delta_from_nearest", "median"), median_unrepresented_harm=("unrepresented_harm_delta", "median"), maximum_unrepresented_harm=("unrepresented_harm_delta", "max")).to_markdown(index=False, floatfmt=".6f")}

{cancellation.to_markdown(index=False, floatfmt=".6f")}

### Unsupported mechanistic-state directions

{support_directions.sort_values(["dataset", "top_three_frequency_severe", "median_absolute_residual_severe"], ascending=False).groupby("dataset", as_index=False).head(8).to_markdown(index=False, floatfmt=".6f")}

{support_direction_sources.groupby(["dataset", "severe_optimism", "panel_source"], as_index=False).agg(median_projection_weight=("projection_weight", "median"), mean_projection_weight=("projection_weight", "mean")).to_markdown(index=False, floatfmt=".6f")}

### Design-spectrum identifiability

{design_rank.to_markdown(index=False, floatfmt=".6g")}

{design_policy_summary.to_markdown(index=False, floatfmt=".6f")}

### Proportional-to-raw-optimum support path

{optimum_path_summary.to_markdown(index=False, floatfmt=".6f")}

### Nonlinear-shape transfer

{shape_transfer.to_markdown(index=False, floatfmt=".4f")}

### Scalar-invariant upper bound

This is a falsification diagnostic, not a candidate model: an isotonic correction is trained directly on development-heldout residuals and evaluated out of fold by training series. If even this arbitrary correction fails, a one-scalar mechanistic penalty is not sufficient.

{scalar_best.to_markdown(index=False, floatfmt=".5f")}

### Sparse multivariate-invariant upper bound

This second deliberately ineligible diagnostic predicts heldout optimism from nine prespecified physical invariants with leave-one-training-series-out nested selection. The mechanistically signed variant selects no invariant in more than 3.6% of outer fits. Letting coefficients take arbitrary signs activates unstable features in at most 21.4% of Table-9 folds and still worsens RMSE, threshold errors, and Regret@1. Even an arbitrary sparse residual map does not reveal a stable low-dimensional omitted penalty.

{multivariate_upper_bound.to_markdown(index=False, floatfmt=".6f")}

{multivariate_stability.loc[multivariate_stability["outer_fold_active_fraction"].gt(0)].to_markdown(index=False, floatfmt=".6f")}

### Worst-policy exposure patterns

The ten most optimistic policies per target are included at bucket resolution. The threshold failures are not a single scalar corner: they range from 0 to 28 buckets below one-quarter of proportional exposure, 7.47 to 99.4 maximum simulated epochs, and phase TV from 0.28 to 0.81. Less-extreme errors also occur without aggregate undercoverage. A single missing-coverage, replay, or phase-divergence surcharge cannot explain this set.

{worst_policy_summary.groupby("dataset", as_index=False).head(5).to_markdown(index=False, floatfmt=".5f")}

### Paired uncertainty for the closest collision extension

Negative deltas favor the candidate. The small Uncheatable RMSE change is uncertain while MAE and Regret@1 worsen; Table-9 calibration improves while MAE worsens and threshold-error count is unchanged.

{bootstrap.to_markdown(index=False, floatfmt=".6f")}

### Cross-swarm transfer of finite collision

Each row is a secondary grouped-OOF test on top of that panel's strongest frozen baseline. A ratio below one favors the collision extension. No non-Delphi panel clears the preregistered 0.5% materiality threshold.

{collision_transfer.to_markdown(index=False, floatfmt=".6f")}

## Frozen-gate outcome

{failures[["candidate", "dataset", "oof_rmse_pass", "regret_at_1_pass", "optimism_count_pass", "calibration_pass", "nested_ablation_pass", "material_improvement_pass", "parameter_stability_pass", "candidate_config_selection_frequency", "family_modal_config", "independent_panel_support_pass", "independent_panel_improvement_count", "raw_optimum_pass", "all_required_gates_pass"]].to_markdown(index=False)}

## Raw optima

{raw.to_markdown(index=False, floatfmt=".5f")}

### Fit-fold optimum stability

{raw_stability.to_markdown(index=False, floatfmt=".5f")}

### Cross-fit value assigned to frozen raw optima

{raw_optimum_crossfit.to_markdown(index=False, floatfmt=".6f")}

## Scientific interpretation

The negative result is stronger than “the search missed a good formula.” Across independent mechanisms, added terms either collapse, select different boundaries across folds/swarms, improve in-panel RMSE while worsening selected regret, or leave the extreme optimism unchanged. The current policy summaries contain enough information to rank ordinary mixtures but not enough interventions to identify how missing coverage, severe repetition, and phase order compose far outside the sampled region.

The appropriate conclusion is therefore **model-form underidentification under support shift**. It would be misleading to publish a new parametric law selected from these development heldouts. A deployment regularizer can keep an existing surrogate near observed policies, but that is a decision constraint, not evidence that its raw surface is correct.

## Recommended next evidence

1. Add targeted interventions that vary one causal invariant at a time: hold aggregate exposure fixed while changing phase order, hold phase order fixed while changing finite-corpus repetition, and vary family coverage without simultaneously changing concentration.
2. Refit only after those interventions enlarge support around plausible optima. Keep the current adversarial stress panel sealed until a new form and gate are frozen independently.
3. Until then, use existing surrogates only for ranking a near-support candidate set. Report a separate deployment trust region and do not call its regularization a mechanistic correction.

## Reproduction

Run the scripts in this directory with `PYTHONPATH=. uv run <script>`. The immutable frozen-manifest digest is `{EXPECTED_MANIFEST_DIGEST}`. `synthesize_discovery.py` refuses to run if it changes or if a sealed-panel token occurs in an input path.
"""
    (output_dir / "final_report.md").write_text(report)


def main() -> None:
    args = parse_args()
    verify_inputs()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    registry = approach_registry()
    validate_registry(registry)
    baselines = pd.read_csv(BASELINE_METRICS)
    candidates = candidate_metrics()
    screen_paths = all_screen_metric_paths()
    screens = all_screen_metrics(screen_paths)
    scorecard = gate_scorecard()
    registry.to_csv(args.output_dir / "approach_registry.csv", index=False)
    baselines.to_csv(args.output_dir / "baseline_metrics.csv", index=False)
    candidates.to_csv(args.output_dir / "candidate_metrics.csv", index=False)
    screens.to_csv(args.output_dir / "all_screen_metrics.csv", index=False)
    scorecard.to_csv(args.output_dir / "acceptance_gate_evaluation.csv", index=False)
    pd.read_csv(FAILURE_ATLAS).to_csv(args.output_dir / "all_3e18_heldout_residuals.csv", index=False)
    pd.read_csv(STABILITY_METRICS).to_csv(args.output_dir / "parameter_identifiability.csv", index=False)
    pd.read_csv(POLICY_NOISE_FLOOR).to_csv(args.output_dir / "policy_only_noise_floor.csv", index=False)
    pd.read_csv(POLICY_DUPLICATES).to_csv(args.output_dir / "exact_duplicate_policy_audit.csv", index=False)
    pd.read_csv(INTERVENTION_TRANSFER).to_csv(args.output_dir / "intervention_source_transfer.csv", index=False)
    pd.read_csv(BASELINE_TRANSFER_SUMMARY).to_csv(args.output_dir / "baseline_family_transfer_summary.csv", index=False)
    pd.read_csv(BASELINE_TRANSFER_RANKS).to_csv(args.output_dir / "baseline_family_rank_transfer.csv", index=False)
    (args.output_dir / "baseline_family_transfer.html").write_bytes(BASELINE_TRANSFER_DASHBOARD.read_bytes())
    pd.read_csv(PHASE_RESTRICTION_SUMMARY).to_csv(args.output_dir / "phase_tied_restriction_summary.csv", index=False)
    (args.output_dir / "phase_tied_restriction.html").write_bytes(PHASE_RESTRICTION_DASHBOARD.read_bytes())
    pd.read_csv(CANDIDATE_COMPLEXITY).to_csv(args.output_dir / "candidate_active_set_complexity.csv", index=False)
    pd.read_csv(CALIBRATION_BOOTSTRAP).to_csv(args.output_dir / "heldout_calibration_bootstrap.csv", index=False)
    (args.output_dir / "heldout_calibration_bootstrap.html").write_bytes(CALIBRATION_BOOTSTRAP_DASHBOARD.read_bytes())
    pd.read_csv(CALIBRATION_PARETO).to_csv(args.output_dir / "calibration_pareto_metrics.csv", index=False)
    (args.output_dir / "calibration_pareto.html").write_bytes(CALIBRATION_PARETO_DASHBOARD.read_bytes())
    pd.read_csv(MULTIVARIATE_UPPER_BOUND).to_csv(
        args.output_dir / "multivariate_invariant_upper_bound_metrics.csv", index=False
    )
    pd.read_csv(MULTIVARIATE_UPPER_BOUND_STABILITY).to_csv(
        args.output_dir / "multivariate_invariant_upper_bound_stability.csv", index=False
    )
    (args.output_dir / "multivariate_invariant_upper_bound.html").write_bytes(
        MULTIVARIATE_UPPER_BOUND_DASHBOARD.read_bytes()
    )
    pd.read_csv(SERIES_STRUCTURE).to_csv(args.output_dir / "series_residual_structure_summary.csv", index=False)
    pd.read_csv(SERIES_STRUCTURE_METRICS).to_csv(args.output_dir / "series_residual_structure_metrics.csv", index=False)
    (args.output_dir / "series_residual_structure.html").write_bytes(SERIES_STRUCTURE_DASHBOARD.read_bytes())
    pd.read_csv(WORST_POLICY_SUMMARY).to_csv(args.output_dir / "worst_policy_summary.csv", index=False)
    pd.read_csv(WORST_POLICY_EXPOSURES).to_csv(args.output_dir / "worst_policy_exposures.csv", index=False)
    for dashboard in WORST_POLICY_DASHBOARDS:
        (args.output_dir / dashboard.name).write_bytes(dashboard.read_bytes())
    pd.read_csv(HYPERPARAMETER_EQUIFINALITY).to_csv(
        args.output_dir / "hyperparameter_equifinality_summary.csv", index=False
    )
    pd.read_csv(HYPERPARAMETER_CROSS_PANEL).to_csv(
        args.output_dir / "hyperparameter_cross_panel_stability.csv", index=False
    )
    (args.output_dir / "hyperparameter_equifinality.html").write_bytes(
        HYPERPARAMETER_EQUIFINALITY_DASHBOARD.read_bytes()
    )
    pd.read_csv(MODEL_DISAGREEMENT_SUMMARY).to_csv(
        args.output_dir / "model_disagreement_warning_summary.csv", index=False
    )
    pd.read_csv(MODEL_DISAGREEMENT_BINS).to_csv(args.output_dir / "model_disagreement_warning_bins.csv", index=False)
    (args.output_dir / "model_disagreement_warning.html").write_bytes(MODEL_DISAGREEMENT_DASHBOARD.read_bytes())
    pd.read_csv(RIDGE_PATH_METRICS).to_csv(args.output_dir / "ridge_calibration_path_metrics.csv", index=False)
    (args.output_dir / "ridge_calibration_path.html").write_bytes(RIDGE_PATH_DASHBOARD.read_bytes())
    pd.read_csv(UNCERTAINTY_TRANSFER).to_csv(args.output_dir / "oof_uncertainty_transfer_summary.csv", index=False)
    (args.output_dir / "oof_uncertainty_transfer.html").write_bytes(UNCERTAINTY_TRANSFER_DASHBOARD.read_bytes())
    pd.read_csv(DECOMPOSITION_SUMMARY).to_csv(
        args.output_dir / "worst_policy_feature_decomposition_summary.csv", index=False
    )
    pd.read_csv(DECOMPOSITION_DETAILS).to_csv(args.output_dir / "worst_policy_feature_contributions.csv", index=False)
    (args.output_dir / "worst_policy_feature_decomposition.html").write_bytes(DECOMPOSITION_DASHBOARD.read_bytes())
    pd.read_csv(CANCELLATION_METRICS).to_csv(args.output_dir / "additive_cancellation_metrics.csv", index=False)
    pd.read_csv(CANCELLATION_QUARTILES).to_csv(args.output_dir / "additive_cancellation_quartiles.csv", index=False)
    (args.output_dir / "additive_cancellation_diagnostic.html").write_bytes(CANCELLATION_DASHBOARD.read_bytes())
    pd.read_csv(CONVEX_SUPPORT_METRICS).to_csv(args.output_dir / "convex_support_metrics.csv", index=False)
    pd.read_csv(CONVEX_SUPPORT_CALIBRATION).to_csv(args.output_dir / "convex_support_calibration.csv", index=False)
    pd.read_csv(CONVEX_SUPPORT_OPTIMA).to_csv(args.output_dir / "raw_optimum_convex_support.csv", index=False)
    (args.output_dir / "convex_support_audit.html").write_bytes(CONVEX_SUPPORT_DASHBOARD.read_bytes())
    pd.read_csv(SUPPORT_STRATIFIED_BASELINES).to_csv(
        args.output_dir / "support_stratified_baseline_metrics.csv", index=False
    )
    pd.read_csv(RAW_OPTIMUM_CROSSFIT).to_csv(args.output_dir / "raw_optimum_crossfit_summary.csv", index=False)
    pd.read_csv(RAW_OPTIMUM_CROSSFIT_ROWS).to_csv(args.output_dir / "raw_optimum_crossfit_predictions.csv", index=False)
    (args.output_dir / "raw_optimum_crossfit_predictions.html").write_bytes(RAW_OPTIMUM_CROSSFIT_DASHBOARD.read_bytes())
    pd.read_csv(SUPPORT_DIRECTION_SUMMARY).to_csv(args.output_dir / "support_direction_summary.csv", index=False)
    pd.read_csv(SUPPORT_DIRECTION_ROWS).to_csv(args.output_dir / "heldout_support_directions.csv", index=False)
    pd.read_csv(SUPPORT_DIRECTION_SOURCES).to_csv(args.output_dir / "heldout_projection_source_mass.csv", index=False)
    (args.output_dir / "convex_support_directions.html").write_bytes(SUPPORT_DIRECTION_DASHBOARD.read_bytes())
    pd.read_csv(DESIGN_RANK).to_csv(args.output_dir / "design_numerical_rank.csv", index=False)
    pd.read_csv(DESIGN_POLICY_ENERGY).to_csv(args.output_dir / "policy_weak_direction_energy.csv", index=False)
    pd.read_csv(DESIGN_POLICY_SUMMARY).to_csv(args.output_dir / "policy_weak_direction_summary.csv", index=False)
    pd.read_csv(DESIGN_SPECTRUM).to_csv(args.output_dir / "design_singular_spectrum.csv", index=False)
    pd.read_csv(DESIGN_LOADINGS).to_csv(args.output_dir / "weak_direction_loadings.csv", index=False)
    (args.output_dir / "design_identifiability.html").write_bytes(DESIGN_DASHBOARD.read_bytes())
    pd.read_csv(OPTIMUM_PATH_SUMMARY).to_csv(args.output_dir / "raw_optimum_support_path_summary.csv", index=False)
    pd.read_csv(OPTIMUM_PATH_ROWS).to_csv(args.output_dir / "raw_optimum_support_paths.csv", index=False)
    (args.output_dir / "raw_optimum_support_paths.html").write_bytes(OPTIMUM_PATH_DASHBOARD.read_bytes())
    pd.read_csv(TRIMMED_CALIBRATION).to_csv(args.output_dir / "trimmed_calibration_metrics.csv", index=False)
    (args.output_dir / "trimmed_calibration.html").write_bytes(TRIMMED_CALIBRATION_DASHBOARD.read_bytes())
    pd.read_csv(PROVENANCE_SUMMARY).to_csv(args.output_dir / "swarm_provenance_summary.csv", index=False)
    pd.read_csv(PROVENANCE_METRIC_RECOMPUTATION).to_csv(args.output_dir / "frozen_metric_recomputation.csv", index=False)
    write_baseline_comparison(args.output_dir, baselines)
    write_calibration_figure(args.output_dir)
    write_support_figure(args.output_dir)
    write_starcoder_figure(args.output_dir)
    write_gate_figure(args.output_dir, scorecard)
    write_report(args.output_dir, registry, scorecard, baselines)
    manifest = {
        "frozen_manifest_digest": EXPECTED_MANIFEST_DIGEST,
        "registry_rows": len(registry),
        "baseline_metric_rows": len(baselines),
        "candidate_metric_rows": len(candidates),
        "all_screen_metric_rows": len(screens),
        "promoted_candidates": int(scorecard["all_required_gates_pass"].sum()),
        "inputs": {
            str(path.relative_to(RESEARCH_DIR)): sha256(path)
            for path in sorted(set((*PROTOCOL_INPUTS, *EVIDENCE_INPUTS, *screen_paths)))
        },
        "source_code": {
            str(path.relative_to(RESEARCH_DIR)): sha256(path)
            for path in sorted((*SCRIPT_DIR.glob("*.py"), *SCRIPT_DIR.glob("*.md")))
        },
        "outputs": {
            path.name: sha256(path)
            for path in sorted(args.output_dir.iterdir())
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
