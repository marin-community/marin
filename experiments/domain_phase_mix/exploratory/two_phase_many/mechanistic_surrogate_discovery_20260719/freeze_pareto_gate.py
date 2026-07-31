# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scipy",
# ]
# ///
"""Freeze the expanded Pareto baseline, acceptance gate, and data-use ledger."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
RESEARCH_DIR = SCRIPT_DIR.parent
DEFAULT_DASHBOARD = RESEARCH_DIR / "mixture_fit_debugger/src/generated/dashboard_data.json"
DEFAULT_ADVERSARIAL_MANIFEST = (
    RESEARCH_DIR / "reference_outputs/delphi_3e18_adversarial_stress_panel_20260716/candidate_manifest.csv"
)
DEFAULT_OUTPUT = RESEARCH_DIR / "reference_outputs/mechanistic_surrogate_discovery_20260719/frozen_gate"
FORBIDDEN_CONFIRMATORY_TOKENS = (
    "frontier_phase_fiber",
    "dm-delphi-frontier-phase-fiber",
    "phase_fiber_3e18_20260719",
)
BASELINE_MODELS = (
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
EXTERNAL_MODELS = (
    "inverse_deficit_log_link",
    "early_family_asymmetric",
)
CORE_OOF_PANELS = (
    ("300m", "uncheatable", "single_phase"),
    ("300m", "uncheatable", "two_phase"),
    ("300m", "table9", "single_phase"),
    ("300m", "table9", "two_phase"),
    ("delphi_3e18", "uncheatable", "two_phase"),
    ("delphi_3e18", "table9", "two_phase"),
    ("production", "uncheatable", "two_phase"),
    ("starcoder_cosine", "starcoder_bpb", "two_phase"),
    ("starcoder_wsd80", "starcoder_bpb", "two_phase"),
)
LOWER_TAIL_FRACTION = 0.15
LOWER_TAIL_MIN_COUNT = 5
OPTIMISM_THRESHOLD = 0.05
CALIBRATION_BINS = 5
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260719


@dataclass(frozen=True)
class Gate:
    """Immutable promotion criteria for the second discovery drive."""

    paired_bootstrap_replicates: int = BOOTSTRAP_REPLICATES
    paired_bootstrap_seed: int = BOOTSTRAP_SEED
    paired_bootstrap_confidence: float = 0.95
    core_oof_rmse_relative_tolerance: float = 0.05
    policy_matched_regret_at_1_absolute_tolerance: float = 0.002
    optimism_threshold_bpb: float = OPTIMISM_THRESHOLD
    optimism_count_rule: str = "candidate <= Pareto baseline on both Delphi 3e18 targets"
    calibration_rule: str = (
        "Move observed-on-predicted slope toward one overall and preserve it within candidate-target, policy, "
        "selection-stratum, and proposer strata."
    )
    material_improvement_rule: str = (
        "Improve at least one primary adversarial diagnostic beyond paired-bootstrap uncertainty; improve both "
        "targets, or at least two primary diagnostics on one target without material regression on the other."
    )
    one_phase_rule: str = (
        "Report both the algebraically phase-tied restriction and an independent fit of the same restricted form."
    )
    transfer_rule: str = "Transfer across at least two independent swarms or both StarCoder schedules."
    mechanism_rule: str = "Every retained mechanism must survive a nested ablation on at least two panels."
    stability_rule: str = (
        "Parameter signs agree in >=80% of grouped folds and dimensionless magnitudes remain comparable across "
        "folds or related swarms."
    )
    raw_optimum_rule: str = (
        "Raw optimum is finite, plausible, support-audited, and bootstrap-stable without deployment regularization."
    )
    adversarial_tuning_rule: str = (
        "No equations, feature definitions, hyperparameters, or output calibration are tuned directly to exposed "
        "adversarial target values; evaluate frozen mechanistic batches only."
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def assert_confirmatory_absent(path: Path) -> None:
    raw = path.read_bytes().lower()
    for token in FORBIDDEN_CONFIRMATORY_TOKENS:
        if token.encode() in raw:
            raise ValueError(f"Untouched confirmatory token {token!r} found in {path}")


def finite_pairs(observed: Iterable[object], predicted: Iterable[object]) -> tuple[np.ndarray, np.ndarray]:
    y = pd.to_numeric(pd.Series(list(observed)), errors="coerce").to_numpy(dtype=float)
    prediction = pd.to_numeric(pd.Series(list(predicted)), errors="coerce").to_numpy(dtype=float)
    valid = np.isfinite(y) & np.isfinite(prediction)
    return y[valid], prediction[valid]


def regret_at_k(observed: np.ndarray, predicted: np.ndarray, k: int) -> float:
    selected = np.argsort(predicted)[: min(k, len(predicted))]
    return float(np.min(observed[selected]) - np.min(observed))


def calibration_bins(observed: np.ndarray, predicted: np.ndarray) -> list[dict[str, float | int]]:
    order = np.argsort(predicted)
    output: list[dict[str, float | int]] = []
    for bin_index, indices in enumerate(np.array_split(order, min(CALIBRATION_BINS, len(order)))):
        if not len(indices):
            continue
        output.append(
            {
                "bin": bin_index,
                "n": len(indices),
                "mean_predicted": float(np.mean(predicted[indices])),
                "mean_observed": float(np.mean(observed[indices])),
                "mean_residual_predicted_minus_observed": float(np.mean(predicted[indices] - observed[indices])),
            }
        )
    return output


def metrics(
    observed: Iterable[object], predicted: Iterable[object]
) -> tuple[dict[str, float | int], list[dict[str, Any]]]:
    y, prediction = finite_pairs(observed, predicted)
    if len(y) < 3:
        raise ValueError(f"Need at least three finite observations, got {len(y)}")
    residual = prediction - y
    optimism = y - prediction
    slope, intercept = np.polyfit(prediction, y, 1)
    tail_count = min(len(y), max(LOWER_TAIL_MIN_COUNT, math.ceil(LOWER_TAIL_FRACTION * len(y))))
    tail = np.argsort(prediction)[:tail_count]
    selected = int(np.argmin(prediction))
    return (
        {
            "n": len(y),
            "rmse": float(np.sqrt(np.mean(residual**2))),
            "mae": float(np.mean(np.abs(residual))),
            "spearman": float(spearmanr(y, prediction).statistic),
            "bias_predicted_minus_observed": float(np.mean(residual)),
            "calibration_slope_observed_on_predicted": float(slope),
            "calibration_intercept_observed_on_predicted": float(intercept),
            "regret_at_1": regret_at_k(y, prediction, 1),
            "regret_at_3": regret_at_k(y, prediction, 3),
            "regret_at_5": regret_at_k(y, prediction, 5),
            "lower_tail_optimism": float(np.mean(np.maximum(optimism[tail], 0.0))),
            "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
            "optimism_gt_0p05_count": int(np.sum(optimism > OPTIMISM_THRESHOLD)),
            "worst_optimism": float(np.max(optimism)),
            "selected_optimism": float(optimism[selected]),
            "selected_observed": float(y[selected]),
            "selected_predicted": float(prediction[selected]),
        },
        calibration_bins(y, prediction),
    )


def delphi_development_layer(row: Mapping[str, Any]) -> str:
    panel = str(row["panel"])
    if panel == "delphi_3e18_adversarial_stress_panel_20260716":
        return "adversarial_120"
    if panel == "delphi_one_phase_augmented_swarm_3e18_20260715":
        return "matched_one_phase_238"
    return "historical_352"


def split_specs(swarm_id: str, rows: list[Mapping[str, Any]], policy: str) -> list[tuple[str, np.ndarray]]:
    fit = np.asarray([row["split"] == "fit" for row in rows], dtype=bool)
    heldout = np.asarray([row["split"] == "heldout" and not bool(row["isSharedAlias"]) for row in rows], dtype=bool)
    specs = [("fit_oof", fit)]
    if not heldout.any():
        return specs
    specs.extend(
        [
            ("heldout_all", heldout),
            (
                "heldout_policy_matched",
                heldout & np.asarray([row["policyFamily"] == policy for row in rows], dtype=bool),
            ),
        ]
    )
    if swarm_id != "delphi_3e18":
        return specs
    for layer in ("historical_352", "matched_one_phase_238", "adversarial_120"):
        layer_mask = heldout & np.asarray([delphi_development_layer(row) == layer for row in rows], dtype=bool)
        specs.append((layer, layer_mask))
        for policy_class in ("single_phase", "two_phase"):
            specs.append(
                (
                    f"{layer}__{policy_class}",
                    layer_mask & np.asarray([row["policyFamily"] == policy_class for row in rows], dtype=bool),
                )
            )
    adversarial = heldout & np.asarray([delphi_development_layer(row) == "adversarial_120" for row in rows], dtype=bool)
    for candidate_target in ("uncheatable", "table9"):
        target_mask = adversarial & np.asarray(
            [row.get("candidateTarget") == candidate_target for row in rows], dtype=bool
        )
        specs.append((f"adversarial_candidate_{candidate_target}", target_mask))
        for policy_class in ("single_phase", "two_phase"):
            specs.append(
                (
                    f"adversarial_candidate_{candidate_target}__{policy_class}",
                    target_mask & np.asarray([row["policyFamily"] == policy_class for row in rows], dtype=bool),
                )
            )
    return specs


def dashboard_metric_rows(bundle: Mapping[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metric_rows: list[dict[str, Any]] = []
    bin_rows: list[dict[str, Any]] = []
    for swarm_id, swarm in bundle["swarms"].items():
        rows = swarm["rows"]
        for target_id, policies in swarm["predictions"].items():
            for policy, models in policies.items():
                for model_id in BASELINE_MODELS:
                    if model_id not in models:
                        continue
                    prediction = np.asarray(models[model_id]["prediction"], dtype=float)
                    observed = np.asarray([row["observed"].get(target_id) for row in rows], dtype=float)
                    for split_name, mask in split_specs(swarm_id, rows, policy):
                        if int(mask.sum()) < 3:
                            continue
                        summary, bins = metrics(observed[mask], prediction[mask])
                        parameter_count = swarm["fits"][target_id][policy][model_id]["parameterCount"]
                        metric_rows.append(
                            {
                                "source": "dashboard_source_predictions",
                                "swarm": swarm_id,
                                "target": target_id,
                                "policy": policy,
                                "model": model_id,
                                "split": split_name,
                                "parameter_count": parameter_count,
                                **summary,
                            }
                        )
                        bin_rows.extend(
                            {
                                "source": "dashboard_source_predictions",
                                "swarm": swarm_id,
                                "target": target_id,
                                "policy": policy,
                                "model": model_id,
                                "split": split_name,
                                **bin_record,
                            }
                            for bin_record in bins
                        )
    return metric_rows, bin_rows


def external_historical_rows(bundle: Mapping[str, Any]) -> list[dict[str, Any]]:
    selections = (
        (
            "model_improvement_round2_conditioned_replay_link_20260716/predictions.csv",
            "inverse_power_deficit_conditioned_replay",
            "log_reducible_bpb",
            "inverse_deficit_log_link",
        ),
        (
            "deficit_output_link_asymmetric_20260716/predictions.csv",
            "inverse_power_deficit_early_family_asymmetric_surplus",
            None,
            "early_family_asymmetric",
        ),
    )
    output: list[dict[str, Any]] = []
    delphi_rows = {
        row["name"]: row
        for row in bundle["swarms"]["delphi_3e18"]["rows"]
        if row["split"] == "heldout" and not row["isSharedAlias"]
    }
    for relative_path, variant, link, model_name in selections:
        prediction_path = RESEARCH_DIR / "reference_outputs" / relative_path
        assert_confirmatory_absent(prediction_path)
        predictions = pd.read_csv(prediction_path)
        selected = predictions.loc[predictions["deficit_variant"].eq(variant)]
        if link is not None:
            selected = selected.loc[selected["link"].eq(link)]
        else:
            selected = selected.loc[
                (selected["dataset"].str.contains("uncheatable") & selected["link"].eq("identity_raw_bpb"))
                | (selected["dataset"].str.contains("table9") & selected["link"].eq("log_reducible_bpb"))
            ]
        for dataset, local in selected.groupby("dataset", sort=True):
            target = "table9" if "table9" in dataset else "uncheatable"
            for source_split, frame in local.groupby("split", sort=True):
                if source_split == "fit_oof":
                    split_frames = [("fit_oof", frame)]
                else:
                    policy_matched = frame.loc[
                        frame["row_id"].map(lambda row_id: delphi_rows[str(row_id)]["policyFamily"]) == "two_phase"
                    ]
                    split_frames = [("historical_352", frame), ("historical_352__two_phase", policy_matched)]
                for split_name, metric_frame in split_frames:
                    summary, _ = metrics(metric_frame["observed"], metric_frame["predicted"])
                    output.append(
                        {
                            "source": str(prediction_path.relative_to(RESEARCH_DIR)),
                            "swarm": "delphi_3e18",
                            "target": target,
                            "policy": "two_phase",
                            "model": model_name,
                            "split": split_name,
                            "parameter_count": "source fit artifact",
                            **summary,
                        }
                    )
    return output


def adversarial_external_rows(
    bundle: Mapping[str, Any], manifest_path: Path
) -> tuple[list[dict[str, Any]], pd.DataFrame]:
    manifest = pd.read_csv(manifest_path)
    rows = bundle["swarms"]["delphi_3e18"]["rows"]
    row_by_candidate: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if row["panel"] != "delphi_3e18_adversarial_stress_panel_20260716":
            continue
        for candidate_id in manifest["candidate_id"]:
            if str(candidate_id) in row["name"]:
                row_by_candidate[str(candidate_id)] = row
                break
    if len(row_by_candidate) != len(manifest):
        raise ValueError(f"Matched {len(row_by_candidate)}/{len(manifest)} adversarial candidates")

    output: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    prediction_columns = {
        "inverse_deficit_log_link": "predicted_inverse_deficit_log_link",
        "early_family_asymmetric": "predicted_early_family_asymmetric",
    }
    for target in ("uncheatable", "table9"):
        target_manifest = manifest.loc[manifest["target"].eq(target)].copy()
        target_manifest["observed"] = [
            row_by_candidate[str(candidate_id)]["observed"][target] for candidate_id in target_manifest["candidate_id"]
        ]
        for model, prediction_column in prediction_columns.items():
            summary, _ = metrics(target_manifest["observed"], target_manifest[prediction_column])
            output.append(
                {
                    "source": str(manifest_path.relative_to(RESEARCH_DIR)),
                    "swarm": "delphi_3e18",
                    "target": target,
                    "policy": "two_phase",
                    "model": model,
                    "split": f"adversarial_candidate_{target}",
                    "parameter_count": "source fit artifact",
                    **summary,
                }
            )
            prediction_rows.extend(
                {
                    "target": target,
                    "candidate_id": row.candidate_id,
                    "policy_class": row.policy_class,
                    "selection_stratum": row.selection_stratum,
                    "origin": row.origin,
                    "proposal_models": row.proposal_models,
                    "observed": row.observed,
                    "model": model,
                    "predicted": getattr(row, prediction_column),
                }
                for row in target_manifest.itertuples(index=False)
            )
    return output, pd.DataFrame(prediction_rows)


def dashboard_adversarial_predictions(bundle: Mapping[str, Any], manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path)
    swarm = bundle["swarms"]["delphi_3e18"]
    candidate_index: dict[str, int] = {}
    for index, row in enumerate(swarm["rows"]):
        if row["panel"] != "delphi_3e18_adversarial_stress_panel_20260716":
            continue
        matches = [candidate_id for candidate_id in manifest["candidate_id"] if str(candidate_id) in row["name"]]
        if len(matches) != 1:
            raise ValueError(f"Expected one adversarial candidate match for {row['name']}, got {matches}")
        candidate_index[str(matches[0])] = index
    prediction_rows: list[dict[str, Any]] = []
    for target in ("uncheatable", "table9"):
        models = swarm["predictions"][target]["two_phase"]
        for candidate in manifest.loc[manifest["target"].eq(target)].itertuples(index=False):
            index = candidate_index[str(candidate.candidate_id)]
            observed = swarm["rows"][index]["observed"][target]
            for model in BASELINE_MODELS:
                prediction_rows.append(
                    {
                        "target": target,
                        "candidate_id": candidate.candidate_id,
                        "policy_class": candidate.policy_class,
                        "selection_stratum": candidate.selection_stratum,
                        "origin": candidate.origin,
                        "proposal_models": candidate.proposal_models,
                        "observed": observed,
                        "model": model,
                        "predicted": models[model]["prediction"][index],
                    }
                )
    return pd.DataFrame(prediction_rows)


def adversarial_strata_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    output: list[dict[str, Any]] = []
    for (target, model), model_frame in predictions.groupby(["target", "model"], sort=True):
        for stratum_column in ("policy_class", "selection_stratum", "origin", "proposal_models"):
            for stratum_value, stratum in model_frame.groupby(stratum_column, sort=True):
                if len(stratum) < 3:
                    continue
                summary, _ = metrics(stratum["observed"], stratum["predicted"])
                output.append(
                    {
                        "target": target,
                        "model": model,
                        "stratum_type": stratum_column,
                        "stratum_value": stratum_value,
                        **summary,
                    }
                )
    return pd.DataFrame(output).sort_values(["target", "stratum_type", "stratum_value", "rmse", "model"])


def bootstrap_statistics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, np.ndarray]:
    residual = predicted - observed
    prediction_centered = predicted - predicted.mean(axis=1, keepdims=True)
    observed_centered = observed - observed.mean(axis=1, keepdims=True)
    denominator = np.sum(prediction_centered**2, axis=1)
    slope = np.divide(
        np.sum(prediction_centered * observed_centered, axis=1),
        denominator,
        out=np.full(len(predicted), np.nan),
        where=denominator > 0,
    )
    selected = np.argmin(predicted, axis=1)
    return {
        "rmse": np.sqrt(np.mean(residual**2, axis=1)),
        "regret_at_1": observed[np.arange(len(observed)), selected] - np.min(observed, axis=1),
        "calibration_distance": np.abs(slope - 1.0),
        "worst_optimism": np.max(observed - predicted, axis=1),
    }


def paired_bootstrap_comparisons(predictions: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    statistics = ("rmse", "regret_at_1", "calibration_distance", "worst_optimism")
    output: list[dict[str, Any]] = []
    for target, target_frame in predictions.groupby("target", sort=True):
        wide = target_frame.pivot(index="candidate_id", columns="model", values="predicted")
        metadata = target_frame.drop_duplicates("candidate_id").set_index("candidate_id")
        wide = wide.loc[metadata.index]
        observed = metadata["observed"].to_numpy(dtype=float)
        cell = metadata["policy_class"].astype(str) + "::" + metadata["selection_stratum"].astype(str)
        cell_indices = [np.flatnonzero(cell.to_numpy() == value) for value in sorted(cell.unique())]
        bootstrap_indices = []
        for _ in range(BOOTSTRAP_REPLICATES):
            bootstrap_indices.append(
                np.concatenate([rng.choice(indices, size=len(indices), replace=True) for indices in cell_indices])
            )
        index_matrix = np.stack(bootstrap_indices)
        observed_samples = observed[index_matrix]
        model_statistics: dict[str, dict[str, np.ndarray]] = {}
        observed_statistics: dict[str, dict[str, float]] = {}
        models = sorted(wide.columns)
        for model in models:
            prediction = wide[model].to_numpy(dtype=float)
            model_statistics[model] = bootstrap_statistics(observed_samples, prediction[index_matrix])
            summary, _ = metrics(observed, prediction)
            observed_statistics[model] = {
                "rmse": float(summary["rmse"]),
                "regret_at_1": float(summary["regret_at_1"]),
                "calibration_distance": abs(float(summary["calibration_slope_observed_on_predicted"]) - 1.0),
                "worst_optimism": float(summary["worst_optimism"]),
            }
        for left_index, left in enumerate(models):
            for right in models[left_index + 1 :]:
                for statistic in statistics:
                    observed_difference = observed_statistics[left][statistic] - observed_statistics[right][statistic]
                    differences = model_statistics[left][statistic] - model_statistics[right][statistic]
                    output.append(
                        {
                            "target": target,
                            "left_model": left,
                            "right_model": right,
                            "statistic": statistic,
                            "left_minus_right": observed_difference,
                            "bootstrap_q025": float(np.quantile(differences, 0.025)),
                            "bootstrap_q975": float(np.quantile(differences, 0.975)),
                            "probability_left_better": float(np.mean(differences < 0.0)),
                            "replicates": BOOTSTRAP_REPLICATES,
                            "stratification": "policy_class x selection_stratum",
                        }
                    )
    return pd.DataFrame(output)


def pareto_models(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    focus = metrics_frame.loc[
        metrics_frame["swarm"].eq("delphi_3e18")
        & metrics_frame["split"].isin(
            [
                "fit_oof",
                "historical_352",
                "matched_one_phase_238",
                "adversarial_candidate_uncheatable",
                "adversarial_candidate_table9",
            ]
        )
    ].copy()
    focus["calibration_distance"] = (focus["calibration_slope_observed_on_predicted"] - 1.0).abs()
    objectives = [
        "rmse",
        "regret_at_1",
        "calibration_distance",
        "optimism_gt_0p05_count",
        "worst_optimism",
    ]
    summaries: list[dict[str, Any]] = []
    for model, frame in focus.groupby("model", sort=True):
        record: dict[str, Any] = {"model": model, "applicable_rows": len(frame)}
        for objective in objectives:
            values = pd.to_numeric(frame[objective], errors="coerce")
            record[f"mean_{objective}"] = float(values.mean()) if values.notna().any() else np.nan
            record[f"wins_{objective}"] = 0
        summaries.append(record)
    summary = pd.DataFrame(summaries)
    for _, panel in focus.groupby(["target", "split"], sort=True):
        for objective in objectives:
            finite = panel.loc[pd.to_numeric(panel[objective], errors="coerce").notna()]
            if finite.empty:
                continue
            winner = finite.sort_values([objective, "model"]).iloc[0]["model"]
            summary.loc[summary["model"].eq(winner), f"wins_{objective}"] += 1
    return summary.sort_values(["wins_rmse", "wins_regret_at_1", "model"], ascending=[False, False, True])


def initial_data_use_ledger(metrics_frame: pd.DataFrame) -> pd.DataFrame:
    now = datetime.now(UTC).isoformat()
    models = sorted(set(metrics_frame["model"]))
    return pd.DataFrame(
        [
            {
                "timestamp": now,
                "round_id": "baseline_reconstruction",
                "candidate_id": model,
                "candidate_family": model,
                "hyperparameters": "pre-existing source-selected configuration",
                "adversarial_outcomes_available_before_proposal": True,
                "adversarial_outcomes_inspected_before_proposal": False,
                "observations_inspiring_mechanism": "pre-existing model; no new proposal",
                "novelty_class": "existing exposed baseline",
                "evaluation_status": "adversarial development outcomes reconstructed",
                "evidence_path": "frozen_gate/baseline_metrics.csv",
                "notes": "Adversarial outcomes were observed before this drive; this row records exposure, not confirmation.",
            }
            for model in models
        ]
    )


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    rows = [header, divider]
    for record in frame[columns].to_dict(orient="records"):
        values = [f"{value:.5f}" if isinstance(value, float) else str(value) for value in record.values()]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dashboard", type=Path, default=DEFAULT_DASHBOARD)
    parser.add_argument("--adversarial-manifest", type=Path, default=DEFAULT_ADVERSARIAL_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    assert_confirmatory_absent(args.dashboard)
    assert_confirmatory_absent(args.adversarial_manifest)
    bundle = json.loads(args.dashboard.read_text())
    metric_rows, bin_rows = dashboard_metric_rows(bundle)
    metric_rows.extend(external_historical_rows(bundle))
    adversarial_rows, external_adversarial_predictions = adversarial_external_rows(bundle, args.adversarial_manifest)
    metric_rows.extend(adversarial_rows)
    dashboard_predictions = dashboard_adversarial_predictions(bundle, args.adversarial_manifest)
    adversarial_predictions = pd.concat(
        [dashboard_predictions, external_adversarial_predictions], ignore_index=True, sort=False
    )

    metrics_frame = pd.DataFrame(metric_rows).sort_values(["swarm", "target", "split", "rmse", "model"])
    bins_frame = pd.DataFrame(bin_rows)
    strata_frame = adversarial_strata_metrics(adversarial_predictions)
    bootstrap_frame = paired_bootstrap_comparisons(adversarial_predictions)
    pareto_frame = pareto_models(metrics_frame)
    ledger_frame = initial_data_use_ledger(metrics_frame)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "baseline_metrics": args.output_dir / "baseline_metrics.csv",
        "calibration_bins": args.output_dir / "calibration_bins.csv",
        "adversarial_strata": args.output_dir / "adversarial_strata_metrics.csv",
        "pareto_summary": args.output_dir / "pareto_summary.csv",
        "data_use_ledger": args.output_dir / "data_use_ledger.csv",
        "adversarial_predictions": args.output_dir / "adversarial_target_matched_predictions.csv",
        "paired_bootstrap": args.output_dir / "paired_bootstrap_comparisons.csv",
    }
    metrics_frame.to_csv(paths["baseline_metrics"], index=False)
    bins_frame.to_csv(paths["calibration_bins"], index=False)
    strata_frame.to_csv(paths["adversarial_strata"], index=False)
    pareto_frame.to_csv(paths["pareto_summary"], index=False)
    ledger_frame.to_csv(paths["data_use_ledger"], index=False)
    adversarial_predictions.to_csv(paths["adversarial_predictions"], index=False)
    bootstrap_frame.to_csv(paths["paired_bootstrap"], index=False)

    gate = Gate()
    gate_record = {
        "frozen_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "dashboard": str(args.dashboard.relative_to(RESEARCH_DIR)),
            "dashboard_sha256": sha256(args.dashboard),
            "adversarial_manifest": str(args.adversarial_manifest.relative_to(RESEARCH_DIR)),
            "adversarial_manifest_sha256": sha256(args.adversarial_manifest),
            "external_prediction_files": {
                "inverse_deficit_log_link": sha256(
                    RESEARCH_DIR
                    / "reference_outputs/model_improvement_round2_conditioned_replay_link_20260716/predictions.csv"
                ),
                "early_family_asymmetric": sha256(
                    RESEARCH_DIR / "reference_outputs/deficit_output_link_asymmetric_20260716/predictions.csv"
                ),
            },
        },
        "forbidden_confirmatory_tokens_checked_absent": list(FORBIDDEN_CONFIRMATORY_TOKENS),
        "baseline_models": list(BASELINE_MODELS),
        "external_models": list(EXTERNAL_MODELS),
        "core_oof_panels": [list(panel) for panel in CORE_OOF_PANELS],
        "acceptance_gate": asdict(gate),
    }
    gate_path = args.output_dir / "acceptance_gate.json"
    gate_path.write_text(json.dumps(gate_record, indent=2, sort_keys=True) + "\n")
    manifest = {
        "acceptance_gate_sha256": sha256(gate_path),
        **{f"{name}_sha256": sha256(path) for name, path in paths.items()},
    }
    manifest_path = args.output_dir / "frozen_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    adversarial_focus = metrics_frame.loc[
        metrics_frame["swarm"].eq("delphi_3e18")
        & metrics_frame["split"].isin(["adversarial_candidate_uncheatable", "adversarial_candidate_table9"])
    ]
    report = [
        "# Frozen Pareto baseline and acceptance gate",
        "",
        f"Frozen manifest digest: `{sha256(manifest_path)}`.",
        "",
        "The exposed adversarial panel is development evidence. Its candidate-target labels and selection strata are preserved. The running frontier phase-fiber panel is absent from every input.",
        "",
        "## Adversarial target-matched Pareto",
        "",
        markdown_table(
            adversarial_focus.sort_values(["target", "rmse", "model"]),
            [
                "target",
                "model",
                "n",
                "rmse",
                "spearman",
                "calibration_slope_observed_on_predicted",
                "regret_at_1",
                "optimism_gt_0p05_count",
                "worst_optimism",
            ],
        ),
        "",
        "## Immutable promotion gate",
        "",
        *[f"- `{key}`: {value}" for key, value in asdict(gate).items()],
    ]
    (args.output_dir / "report.md").write_text("\n".join(report) + "\n")
    print(json.dumps({**manifest, "manifest_sha256": sha256(manifest_path)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
