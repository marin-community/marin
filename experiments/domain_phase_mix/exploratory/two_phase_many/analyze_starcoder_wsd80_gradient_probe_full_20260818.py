# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E501  # Generated Markdown tables keep their source rows intact.

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs>=2025.5.1",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scipy>=1.14",
# ]
# ///
"""Analyze the frozen full StarCoder WSD80 gradient-conflict panel."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
DESIGN_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_gradient_conflict_design_20260811_v9"
RELEASE_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_gradient_probe_release_v6_20260816"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "starcoder_wsd80_gradient_probe_full_results_20260818"
RESULT_ROOT = (
    "gs://marin-us-central1/analysis/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_probe_review_v9_release_v6_20260816"
)
TRAINING_ROOT = (
    "gs://marin-us-central1/checkpoints/pinlin_calvin_xu/data_mixture/"
    "starcoder_wsd80_gradient_conflict_review_v9_20260811/trajectories"
)
TRAINING_VERSION = "2026.08.11.9"
PROBE_VERSION = "2026.08.16.6"
BOOTSTRAP_DRAWS = 100_000
BOOTSTRAP_SEED = 20260818
MAX_WORKERS = 64

PL_TARGET = "paloma_programming_languages"
C4_TARGET = "paloma_c4_en"
GITHUB_TARGET = "uncheatable_github_python"
WIKIPEDIA_TARGET = "uncheatable_wikipedia_english"
TARGETS = (PL_TARGET, C4_TARGET, GITHUB_TARGET, WIKIPEDIA_TARGET)
TARGET_LABELS = {
    PL_TARGET: "Programming Languages",
    C4_TARGET: "C4 English",
    GITHUB_TARGET: "GitHub Python",
    WIKIPEDIA_TARGET: "Wikipedia English",
}
METRIC_KEYS = {
    PL_TARGET: "eval/paloma/dolma_100_programing_languages-llama3/bpb",
    C4_TARGET: "eval/paloma/c4_en-llama3/bpb",
    GITHUB_TARGET: "eval/uncheatable_eval/github_python-llama3/bpb",
    WIKIPEDIA_TARGET: "eval/uncheatable_eval/wikipedia_english-llama3/bpb",
    "uncheatable": "eval/uncheatable_eval/bpb",
}
PRIMARY_Q = (0.25, 0.35, 0.45, 0.55)
H2_STATES = {
    "fraction_0p40": "mid",
    "fraction_0p55": "mid",
    "decay_minus_256": "late_pre_decay",
    "decay_minus_64": "late_pre_decay",
}
PROBE_ANALYSIS_ROLES = {
    "h2_primary",
    "h3_full_support_pair",
    "h3_second_pool_sensitivity",
    "h5_preregistered_profile",
    "h2_aggregate_matched",
}
SOURCE_IDS = (
    "starcoder_excluded_global",
    "starcoder_support_reference",
    "starcoder_on_policy",
    "nemotron_aggregate",
)
PLOT_TEMPLATE = "plotly_white"
PLOT_COLORS = {
    "tied": "#d73027",
    "two_phase": "#1a9850",
    "calibration": "#4575b4",
    "validation": "#d73027",
    "full": "#313695",
    "m100a": "#f46d43",
    "m100b": "#fee08b",
}


def _gcs_path(path: str) -> str:
    return path.removeprefix("gs://")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_json(fs: gcsfs.GCSFileSystem, path: str) -> dict[str, Any]:
    with fs.open(_gcs_path(path), "rb") as handle:
        return json.load(handle)


def _read_jsonl_last(fs: gcsfs.GCSFileSystem, path: str) -> dict[str, Any]:
    with fs.open(_gcs_path(path), "rb") as handle:
        lines = [line for line in handle.read().splitlines() if line.strip()]
    if not lines:
        raise RuntimeError(f"Empty JSONL artifact: {path}")
    return json.loads(lines[-1])


def _parallel_map(function: Callable[[Any], Any], values: Iterable[Any]) -> list[Any]:
    values = list(values)
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        return list(executor.map(function, values))


def exact_sign_flip_p(differences: Sequence[float], *, alternative: str) -> float:
    """Return an exact paired sign-flip p-value for a mean contrast."""
    values = np.asarray(differences, dtype=float)
    observed = float(values.mean())
    null = np.asarray([np.mean(values * signs) for signs in itertools.product((-1.0, 1.0), repeat=len(values))])
    tolerance = 1e-15
    if alternative == "greater":
        return float(np.mean(null >= observed - tolerance))
    if alternative == "less":
        return float(np.mean(null <= observed + tolerance))
    if alternative == "two-sided":
        return float(np.mean(np.abs(null) >= abs(observed) - tolerance))
    raise ValueError(f"Unknown alternative: {alternative}")


def bootstrap_mean_interval(values: Sequence[float]) -> tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    indices = rng.integers(0, len(array), size=(BOOTSTRAP_DRAWS, len(array)))
    means = array[indices].mean(axis=1)
    return tuple(float(value) for value in np.quantile(means, (0.025, 0.975)))


def _contrast_summary(
    values: Sequence[float],
    *,
    name: str,
    alternative: str,
    interpretation: str,
    evidence_role: str,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=float)
    ci_low, ci_high = bootstrap_mean_interval(array)
    return {
        "contrast": name,
        "n_paired_seeds": len(array),
        "mean_effect_bpb": float(array.mean()),
        "sd_effect_bpb": float(array.std(ddof=1)),
        "bootstrap_ci95_low": ci_low,
        "bootstrap_ci95_high": ci_high,
        "exact_sign_flip_p": exact_sign_flip_p(array, alternative=alternative),
        "alternative": alternative,
        "positive_pairs": int(np.sum(array > 0)),
        "interpretation": interpretation,
        "evidence_role": evidence_role,
    }


def load_endpoint_metrics(fs: gcsfs.GCSFileSystem, trajectories: pd.DataFrame) -> pd.DataFrame:
    def load(row: Mapping[str, Any]) -> dict[str, Any]:
        path = f"{TRAINING_ROOT}/{row['trajectory_id']}/{TRAINING_VERSION}/" "checkpoints/eval_metrics.jsonl"
        final = _read_jsonl_last(fs, path)
        expected_step = int(row["total_steps"]) - 1
        if int(final["step"]) != expected_step:
            raise RuntimeError(
                f"Final endpoint step mismatch for {row['trajectory_id']}: " f"{final['step']} != {expected_step}"
            )
        result = dict(row)
        result["endpoint_step"] = int(final["step"])
        result["endpoint_uri"] = path
        for target, key in METRIC_KEYS.items():
            if key not in final:
                raise RuntimeError(f"Missing endpoint metric {key} for {row['trajectory_id']}")
            result[f"{target}_bpb"] = float(final[key])
        return result

    return pd.DataFrame(_parallel_map(load, trajectories.to_dict("records")))


def load_result_documents(
    fs: gcsfs.GCSFileSystem,
    manifest: pd.DataFrame,
    *,
    kind: str,
    release_sha256: str,
) -> list[dict[str, Any]]:
    def load(row: Mapping[str, Any]) -> dict[str, Any]:
        path = f"{RESULT_ROOT}/full/{kind}/{row['group_id']}/{PROBE_VERSION}/" f"rows/{row['row_id']}.json"
        document = _read_json(fs, path)
        if document["release_sha256"] != release_sha256:
            raise RuntimeError(f"Release mismatch at {path}")
        if str(document["row"]["row_id"]) != str(row["row_id"]):
            raise RuntimeError(f"Row identity mismatch at {path}")
        if document.get("endpoint_metrics_read") is not False:
            raise RuntimeError(f"Endpoint leakage marker at {path}")
        document["source_uri"] = path
        return document

    return _parallel_map(load, manifest.to_dict("records"))


def flatten_optimizer(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for document in documents:
        row = document["row"]
        for target, statistics in document["target_utility_statistics"].items():
            for component in ("trunk", "full", "head", "embedding"):
                projected = statistics["projected_optimizer_update"][component]
                rows.append(
                    {
                        "parent_trajectory_id": row["parent_trajectory_id"],
                        "parent_checkpoint_label": row["parent_checkpoint_label"],
                        "starcoder_weight": float(row["starcoder_weight"]),
                        "target": target,
                        "component": component,
                        "utility_dot": float(projected["dot"]),
                        "utility_cosine": float(projected["cosine"]),
                        "target_gradient_norm": float(projected["left_norm"]),
                        "mixture_update_norm": float(projected["right_norm"]),
                        "row_id": row["row_id"],
                        "source_uri": document["source_uri"],
                    }
                )
    return pd.DataFrame(rows)


def flatten_rollouts(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for document in documents:
        row = document["row"]
        for readout in document["readouts"]:
            rows.append(
                {
                    "parent_trajectory_id": row["parent_trajectory_id"],
                    "parent_checkpoint_label": row["parent_checkpoint_label"],
                    "starcoder_weight": float(row["starcoder_weight"]),
                    "analysis_role": row["analysis_role"],
                    "rollout_order_seed": int(row["rollout_order_seed"]),
                    "updates": int(readout["updates"]),
                    "bpb": float(readout["bpb"]),
                    "bpb_standard_error": float(readout["bpb_standard_error"]),
                    "row_id": row["row_id"],
                    "source_uri": document["source_uri"],
                }
            )
    return pd.DataFrame(rows)


def flatten_probe_geometry(documents: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for document in documents:
        row = document["row"]
        target = row["distribution_id"]
        for source, statistics in document["pairwise_statistics"].items():
            if source not in SOURCE_IDS:
                continue
            for component in ("trunk", "full", "head", "embedding"):
                gradient = statistics["projected_gradient"][component]
                update = statistics["projected_optimizer_update"][component]
                rows.append(
                    {
                        "trajectory_id": row["trajectory_id"],
                        "checkpoint_label": row["checkpoint_label"],
                        "target": target,
                        "source": source,
                        "component": component,
                        "analysis_role": row["analysis_role"],
                        "gradient_dot": float(gradient["dot"]),
                        "gradient_cosine": float(gradient["cosine"]),
                        "target_update_source_update_dot": float(update["dot"]),
                        "target_update_source_update_cosine": float(update["cosine"]),
                        "row_id": row["row_id"],
                        "source_uri": document["source_uri"],
                    }
                )
    return pd.DataFrame(rows)


def endpoint_contrasts(endpoint: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = endpoint[endpoint["arm"].isin(("p2", "p3", "p4"))]
    pivot = primary.pivot_table(
        index=["training_seed", "support_id"],
        columns="arm",
        values=f"{PL_TARGET}_bpb",
        aggfunc="first",
    ).reset_index()
    if pivot[["p2", "p3", "p4"]].isna().any().any():
        raise RuntimeError("P2/P3/P4 endpoint pairing is incomplete")
    pivot["p4_minus_p3_gain"] = pivot["p4"] - pivot["p3"]
    pivot["p2_minus_p3_joint_gain"] = pivot["p2"] - pivot["p3"]

    summaries: list[dict[str, Any]] = []
    for support, group in pivot.groupby("support_id", sort=True):
        summaries.append(
            _contrast_summary(
                group["p4_minus_p3_gain"],
                name=f"P3 two-phase gain over aggregate-matched P4 ({support})",
                alternative="greater",
                interpretation="Positive favors the two-phase P3 policy.",
                evidence_role="preregistered_family_component_under_specified",
            )
        )
    seed_average = pivot.groupby("training_seed", as_index=False)["p4_minus_p3_gain"].mean()
    summaries.append(
        _contrast_summary(
            seed_average["p4_minus_p3_gain"],
            name="P3 two-phase gain over P4 (support-average sensitivity)",
            alternative="greater",
            interpretation="Positive favors P3; support averaging was not fully specified in the freeze.",
            evidence_role="post_hoc_support_average_sensitivity",
        )
    )
    for support, group in pivot.groupby("support_id", sort=True):
        summaries.append(
            _contrast_summary(
                group["p2_minus_p3_joint_gain"],
                name=f"Secondary P3 gain over selected tied P2 ({support})",
                alternative="two-sided",
                interpretation=(
                    "Positive favors P3, but this jointly changes aggregate and phase schedule and is not "
                    "a pure ordering contrast."
                ),
                evidence_role="secondary_historical_joint_policy_contrast",
            )
        )
    support_wide = pivot.pivot(
        index="training_seed",
        columns="support_id",
        values="p2_minus_p3_joint_gain",
    )
    if not {"m100a", "full"}.issubset(support_wide.columns):
        raise RuntimeError("P2/P3 support interaction is incomplete")
    summaries.append(
        _contrast_summary(
            support_wide["m100a"] - support_wide["full"],
            name="Secondary P3-versus-P2 gain interaction: m100a minus full",
            alternative="two-sided",
            interpretation=(
                "Positive means finite support increases P3's gain over P2; this is endpoint evidence, not "
                "the frozen H3 gradient-space estimand."
            ),
            evidence_role="secondary_endpoint_support_interaction",
        )
    )

    h5 = endpoint[endpoint["arm"].eq("b")].copy()
    h5_pivot = h5.pivot_table(
        index="training_seed",
        columns="policy_role",
        values=f"{PL_TARGET}_bpb",
        aggfunc="first",
    ).reset_index()
    required = {"boundary_beta_0p60", "boundary_beta_0p85"}
    if not required.issubset(h5_pivot.columns):
        raise RuntimeError("H5 endpoint pairing is incomplete")
    h5_pivot["beta_0p85_minus_0p60"] = h5_pivot["boundary_beta_0p85"] - h5_pivot["boundary_beta_0p60"]
    summaries.append(
        _contrast_summary(
            h5_pivot["beta_0p85_minus_0p60"],
            name="H5 endpoint: beta 0.85 minus beta 0.60",
            alternative="two-sided",
            interpretation="Positive means the earlier beta=0.60 data switch has lower BPB.",
            evidence_role="preregistered_primary",
        )
    )
    return pivot, pd.DataFrame(summaries)


def h4_validation(
    optimizer: pd.DataFrame,
    rollouts: pd.DataFrame,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    utility = optimizer[optimizer["target"].eq(PL_TARGET) & optimizer["component"].eq("trunk")].copy()
    final = rollouts[
        rollouts["updates"].eq(512)
        & rollouts["analysis_role"].isin(("h4_independent_calibration", "h4_primary_validation"))
    ].copy()
    merged = final.merge(
        utility,
        on=["parent_trajectory_id", "parent_checkpoint_label", "starcoder_weight"],
        how="left",
        validate="one_to_one",
    )
    if merged["utility_dot"].isna().any():
        raise RuntimeError("H4 rollout rows do not have matching optimizer utilities")
    merged = merged[merged["starcoder_weight"].isin(PRIMARY_Q)].copy()
    baseline = merged[merged["starcoder_weight"].eq(0.35)][["parent_trajectory_id", "utility_dot", "bpb"]].rename(
        columns={"utility_dot": "utility_at_q035", "bpb": "bpb_at_q035"}
    )
    merged = merged.merge(baseline, on="parent_trajectory_id", how="left", validate="many_to_one")
    merged["delta_utility"] = merged["utility_dot"] - merged["utility_at_q035"]
    merged["delta_bpb"] = merged["bpb"] - merged["bpb_at_q035"]

    calibration = merged[merged["analysis_role"].eq("h4_independent_calibration")]
    validation = merged[merged["analysis_role"].eq("h4_primary_validation")].copy()
    denominator = float(np.square(calibration["delta_utility"]).sum())
    if denominator == 0:
        raise RuntimeError("H4 calibration utility has zero spread")
    slope = float((calibration["delta_utility"] * calibration["delta_bpb"]).sum() / denominator)
    validation["predicted_delta_bpb"] = slope * validation["delta_utility"]
    error = validation["predicted_delta_bpb"] - validation["delta_bpb"]
    centered_total = float(np.square(validation["delta_bpb"] - validation["delta_bpb"].mean()).sum())
    r2 = 1.0 - float(np.square(error).sum()) / centered_total
    pearson = stats.pearsonr(validation["predicted_delta_bpb"], validation["delta_bpb"])
    spearman = stats.spearmanr(validation["predicted_delta_bpb"], validation["delta_bpb"])

    selections: list[dict[str, Any]] = []
    for parent, group in validation.groupby("parent_trajectory_id", sort=True):
        predicted = group.loc[group["predicted_delta_bpb"].idxmin()]
        observed = group.loc[group["bpb"].idxmin()]
        selections.append(
            {
                "parent_trajectory_id": parent,
                "predicted_q": float(predicted["starcoder_weight"]),
                "observed_q": float(observed["starcoder_weight"]),
                "selection_regret_bpb": float(predicted["bpb"] - observed["bpb"]),
            }
        )
    selection = pd.DataFrame(selections)

    spread_checks: list[dict[str, Any]] = []
    for parent, group in validation.groupby("parent_trajectory_id", sort=True):
        low = group.loc[group["bpb"].idxmin()]
        high = group.loc[group["bpb"].idxmax()]
        spread = float(high["bpb"] - low["bpb"])
        three_se = 3.0 * math.sqrt(float(low["bpb_standard_error"]) ** 2 + float(high["bpb_standard_error"]) ** 2)
        spread_checks.append(
            {
                "parent_trajectory_id": parent,
                "bpb_spread": spread,
                "three_measurement_se": three_se,
                "clears_three_se": spread > three_se,
            }
        )
    spread = pd.DataFrame(spread_checks)
    monotonic_checks: list[dict[str, Any]] = []
    for parent, group in validation.groupby("parent_trajectory_id", sort=True):
        ordered = group.sort_values("starcoder_weight")
        monotonic_checks.append(
            {
                "parent_trajectory_id": parent,
                "strict_bpb_improvement": bool(np.all(np.diff(ordered["bpb"]) < 0)),
                "strict_utility_increase": bool(np.all(np.diff(ordered["delta_utility"]) > 0)),
            }
        )
    monotonic = pd.DataFrame(monotonic_checks)
    median_spread = float(spread["bpb_spread"].median())
    validation_rmse = float(np.sqrt(np.mean(np.square(error))))
    summary = {
        "status": "exploratory_only_mapping_not_numerically_frozen",
        "mapping": "within-parent q=0.35-centered OLS through the origin; calibration seeds only",
        "calibration_seed_count": int(calibration["parent_trajectory_id"].nunique()),
        "validation_seed_count": int(validation["parent_trajectory_id"].nunique()),
        "slope_delta_bpb_per_utility": slope,
        "validation_rmse_bpb": validation_rmse,
        "validation_mae_bpb": float(np.mean(np.abs(error))),
        "validation_r2": r2,
        "validation_pearson_r": float(pearson.statistic),
        "validation_pearson_p": float(pearson.pvalue),
        "validation_spearman_rho": float(spearman.statistic),
        "validation_spearman_p": float(spearman.pvalue),
        "median_selection_regret_bpb": float(selection["selection_regret_bpb"].median()),
        "mean_selection_regret_bpb": float(selection["selection_regret_bpb"].mean()),
        "exact_q_match_fraction": float(np.mean(selection["predicted_q"] == selection["observed_q"])),
        "three_se_clear_fraction": float(spread["clears_three_se"].mean()),
        "median_observed_spread_bpb": median_spread,
        "rmse_fraction_of_median_spread": validation_rmse / median_spread,
        "strict_bpb_improvement_fraction": float(monotonic["strict_bpb_improvement"].mean()),
        "strict_utility_increase_fraction": float(monotonic["strict_utility_increase"].mean()),
        "predicted_q_values": sorted(selection["predicted_q"].unique().tolist()),
        "observed_q_values": sorted(selection["observed_q"].unique().tolist()),
    }
    merged = merged.merge(
        selection,
        on="parent_trajectory_id",
        how="left",
        validate="many_to_one",
    ).merge(
        spread,
        on="parent_trajectory_id",
        how="left",
        validate="many_to_one",
    )
    if "predicted_delta_bpb" not in merged:
        merged["predicted_delta_bpb"] = np.nan
    merged.loc[merged["analysis_role"].eq("h4_primary_validation"), "predicted_delta_bpb"] = (
        validation.set_index(["parent_trajectory_id", "starcoder_weight"])["predicted_delta_bpb"]
        .reindex(
            pd.MultiIndex.from_frame(
                merged.loc[
                    merged["analysis_role"].eq("h4_primary_validation"),
                    ["parent_trajectory_id", "starcoder_weight"],
                ]
            )
        )
        .to_numpy()
    )
    return merged, summary


def gradient_proxy_analysis(
    geometry: pd.DataFrame,
    trajectories: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    primary = geometry[geometry["component"].eq("trunk")].merge(
        trajectories[["trajectory_id", "training_seed", "support_id", "policy_role", "arm"]],
        on="trajectory_id",
        how="left",
        validate="many_to_one",
    )
    pivot = primary.pivot_table(
        index=[
            "trajectory_id",
            "training_seed",
            "support_id",
            "policy_role",
            "arm",
            "checkpoint_label",
            "target",
            "analysis_role",
        ],
        columns="source",
        values="gradient_cosine",
        aggfunc="first",
    ).reset_index()
    required = {"starcoder_excluded_global", "nemotron_aggregate"}
    if not required.issubset(pivot.columns):
        raise RuntimeError("Gradient proxy rows omit global StarCoder or Nemotron")
    pivot["source_choice_gradient_cosine"] = pivot["starcoder_excluded_global"] - pivot["nemotron_aggregate"]

    h2 = pivot[
        pivot["analysis_role"].isin(("h2_primary", "h3_full_support_pair")) & pivot["checkpoint_label"].isin(H2_STATES)
    ].copy()
    h2["period"] = h2["checkpoint_label"].map(H2_STATES)
    temporal = h2.groupby(["training_seed", "support_id", "target", "period"], as_index=False)[
        "source_choice_gradient_cosine"
    ].mean()
    temporal = temporal.pivot_table(
        index=["training_seed", "support_id", "target"],
        columns="period",
        values="source_choice_gradient_cosine",
        aggfunc="first",
    ).reset_index()
    temporal["late_minus_mid"] = temporal["late_pre_decay"] - temporal["mid"]
    target_wide = temporal.pivot_table(
        index=["training_seed", "support_id"],
        columns="target",
        values="late_minus_mid",
        aggfunc="first",
    ).reset_index()
    target_wide["pl_minus_c4_temporal_proxy"] = target_wide[PL_TARGET] - target_wide[C4_TARGET]
    target_wide["github_minus_c4_temporal_proxy"] = target_wide[GITHUB_TARGET] - target_wide[C4_TARGET]
    target_wide["wikipedia_minus_c4_temporal_proxy"] = target_wide[WIKIPEDIA_TARGET] - target_wide[C4_TARGET]

    summaries: list[dict[str, Any]] = []
    for support, group in target_wide.groupby("support_id", sort=True):
        summaries.append(
            {
                **_contrast_summary(
                    group["pl_minus_c4_temporal_proxy"],
                    name=f"Exploratory H2 gradient-cosine proxy ({support})",
                    alternative="greater",
                    interpretation=(
                        "Positive means relative PL alignment with StarCoder versus Nemotron rises more "
                        "than C4 alignment; this is not the frozen optimizer-aware A_y statistic."
                    ),
                    evidence_role="exploratory_proxy",
                ),
                "github_same_sign": bool(
                    np.sign(group["github_minus_c4_temporal_proxy"].mean())
                    == np.sign(group["pl_minus_c4_temporal_proxy"].mean())
                ),
                "wikipedia_negative_control_mean": float(group["wikipedia_minus_c4_temporal_proxy"].mean()),
            }
        )
    paired = target_wide.pivot_table(
        index="training_seed",
        columns="support_id",
        values="pl_minus_c4_temporal_proxy",
        aggfunc="first",
    ).dropna(subset=["m100a", "full"])
    summaries.append(
        _contrast_summary(
            paired["m100a"] - paired["full"],
            name="Exploratory H3 m100a-minus-full gradient-cosine proxy",
            alternative="two-sided",
            interpretation="Not the frozen optimizer-aware H3 statistic.",
            evidence_role="exploratory_proxy",
        )
    )
    return pivot, pd.DataFrame(summaries)


def write_endpoint_plot(endpoint: pd.DataFrame, output_path: Path) -> None:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Aggregate-matched phase contrast", "Moved switch boundary"),
    )
    endpoint_primary = endpoint[endpoint["arm"].isin(("p3", "p4"))]
    for support in ("m100a", "full"):
        group = endpoint_primary[endpoint_primary["support_id"].eq(support)]
        for arm, label, color in (
            ("p4", "Tied P4", PLOT_COLORS["tied"]),
            ("p3", "Two-phase P3", PLOT_COLORS["two_phase"]),
        ):
            arm_group = group[group["arm"].eq(arm)]
            fig.add_trace(
                go.Box(
                    x=[f"{support}<br>{label}"] * len(arm_group),
                    y=arm_group[f"{PL_TARGET}_bpb"],
                    name=f"{support} {label}",
                    marker_color=color,
                    boxpoints="all",
                    jitter=0.2,
                    pointpos=0,
                    showlegend=False,
                ),
                row=1,
                col=1,
            )
    h5 = endpoint[endpoint["arm"].eq("b")].copy()
    h5["beta"] = h5["policy_role"].str.extract(r"beta_(0p\d+)", expand=False).str.replace("p", ".")
    h5.loc[h5["policy_role"].eq("boundary_tied_018"), "beta"] = "tied"
    order = ["0.60", "0.70", "0.80", "0.85", "0.90", "tied"]
    for beta in order:
        group = h5[h5["beta"].eq(beta)]
        fig.add_trace(
            go.Box(
                x=[beta] * len(group),
                y=group[f"{PL_TARGET}_bpb"],
                name=beta,
                marker_color=(PLOT_COLORS["tied"] if beta == "tied" else PLOT_COLORS["two_phase"]),
                boxpoints="all",
                jitter=0.18,
                pointpos=0,
                showlegend=False,
            ),
            row=1,
            col=2,
        )
    fig.update_yaxes(title_text="Programming Languages BPB (lower is better)", row=1, col=1)
    fig.update_yaxes(title_text="Programming Languages BPB (lower is better)", row=1, col=2)
    fig.update_xaxes(title_text="Support and policy", row=1, col=1)
    fig.update_xaxes(title_text="Data-switch fraction beta", row=1, col=2)
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Preregistered endpoint interventions",
        height=620,
        width=1250,
        margin=dict(l=80, r=40, t=100, b=100),
    )
    fig.write_html(output_path, include_plotlyjs="cdn", config={"toImageButtonOptions": {"scale": 4}})


def write_h4_plot(h4: pd.DataFrame, output_path: Path) -> None:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Calibration mapping and held-out validation", "Validation rollout profiles"),
    )
    for role, label, color in (
        ("h4_independent_calibration", "Calibration seeds", PLOT_COLORS["calibration"]),
        ("h4_primary_validation", "Held-out validation seeds", PLOT_COLORS["validation"]),
    ):
        group = h4[h4["analysis_role"].eq(role)]
        fig.add_trace(
            go.Scatter(
                x=group["delta_utility"],
                y=group["delta_bpb"],
                mode="markers",
                name=label,
                marker=dict(color=color, size=7, opacity=0.65),
                customdata=np.stack([group["parent_trajectory_id"], group["starcoder_weight"]], axis=1),
                hovertemplate=(
                    "%{customdata[0]}<br>q=%{customdata[1]:.2f}<br>"
                    "delta utility=%{x:.5g}<br>delta BPB=%{y:.5f}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    validation = h4[h4["analysis_role"].eq("h4_primary_validation")]
    for _, group in validation.groupby("parent_trajectory_id", sort=True):
        fig.add_trace(
            go.Scatter(
                x=group["starcoder_weight"],
                y=group["bpb"],
                mode="lines+markers",
                line=dict(color=PLOT_COLORS["validation"], width=1),
                marker=dict(size=5),
                opacity=0.25,
                showlegend=False,
                hovertemplate="q=%{x:.2f}<br>BPB=%{y:.5f}<extra></extra>",
            ),
            row=1,
            col=2,
        )
    mean_profile = validation.groupby("starcoder_weight", as_index=False)["bpb"].agg(["mean", "sem"]).reset_index()
    fig.add_trace(
        go.Scatter(
            x=mean_profile["starcoder_weight"],
            y=mean_profile["mean"],
            error_y=dict(type="data", array=1.96 * mean_profile["sem"], visible=True),
            mode="lines+markers",
            name="Validation mean +/- 95% normal CI",
            line=dict(color="#252525", width=3),
            marker=dict(size=9),
        ),
        row=1,
        col=2,
    )
    fig.update_xaxes(title_text="Utility difference from q=0.35", row=1, col=1)
    fig.update_yaxes(title_text="BPB difference from q=0.35", row=1, col=1)
    fig.update_xaxes(title_text="StarCoder share q", row=1, col=2)
    fig.update_yaxes(title_text="Programming Languages BPB", row=1, col=2)
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="H4 exact-state short-rollout audit",
        height=620,
        width=1250,
        margin=dict(l=80, r=40, t=100, b=80),
    )
    fig.write_html(output_path, include_plotlyjs="cdn", config={"toImageButtonOptions": {"scale": 4}})


def write_proxy_plot(proxy: pd.DataFrame, output_path: Path) -> None:
    h2 = proxy[
        proxy["analysis_role"].isin(("h2_primary", "h3_full_support_pair"))
        & proxy["checkpoint_label"].isin(H2_STATES)
        & proxy["target"].isin((PL_TARGET, C4_TARGET, GITHUB_TARGET))
    ].copy()
    labels = ["fraction_0p40", "fraction_0p55", "decay_minus_256", "decay_minus_64"]
    fig = make_subplots(rows=1, cols=2, subplot_titles=("Finite m100a support", "Full support"))
    for col, support in enumerate(("m100a", "full"), start=1):
        support_data = h2[h2["support_id"].eq(support)]
        for target in (PL_TARGET, C4_TARGET, GITHUB_TARGET):
            target_data = support_data[support_data["target"].eq(target)]
            summary = (
                target_data.groupby("checkpoint_label")["source_choice_gradient_cosine"]
                .agg(["mean", "sem"])
                .reindex(labels)
            )
            fig.add_trace(
                go.Scatter(
                    x=labels,
                    y=summary["mean"],
                    error_y=dict(type="data", array=1.96 * summary["sem"], visible=True),
                    mode="lines+markers",
                    name=TARGET_LABELS[target],
                    legendgroup=target,
                    showlegend=col == 1,
                ),
                row=1,
                col=col,
            )
    fig.update_yaxes(
        title_text="cos(target, StarCoder gradient) - cos(target, Nemotron gradient)",
        row=1,
        col=1,
    )
    fig.update_layout(
        template=PLOT_TEMPLATE,
        title="Exploratory gradient-space proxy (not preregistered A_y)",
        height=620,
        width=1250,
        margin=dict(l=100, r=40, t=100, b=100),
    )
    fig.write_html(output_path, include_plotlyjs="cdn", config={"toImageButtonOptions": {"scale": 4}})


def write_report(
    output_dir: Path,
    endpoint: pd.DataFrame,
    endpoint_summary: pd.DataFrame,
    h4_summary: Mapping[str, Any],
    proxy_summary: pd.DataFrame,
    audit: Mapping[str, Any],
) -> None:
    p3_rows = endpoint_summary[endpoint_summary["evidence_role"].eq("preregistered_family_component_under_specified")]
    secondary_rows = endpoint_summary[endpoint_summary["evidence_role"].str.startswith("secondary_")]
    h5 = endpoint_summary[endpoint_summary["contrast"].str.startswith("H5")].iloc[0]
    p3_lines = "\n".join(
        f"| {row.contrast} | {row.mean_effect_bpb:+.6f} | "
        f"[{row.bootstrap_ci95_low:+.6f}, {row.bootstrap_ci95_high:+.6f}] | "
        f"{row.exact_sign_flip_p:.6f} | {row.positive_pairs}/{row.n_paired_seeds} |"
        for row in p3_rows.itertuples()
    )
    secondary_lines = "\n".join(
        f"| {row.contrast} | {row.mean_effect_bpb:+.6f} | "
        f"[{row.bootstrap_ci95_low:+.6f}, {row.bootstrap_ci95_high:+.6f}] | "
        f"{row.exact_sign_flip_p:.6f} | {row.positive_pairs}/{row.n_paired_seeds} |"
        for row in secondary_rows.itertuples()
    )
    h5_means = (
        endpoint[endpoint["arm"].eq("b")]
        .groupby("policy_role", as_index=False)[f"{PL_TARGET}_bpb"]
        .agg(["mean", "std"])
        .reset_index()
    )
    h5_order = {
        "boundary_beta_0p60": 0,
        "boundary_beta_0p70": 1,
        "boundary_beta_0p80": 2,
        "boundary_beta_0p85": 3,
        "boundary_beta_0p90": 4,
        "boundary_tied_018": 5,
    }
    h5_means["order"] = h5_means["policy_role"].map(h5_order)
    h5_mean_lines = "\n".join(
        f"| {row.policy_role.removeprefix('boundary_')} | {row.mean:.6f} | {row.std:.6f} |"
        for row in h5_means.sort_values("order").itertuples()
    )
    proxy_lines = "\n".join(
        f"| {row.contrast} | {row.mean_effect_bpb:+.6f} | "
        f"[{row.bootstrap_ci95_low:+.6f}, {row.bootstrap_ci95_high:+.6f}] | "
        f"{row.exact_sign_flip_p:.6f} |"
        for row in proxy_summary.itertuples()
    )
    report = f"""# StarCoder WSD80 gradient-conflict full-panel analysis

## Bottom line

The panel is operationally complete, but it does **not** support a clean five-family confirmatory verdict. The frozen
H1 source-source cosine and H2/H3 normalized target-utility statistic cannot be reconstructed from the persisted
probe rows. H4's calibration mapping and decision rule were never numerically frozen. P3-versus-P4 also did not freeze
how the two support strata enter one family-level p-value. A Holm-corrected five-family claim would therefore be a
post-hoc reconstruction and is not reported.

Two endpoint interventions remain directly interpretable. The P3/P4 comparison isolates phase schedule at nearly
fixed aggregate exposure. The H5 comparison moves the data-switch boundary at fixed aggregate and contrast. They
establish trajectory dependence, but P3/P4 alone does not establish that the global two-phase optimum beats the
global tied optimum.

## Endpoint interventions

| contrast | mean BPB effect | paired bootstrap 95% CI | raw exact sign-flip p | positive pairs |
|---|---:|---:|---:|---:|
{p3_lines}
| {h5.contrast} | {h5.mean_effect_bpb:+.6f} | [{h5.bootstrap_ci95_low:+.6f}, {h5.bootstrap_ci95_high:+.6f}] | {h5.exact_sign_flip_p:.6f} | {h5.positive_pairs}/{h5.n_paired_seeds} |

Positive P3/P4 effects favor the two-phase policy. Positive H5 effects mean the beta=0.60 switch has lower BPB than
the beta=0.85 switch. The observed negative H5 effect therefore favors the later beta=0.85 switch. Its confidence
interval excludes zero but crosses -0.001 BPB: the policies differ, but the result does not establish that the effect
exceeds the frozen practical-equivalence margin.

Both P3/P4 support strata show a very large ordering gain, including full support. Finite-support repetition is
therefore not necessary for phase scheduling to rescue this aggregate. The secondary P3/P2 comparison asks the
different global-policy question and changes sign across support:

| secondary contrast | mean BPB effect | paired bootstrap 95% CI | raw exact sign-flip p | positive pairs |
|---|---:|---:|---:|---:|
{secondary_lines}

The m100a-minus-full interaction in P3's gain over P2 is +0.010831 BPB. This is secondary endpoint evidence that
finite support changes whether the selected two-phase policy beats the selected tied comparator; it is not the
unavailable H3 gradient-space mechanism test.

The complete H5 ladder is non-monotone and descriptively best at beta=0.80, where the data switch coincides with
optimizer decay. Only beta=0.60 versus beta=0.85 was preregistered, so the beta=0.80 minimum is hypothesis-generating
rather than causal evidence for decay alignment.

| H5 policy | mean BPB | seed SD |
|---|---:|---:|
{h5_mean_lines}

## H4 short-rollout validity

This audit uses a transparent, post-hoc mapping because no exact mapping was frozen: within each parent, both utility
and 512-update BPB are centered at q=0.35; a zero-intercept slope is fit on eight independent calibration seeds and
then applied unchanged to sixteen validation seeds.

| metric | value |
|---|---:|
| calibration seeds | {h4_summary['calibration_seed_count']} |
| validation seeds | {h4_summary['validation_seed_count']} |
| fitted slope | {h4_summary['slope_delta_bpb_per_utility']:.6g} |
| validation RMSE | {h4_summary['validation_rmse_bpb']:.6f} BPB |
| validation R2 | {h4_summary['validation_r2']:.4f} |
| validation Pearson r | {h4_summary['validation_pearson_r']:.4f} |
| validation Spearman rho | {h4_summary['validation_spearman_rho']:.4f} |
| mean selected-q regret | {h4_summary['mean_selection_regret_bpb']:.6f} BPB |
| exact q match | {h4_summary['exact_q_match_fraction']:.1%} |
| parents clearing 3 measurement SE | {h4_summary['three_se_clear_fraction']:.1%} |
| RMSE / median observed q-range | {h4_summary['rmse_fraction_of_median_spread']:.1%} |
| strictly monotone BPB curves | {h4_summary['strict_bpb_improvement_fraction']:.1%} |
| strictly monotone utility curves | {h4_summary['strict_utility_increase_fraction']:.1%} |
| predicted q values | {h4_summary['predicted_q_values']} |
| observed q values | {h4_summary['observed_q_values']} |

This can establish whether local optimizer-aware utility predicts nearby short rollouts. It cannot establish endpoint
mediation, on-policy transport, or the original H4 confirmatory p-value. Every validation curve improves monotonically
through the upper edge q=0.55, so the 100% selected-q match is a boundary-direction result, not evidence that the
utility identifies an interior optimum.

## Gradient-space diagnostics

Because the source-source and target-gradient-versus-source-update cross statistics are absent, these diagnostics use
the explicitly different quantity
`cos(g_target, g_StarCoder) - cos(g_target, g_Nemotron)`. They are exploratory and cannot replace H1-H3.

| diagnostic | mean effect | paired bootstrap 95% CI | raw sign-flip p |
|---|---:|---:|---:|
{proxy_lines}

## Recoverability audit

| family | status | exact reason |
|---|---|---|
| H1 source conflict | blocked | Source rows were compared only with targets; StarCoder-versus-Nemotron gradient/update dot products were not persisted. |
| H2 temporal revaluation | blocked | Probe rows compare target gradients to source gradients and target updates to source updates, not target gradients to source updates; the norm of Delta(S)-Delta(N) is also absent. |
| H3 repetition interaction | blocked | It inherits the unavailable H2 statistic. |
| H4 held-out rollout prediction | exploratory only | Calibration mapping, reliability threshold, and family-level p-value were never numerically frozen before full outcomes. |
| P3 versus P4 | endpoint effect estimable; family p under-specified | m100a and full-support contrasts are paired by seed, but the freeze did not define whether to pool, average, or privilege one support. |
| H5 endpoint | estimable | The 16 paired beta=0.60 versus beta=0.85 endpoint contrast is fully specified. |
| H5 mechanism profile | blocked | It requires the same unavailable A_y statistic as H2. |

## Hypothesis verdicts

| hypothesis | verdict | implication |
|---|---|---|
| H1 source conflict | blocked | The required StarCoder-versus-Nemotron gradient and update cosines were not persisted. |
| H2 temporal target revaluation | blocked; exploratory proxy null | The substitute raw-gradient proxy is near zero with intervals spanning both signs, so it supplies no positive mechanism evidence but cannot reject the frozen optimizer-aware hypothesis. |
| H3 repetition interaction | blocked; secondary endpoint interaction present | The frozen gradient statistic is unavailable. Endpoint behavior changes with support, but that does not identify the preregistered utility mechanism. |
| H4 local behavioral validity | exploratory support | One-step optimizer-aware utility predicts the direction and scale of 512-update interventions on held-out seeds, but the mapping was not frozen and all candidate curves are monotone to the q boundary. |
| H5 moved switch | endpoint supported; mechanism profile blocked | Beta=0.85 beats beta=0.60 by 0.001324 BPB, demonstrating switch-time sensitivity at fixed aggregate and contrast; the planned gradient-profile explanation is unavailable. |

The release and row identities passed the final postflight audit. Analysis inputs and counts are recorded in
`analysis_audit.json`; flattened source tables retain durable GCS row URIs.

## Artifacts

- `endpoint_interventions.html`
- `h4_rollout_validation.html`
- `gradient_proxy_temporal.html`
- `endpoint_metrics.csv`
- `optimizer_utility.csv`
- `rollout_readouts.csv`
- `probe_target_source_geometry.csv`
- `endpoint_contrasts.csv`
- `h4_validation_rows.csv`
- `gradient_proxy_contrasts.csv`
- `analysis_audit.json`
"""
    (output_dir / "report.md").write_text(report)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    release_path = RELEASE_DIR / "release.json"
    release = json.loads(release_path.read_text())
    release_sha256 = str(release["release_sha256"])
    if release["endpoint_metrics_read"] is not False:
        raise RuntimeError("Frozen release was not endpoint-blind")

    trajectories = pd.read_csv(DESIGN_DIR / "trajectory_manifest.csv")
    probe_manifest = pd.read_csv(RELEASE_DIR / "full_probe_manifest.csv")
    optimizer_manifest = pd.read_csv(RELEASE_DIR / "full_optimizer_manifest.csv")
    rollout_manifest = pd.read_csv(RELEASE_DIR / "full_rollout_manifest.csv")
    selected_probe_manifest = probe_manifest[
        probe_manifest["analysis_role"].isin(PROBE_ANALYSIS_ROLES) & probe_manifest["distribution_id"].isin(TARGETS)
    ].copy()

    fs = gcsfs.GCSFileSystem(token="google_default")
    endpoint = load_endpoint_metrics(fs, trajectories)
    optimizer_documents = load_result_documents(
        fs,
        optimizer_manifest,
        kind="optimizer",
        release_sha256=release_sha256,
    )
    rollout_documents = load_result_documents(
        fs,
        rollout_manifest,
        kind="rollout",
        release_sha256=release_sha256,
    )
    probe_documents = load_result_documents(
        fs,
        selected_probe_manifest,
        kind="probe",
        release_sha256=release_sha256,
    )

    optimizer = flatten_optimizer(optimizer_documents)
    rollouts = flatten_rollouts(rollout_documents)
    geometry = flatten_probe_geometry(probe_documents)
    endpoint_pairs, endpoint_summary = endpoint_contrasts(endpoint)
    h4_rows, h4_summary = h4_validation(optimizer, rollouts)
    proxy_rows, proxy_summary = gradient_proxy_analysis(geometry, trajectories)

    endpoint.to_csv(output_dir / "endpoint_metrics.csv", index=False)
    endpoint_pairs.to_csv(output_dir / "endpoint_pair_rows.csv", index=False)
    endpoint_summary.to_csv(output_dir / "endpoint_contrasts.csv", index=False)
    optimizer.to_csv(output_dir / "optimizer_utility.csv", index=False)
    rollouts.to_csv(output_dir / "rollout_readouts.csv", index=False)
    geometry.to_csv(output_dir / "probe_target_source_geometry.csv", index=False)
    h4_rows.to_csv(output_dir / "h4_validation_rows.csv", index=False)
    proxy_rows.to_csv(output_dir / "gradient_proxy_rows.csv", index=False)
    proxy_summary.to_csv(output_dir / "gradient_proxy_contrasts.csv", index=False)

    write_endpoint_plot(endpoint, output_dir / "endpoint_interventions.html")
    write_h4_plot(h4_rows, output_dir / "h4_rollout_validation.html")
    write_proxy_plot(proxy_rows, output_dir / "gradient_proxy_temporal.html")

    audit = {
        "analysis_version": "2026-08-18-gradient-probe-full-v1",
        "release_sha256": release_sha256,
        "release_file_sha256": _sha256(release_path),
        "design_sha256": release["design_sha256"],
        "source_counts": {
            "trajectory_rows": len(trajectories),
            "endpoint_rows": len(endpoint),
            "optimizer_manifest_rows": len(optimizer_manifest),
            "optimizer_documents": len(optimizer_documents),
            "rollout_manifest_rows": len(rollout_manifest),
            "rollout_documents": len(rollout_documents),
            "selected_probe_manifest_rows": len(selected_probe_manifest),
            "probe_documents": len(probe_documents),
        },
        "confirmatory_recoverability": {
            "p3_p4_endpoint": "effect_recoverable_family_aggregation_under_specified",
            "h1": "blocked_missing_source_source_statistics",
            "h2": "blocked_missing_target_gradient_source_update_cross_statistics_and_difference_norm",
            "h3": "blocked_inherits_h2",
            "h4": "exploratory_mapping_not_numerically_frozen",
            "h5_endpoint": "recoverable",
            "h5_profile": "blocked_inherits_h2_statistic",
            "holm_family": "not_computable_without_post_hoc_substitution",
        },
        "h4_summary": h4_summary,
        "bootstrap": {"draws": BOOTSTRAP_DRAWS, "seed": BOOTSTRAP_SEED},
        "endpoint_metrics_read_during_probe_execution": False,
        "endpoint_metrics_read_during_analysis": True,
    }
    (output_dir / "analysis_audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
    write_report(output_dir, endpoint, endpoint_summary, h4_summary, proxy_summary, audit)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
