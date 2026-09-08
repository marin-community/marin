# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
"""Measure signal-to-noise for fixed-aggregate phase-asymmetry interventions.

The historical whole-swarm diagnostic defines amplitude SNR as

    SD(metric across policies) / SD(metric across repeated controls).

This script reproduces that quantity for the Delphi 3e18 fit swarm and for
aggregate-matched phase fibers. It additionally uses same-seed tied controls
and antithetic pairs to separate:

* total fixed-aggregate phase response;
* the odd phase-order response, (L(+d) - L(-d)) / 2;
* the even asymmetry response, (L(+d) + L(-d)) / 2 - L(0).

The targeted pairwise phase-order panel is sealed and is never read here.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
from plotly.subplots import make_subplots
from scipy.stats import chi2

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_COMPONENT_PANEL = REFERENCE_OUTPUTS / "delphi_3e18_observed_components_20260724" / "observed_component_panel.csv"
DEFAULT_HELDOUTS = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714" / "heldout_current.csv"
DEFAULT_AGGRESSIVE_RESULTS = (
    REFERENCE_OUTPUTS
    / "delphi_3e18_aggressive_phase_asymmetry_results_20260723"
    / "observed_results_with_control_deltas.csv"
)
DEFAULT_ANTITHETIC_PAIRS = (
    REFERENCE_OUTPUTS / "delphi_3e18_aggressive_phase_asymmetry_results_20260723" / "balanced_antithetic_pairs.csv"
)
DEFAULT_PROPORTIONAL_NOISE = (
    REFERENCE_OUTPUTS / "delphi_3e18_proportional_noise_floor_20260703" / "noise_floor_summary.json"
)
DEFAULT_TABLE9_COMPONENT_NOISE = (
    REFERENCE_OUTPUTS / "delphi_3e18_proportional_noise_floor_20260703" / "noise_component_matrix.csv"
)
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_3e18_fixed_aggregate_phase_snr_20260724"

RANDOM_SERIES = "delphi_3e18_frontier_random_phase_population_20260720"
SEALED_SERIES_FRAGMENT = "targeted_pairwise"
TARGET_COLUMNS = {
    "uncheatable": "uncheatable_bpb",
    "table9": "table9_macro_bpb",
}
TARGET_ANCHORS = {
    "uncheatable": "uncheatable_frontier",
    "table9": "table9_frontier",
}
PHASE_0_PREFIX = "phase_0_"
PHASE_1_PREFIX = "phase_1_"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
EPSILON = 1e-12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--component-panel", type=Path, default=DEFAULT_COMPONENT_PANEL)
    parser.add_argument("--heldouts", type=Path, default=DEFAULT_HELDOUTS)
    parser.add_argument("--aggressive-results", type=Path, default=DEFAULT_AGGRESSIVE_RESULTS)
    parser.add_argument("--antithetic-pairs", type=Path, default=DEFAULT_ANTITHETIC_PAIRS)
    parser.add_argument("--proportional-noise", type=Path, default=DEFAULT_PROPORTIONAL_NOISE)
    parser.add_argument("--table9-component-noise", type=Path, default=DEFAULT_TABLE9_COMPONENT_NOISE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def metric_columns(frame: pd.DataFrame) -> list[str]:
    metadata = {
        "panel",
        "row_name",
        "source_row_name",
        "training_run_id",
        "table9_eval_run_id",
        "anchor_id",
        "contrast_family",
        "direction_id",
        "sign",
        "seed_block",
    }
    return [
        column
        for column in frame.columns
        if column not in metadata and not column.startswith(PHASE_0_PREFIX) and not column.startswith(PHASE_1_PREFIX)
    ]


def metric_family(metric: str) -> str:
    if metric == "uncheatable_bpb":
        return "Uncheatable macro"
    if metric == "table9_macro_bpb":
        return "Table-9 macro"
    if metric.startswith("eval/uncheatable_eval/"):
        return "Uncheatable component"
    return "Table-9 component"


def metric_display_name(metric: str) -> str:
    if metric == "uncheatable_bpb":
        return "Uncheatable macro"
    if metric == "table9_macro_bpb":
        return "Table-9 macro"
    if metric.startswith("eval/uncheatable_eval/"):
        return metric.removeprefix("eval/uncheatable_eval/").removesuffix("/bpb")
    if metric.startswith("olmo_base_eval/easy_bpb/"):
        return metric.removeprefix("olmo_base_eval/easy_bpb/").removesuffix("/bpb")
    return metric


def table9_noise_column(metric: str) -> str | None:
    if metric.startswith("olmo_base_eval/easy_bpb/"):
        return metric.replace("olmo_base_eval/easy_bpb/", "olmo_base_easy/table9/")
    if metric.startswith("mmlu_"):
        return f"olmo_base_easy/table9/{metric}/bpb"
    return None


def pooled_within_sd(frame: pd.DataFrame, value_column: str, group_column: str) -> float:
    residuals = frame[value_column] - frame.groupby(group_column)[value_column].transform("mean")
    degrees_of_freedom = len(frame) - frame[group_column].nunique()
    assert degrees_of_freedom > 0
    return float(math.sqrt(float(np.dot(residuals, residuals)) / degrees_of_freedom))


def pooled_control_sd(frame: pd.DataFrame, metric: str) -> tuple[float, int, int]:
    controls = frame.loc[frame["sign"].eq("center"), ["anchor_id", metric]]
    return (
        pooled_within_sd(controls, metric, "anchor_id"),
        len(controls),
        len(controls) - controls["anchor_id"].nunique(),
    )


def noise_sd_interval(noise_sd: float, degrees_of_freedom: int) -> tuple[float, float]:
    lower = math.sqrt(degrees_of_freedom * noise_sd**2 / chi2.ppf(0.975, degrees_of_freedom))
    upper = math.sqrt(degrees_of_freedom * noise_sd**2 / chi2.ppf(0.025, degrees_of_freedom))
    return lower, upper


def latent_amplitude_snr(effect_rms: float, noise_sd: float) -> float:
    return math.sqrt(max(effect_rms**2 - noise_sd**2, 0.0)) / noise_sd


def effect_snr_fields(effect: np.ndarray, noise_sd: float, noise_degrees_of_freedom: int) -> dict[str, float]:
    effect = effect[np.isfinite(effect)]
    assert len(effect) >= 2
    effect_rms = float(np.sqrt(np.mean(np.square(effect))))
    effect_sd = float(np.std(effect, ddof=1))
    noise_low, noise_high = noise_sd_interval(noise_sd, noise_degrees_of_freedom)
    return {
        "mean_effect_bpb": float(np.mean(effect)),
        "sd_effect_bpb": effect_sd,
        "rms_effect_bpb": effect_rms,
        "noise_sd_bpb": noise_sd,
        "noise_df": noise_degrees_of_freedom,
        "noise_sd_ci95_low_bpb": noise_low,
        "noise_sd_ci95_high_bpb": noise_high,
        "raw_rms_snr": effect_rms / noise_sd,
        "raw_rms_snr_noise_ci95_low": effect_rms / noise_high,
        "raw_rms_snr_noise_ci95_high": effect_rms / noise_low,
        "latent_rms_snr": latent_amplitude_snr(effect_rms, noise_sd),
        "latent_rms_snr_noise_ci95_low": latent_amplitude_snr(effect_rms, noise_high),
        "latent_rms_snr_noise_ci95_high": latent_amplitude_snr(effect_rms, noise_low),
    }


def load_primary_noise(
    metrics: list[str],
    fiber: pd.DataFrame,
    proportional_summary: dict[str, Any],
    table9_component_noise: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    table9_component_sds = table9_component_noise.std(axis=0, ddof=1)
    for metric in metrics:
        matched_sd, matched_n, matched_df = pooled_control_sd(fiber, metric)
        table9_column = table9_noise_column(metric)
        if metric == "uncheatable_bpb":
            source = "proportional_repeats_n10"
            noise_sd = float(proportional_summary["uncheatable_bpb_sd"])
            noise_n = int(proportional_summary["n_repeats"])
            noise_df = noise_n - 1
        elif metric == "table9_macro_bpb":
            source = "proportional_repeats_n10"
            noise_sd = float(proportional_summary["table9_macro_bpb_sd"])
            noise_n = int(proportional_summary["n_repeats"])
            noise_df = noise_n - 1
        elif table9_column is not None:
            assert table9_column in table9_component_sds.index, (metric, table9_column)
            source = "proportional_repeats_n10"
            noise_sd = float(table9_component_sds.loc[table9_column])
            noise_n = len(table9_component_noise)
            noise_df = noise_n - 1
        else:
            source = "matched_fiber_controls_n8"
            noise_sd = matched_sd
            noise_n = matched_n
            noise_df = matched_df
        noise_low, noise_high = noise_sd_interval(noise_sd, noise_df)
        rows.append(
            {
                "metric": metric,
                "metric_display": metric_display_name(metric),
                "metric_family": metric_family(metric),
                "primary_noise_source": source,
                "primary_noise_sd_bpb": noise_sd,
                "primary_noise_n": noise_n,
                "primary_noise_df": noise_df,
                "primary_noise_sd_ci95_low_bpb": noise_low,
                "primary_noise_sd_ci95_high_bpb": noise_high,
                "matched_control_noise_sd_bpb": matched_sd,
                "matched_control_n": matched_n,
                "matched_control_df": matched_df,
            }
        )
    return pd.DataFrame(rows)


def phase_total_variation(frame: pd.DataFrame) -> pd.Series:
    phase_0 = [column for column in frame.columns if column.startswith(PHASE_0_PREFIX)]
    phase_1 = [f"{PHASE_1_PREFIX}{column.removeprefix(PHASE_0_PREFIX)}" for column in phase_0]
    assert phase_0 and all(column in frame.columns for column in phase_1)
    return pd.Series(
        0.5 * np.abs(frame[phase_0].to_numpy() - frame[phase_1].to_numpy()).sum(axis=1),
        index=frame.index,
    )


def metric_phase_snr(
    component_panel: pd.DataFrame,
    noise_table: pd.DataFrame,
    metrics: list[str],
) -> pd.DataFrame:
    fit = component_panel.loc[component_panel["panel"].eq("two_phase_fit")]
    fiber = component_panel.loc[component_panel["panel"].eq("frontier_phase_fiber")].copy()
    fiber["phase_tv"] = phase_total_variation(fiber)
    controls = fiber.loc[fiber["sign"].eq("center")]
    treatments = fiber.loc[fiber["sign"].isin(["plus", "minus"])].copy()
    assert len(fit) == 280
    assert len(controls) == 8
    assert len(treatments) == 192
    assert treatments.groupby(["anchor_id", "direction_id"]).size().eq(2).all()

    noise_by_metric = noise_table.set_index("metric")
    rows: list[dict[str, Any]] = []
    scopes = [("pooled_anchor_centered", None), *[(anchor, anchor) for anchor in sorted(controls["anchor_id"].unique())]]
    for metric in metrics:
        whole_swarm_sd = float(fit[metric].std(ddof=1))
        noise = noise_by_metric.loc[metric]
        for scope, anchor in scopes:
            scoped_treatments = treatments if anchor is None else treatments.loc[treatments["anchor_id"].eq(anchor)]
            scoped_controls = controls if anchor is None else controls.loc[controls["anchor_id"].eq(anchor)]
            control_index = scoped_controls.set_index(["anchor_id", "seed_block"])[metric]
            scoped_treatments = scoped_treatments.copy()
            scoped_treatments["same_seed_control"] = [
                control_index.loc[(row.anchor_id, row.seed_block)]
                for row in scoped_treatments[["anchor_id", "seed_block"]].itertuples(index=False)
            ]
            scoped_treatments["delta"] = scoped_treatments[metric] - scoped_treatments["same_seed_control"]

            if anchor is None:
                outcome_sd = pooled_within_sd(scoped_treatments, metric, "anchor_id")
                delta_sd = pooled_within_sd(scoped_treatments, "delta", "anchor_id")
                centered_delta = (
                    scoped_treatments["delta"] - scoped_treatments.groupby("anchor_id")["delta"].transform("mean")
                ).to_numpy()
                delta_rms = float(np.sqrt(np.mean(np.square(centered_delta))))
                matched_noise_sd, matched_noise_n, matched_noise_df = pooled_control_sd(fiber, metric)
            else:
                outcome_sd = float(scoped_treatments[metric].std(ddof=1))
                delta_sd = float(scoped_treatments["delta"].std(ddof=1))
                delta_rms = float(np.sqrt(np.mean(np.square(scoped_treatments["delta"].to_numpy()))))
                matched_noise_sd = float(scoped_controls[metric].std(ddof=1))
                matched_noise_n = len(scoped_controls)
                matched_noise_df = matched_noise_n - 1

            primary_noise_sd = float(noise["primary_noise_sd_bpb"])
            independent_delta_noise_sd = math.sqrt(2) * primary_noise_sd
            rows.append(
                {
                    "metric": metric,
                    "metric_display": noise["metric_display"],
                    "metric_family": noise["metric_family"],
                    "scope": scope,
                    "anchor_id": anchor,
                    "fit_swarm_rows": len(fit),
                    "fixed_aggregate_rows": len(scoped_treatments),
                    "fixed_aggregate_direction_count": scoped_treatments["direction_id"].nunique(),
                    "median_phase_tv": float(scoped_treatments["phase_tv"].median()),
                    "max_phase_tv": float(scoped_treatments["phase_tv"].max()),
                    "primary_noise_source": noise["primary_noise_source"],
                    "primary_noise_sd_bpb": primary_noise_sd,
                    "primary_noise_n": int(noise["primary_noise_n"]),
                    "primary_noise_df": int(noise["primary_noise_df"]),
                    "matched_control_noise_sd_bpb": matched_noise_sd,
                    "matched_control_n": matched_noise_n,
                    "matched_control_df": matched_noise_df,
                    "whole_swarm_signal_sd_bpb": whole_swarm_sd,
                    "whole_swarm_historical_snr": whole_swarm_sd / primary_noise_sd,
                    "fixed_aggregate_outcome_sd_bpb": outcome_sd,
                    "fixed_aggregate_historical_snr": outcome_sd / primary_noise_sd,
                    "fixed_aggregate_delta_sd_bpb": delta_sd,
                    "fixed_aggregate_delta_rms_bpb": delta_rms,
                    "independent_delta_noise_sd_bpb": independent_delta_noise_sd,
                    "paired_delta_raw_rms_snr_independence": delta_rms / independent_delta_noise_sd,
                    "paired_delta_latent_rms_snr_independence": latent_amplitude_snr(
                        delta_rms,
                        independent_delta_noise_sd,
                    ),
                    "phase_to_whole_swarm_sd_ratio": outcome_sd / whole_swarm_sd,
                    "phase_to_whole_swarm_variance_ratio": (outcome_sd / whole_swarm_sd) ** 2,
                }
            )
    return pd.DataFrame(rows)


def same_seed_delta_noise(aggressive: pd.DataFrame) -> pd.DataFrame:
    replicated = aggressive.loc[aggressive["contrast_family"].eq("dolmino_late_continuum")]
    rows: list[dict[str, Any]] = []
    for anchor, anchor_frame in replicated.groupby("anchor_id", sort=True):
        for target in TARGET_COLUMNS:
            pooled_sum_squares = 0.0
            degrees_of_freedom = 0
            for _, group in anchor_frame.groupby("direction_id", sort=True):
                values = group[f"{target}_delta_vs_control"].to_numpy(dtype=float)
                assert len(values) == 3
                pooled_sum_squares += (len(values) - 1) * float(np.var(values, ddof=1))
                degrees_of_freedom += len(values) - 1
            noise_sd = math.sqrt(pooled_sum_squares / degrees_of_freedom)
            low, high = noise_sd_interval(noise_sd, degrees_of_freedom)
            rows.append(
                {
                    "anchor_id": anchor,
                    "target": target,
                    "same_seed_delta_noise_sd_bpb": noise_sd,
                    "noise_df": degrees_of_freedom,
                    "noise_sd_ci95_low_bpb": low,
                    "noise_sd_ci95_high_bpb": high,
                    "replicated_policy_count": anchor_frame["direction_id"].nunique(),
                    "replicates_per_policy": 3,
                }
            )
    return pd.DataFrame(rows)


def macro_snr_row(
    *,
    panel: str,
    design_family: str,
    effect_channel: str,
    anchor: str,
    target: str,
    asymmetry_level: float,
    effects: np.ndarray,
    direction_count: int,
    delta_noise_sd: float,
    noise_df: int,
    noise_scale: float = 1.0,
) -> dict[str, Any]:
    fields = effect_snr_fields(effects, delta_noise_sd * noise_scale, noise_df)
    return {
        "panel": panel,
        "design_family": design_family,
        "effect_channel": effect_channel,
        "anchor_id": anchor,
        "target": target,
        "target_matched": TARGET_ANCHORS[target] == anchor,
        "asymmetry_level": asymmetry_level,
        "n_observations": len(effects),
        "n_directions": direction_count,
        **fields,
    }


def random_macro_snr(
    heldouts: pd.DataFrame,
    noise: pd.DataFrame,
) -> list[dict[str, Any]]:
    assert not heldouts["training_series"].astype(str).str.contains(SEALED_SERIES_FRAGMENT, case=False).any()
    random = heldouts.loc[heldouts["training_series"].eq(RANDOM_SERIES)].copy()
    assert len(random) == 296
    metadata = random["proposal_metadata_json"].map(json.loads)
    random["phase_tv"] = metadata.map(lambda row: float(row["phase_tv"]))
    controls = random.loc[random["candidate_kind"].eq("center_control")]
    treatments = random.loc[random["candidate_kind"].eq("random_isotropic")].copy()
    noise_index = noise.set_index(["anchor_id", "target"])
    rows: list[dict[str, Any]] = []
    for anchor, anchor_frame in treatments.groupby("anchor_id", sort=True):
        anchor_controls = controls.loc[controls["anchor_id"].eq(anchor)].set_index("seed_block")
        for target, metric in TARGET_COLUMNS.items():
            anchor_frame = anchor_frame.copy()
            anchor_frame["delta"] = anchor_frame[metric] - anchor_frame["seed_block"].map(anchor_controls[metric])
            noise_row = noise_index.loc[(anchor, target)]
            for radius, group in anchor_frame.groupby("radius_fraction", sort=True):
                rows.append(
                    macro_snr_row(
                        panel="random_frontier_population",
                        design_family="random_isotropic",
                        effect_channel="total_delta",
                        anchor=anchor,
                        target=target,
                        asymmetry_level=float(group["phase_tv"].mean()),
                        effects=group["delta"].to_numpy(dtype=float),
                        direction_count=group["direction_id"].nunique(),
                        delta_noise_sd=float(noise_row["same_seed_delta_noise_sd_bpb"]),
                        noise_df=int(noise_row["noise_df"]),
                    )
                    | {"radius_fraction": float(radius)}
                )
    return rows


def fiber_macro_snr(
    component_panel: pd.DataFrame,
    noise: pd.DataFrame,
) -> list[dict[str, Any]]:
    fiber = component_panel.loc[component_panel["panel"].eq("frontier_phase_fiber")].copy()
    fiber["phase_tv"] = phase_total_variation(fiber)
    controls = fiber.loc[fiber["sign"].eq("center")].set_index(["anchor_id", "seed_block"])
    treatments = fiber.loc[fiber["sign"].isin(["plus", "minus"])].copy()
    noise_index = noise.set_index(["anchor_id", "target"])
    rows: list[dict[str, Any]] = []
    for anchor, anchor_frame in treatments.groupby("anchor_id", sort=True):
        for target, metric in TARGET_COLUMNS.items():
            anchor_frame = anchor_frame.copy()
            anchor_frame["delta"] = [
                row_value - controls.loc[(row_anchor, seed), metric]
                for row_value, row_anchor, seed in anchor_frame[[metric, "anchor_id", "seed_block"]].itertuples(
                    index=False,
                    name=None,
                )
            ]
            noise_row = noise_index.loc[(anchor, target)]
            for family, group in anchor_frame.groupby("contrast_family", sort=True):
                rows.append(
                    macro_snr_row(
                        panel="frontier_phase_fiber",
                        design_family=str(family),
                        effect_channel="total_delta",
                        anchor=anchor,
                        target=target,
                        asymmetry_level=float(group["phase_tv"].median()),
                        effects=group["delta"].to_numpy(dtype=float),
                        direction_count=group["direction_id"].nunique(),
                        delta_noise_sd=float(noise_row["same_seed_delta_noise_sd_bpb"]),
                        noise_df=int(noise_row["noise_df"]),
                    )
                )
    return rows


def aggressive_macro_snr(
    aggressive: pd.DataFrame,
    antithetic: pd.DataFrame,
    noise: pd.DataFrame,
) -> list[dict[str, Any]]:
    noise_index = noise.set_index(["anchor_id", "target"])
    rows: list[dict[str, Any]] = []
    treatments = aggressive.loc[aggressive["contrast_family"].ne("center_control")]
    for (anchor, family), family_frame in treatments.groupby(["anchor_id", "contrast_family"], sort=True):
        for target in TARGET_COLUMNS:
            noise_row = noise_index.loc[(anchor, target)]
            group_columns = ["target_phase_tv"]
            if family == "dolmino_late_continuum":
                group_columns = ["direction_id", "target_phase_tv"]
            for group_key, group in family_frame.groupby(group_columns, sort=True):
                asymmetry_level = float(group["target_phase_tv"].iloc[0])
                rows.append(
                    macro_snr_row(
                        panel="aggressive_phase_asymmetry",
                        design_family=str(family),
                        effect_channel="total_delta",
                        anchor=anchor,
                        target=target,
                        asymmetry_level=asymmetry_level,
                        effects=group[f"{target}_delta_vs_control"].to_numpy(dtype=float),
                        direction_count=group["direction_id"].nunique(),
                        delta_noise_sd=float(noise_row["same_seed_delta_noise_sd_bpb"]),
                        noise_df=int(noise_row["noise_df"]),
                    )
                    | {
                        "direction_group": (
                            str(group_key[0])
                            if isinstance(group_key, tuple) and family == "dolmino_late_continuum"
                            else None
                        )
                    }
                )

    for anchor, anchor_frame in antithetic.groupby("anchor_id", sort=True):
        for target in TARGET_COLUMNS:
            noise_row = noise_index.loc[(anchor, target)]
            delta_noise_sd = float(noise_row["same_seed_delta_noise_sd_bpb"])
            noise_df = int(noise_row["noise_df"])
            for phase_tv, group in anchor_frame.groupby("target_phase_tv", sort=True):
                rows.append(
                    macro_snr_row(
                        panel="aggressive_phase_asymmetry",
                        design_family="balanced_partition",
                        effect_channel="odd_order",
                        anchor=anchor,
                        target=target,
                        asymmetry_level=float(phase_tv),
                        effects=group[f"{target}_odd_effect"].to_numpy(dtype=float),
                        direction_count=len(group),
                        delta_noise_sd=delta_noise_sd,
                        noise_df=noise_df,
                        noise_scale=0.5,
                    )
                )
                rows.append(
                    macro_snr_row(
                        panel="aggressive_phase_asymmetry",
                        design_family="balanced_partition",
                        effect_channel="even_asymmetry",
                        anchor=anchor,
                        target=target,
                        asymmetry_level=float(phase_tv),
                        effects=group[f"{target}_curvature"].to_numpy(dtype=float),
                        direction_count=len(group),
                        delta_noise_sd=delta_noise_sd,
                        noise_df=noise_df,
                        noise_scale=math.sqrt(3) / 2,
                    )
                )
    return rows


def build_macro_snr(
    heldouts: pd.DataFrame,
    component_panel: pd.DataFrame,
    aggressive: pd.DataFrame,
    antithetic: pd.DataFrame,
    noise: pd.DataFrame,
) -> pd.DataFrame:
    rows = [
        *random_macro_snr(heldouts, noise),
        *fiber_macro_snr(component_panel, noise),
        *aggressive_macro_snr(aggressive, antithetic, noise),
    ]
    result = pd.DataFrame(rows)
    result["radius_fraction"] = result.get("radius_fraction", np.nan)
    result["direction_group"] = result.get("direction_group", None)
    return result.sort_values(
        ["target_matched", "target", "anchor_id", "panel", "design_family", "effect_channel", "asymmetry_level"],
        ascending=[False, True, True, True, True, True, True],
    ).reset_index(drop=True)


def metric_rollup(metric_snr: pd.DataFrame) -> pd.DataFrame:
    pooled = metric_snr.loc[metric_snr["scope"].eq("pooled_anchor_centered")]
    rows: list[dict[str, Any]] = []
    for family, group in pooled.groupby("metric_family", sort=True):
        rows.append(
            {
                "metric_family": family,
                "metric_count": len(group),
                "median_whole_swarm_historical_snr": float(group["whole_swarm_historical_snr"].median()),
                "median_fixed_aggregate_historical_snr": float(group["fixed_aggregate_historical_snr"].median()),
                "median_paired_delta_raw_rms_snr_independence": float(
                    group["paired_delta_raw_rms_snr_independence"].median()
                ),
                "median_phase_to_whole_swarm_sd_ratio": float(group["phase_to_whole_swarm_sd_ratio"].median()),
                "count_fixed_aggregate_snr_below_1": int((group["fixed_aggregate_historical_snr"] < 1).sum()),
                "count_fixed_aggregate_snr_above_2": int((group["fixed_aggregate_historical_snr"] > 2).sum()),
            }
        )
    return pd.DataFrame(rows)


def write_metric_comparison_plot(metric_snr: pd.DataFrame, output_path: Path) -> None:
    pooled = metric_snr.loc[metric_snr["scope"].eq("pooled_anchor_centered")].copy()
    pooled["ratio"] = pooled["phase_to_whole_swarm_sd_ratio"]
    log_ratio = np.log10(pooled["ratio"].clip(lower=1e-4))
    normalized = (log_ratio - log_ratio.min()) / max(log_ratio.max() - log_ratio.min(), EPSILON)
    colors = sample_colorscale("RdYlGn_r", normalized.tolist())
    symbols = pooled["metric_family"].map(
        {
            "Uncheatable macro": "star",
            "Table-9 macro": "star",
            "Uncheatable component": "circle",
            "Table-9 component": "diamond",
        }
    )
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=pooled["whole_swarm_historical_snr"],
            y=pooled["fixed_aggregate_historical_snr"],
            mode="markers",
            marker={
                "size": np.where(pooled["metric_family"].str.contains("macro"), 17, 10),
                "color": colors,
                "symbol": symbols,
                "line": {"color": "#173246", "width": 1},
            },
            customdata=np.column_stack(
                [
                    pooled["metric_display"],
                    pooled["metric_family"],
                    pooled["primary_noise_source"],
                    pooled["primary_noise_sd_bpb"],
                    pooled["phase_to_whole_swarm_sd_ratio"],
                    pooled["paired_delta_raw_rms_snr_independence"],
                ]
            ),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "%{customdata[1]}<br>"
                "whole-swarm SNR: %{x:.2f}<br>"
                "fixed-aggregate phase SNR: %{y:.2f}<br>"
                "phase / whole SD: %{customdata[4]:.3f}<br>"
                "paired-delta raw SNR (rho=0): %{customdata[5]:.2f}<br>"
                "noise: %{customdata[3]:.5f} (%{customdata[2]})<extra></extra>"
            ),
        )
    )
    maximum = max(
        float(pooled["whole_swarm_historical_snr"].max()),
        float(pooled["fixed_aggregate_historical_snr"].max()),
    )
    figure.add_trace(
        go.Scatter(
            x=[0.1, maximum * 1.15],
            y=[0.1, maximum * 1.15],
            mode="lines",
            line={"color": "#8b9aa2", "dash": "dash"},
            hoverinfo="skip",
            showlegend=False,
        )
    )
    figure.add_hline(y=1, line={"color": "#d75d2b", "dash": "dot"})
    figure.add_vline(x=1, line={"color": "#d75d2b", "dash": "dot"})
    figure.update_layout(
        title={
            "text": (
                "3e18 metric SNR: whole mixture variation versus fixed-aggregate phase variation"
                "<br><sup>Historical amplitude SNR = policy SD / repeat-control SD; log scales</sup>"
            ),
            "x": 0.5,
        },
        xaxis={"title": "Whole two-phase fit-swarm SNR", "type": "log"},
        yaxis={"title": "Fixed-aggregate phase-fiber SNR", "type": "log"},
        template="plotly_white",
        width=1180,
        height=820,
        margin={"l": 90, "r": 60, "t": 110, "b": 90},
        font={"family": "Avenir Next, sans-serif", "color": "#173246"},
        paper_bgcolor="#fbf8ef",
        plot_bgcolor="#fffdf8",
    )
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_macro_plot(macro_snr: pd.DataFrame, output_path: Path) -> None:
    matched = macro_snr.loc[macro_snr["target_matched"]].copy()
    selected = matched.loc[
        matched["effect_channel"].isin(["total_delta", "odd_order", "even_asymmetry"])
        & matched["design_family"].isin(
            [
                "random_isotropic",
                "domain_vs_rest",
                "high_mass_pair",
                "balanced_partition",
                "handcrafted_late_quality",
                "dolmino_late_continuum",
            ]
        )
    ].copy()
    selected["series"] = selected["design_family"] + " / " + selected["effect_channel"]
    series_order = sorted(selected["series"].unique())
    colors = dict(
        zip(series_order, sample_colorscale("RdYlGn_r", np.linspace(0.05, 0.95, len(series_order))), strict=True)
    )
    figure = make_subplots(rows=1, cols=2, subplot_titles=["Uncheatable", "Table-9 macro"])
    for column, target in enumerate(TARGET_COLUMNS, start=1):
        target_frame = selected.loc[selected["target"].eq(target)]
        for series, group in target_frame.groupby("series", sort=True):
            group = group.sort_values("asymmetry_level")
            figure.add_trace(
                go.Scatter(
                    x=group["asymmetry_level"],
                    y=group["latent_rms_snr"],
                    mode="markers+lines",
                    name=series,
                    legendgroup=series,
                    showlegend=column == 1,
                    marker={"size": 9, "color": colors[series], "line": {"color": "#173246", "width": 0.6}},
                    line={"color": colors[series], "width": 1.5},
                    customdata=np.column_stack(
                        [
                            group["raw_rms_snr"],
                            group["mean_effect_bpb"],
                            group["rms_effect_bpb"],
                            group["noise_sd_bpb"],
                            group["n_directions"],
                            group["panel"],
                        ]
                    ),
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        "phase TV: %{x:.3f}<br>"
                        "latent RMS SNR: %{y:.2f}<br>"
                        "raw RMS SNR: %{customdata[0]:.2f}<br>"
                        "mean effect: %{customdata[1]:+.5f} BPB<br>"
                        "RMS effect: %{customdata[2]:.5f} BPB<br>"
                        "noise SD: %{customdata[3]:.5f} BPB<br>"
                        "directions: %{customdata[4]:.0f}<br>"
                        "%{customdata[5]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.add_hline(y=1, line={"color": "#d75d2b", "dash": "dot"}, row=1, col=column)
    figure.update_xaxes(title_text="Phase total variation", range=[-0.01, 0.78])
    figure.update_yaxes(title_text="Noise-corrected RMS SNR", rangemode="tozero", row=1, col=1)
    figure.update_layout(
        title={
            "text": (
                "Where phase asymmetry becomes statistically visible"
                "<br><sup>SNR removes the empirically estimated same-seed delta noise floor; "
                "SNR = 1 is a reference</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        width=1450,
        height=760,
        margin={"l": 90, "r": 50, "t": 120, "b": 90},
        legend={"orientation": "h", "y": -0.18, "x": 0.5, "xanchor": "center"},
        font={"family": "Avenir Next, sans-serif", "color": "#173246"},
        paper_bgcolor="#fbf8ef",
        plot_bgcolor="#fffdf8",
    )
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def write_metric_heatmap(metric_snr: pd.DataFrame, output_path: Path) -> None:
    pooled = metric_snr.loc[metric_snr["scope"].eq("pooled_anchor_centered")].copy()
    pooled = pooled.sort_values("fixed_aggregate_historical_snr", ascending=True)
    columns = [
        "whole_swarm_historical_snr",
        "fixed_aggregate_historical_snr",
        "paired_delta_raw_rms_snr_independence",
    ]
    values = pooled[columns].clip(lower=1e-3).to_numpy()
    z = np.log10(values)
    figure = go.Figure(
        go.Heatmap(
            z=z,
            x=["Whole swarm", "Fixed aggregate", "Paired delta (rho=0)"],
            y=pooled["metric_display"],
            text=np.vectorize(lambda value: f"{value:.2f}")(values),
            texttemplate="%{text}",
            colorscale="RdYlGn_r",
            colorbar={"title": "log10 SNR"},
            customdata=np.repeat(pooled["metric_family"].to_numpy()[:, None], len(columns), axis=1),
            hovertemplate="<b>%{y}</b><br>%{customdata}<br>%{x}: %{text}<extra></extra>",
        )
    )
    figure.update_layout(
        title={
            "text": (
                "Per-metric SNR at 3e18"
                "<br><sup>Fixed-aggregate fibers preserve total exposure; paired delta subtracts "
                "the same-seed tied control</sup>"
            ),
            "x": 0.5,
        },
        template="plotly_white",
        width=1200,
        height=1850,
        margin={"l": 310, "r": 90, "t": 110, "b": 80},
        font={"family": "Avenir Next, sans-serif", "color": "#173246"},
        paper_bgcolor="#fbf8ef",
        plot_bgcolor="#fffdf8",
    )
    figure.write_html(output_path, include_plotlyjs="cdn", config=PLOT_CONFIG)


def report_markdown(
    metric_snr: pd.DataFrame,
    rollup: pd.DataFrame,
    macro_snr: pd.DataFrame,
    same_seed_noise_table: pd.DataFrame,
) -> str:
    pooled = metric_snr.loc[metric_snr["scope"].eq("pooled_anchor_centered")].set_index("metric")
    target_matched_fiber = macro_snr.loc[
        macro_snr["target_matched"]
        & macro_snr["panel"].eq("frontier_phase_fiber")
        & macro_snr["effect_channel"].eq("total_delta")
    ]
    target_matched_random = macro_snr.loc[
        macro_snr["target_matched"]
        & macro_snr["panel"].eq("random_frontier_population")
        & macro_snr["effect_channel"].eq("total_delta")
    ]
    aggressive_channels = macro_snr.loc[
        macro_snr["target_matched"]
        & macro_snr["panel"].eq("aggressive_phase_asymmetry")
        & macro_snr["design_family"].eq("balanced_partition")
        & macro_snr["effect_channel"].isin(["odd_order", "even_asymmetry"])
    ][
        [
            "target",
            "effect_channel",
            "asymmetry_level",
            "n_directions",
            "mean_effect_bpb",
            "rms_effect_bpb",
            "noise_sd_bpb",
            "raw_rms_snr",
            "latent_rms_snr",
        ]
    ]
    macro_rows = pooled.loc[["uncheatable_bpb", "table9_macro_bpb"]][
        [
            "metric_display",
            "primary_noise_sd_bpb",
            "whole_swarm_signal_sd_bpb",
            "whole_swarm_historical_snr",
            "fixed_aggregate_outcome_sd_bpb",
            "fixed_aggregate_historical_snr",
            "phase_to_whole_swarm_sd_ratio",
            "paired_delta_raw_rms_snr_independence",
        ]
    ].reset_index(drop=True)

    uncheatable_fiber = target_matched_fiber.loc[target_matched_fiber["target"].eq("uncheatable")]
    table9_fiber = target_matched_fiber.loc[target_matched_fiber["target"].eq("table9")]
    uncheatable_fiber_latent = float(
        np.average(uncheatable_fiber["latent_rms_snr"], weights=uncheatable_fiber["n_observations"])
    )
    table9_fiber_latent = float(np.average(table9_fiber["latent_rms_snr"], weights=table9_fiber["n_observations"]))
    random_max_latent = target_matched_random.groupby("target")["latent_rms_snr"].max()

    return f"""# Fixed-aggregate phase-asymmetry signal-to-noise at Delphi 3e18

## Executive result

The phase-ordering problem has dramatically less signal than the whole mixture
problem at the same training configuration.

{macro_rows.to_markdown(index=False, floatfmt=".6f")}

The direct historical SNR definition is

$$
\\operatorname{{SNR}}_\\text{{historical}}
=
\\frac{{\\operatorname{{SD}}_p[L(p)]}}
{{\\operatorname{{SD}}[L(p_\\text{{repeat}})]}}.
$$

On the two macro objectives, the full two-phase fit swarm has amplitude SNR
{pooled.loc["uncheatable_bpb", "whole_swarm_historical_snr"]:.1f}
(Uncheatable) and
{pooled.loc["table9_macro_bpb", "whole_swarm_historical_snr"]:.1f}
(Table-9). Holding the aggregate mixture fixed collapses this to
{pooled.loc["uncheatable_bpb", "fixed_aggregate_historical_snr"]:.2f} and
{pooled.loc["table9_macro_bpb", "fixed_aggregate_historical_snr"]:.2f}.
The fixed-aggregate phase-fiber SD is only
{100 * pooled.loc["uncheatable_bpb", "phase_to_whole_swarm_sd_ratio"]:.1f}% and
{100 * pooled.loc["table9_macro_bpb", "phase_to_whole_swarm_sd_ratio"]:.1f}%
of the full-swarm SD, respectively.

This is strong evidence that learning aggregate quality is easy relative to
learning phase order. It is not evidence that phase order is universally
irrelevant.

## Paired phase-effect SNR

For a phase contrast \\(d\\) around tied aggregate \\(a\\), define

$$
\\Delta(d)=L(a,d)-L(a,0).
$$

For antithetic pairs, decompose this response as

$$
O(d)=\\frac{{L(a,+d)-L(a,-d)}}{{2}},
\\qquad
C(d)=\\frac{{L(a,+d)+L(a,-d)}}{{2}}-L(a,0).
$$

\\(O\\) is the odd order effect and \\(C\\) is the even cost or benefit of making
the phases different. Same-seed delta noise is estimated from three policies
with three independently seeded replications each:

{same_seed_noise_table.to_markdown(index=False, floatfmt=".6f")}

After subtracting this noise variance, the target-matched phase-fiber latent
RMS SNR is approximately {uncheatable_fiber_latent:.2f} on Uncheatable and
{table9_fiber_latent:.2f} on Table-9. The low-radius isotropic panel reaches at
most {random_max_latent.get("uncheatable", float("nan")):.2f} and
{random_max_latent.get("table9", float("nan")):.2f}, respectively. In other
words, ordinary near-tied random phase perturbations are predominantly
noise-scale.

Aggressive antithetic partitions become identifiable:

{aggressive_channels.to_markdown(index=False, floatfmt=".6f")}

The important qualification is that the high-SNR response at phase TV 0.50 is
mostly an even asymmetry cost: large phase divergence is reliably harmful on
average. The scientifically useful order signal is the odd channel, which
becomes resolvable more slowly and does not by itself identify a beneficial
direction.

## Per-metric result

{rollup.to_markdown(index=False, floatfmt=".4f")}

The component table uses the ten-run proportional repeat panel whenever it is
available. The seven Uncheatable components use eight matched frontier controls
pooled after removing the two anchor means; that denominator has only six
degrees of freedom and should be interpreted as exploratory.

## Interpretation

1. **The bottleneck is local phase identification, not global metric noise.**
   Whole-mixture variation is high-SNR, while the response on aggregate-matched
   fibers is around one noise SD.
2. **Small random asymmetry is an inefficient discovery design.** At the
   radii tested, its latent response variance is indistinguishable from or only
   comparable to same-seed run noise.
3. **Large asymmetry supplies signal, but mostly about failure.** It is useful
   for learning overload and forgetting costs, not sufficient for finding a
   better-than-tied order.
4. **A 0.003 or 0.009 selected dip can coexist with low population SNR.**
   Selection searches the extreme tail; the SNR here measures response
   variation over a preregistered design distribution. A repeatable selected
   gain requires a direction model, not merely a wider random sweep.

## Statistical cautions

- SNR depends on the intervention distribution. Values for phase TV 0.10,
  0.25, and 0.50 are not interchangeable.
- The historical SNR contains both latent policy signal and run noise. The
  paired tables additionally report
  \\(\\sqrt{{\\max(\\operatorname{{RMS}}^2-\\sigma^2,0)}}/\\sigma\\).
- Noise confidence bounds propagate denominator uncertainty only; they are not
  full confidence intervals for the SNR.
- Same-seed covariance is identified from six variance degrees of freedom.
  The estimates are useful but still imprecise.
- The pending targeted pairwise panel is sealed and was not read.

## Artifacts

- `metric_phase_snr.csv`: all 60 smooth metrics, two anchors, and pooled values.
- `metric_phase_snr_rollup.csv`: objective-family summaries.
- `macro_phase_snr_by_design.csv`: random, fiber, antithetic, handcrafted, and
  Dolmino-late phase responses by radius.
- `same_seed_delta_noise.csv`: empirical paired-noise estimates.
- `metric_snr_comparison.html`: whole-swarm versus fixed-aggregate SNR.
- `metric_phase_snr_heatmap.html`: per-metric SNR table.
- `macro_phase_snr_by_radius.html`: noise-corrected macro SNR versus phase TV.
"""


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    component_panel = pd.read_csv(args.component_panel)
    heldouts = pd.read_csv(args.heldouts)
    aggressive = pd.read_csv(args.aggressive_results)
    antithetic = pd.read_csv(args.antithetic_pairs)
    proportional_summary = json.loads(args.proportional_noise.read_text())
    table9_component_noise = pd.read_csv(args.table9_component_noise, index_col=0)

    assert not heldouts["training_series"].astype(str).str.contains(SEALED_SERIES_FRAGMENT, case=False).any()
    assert set(component_panel["panel"]) == {"two_phase_fit", "one_phase_fit", "frontier_phase_fiber"}
    metrics = metric_columns(component_panel)
    assert len(metrics) == 60
    assert component_panel[metrics].notna().all().all()
    assert len(table9_component_noise) == int(proportional_summary["n_repeats"]) == 10

    fiber = component_panel.loc[component_panel["panel"].eq("frontier_phase_fiber")]
    noise_table = load_primary_noise(metrics, fiber, proportional_summary, table9_component_noise)
    metric_snr = metric_phase_snr(component_panel, noise_table, metrics)
    rollup = metric_rollup(metric_snr)
    same_seed_noise_table = same_seed_delta_noise(aggressive)
    macro_snr = build_macro_snr(
        heldouts,
        component_panel,
        aggressive,
        antithetic,
        same_seed_noise_table,
    )

    noise_table.to_csv(args.output_dir / "metric_noise_sources.csv", index=False)
    metric_snr.to_csv(args.output_dir / "metric_phase_snr.csv", index=False)
    rollup.to_csv(args.output_dir / "metric_phase_snr_rollup.csv", index=False)
    same_seed_noise_table.to_csv(args.output_dir / "same_seed_delta_noise.csv", index=False)
    macro_snr.to_csv(args.output_dir / "macro_phase_snr_by_design.csv", index=False)
    write_metric_comparison_plot(metric_snr, args.output_dir / "metric_snr_comparison.html")
    write_metric_heatmap(metric_snr, args.output_dir / "metric_phase_snr_heatmap.html")
    write_macro_plot(macro_snr, args.output_dir / "macro_phase_snr_by_radius.html")
    report = report_markdown(metric_snr, rollup, macro_snr, same_seed_noise_table)
    (args.output_dir / "report.md").write_text(report)

    pooled = metric_snr.loc[metric_snr["scope"].eq("pooled_anchor_centered")].set_index("metric")
    summary = {
        "metric_count": len(metrics),
        "component_panel_rows": len(component_panel),
        "fixed_aggregate_fiber_rows": len(fiber),
        "whole_swarm_snr": {
            target: float(pooled.loc[metric, "whole_swarm_historical_snr"]) for target, metric in TARGET_COLUMNS.items()
        },
        "fixed_aggregate_snr": {
            target: float(pooled.loc[metric, "fixed_aggregate_historical_snr"])
            for target, metric in TARGET_COLUMNS.items()
        },
        "phase_to_whole_swarm_sd_ratio": {
            target: float(pooled.loc[metric, "phase_to_whole_swarm_sd_ratio"])
            for target, metric in TARGET_COLUMNS.items()
        },
        "sealed_targeted_pairwise_panel_read": False,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")


if __name__ == "__main__":
    main()
