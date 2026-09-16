# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "plotly",
#   "tabulate",
# ]
# ///

"""Audit whether a phase-0 epoch cap preserves dense StarCoder surface optima."""

from __future__ import annotations

import argparse
import html
import json
import math
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from starcoder_wsd80_epoch_accounting import (
    SIMULATED_EPOCH_TARGET_BUDGET,
    STARCODER_SOURCE_TOKENS,
    simulated_materialized_epochs,
)

SCRIPT_DIR = Path(__file__).resolve().parent
EXPLORATORY_DIR = SCRIPT_DIR.parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_prefix_admissibility_20260824"

PRIMARY_CAP = 10.0
CAP_SWEEP = (4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0)
PROGRAMMING_LANGUAGES_BPB = "Programming Languages BPB"

HISTORICAL_COSINE_PATH = EXPLORATORY_DIR / "paper_plots" / "data" / "two_phase_starcoder_combined_143_from_wandb.csv"
HISTORICAL_WSD50_PATH = (
    EXPLORATORY_DIR
    / "starcoder_wsd_boundary_aligned_repeat_outputs"
    / "two_phase_feature_bayes_linear_20260313_211537"
    / "proxy_results.csv"
)
TOKEN_LADDER_PATH = REFERENCE_OUTPUTS / "starcoder_wsd80_token_budget_surfaces_20260731" / "surface_coordinates.csv"
MATCHED_ND_PATH = (
    REFERENCE_OUTPUTS
    / "starcoder_wsd80_matched_nd_stage1_20260731"
    / "stage3_dense_surface_results_20260802"
    / "combined_discovery_observations.csv"
)
DENSE_REPLAY_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_dense_support_calibration_results_20260813"
DENSE_REPLAY_PATH = DENSE_REPLAY_DIR / "coverage_with_calibration_weights.csv"
WEIGHTED_SURFACE_SUMMARY_PATH = DENSE_REPLAY_DIR / "weighted_surface_summary.csv"
ATOMIC_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_full_pool_atomic_surface_explorer_20260811"
ATOMIC_OBSERVATIONS_PATH = ATOMIC_DIR / "full_pool_atomic_metric_observations.csv"
ATOMIC_OPTIMA_PATH = ATOMIC_DIR / "atomic_metric_raw_optima.csv"
CONFIRMATION_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_dense_support_empirical_optimum_confirmation_results_20260811"
CONFIRMATION_SUMMARY_PATH = CONFIRMATION_DIR / "block_summary.csv"
CALIBRATION_REPEATS_PATH = DENSE_REPLAY_DIR / "calibration_repeat_observations.csv"

FIXED_N_CELL_IDS = {
    "r0_shared_h0640_s03820",
    "r1_increase_d_h0640_s07320",
    "r2_increase_d_h0640_s14960",
    "r3_increase_d_h0640_s28260",
}
MISS_CELL_ID = "r1_increase_d_h0640_s07320"
MISS_SUPPORT_ID = "m400"

PANEL_LABELS = {
    "historical_schedule": "Historical schedules",
    "fixed_model_token_ladder": "Fixed-model token ladder",
    "matched_nd": "Matched N,D grid",
    "dense_horizon_replay": "Dense horizon x replay",
}
PANEL_COLORS = {
    "historical_schedule": "#1b9e77",
    "fixed_model_token_ladder": "#66a61e",
    "matched_nd": "#e6ab02",
    "dense_horizon_replay": "#d95f02",
}


@dataclass(frozen=True)
class MissEvidence:
    cell_id: str
    support_id: str
    raw_coordinate_id: str
    raw_phase_0_epochs: float
    raw_bpb: float
    replacement_coordinate_id: str
    replacement_phase_0_epochs: float
    replacement_bpb: float
    replacement_l2: float
    observed_regret_bpb: float
    confirmation_coordinate_id: str
    confirmation_phase_0_epochs: float
    fresh_raw_mean_bpb: float
    fresh_confirmation_mean_bpb: float
    raw_minus_confirmation_bpb: float
    ci95_low: float
    ci95_high: float
    one_sided_p: float
    holm_p: float
    confirmation_win_count: int
    pair_count: int
    winner_curse_bpb: float
    raw_discovery_to_fresh_shift_bpb: float
    confirmation_discovery_to_fresh_shift_bpb: float
    replacement_all_mean_bpb: float
    replacement_repeats_only_mean_bpb: float
    replacement_all_sd_bpb: float
    replacement_observation_count: int
    replacement_predicted_sd_bpb: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def _standard_frame(
    frame: pd.DataFrame,
    *,
    panel: str,
    surface_id: str,
    configuration_key: str,
    objective_column: str,
    phase_0_column: str,
    phase_1_column: str,
    phase_0_epochs: pd.Series,
    coordinate_column: str,
    source_path: Path,
    epoch_accounting: str,
) -> pd.DataFrame:
    required = {objective_column, phase_0_column, phase_1_column, coordinate_column}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source_path} is missing columns: {sorted(missing)}")
    if len(phase_0_epochs) != len(frame):
        raise ValueError(f"Epoch accounting length mismatch for {surface_id}")
    result = pd.DataFrame(
        {
            "panel": panel,
            "surface_id": surface_id,
            "configuration_key": configuration_key,
            "objective": PROGRAMMING_LANGUAGES_BPB,
            "coordinate_id": frame[coordinate_column].astype(str).to_numpy(),
            "phase_0_starcoder": frame[phase_0_column].astype(float).to_numpy(),
            "phase_1_starcoder": frame[phase_1_column].astype(float).to_numpy(),
            "phase_0_starcoder_epochs": phase_0_epochs.astype(float).to_numpy(),
            "bpb": frame[objective_column].astype(float).to_numpy(),
            "source_path": str(source_path),
            "epoch_accounting": epoch_accounting,
        }
    )
    if result["bpb"].isna().any():
        raise ValueError(f"{surface_id} has missing {PROGRAMMING_LANGUAGES_BPB}")
    return result


def _historical_schedule_observations() -> list[pd.DataFrame]:
    cosine = pd.read_csv(HISTORICAL_COSINE_PATH)
    cosine_frame = _standard_frame(
        cosine,
        panel="historical_schedule",
        surface_id="cosine_50_50",
        configuration_key="historical_cosine_50_50",
        objective_column="eval/paloma/dolma_100_programing_languages/bpb",
        phase_0_column="phase_0_starcoder",
        phase_1_column="phase_1_starcoder",
        phase_0_epochs=cosine["phase_0_starcoder_epochs"],
        coordinate_column="run_id",
        source_path=HISTORICAL_COSINE_PATH,
        epoch_accounting="50/50 phase share times fixed simulated materialized support, recorded in source table",
    )

    wsd50 = pd.read_csv(HISTORICAL_WSD50_PATH)
    starcoder_multiplier = SIMULATED_EPOCH_TARGET_BUDGET / STARCODER_SOURCE_TOKENS
    wsd50_epochs = 0.5 * wsd50["phase_0_starcoder"].astype(float) * starcoder_multiplier
    wsd50_frame = _standard_frame(
        wsd50,
        panel="historical_schedule",
        surface_id="wsd_50_50",
        configuration_key="historical_wsd_50_50",
        objective_column="actual_bpb",
        phase_0_column="phase_0_starcoder",
        phase_1_column="phase_1_starcoder",
        phase_0_epochs=wsd50_epochs,
        coordinate_column="run_name",
        source_path=HISTORICAL_WSD50_PATH,
        epoch_accounting="50/50 phase share times fixed simulated materialized support",
    )
    return [cosine_frame, wsd50_frame]


def _wsd80_phase_0_epochs(frame: pd.DataFrame, phase_0_column: str, phase_1_column: str) -> pd.Series:
    values = [
        simulated_materialized_epochs(float(phase_0), float(phase_1)).starcoder.phase_0
        for phase_0, phase_1 in zip(frame[phase_0_column], frame[phase_1_column], strict=True)
    ]
    return pd.Series(values, index=frame.index, dtype=float)


def _token_ladder_observations() -> list[pd.DataFrame]:
    frame = pd.read_csv(TOKEN_LADDER_PATH)
    frame["phase_0_epochs"] = _wsd80_phase_0_epochs(frame, "p0", "p1")
    observations: list[pd.DataFrame] = []
    for token_label, block in frame.groupby("token_budget_label", sort=False):
        observations.append(
            _standard_frame(
                block,
                panel="fixed_model_token_ladder",
                surface_id=f"token_ladder_{token_label}",
                configuration_key=f"fixed_model_{token_label}",
                objective_column="bpb",
                phase_0_column="p0",
                phase_1_column="p1",
                phase_0_epochs=block["phase_0_epochs"],
                coordinate_column="wandb_id",
                source_path=TOKEN_LADDER_PATH,
                epoch_accounting="fixed WSD80 simulated materialized support",
            )
        )
    return observations


def _matched_nd_observations() -> list[pd.DataFrame]:
    frame = pd.read_csv(MATCHED_ND_PATH)
    frame["phase_0_epochs"] = _wsd80_phase_0_epochs(
        frame,
        "phase_0_starcoder",
        "phase_1_starcoder",
    )
    observations: list[pd.DataFrame] = []
    for cell_id, block in frame.groupby("cell_id", sort=False):
        configuration_key = f"fixed_n_{cell_id}" if cell_id in FIXED_N_CELL_IDS else f"matched_nd_{cell_id}"
        observations.append(
            _standard_frame(
                block,
                panel="matched_nd",
                surface_id=f"matched_nd_{cell_id}",
                configuration_key=configuration_key,
                objective_column="starcoder_bpb",
                phase_0_column="phase_0_starcoder",
                phase_1_column="phase_1_starcoder",
                phase_0_epochs=block["phase_0_epochs"],
                coordinate_column="run_name",
                source_path=MATCHED_ND_PATH,
                epoch_accounting="fixed WSD80 simulated materialized support",
            )
        )
    return observations


def _dense_replay_observations() -> list[pd.DataFrame]:
    frame = pd.read_csv(DENSE_REPLAY_PATH)
    observations: list[pd.DataFrame] = []
    for (cell_id, support_id), block in frame.groupby(["cell_id", "support_id"], sort=False):
        configuration_key = (
            f"fixed_n_{cell_id}"
            if support_id == "m100" and cell_id in FIXED_N_CELL_IDS
            else f"replay_{cell_id}_{support_id}"
        )
        observations.append(
            _standard_frame(
                block,
                panel="dense_horizon_replay",
                surface_id=f"dense_replay_{cell_id}_{support_id}",
                configuration_key=configuration_key,
                objective_column="bpb",
                phase_0_column="phase_0_starcoder",
                phase_1_column="phase_1_starcoder",
                phase_0_epochs=block["starcoder_phase_0_epochs"],
                coordinate_column="coordinate_id",
                source_path=DENSE_REPLAY_PATH,
                epoch_accounting="exact realized support and MixtureDataset allocation",
            )
        )
    return observations


def load_primary_observations() -> pd.DataFrame:
    blocks = [
        *_historical_schedule_observations(),
        *_token_ladder_observations(),
        *_matched_nd_observations(),
        *_dense_replay_observations(),
    ]
    observations = pd.concat(blocks, ignore_index=True)
    surface_count = observations["surface_id"].nunique()
    if surface_count != 44:
        raise ValueError(f"Expected 44 primary surface instances, found {surface_count}")
    if len(observations) != 5036:
        raise ValueError(f"Expected 5,036 primary observations, found {len(observations)}")
    panel_counts = observations.groupby("panel")["surface_id"].nunique().to_dict()
    expected_panel_counts = {
        "historical_schedule": 2,
        "fixed_model_token_ladder": 4,
        "matched_nd": 10,
        "dense_horizon_replay": 28,
    }
    if panel_counts != expected_panel_counts:
        raise ValueError(f"Unexpected panel decomposition: {panel_counts}")
    return observations


def audit_surfaces(observations: pd.DataFrame, cap: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for surface_id, block in observations.groupby("surface_id", sort=False):
        raw = block.loc[block["bpb"].idxmin()]
        admissible = block.loc[block["phase_0_starcoder_epochs"] <= cap + 1e-12]
        if admissible.empty:
            raise ValueError(f"Cap {cap} removes every coordinate in {surface_id}")
        admissible_best = admissible.loc[admissible["bpb"].idxmin()]
        rows.append(
            {
                "panel": raw["panel"],
                "surface_id": surface_id,
                "configuration_key": raw["configuration_key"],
                "objective": raw["objective"],
                "cap_phase_0_epochs": cap,
                "observation_count": len(block),
                "admissible_count": len(admissible),
                "admissible_fraction": len(admissible) / len(block),
                "raw_coordinate_id": raw["coordinate_id"],
                "raw_phase_0_starcoder": raw["phase_0_starcoder"],
                "raw_phase_1_starcoder": raw["phase_1_starcoder"],
                "raw_phase_0_starcoder_epochs": raw["phase_0_starcoder_epochs"],
                "raw_min_bpb": raw["bpb"],
                "exact_raw_optimum_retained": bool(raw["phase_0_starcoder_epochs"] <= cap + 1e-12),
                "best_admissible_coordinate_id": admissible_best["coordinate_id"],
                "best_admissible_phase_0_starcoder": admissible_best["phase_0_starcoder"],
                "best_admissible_phase_1_starcoder": admissible_best["phase_1_starcoder"],
                "best_admissible_phase_0_starcoder_epochs": admissible_best["phase_0_starcoder_epochs"],
                "best_admissible_bpb": admissible_best["bpb"],
                "observed_regret_bpb": admissible_best["bpb"] - raw["bpb"],
                "epoch_accounting": raw["epoch_accounting"],
                "source_path": raw["source_path"],
            }
        )
    result = pd.DataFrame(rows)
    if len(result) != 44:
        raise ValueError(f"Expected 44 audited surfaces, found {len(result)}")
    return result


def cap_sweep(observations: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for cap in CAP_SWEEP:
        audit = audit_surfaces(observations, cap)
        rows.append(
            {
                "cap_phase_0_epochs": cap,
                "surface_instances_retained": int(audit["exact_raw_optimum_retained"].sum()),
                "surface_instance_count": len(audit),
                "surface_instances_pruned": int(audit["admissible_fraction"].lt(1.0).sum()),
                "coordinate_fraction_retained": float((observations["phase_0_starcoder_epochs"] <= cap + 1e-12).mean()),
                "maximum_observed_regret_bpb": float(audit["observed_regret_bpb"].max()),
            }
        )
    return pd.DataFrame(rows)


def atomic_metric_sensitivity(cap: float) -> pd.DataFrame:
    coverage = pd.read_csv(DENSE_REPLAY_PATH)
    metadata = coverage[
        [
            "run_name",
            "is_alias",
            "alias_of_run_name",
            "cell_id",
            "support_id",
            "coordinate_id",
            "phase_0_starcoder",
            "phase_1_starcoder",
            "starcoder_phase_0_epochs",
        ]
    ].copy()
    metadata["metric_run_name"] = metadata["alias_of_run_name"].where(metadata["is_alias"], metadata["run_name"])
    metric_values = pd.read_csv(ATOMIC_OBSERVATIONS_PATH).rename(columns={"run_name": "metric_run_name"})
    atomic = metadata.merge(
        metric_values,
        on="metric_run_name",
        how="left",
        validate="many_to_one",
    )
    metric_columns = [column for column in metric_values.columns if column != "metric_run_name"]
    if len(metric_columns) != 23:
        raise ValueError(f"Expected 23 atomic metrics, found {len(metric_columns)}")
    if int(metadata["is_alias"].sum()) != 396:
        raise ValueError(f"Expected 396 finite-support aliases, found {int(metadata['is_alias'].sum())}")
    if len(atomic) != 3500 or atomic[metric_columns].isna().any().any():
        raise ValueError("Atomic alias recovery must yield 3,500 complete coordinate rows")
    if set(atomic.groupby(["cell_id", "support_id"]).size()) != {125}:
        raise ValueError("Expected 125 atomic coordinates in every dense surface")
    labels = (
        pd.read_csv(ATOMIC_OPTIMA_PATH)[["metric_key", "metric_label"]]
        .drop_duplicates("metric_key")
        .set_index("metric_key")["metric_label"]
        .to_dict()
    )
    rows: list[dict[str, object]] = []
    for (cell_id, support_id), block in atomic.groupby(["cell_id", "support_id"], sort=False):
        for metric in metric_columns:
            available = block.dropna(subset=[metric])
            raw = available.loc[available[metric].idxmin()]
            admissible = available.loc[available["starcoder_phase_0_epochs"] <= cap + 1e-12]
            if admissible.empty:
                raise ValueError(f"Cap {cap} removes every coordinate in {cell_id}/{support_id}/{metric}")
            admissible_best = admissible.loc[admissible[metric].idxmin()]
            rows.append(
                {
                    "metric_key": metric,
                    "metric_label": labels[metric],
                    "cell_id": cell_id,
                    "support_id": support_id,
                    "cap_phase_0_epochs": cap,
                    "observation_count": len(available),
                    "raw_coordinate_id": raw["coordinate_id"],
                    "raw_phase_0_starcoder": raw["phase_0_starcoder"],
                    "raw_phase_1_starcoder": raw["phase_1_starcoder"],
                    "raw_phase_0_starcoder_epochs": raw["starcoder_phase_0_epochs"],
                    "raw_min_bpb": raw[metric],
                    "exact_raw_optimum_retained": bool(raw["starcoder_phase_0_epochs"] <= cap + 1e-12),
                    "best_admissible_coordinate_id": admissible_best["coordinate_id"],
                    "best_admissible_bpb": admissible_best[metric],
                    "observed_regret_bpb": admissible_best[metric] - raw[metric],
                }
            )
    result = pd.DataFrame(rows)
    if len(result) != 644:
        raise ValueError(f"Expected 644 atomic objective surfaces, found {len(result)}")
    return result


def confirmation_cap_contrasts(cap: float) -> pd.DataFrame:
    summary = pd.read_csv(CONFIRMATION_SUMMARY_PATH)
    coverage = pd.read_csv(DENSE_REPLAY_PATH)[["cell_id", "support_id", "coordinate_id", "starcoder_phase_0_epochs"]]
    tied = coverage.rename(
        columns={"coordinate_id": "tied_coordinate_id", "starcoder_phase_0_epochs": "tied_phase_0_epochs"}
    )
    untied = coverage.rename(
        columns={"coordinate_id": "untied_coordinate_id", "starcoder_phase_0_epochs": "untied_phase_0_epochs"}
    )
    merged = summary.merge(
        tied,
        on=["cell_id", "support_id", "tied_coordinate_id"],
        validate="one_to_one",
    ).merge(
        untied,
        on=["cell_id", "support_id", "untied_coordinate_id"],
        validate="one_to_one",
    )
    contrasts = merged.loc[
        merged["tied_phase_0_epochs"].gt(cap + 1e-12) & merged["untied_phase_0_epochs"].le(cap + 1e-12)
    ].copy()
    contrasts["admissible_wins_mean"] = contrasts["mean_gain_bpb"].gt(0.0)
    contrasts["admissible_wins_all_seeds"] = contrasts["untied_win_count"].eq(contrasts["pair_count"])
    if len(contrasts) != 8:
        raise ValueError(f"Expected eight direct cap-crossing confirmation contrasts, found {len(contrasts)}")
    return contrasts.sort_values(["cell_order", "support_order"]).reset_index(drop=True)


def miss_evidence(audit: pd.DataFrame, confirmations: pd.DataFrame) -> MissEvidence:
    misses = audit.loc[~audit["exact_raw_optimum_retained"]]
    if len(misses) != 1:
        raise ValueError(f"Expected one cap-10 raw-optimum miss, found {len(misses)}")
    miss = misses.iloc[0]
    if miss["surface_id"] != f"dense_replay_{MISS_CELL_ID}_{MISS_SUPPORT_ID}":
        raise ValueError(f"Unexpected cap-10 miss: {miss['surface_id']}")

    selected = confirmations.loc[
        confirmations["cell_id"].eq(MISS_CELL_ID) & confirmations["support_id"].eq(MISS_SUPPORT_ID)
    ]
    if len(selected) != 1:
        raise ValueError("Expected one cap-crossing fresh-confirmation row for the cap-10 miss")
    row = selected.iloc[0]
    if str(row["tied_coordinate_id"]) != str(miss["raw_coordinate_id"]):
        raise ValueError("Fresh confirmation does not target the computed cap-10 raw argmin")

    coverage = pd.read_csv(DENSE_REPLAY_PATH)
    repeats = pd.read_csv(CALIBRATION_REPEATS_PATH)
    replacement_coordinate_id = str(miss["best_admissible_coordinate_id"])
    discovery = coverage.loc[
        coverage["cell_id"].eq(MISS_CELL_ID)
        & coverage["support_id"].eq(MISS_SUPPORT_ID)
        & coverage["coordinate_id"].eq(replacement_coordinate_id)
    ]
    if len(discovery) != 1:
        raise ValueError("Expected one discovery row for the cap-10 replacement")
    repeat_values = repeats.loc[
        repeats["cell_id"].eq(MISS_CELL_ID)
        & repeats["support_id"].eq(MISS_SUPPORT_ID)
        & repeats["coordinate_id"].eq(replacement_coordinate_id),
        "bpb",
    ]
    calibration = pd.concat(
        [
            discovery["bpb"],
            repeat_values,
        ],
        ignore_index=True,
    )
    if len(calibration) != 4:
        raise ValueError(f"Expected four replacement observations, found {len(calibration)}")
    return MissEvidence(
        cell_id=MISS_CELL_ID,
        support_id=MISS_SUPPORT_ID,
        raw_coordinate_id=str(miss["raw_coordinate_id"]),
        raw_phase_0_epochs=float(miss["raw_phase_0_starcoder_epochs"]),
        raw_bpb=float(miss["raw_min_bpb"]),
        replacement_coordinate_id=replacement_coordinate_id,
        replacement_phase_0_epochs=float(miss["best_admissible_phase_0_starcoder_epochs"]),
        replacement_bpb=float(miss["best_admissible_bpb"]),
        replacement_l2=math.hypot(
            float(miss["raw_phase_0_starcoder"] - miss["best_admissible_phase_0_starcoder"]),
            float(miss["raw_phase_1_starcoder"] - miss["best_admissible_phase_1_starcoder"]),
        ),
        observed_regret_bpb=float(miss["observed_regret_bpb"]),
        confirmation_coordinate_id=str(row["untied_coordinate_id"]),
        confirmation_phase_0_epochs=float(row["untied_phase_0_epochs"]),
        fresh_raw_mean_bpb=float(row["fresh_tied_mean_bpb"]),
        fresh_confirmation_mean_bpb=float(row["fresh_untied_mean_bpb"]),
        raw_minus_confirmation_bpb=float(row["mean_gain_bpb"]),
        ci95_low=float(row["ci95_low"]),
        ci95_high=float(row["ci95_high"]),
        one_sided_p=float(row["paired_t_one_sided_p"]),
        holm_p=float(row["paired_t_holm_p"]),
        confirmation_win_count=int(row["untied_win_count"]),
        pair_count=int(row["pair_count"]),
        winner_curse_bpb=float(row["winner_curse_bpb"]),
        raw_discovery_to_fresh_shift_bpb=float(row["fresh_tied_mean_bpb"] - row["discovery_tied_bpb"]),
        confirmation_discovery_to_fresh_shift_bpb=float(row["fresh_untied_mean_bpb"] - row["discovery_untied_bpb"]),
        replacement_all_mean_bpb=float(calibration.mean()),
        replacement_repeats_only_mean_bpb=float(repeat_values.mean()),
        replacement_all_sd_bpb=float(calibration.std(ddof=1)),
        replacement_observation_count=len(calibration),
        replacement_predicted_sd_bpb=float(discovery.iloc[0]["predicted_sd_bpb"]),
    )


def _panel_summary(audit: pd.DataFrame) -> pd.DataFrame:
    return (
        audit.groupby("panel", sort=False)
        .agg(
            surface_instances=("surface_id", "size"),
            exact_raw_optima_retained=("exact_raw_optimum_retained", "sum"),
            maximum_raw_phase_0_epochs=("raw_phase_0_starcoder_epochs", "max"),
            mean_coordinate_fraction_retained=("admissible_fraction", "mean"),
            maximum_observed_regret_bpb=("observed_regret_bpb", "max"),
        )
        .reset_index()
    )


def _render_figure(audit: pd.DataFrame, sweep: pd.DataFrame) -> str:
    retrospective_cap = math.ceil(float(audit["raw_phase_0_starcoder_epochs"].max()))
    if retrospective_cap != 12:
        raise ValueError(f"Expected retrospective integer cap 12, found {retrospective_cap}")
    ordered = audit.sort_values(["panel", "raw_phase_0_starcoder_epochs", "surface_id"]).reset_index(drop=True)
    ordered["surface_index"] = range(1, len(ordered) + 1)
    figure = make_subplots(
        rows=1,
        cols=2,
        column_widths=[0.63, 0.37],
        horizontal_spacing=0.12,
        subplot_titles=("Phase-0 epochs at each raw observed optimum", "Cap recall versus search-space retention"),
    )
    for panel, block in ordered.groupby("panel", sort=False):
        figure.add_trace(
            go.Scatter(
                x=block["surface_index"],
                y=block["raw_phase_0_starcoder_epochs"],
                mode="markers",
                name=PANEL_LABELS[panel],
                marker={"size": 11, "color": PANEL_COLORS[panel], "line": {"color": "#17324d", "width": 0.8}},
                customdata=block[
                    [
                        "surface_id",
                        "raw_phase_0_starcoder",
                        "raw_phase_1_starcoder",
                        "raw_min_bpb",
                        "exact_raw_optimum_retained",
                    ]
                ],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "raw optimum=(%{customdata[1]:.4f}, %{customdata[2]:.4f})<br>"
                    "phase-0 StarCoder epochs=%{y:.4f}<br>"
                    "BPB=%{customdata[3]:.6f}<br>"
                    "retained at 10 epochs=%{customdata[4]}<extra></extra>"
                ),
            ),
            row=1,
            col=1,
        )
    figure.add_hline(y=PRIMARY_CAP, line={"color": "#d95f02", "width": 2.3}, row=1, col=1)
    figure.add_hline(y=retrospective_cap, line={"color": "#7570b3", "width": 1.6, "dash": "dot"}, row=1, col=1)
    figure.add_annotation(
        x=44,
        y=PRIMARY_CAP,
        text="pre-existing candidate cap = 10",
        showarrow=False,
        xanchor="right",
        yshift=13,
        font={"color": "#d95f02", "size": 12},
        row=1,
        col=1,
    )
    figure.add_annotation(
        x=44,
        y=retrospective_cap,
        text=f"retrospective exact raw-primary recall = {retrospective_cap}",
        showarrow=False,
        xanchor="right",
        yshift=13,
        font={"color": "#7570b3", "size": 12},
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Scatter(
            x=sweep["cap_phase_0_epochs"],
            y=sweep["surface_instances_retained"] / sweep["surface_instance_count"],
            mode="lines+markers",
            name="Raw optimum recall",
            line={"color": "#1b9e77", "width": 3},
            marker={"size": 10},
            hovertemplate="cap=%{x:.0f}<br>raw optimum recall=%{y:.1%}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.add_trace(
        go.Scatter(
            x=sweep["cap_phase_0_epochs"],
            y=sweep["coordinate_fraction_retained"],
            mode="lines+markers",
            name="Coordinates retained",
            line={"color": "#d95f02", "width": 3},
            marker={"size": 10, "symbol": "diamond"},
            hovertemplate="cap=%{x:.0f}<br>coordinates retained=%{y:.1%}<extra></extra>",
        ),
        row=1,
        col=2,
    )
    figure.add_vline(x=PRIMARY_CAP, line={"color": "#17324d", "width": 1.5}, row=1, col=2)
    figure.update_xaxes(title_text="Surface instance", showticklabels=False, row=1, col=1)
    figure.update_yaxes(title_text="Phase-0 StarCoder materialized epochs", rangemode="tozero", row=1, col=1)
    figure.update_xaxes(title_text="Phase-0 epoch cap", row=1, col=2)
    figure.update_yaxes(title_text="Fraction", tickformat=".0%", range=[0.45, 1.03], row=1, col=2)
    figure.update_layout(
        width=1500,
        height=650,
        margin={"l": 90, "r": 45, "t": 105, "b": 80},
        paper_bgcolor="#f7f3e8",
        plot_bgcolor="#fffdf8",
        font={"family": "Avenir Next, Source Sans Pro, sans-serif", "size": 14, "color": "#17324d"},
        legend={
            "orientation": "h",
            "x": 0,
            "y": 1.13,
            "bgcolor": "rgba(255,253,248,0.92)",
            "bordercolor": "#d8d1c2",
            "borderwidth": 1,
        },
        hoverlabel={"align": "left"},
    )
    figure.update_xaxes(gridcolor="#e3ddcf", linecolor="#17324d", showline=True)
    figure.update_yaxes(gridcolor="#e3ddcf", linecolor="#17324d", showline=True)
    return figure.to_html(
        full_html=False,
        include_plotlyjs=True,
        config={
            "displaylogo": False,
            "responsive": True,
            "toImageButtonOptions": {"format": "png", "filename": "starcoder_prefix_admissibility", "scale": 4},
        },
    )


def _format_panel_table(summary: pd.DataFrame) -> str:
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{html.escape(PANEL_LABELS[row.panel])}</td>"
            f"<td>{row.exact_raw_optima_retained}/{row.surface_instances}</td>"
            f"<td>{row.maximum_raw_phase_0_epochs:.3f}</td>"
            f"<td>{row.mean_coordinate_fraction_retained:.1%}</td>"
            f"<td>{row.maximum_observed_regret_bpb:.6f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _format_confirmation_table(confirmations: pd.DataFrame) -> str:
    rows = []
    for row in confirmations.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td><code>{html.escape(row.cell_id)}/{html.escape(row.support_id)}</code></td>"
            f"<td><code>{html.escape(row.tied_coordinate_id)}</code> ({row.tied_phase_0_epochs:.3f})</td>"
            f"<td><code>{html.escape(row.untied_coordinate_id)}</code> ({row.untied_phase_0_epochs:.3f})</td>"
            f"<td>{row.mean_gain_bpb:+.6f}</td>"
            f"<td>{row.untied_win_count}/{row.pair_count}</td>"
            f"<td>{row.paired_t_holm_p:.4f}</td>"
            "</tr>"
        )
    return "".join(rows)


def _weighted_surface_caveat() -> pd.DataFrame:
    surfaces = pd.read_csv(WEIGHTED_SURFACE_SUMMARY_PATH)
    selected = surfaces.loc[
        surfaces["support_id"].eq("m100")
        & surfaces["cell_id"].isin(
            [
                "r1_increase_d_h0640_s07320",
                "r2_increase_d_h0640_s14960",
                "r3_increase_d_h0640_s28260",
            ]
        )
    ].copy()
    coverage = pd.read_csv(DENSE_REPLAY_PATH)
    observed_minima = (
        coverage.groupby(["cell_id", "support_id"], as_index=False)["bpb"]
        .min()
        .rename(columns={"bpb": "observed_min_bpb"})
    )
    selected = selected.merge(observed_minima, on=["cell_id", "support_id"], validate="one_to_one")
    if len(selected) != 3:
        raise ValueError(f"Expected three unstable m100 weighted-surface fits, found {len(selected)}")
    if not selected["weighted_surface_untied_bpb"].lt(selected["observed_min_bpb"]).all():
        raise ValueError("Expected unstable weighted fits to extrapolate below every observed minimum")
    return selected


def _nominal_pair_summary() -> pd.DataFrame:
    matched = pd.read_csv(MATCHED_ND_PATH)[["cell_id", "phase_0_starcoder", "phase_1_starcoder", "starcoder_bpb"]]
    dense = pd.read_csv(DENSE_REPLAY_PATH).loc[
        lambda frame: frame["support_id"].eq("m100"),
        ["cell_id", "phase_0_starcoder", "phase_1_starcoder", "bpb"],
    ]
    overlaps = matched.merge(
        dense,
        on=["cell_id", "phase_0_starcoder", "phase_1_starcoder"],
        validate="one_to_one",
    )
    overlaps["matched_minus_dense_bpb"] = overlaps["starcoder_bpb"] - overlaps["bpb"]
    summary = overlaps.groupby("cell_id")["matched_minus_dense_bpb"].agg(["count", "mean"]).reset_index()
    if set(summary["cell_id"]) != FIXED_N_CELL_IDS:
        raise ValueError(f"Expected all four nominally matched fixed-N cells, found {set(summary['cell_id'])}")
    if summary["count"].min() < 40:
        raise ValueError("Expected at least 40 shared coordinates in every nominally matched panel pair")
    return summary


def write_html(
    output_path: Path,
    audit: pd.DataFrame,
    sweep: pd.DataFrame,
    atomic: pd.DataFrame,
    confirmations: pd.DataFrame,
    miss: MissEvidence,
) -> None:
    summary = _panel_summary(audit)
    exact = int(audit["exact_raw_optimum_retained"].sum())
    retained = int(audit["admissible_count"].sum())
    total = int(audit["observation_count"].sum())
    atomic_exact = int(atomic["exact_raw_optimum_retained"].sum())
    code_atomic = atomic.loc[
        atomic["metric_label"].isin(
            ["Paloma · Programming Languages", "Uncheatable · GitHub C++", "Uncheatable · GitHub Python"]
        )
    ]
    code_atomic_exact = int(code_atomic["exact_raw_optimum_retained"].sum())
    confirmation_wins = int(confirmations["admissible_wins_mean"].sum())
    confirmation_sweeps = int(confirmations["admissible_wins_all_seeds"].sum())
    holm_wins = int((confirmations["mean_gain_bpb"].gt(0.0) & confirmations["paired_t_holm_p"].lt(0.05)).sum())
    vacuous = int(audit["admissible_fraction"].eq(1.0).sum())
    near_binding = int(audit["raw_phase_0_starcoder_epochs"].gt(8.0).sum())
    weighted = _weighted_surface_caveat()
    nominal_pairs = _nominal_pair_summary()
    weighted_confirmations = confirmations.loc[
        confirmations["support_id"].eq("m100") & confirmations["cell_id"].isin(weighted["cell_id"])
    ]
    if len(weighted_confirmations) != 2 or not weighted_confirmations["admissible_wins_all_seeds"].all():
        raise ValueError("Expected two 5/5 cap-crossing confirmations for unstable m100 fits")
    weighted_unconfirmed = sorted(set(weighted["cell_id"]) - set(weighted_confirmations["cell_id"]))
    if weighted_unconfirmed != ["r1_increase_d_h0640_s07320"]:
        raise ValueError(f"Unexpected unstable-fit confirmation gap: {weighted_unconfirmed}")
    practical_exact_cap = math.ceil(float(audit["raw_phase_0_starcoder_epochs"].max()))
    if practical_exact_cap != 12:
        raise ValueError(f"Expected retrospective integer cap 12, found {practical_exact_cap}")
    simulated_epoch_scale = SIMULATED_EPOCH_TARGET_BUDGET / STARCODER_SOURCE_TOKENS
    historical_weight_cap = PRIMARY_CAP / (0.5 * simulated_epoch_scale)
    wsd80_weight_cap = PRIMARY_CAP / (0.8 * simulated_epoch_scale)
    plot = _render_figure(audit, sweep)
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>StarCoder prefix-admissibility audit</title>
<style>
:root {{ --ink:#17324d; --paper:#f7f3e8; --card:#fffdf8; --rule:#d8d1c2; --green:#1b9e77; --orange:#d95f02; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; color:var(--ink); background:var(--paper); font-family:"Avenir Next","Source Sans Pro",sans-serif; }}
main {{ max-width:1540px; margin:0 auto; padding:54px 34px 72px; }}
h1,h2 {{ font-family:Georgia,"Times New Roman",serif; margin:0; }}
h1 {{ font-size:48px; line-height:1.05; max-width:1050px; }}
h2 {{ font-size:30px; margin-top:42px; }}
.lede {{ font-size:21px; line-height:1.5; max-width:1120px; color:#52687a; }}
.cards {{ display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:18px; margin:32px 0; }}
.card {{ background:var(--card); border:1px solid var(--rule); padding:24px; min-height:150px; }}
.value {{ display:block; font-family:Georgia,"Times New Roman",serif; font-size:44px; color:var(--green); }}
.label {{ display:block; margin-top:9px; font-size:16px; line-height:1.35; }}
.verdict {{ border-left:7px solid var(--green); background:var(--card); padding:22px 28px; font-size:20px; line-height:1.5; }}
.caveat {{ border-left-color:var(--orange); margin-top:18px; }}
.plot {{ margin:28px -18px 0; }}
table {{ width:100%; border-collapse:collapse; background:var(--card); margin-top:18px; }}
th,td {{ padding:13px 15px; border-bottom:1px solid var(--rule); text-align:right; }}
th:first-child,td:first-child {{ text-align:left; }}
th {{ font-size:13px; text-transform:uppercase; letter-spacing:.06em; }}
p,li {{ font-size:18px; line-height:1.55; }}
code {{ background:#efe9dc; padding:2px 5px; }}
.small {{ font-size:15px; color:#667987; }}
@media (max-width:900px) {{ .cards {{ grid-template-columns:1fr; }} h1 {{ font-size:38px; }} main {{ padding:34px 18px 56px; }} }}
</style>
</head>
<body><main>
<h1>Does a simple phase-0 epoch cap preserve the StarCoder optimum?</h1>
<p class="lede">This is a recall audit of the admissibility filter only. A prefix is retained when phase-0 StarCoder materialized epochs do not exceed the cap; Nemotron's broad pool never binds in these panels. We do not fit or choose a prefix here. We only ask whether a competitive observed continuation remains searchable.</p>
<div class="cards">
  <div class="card"><span class="value">{exact}/44</span><span class="label">one-seed raw Programming Languages BPB argmins retained by the pre-existing candidate cap of 10 epochs</span></div>
  <div class="card"><span class="value">{retained / total:.1%}</span><span class="label">of {total:,} observed coordinates retained, so the filter removes {1 - retained / total:.1%} of the search space</span></div>
  <div class="card"><span class="value">1 miss</span><span class="label">the excluded raw winner is only {miss.replacement_l2:.4f} L2 from the best admissible point and worsens {miss.raw_discovery_to_fresh_shift_bpb:.4f} BPB on fresh seeds</span></div>
</div>
<div class="verdict"><strong>Answer:</strong> cap 10 is supported as a high-recall candidate filter, but the archive does not certify basin preservation in every setting. It retains 43/44 exact one-seed raw primary argmins and a nearby tied point in the sole miss; that excluded draw is not stable under fresh seeds.</div>
<div class="verdict caveat"><strong>Limits:</strong> {vacuous} of 44 instances are vacuous because the cap removes no sampled coordinate, and only {near_binding} raw argmins exceed 8 phase-0 epochs. The eight fresh cap-crossing pairs are directionally supportive ({confirmation_wins}/8 favor admissible; {confirmation_sweeps}/8 sweep all seeds; {holm_wins}/8 Holm-significant), but every excluded point is tied and every admissible point untied. They cannot separate an epoch-cap effect from the two-phase policy-class benefit.</div>
<div class="plot">{plot}</div>
<h2>Fresh paired cap-crossing evidence</h2>
<p>Each row compares a tied coordinate above 10 phase-0 StarCoder epochs with an untied coordinate at or below 10 in the same N,D,replay cell, using five fresh paired seeds. Gain is excluded-minus-admissible BPB. This is a post hoc reuse of a tied-versus-untied confirmation panel, not a cap intervention: the policy-class and cap contrasts are perfectly confounded. Holm p-values retain the original, conservative 28-block correction family.</p>
<table><thead><tr><th>Cell</th><th>Excluded coordinate (epochs)</th><th>Admissible coordinate (epochs)</th><th>Gain</th><th>Admissible seed wins</th><th>Holm p</th></tr></thead><tbody>{_format_confirmation_table(confirmations)}</tbody></table>
<h2>The one exact miss</h2>
<p>The miss is <code>{miss.cell_id}/{miss.support_id}</code>, the 1.92B-token, 4x-replay cell. Its one-seed raw argmin is <code>{miss.raw_coordinate_id}</code> at {miss.raw_phase_0_epochs:.6f} epochs and {miss.raw_bpb:.6f} BPB. The actual best cap-10 grid point is <code>{miss.replacement_coordinate_id}</code> at {miss.replacement_phase_0_epochs:.6f} epochs and {miss.replacement_bpb:.6f} BPB, giving apparent one-seed regret {miss.observed_regret_bpb:.6f} BPB. Both lie on the tied diagonal and are only {miss.replacement_l2:.4f} apart in (p0,p1) L2 distance, so the filter retains a nearby tied point even though it drops the exact draw.</p>
<p>The available fresh comparison was designed for constant-versus-two-phase confirmation, not this cap audit. It compares the excluded <code>{miss.raw_coordinate_id}</code> with a different cap-admissible coordinate, <code>{miss.confirmation_coordinate_id}</code> at {miss.confirmation_phase_0_epochs:.3f} epochs, rather than with <code>{miss.replacement_coordinate_id}</code>. Over five paired seeds the excluded mean was {miss.fresh_raw_mean_bpb:.6f} and the admissible mean was {miss.fresh_confirmation_mean_bpb:.6f}; excluded-minus-admissible was {miss.raw_minus_confirmation_bpb:+.6f} BPB, 95% CI [{miss.ci95_low:+.6f}, {miss.ci95_high:+.6f}], one-sided p={miss.one_sided_p:.3f}, Holm p={miss.holm_p:.3f}, with the admissible point winning {miss.confirmation_win_count}/{miss.pair_count} seeds.</p>
<p>The discovery draw is strongly winner's-curse sensitive: <code>{miss.raw_coordinate_id}</code> worsened by {miss.raw_discovery_to_fresh_shift_bpb:+.6f} BPB from discovery to fresh mean, versus {miss.confirmation_discovery_to_fresh_shift_bpb:+.6f} for <code>{miss.confirmation_coordinate_id}</code>; the discovery gain was inflated by {-miss.winner_curse_bpb:.6f} BPB. Four observations of the actual replacement <code>{miss.replacement_coordinate_id}</code> average {miss.replacement_all_mean_bpb:.6f} BPB ({miss.replacement_repeats_only_mean_bpb:.6f} over repeats only; SD {miss.replacement_all_sd_bpb:.6f}), against a calibration-predicted SD of {miss.replacement_predicted_sd_bpb:.6f}. No fresh paired comparison directly tests <code>{miss.raw_coordinate_id}</code> against <code>{miss.replacement_coordinate_id}</code>; the excluded discovery argmin is unsupported as the expected optimum, not directly disproved against its actual replacement.</p>
<h2>Coverage by panel</h2>
<table><thead><tr><th>Panel</th><th>Raw argmins retained</th><th>Max raw optimum epochs</th><th>Mean coordinates retained</th><th>Max observed regret</th></tr></thead><tbody>{_format_panel_table(summary)}</tbody></table>
<p class="small">The inventory contains 44 distinct measured surface instances. Four fixed-N matched-grid cells have nominally matched dense-panel <code>m100</code> counterparts, but they are not identical re-sweeps: each pair shares {nominal_pairs['count'].min():.0f}-{nominal_pairs['count'].max():.0f} coordinates and the matched-minus-dense mean offset is {nominal_pairs['mean'].min():.4f}-{nominal_pairs['mean'].max():.4f} BPB. Data-cache realizations and epoch burdens differ. Both instances are audited and are not treated as independent replications.</p>
<p class="small">Panel coverage is heterogeneous. The historical WSD-50/50 panel is a proxy-selected candidate list; only the 1B token-ladder rung is a dense grid, while the 2B/4B/8B rungs include scale-specific fibers, Bayesian-refinement, scaffold, and tied-diagonal points. The matched N,D and replay panels are the densest. Raw argmins over adaptive point sets carry extra selection bias.</p>
<p class="small">Epoch accounting is also panel-specific and recorded per row in <code>surface_audit.csv</code>. Historical panels use target-budget simulated epochs at a 0.5 phase share; the token ladder and matched N,D panels use the same construction at 0.8; dense replay uses exact realized epochs over deliberately varied support. Under simulated accounting, cap 10 is equivalent to phase-0 StarCoder weight ≤{historical_weight_cap:.3f} and ≤{wsd80_weight_cap:.3f}, respectively.</p>
<h2>Atomic-objective sensitivity</h2>
<p>The same 28 dense horizon-by-replay cells expose 23 atomic BPB objectives, yielding 644 objective surfaces with all 125 coordinates each. For the 396 finite-support aliases, the audit reuses the metric of the bit-identical full-pool source run. At 10 epochs, {atomic_exact}/644 raw atomic argmins are retained. On the three code-relevant objectives the figure is {code_atomic_exact}/{len(code_atomic)}. The misses' best-admissible raw-grid regrets range from {atomic.loc[~atomic['exact_raw_optimum_retained'], 'observed_regret_bpb'].min():.6f} to {atomic.loc[~atomic['exact_raw_optimum_retained'], 'observed_regret_bpb'].max():.6f} BPB. These objectives reuse checkpoints and are not independent experiments.</p>
<h2>Model-based caveat</h2>
<p>Three noise-weighted <code>m100</code> ridge fits place their untied optima at phase-0 StarCoder weight {weighted['weighted_surface_untied_p0'].min():.3f}-{weighted['weighted_surface_untied_p0'].max():.3f}, outside cap 10. We do not treat these extrapolated locations as expected optima: spatial-CV RMSE is {weighted['weighted_spatial_cv_rmse'].min():.3f}-{weighted['weighted_spatial_cv_rmse'].max():.3f} BPB, and each fitted optimum predicts below every observed point on its own surface. Cap-crossing fresh evidence exists for {len(weighted_confirmations)}/3 fits (<code>r2/m100</code> and <code>r3/m100</code>), where admissible points win {int(weighted_confirmations['untied_win_count'].min())}/{int(weighted_confirmations['pair_count'].max())} seeds; <code>r1/m100</code> has no cap-crossing confirmation. The disagreement is disclosed rather than counted as observed evidence.</p>
<h2>Interpretation</h2>
<p>The present two-bucket archive supports cap 10 as a high-recall candidate: it removes a meaningful fraction of risky prefixes, retains 43/44 exact raw primary argmins, and leaves a nearby tied point in the only miss. It does not certify the unobserved continuous optimum, isolate an epoch-cap effect in the fresh confirmation panel, validate the later lexicographic prefix-ranking step, or prove that a uniform 10-epoch cap transfers unchanged to 39 buckets.</p>
<p class="small">A cap of {audit['raw_phase_0_starcoder_epochs'].max():.6f} epochs, rounded to {practical_exact_cap}, is the smallest retrospective threshold containing all raw primary argmins. It is set entirely by the unconfirmed one-seed miss above and does not contain the atomic PTB argmin, so it should not replace the pre-existing candidate cap 10.</p>
</main></body></html>"""
    output_path.write_text(document)


def write_report(
    output_path: Path,
    audit: pd.DataFrame,
    sweep: pd.DataFrame,
    atomic: pd.DataFrame,
    confirmations: pd.DataFrame,
    miss: MissEvidence,
) -> None:
    summary = _panel_summary(audit)
    exact = int(audit["exact_raw_optimum_retained"].sum())
    retained = int(audit["admissible_count"].sum())
    total = int(audit["observation_count"].sum())
    minimum_exact_cap = float(audit["raw_phase_0_starcoder_epochs"].max())
    atomic_misses = atomic.loc[~atomic["exact_raw_optimum_retained"]]
    code_atomic = atomic.loc[
        atomic["metric_label"].isin(
            ["Paloma · Programming Languages", "Uncheatable · GitHub C++", "Uncheatable · GitHub Python"]
        )
    ]
    weighted = _weighted_surface_caveat()
    nominal_pairs = _nominal_pair_summary()
    weighted_confirmations = confirmations.loc[
        confirmations["support_id"].eq("m100") & confirmations["cell_id"].isin(weighted["cell_id"])
    ]
    if len(weighted_confirmations) != 2 or not weighted_confirmations["admissible_wins_all_seeds"].all():
        raise ValueError("Expected two 5/5 cap-crossing confirmations for unstable m100 fits")
    weighted_unconfirmed = sorted(set(weighted["cell_id"]) - set(weighted_confirmations["cell_id"]))
    if weighted_unconfirmed != ["r1_increase_d_h0640_s07320"]:
        raise ValueError(f"Unexpected unstable-fit confirmation gap: {weighted_unconfirmed}")
    practical_exact_cap = math.ceil(minimum_exact_cap)
    if practical_exact_cap != 12:
        raise ValueError(f"Expected retrospective integer cap 12, found {practical_exact_cap}")
    simulated_epoch_scale = SIMULATED_EPOCH_TARGET_BUDGET / STARCODER_SOURCE_TOKENS
    historical_weight_cap = PRIMARY_CAP / (0.5 * simulated_epoch_scale)
    wsd80_weight_cap = PRIMARY_CAP / (0.8 * simulated_epoch_scale)
    vacuous = int(audit["admissible_fraction"].eq(1.0).sum())
    near_binding = int(audit["raw_phase_0_starcoder_epochs"].gt(8.0).sum())
    confirmation_wins = int(confirmations["admissible_wins_mean"].sum())
    confirmation_sweeps = int(confirmations["admissible_wins_all_seeds"].sum())
    holm_wins = int((confirmations["mean_gain_bpb"].gt(0.0) & confirmations["paired_t_holm_p"].lt(0.05)).sum())
    panel_lines = [
        "| Panel | Raw argmins retained | Max raw-optimum phase-0 epochs | Mean coordinates retained | Max observed regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in summary.itertuples(index=False):
        panel_lines.append(
            f"| {PANEL_LABELS[row.panel]} | {row.exact_raw_optima_retained}/{row.surface_instances} | "
            f"{row.maximum_raw_phase_0_epochs:.6f} | {row.mean_coordinate_fraction_retained:.1%} | "
            f"{row.maximum_observed_regret_bpb:.6f} |"
        )
    atomic_lines = [
        "| Metric | Cell | Support | Raw phase-0 epochs | Best-admissible observed regret |",
        "|---|---|---|---:|---:|",
    ]
    for row in atomic_misses.sort_values("observed_regret_bpb", ascending=False).itertuples(index=False):
        atomic_lines.append(
            f"| {row.metric_label} | `{row.cell_id}` | `{row.support_id}` | "
            f"{row.raw_phase_0_starcoder_epochs:.6f} | {row.observed_regret_bpb:.6f} |"
        )
    confirmation_lines = [
        "| Cell | Excluded point (epochs) | Admissible point (epochs) | Excluded-minus-admissible BPB | Admissible seed wins | Holm p |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in confirmations.itertuples(index=False):
        confirmation_lines.append(
            f"| `{row.cell_id}/{row.support_id}` | `{row.tied_coordinate_id}` ({row.tied_phase_0_epochs:.3f}) | "
            f"`{row.untied_coordinate_id}` ({row.untied_phase_0_epochs:.3f}) | {row.mean_gain_bpb:+.6f} | "
            f"{row.untied_win_count}/{row.pair_count} | {row.paired_t_holm_p:.4f} |"
        )
    report = "\n".join(
        [
            "# StarCoder prefix-admissibility audit",
            "",
            "## Verdict",
            "",
            f"The pre-existing candidate cap of 10 phase-0 StarCoder epochs retains the exact one-seed raw Programming Languages BPB argmin in **{exact}/44 measured surface instances** and **{retained:,}/{total:,} coordinates ({retained / total:.1%})**.",
            "",
            "The result supports cap 10 as a high-recall candidate filter, but does not certify basin preservation in every setting. It retains a nearby tied point in the sole miss, and that excluded discovery draw is unstable under fresh seeds.",
            "",
            f"This is still an easy test on part of the inventory: in **{vacuous}/44** instances the cap removes no sampled coordinate, and only **{near_binding}/44** raw argmins exceed 8 epochs. Eight fresh paired comparisons cross the cap and directionally favor admissible policies ({confirmation_wins}/8 in mean, {confirmation_sweeps}/8 across all seeds, {holm_wins}/8 Holm-significant), but every excluded point is tied and every admissible point untied. Those repeats cannot separate an epoch-cap effect from the two-phase policy-class benefit. The audit does not prove transfer to 39 buckets or test the later lexicographic prefix-selection step.",
            "",
            f"A cap of **{minimum_exact_cap:.6f} epochs** is the literal minimum containing every raw primary argmin; **{practical_exact_cap} epochs** is the smallest practical integer. It is set entirely by the unconfirmed one-seed miss and does not contain the atomic PTB argmin. It is retrospective and should not replace 10.",
            "",
            "## Scope",
            "",
            "- 44 measured surface instances: two historical schedule panels, four fixed-model token horizons, ten matched-N,D cells, and 28 dense horizon-by-replay cells.",
            f"- Four fixed-N matched-grid cells have nominally matched dense-panel `m100` counterparts. They are not identical re-sweeps: each pair shares {nominal_pairs['count'].min():.0f}-{nominal_pairs['count'].max():.0f} coordinates and the matched-minus-dense mean offset is {nominal_pairs['mean'].min():.4f}-{nominal_pairs['mean'].max():.4f} BPB. Data-cache realizations and epoch burdens differ. Both are audited and are not independent replications.",
            "- Panel composition is heterogeneous. The WSD-50/50 panel is a proxy-selected candidate list; only the 1B token-ladder rung is a dense grid, while the 2B/4B/8B rungs mix scale-specific fibers, Bayesian-refinement, scaffold, and tied-diagonal points. Raw argmins over adaptive point sets carry additional selection bias.",
            f"- Epoch accounting is panel-specific and recorded in `surface_audit.csv`. Historical panels use target-budget simulated epochs at a 0.5 phase share; the token ladder and matched N,D panels use that construction at 0.8; dense replay uses exact realized epochs over varied support. Under simulated accounting, cap 10 is equivalent to phase-0 StarCoder weight <= {historical_weight_cap:.3f} and <= {wsd80_weight_cap:.3f}, respectively.",
            "- Primary objective: Llama-3-tokenized Paloma Programming Languages BPB.",
            "- Excluded: standalone one-dimensional-only experiments, sparse fixed-policy interventions, and surrogate-rendered surfaces. Fiber points embedded inside broader token-ladder panels remain included.",
            "- The audit asks only whether the cap preserves the observed optimum. It does not optimize or rank the surviving prefixes.",
            "",
            "## Primary results",
            "",
            *panel_lines,
            "",
            "## Fresh paired cap-crossing evidence",
            "",
            "Each row compares a tied coordinate above 10 phase-0 StarCoder epochs with an untied coordinate at or below 10 in the same N,D,replay cell, using five fresh paired seeds. Positive gain means the admissible coordinate is better. This is a post hoc reuse of a tied-versus-untied confirmation panel, not a cap intervention: policy class and cap status are perfectly confounded. Holm p-values retain the original, conservative 28-block correction family.",
            "",
            *confirmation_lines,
            "",
            "## The one exact miss",
            "",
            f"`{miss.cell_id}/{miss.support_id}` has raw one-seed argmin `{miss.raw_coordinate_id}` at {miss.raw_phase_0_epochs:.6f} phase-0 StarCoder epochs and {miss.raw_bpb:.6f} BPB. The actual cap-10 grid replacement is `{miss.replacement_coordinate_id}` at {miss.replacement_phase_0_epochs:.6f} epochs and {miss.replacement_bpb:.6f} BPB, for apparent one-seed regret {miss.observed_regret_bpb:.6f} BPB. Both are tied points and only {miss.replacement_l2:.4f} apart in `(p0,p1)` L2 distance, so the cap retains a nearby tied point even though it drops the exact draw.",
            "",
            f"The available fresh comparison was designed for constant-versus-two-phase confirmation, not this cap audit. It compares `{miss.raw_coordinate_id}` against another admissible coordinate, `{miss.confirmation_coordinate_id}` at {miss.confirmation_phase_0_epochs:.3f} epochs, not against `{miss.replacement_coordinate_id}`. Over five paired seeds, excluded-minus-admissible is {miss.raw_minus_confirmation_bpb:+.6f} BPB, 95% CI [{miss.ci95_low:+.6f}, {miss.ci95_high:+.6f}], one-sided p={miss.one_sided_p:.3f}, Holm p={miss.holm_p:.3f}; the admissible point wins {miss.confirmation_win_count}/{miss.pair_count} seeds.",
            "",
            f"The excluded discovery argmin worsens by {miss.raw_discovery_to_fresh_shift_bpb:+.6f} BPB from discovery to fresh mean, versus {miss.confirmation_discovery_to_fresh_shift_bpb:+.6f} for the confirmation alternative; the discovery gain was inflated by {-miss.winner_curse_bpb:.6f} BPB. Four observations of the actual replacement average {miss.replacement_all_mean_bpb:.6f} BPB ({miss.replacement_repeats_only_mean_bpb:.6f} over repeats only; SD {miss.replacement_all_sd_bpb:.6f}), against calibration-predicted SD {miss.replacement_predicted_sd_bpb:.6f}. There is no fresh paired comparison against the actual replacement. The excluded discovery argmin is unsupported as the expected optimum, not directly refuted against `{miss.replacement_coordinate_id}`.",
            "",
            "## Atomic-objective sensitivity",
            "",
            f"Across 23 atomic BPB metrics on the 28 dense horizon-by-replay cells, the cap retains **{int(atomic['exact_raw_optimum_retained'].sum())}/644 raw argmins**. Every cell uses all 125 coordinates; metrics for 396 finite-support aliases are inherited from their bit-identical full-pool source runs. On Programming Languages, GitHub C++, and GitHub Python, retention is **{int(code_atomic['exact_raw_optimum_retained'].sum())}/{len(code_atomic)}**. These reuse checkpoints and are not independent experiments.",
            "",
            *atomic_lines,
            "",
            "## Model-based caveat",
            "",
            f"Three noise-weighted `m100` ridge fits place their untied optima at phase-0 StarCoder weight {weighted['weighted_surface_untied_p0'].min():.3f}-{weighted['weighted_surface_untied_p0'].max():.3f}, outside cap 10. We do not treat those extrapolations as expected optima: spatial-CV RMSE is {weighted['weighted_spatial_cv_rmse'].min():.3f}-{weighted['weighted_spatial_cv_rmse'].max():.3f} BPB and each fitted optimum predicts below every observed point on its own surface. Cap-crossing fresh evidence exists for {len(weighted_confirmations)}/3 fits (`r2/m100` and `r3/m100`), where admissible points win {int(weighted_confirmations['untied_win_count'].min())}/{int(weighted_confirmations['pair_count'].max())} seeds; `r1/m100` has no cap-crossing confirmation.",
            "",
            "## Cap sweep",
            "",
            sweep.to_markdown(index=False, floatfmt=".6f"),
            "",
            "## Outputs",
            "",
            "- `surface_audit.csv`: one row per primary surface instance at cap 10.",
            "- `cap_sweep.csv`: exact raw-optimum recall and coordinate retention across caps.",
            "- `atomic_metric_sensitivity.csv`: 644 atomic-objective checks at cap 10.",
            "- `fresh_confirmation_cap_contrasts.csv`: all eight fresh paired comparisons that cross cap 10.",
            "- `prefix_admissibility.html`: self-contained interactive report.",
            "",
        ]
    )
    output_path.write_text(report)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    observations = load_primary_observations()
    audit = audit_surfaces(observations, PRIMARY_CAP)
    sweep = cap_sweep(observations)
    atomic = atomic_metric_sensitivity(PRIMARY_CAP)
    confirmations = confirmation_cap_contrasts(PRIMARY_CAP)
    miss = miss_evidence(audit, confirmations)

    audit.to_csv(args.output_dir / "surface_audit.csv", index=False)
    sweep.to_csv(args.output_dir / "cap_sweep.csv", index=False)
    atomic.to_csv(args.output_dir / "atomic_metric_sensitivity.csv", index=False)
    confirmations.to_csv(args.output_dir / "fresh_confirmation_cap_contrasts.csv", index=False)
    write_report(args.output_dir / "report.md", audit, sweep, atomic, confirmations, miss)
    write_html(args.output_dir / "prefix_admissibility.html", audit, sweep, atomic, confirmations, miss)

    summary = {
        "output_dir": str(args.output_dir),
        "primary_cap_phase_0_epochs": PRIMARY_CAP,
        "surface_instances": len(audit),
        "exact_raw_optima_retained": int(audit["exact_raw_optimum_retained"].sum()),
        "coordinates": int(audit["observation_count"].sum()),
        "coordinates_retained": int(audit["admissible_count"].sum()),
        "fresh_cap_crossing_comparisons": len(confirmations),
        "fresh_comparisons_favoring_admissible": int(confirmations["admissible_wins_mean"].sum()),
        "minimum_cap_for_all_primary_raw_optima": float(audit["raw_phase_0_starcoder_epochs"].max()),
        "atomic_objective_surfaces": len(atomic),
        "atomic_raw_optima_retained": int(atomic["exact_raw_optimum_retained"].sum()),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
