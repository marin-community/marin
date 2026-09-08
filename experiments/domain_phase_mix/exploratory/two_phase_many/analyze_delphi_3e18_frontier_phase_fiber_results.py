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
#   "wandb",
# ]
# ///
"""Collect and analyze the completed Delphi 3e18 frontier phase-fiber DOE."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_PANEL_DIR = REFERENCE_OUTPUTS / "delphi_3e18_frontier_phase_fiber_20260719"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_3e18_frontier_phase_fiber_results_20260719"

TRAIN_PROJECT = "marin-community/marin"
EVAL_PROJECT = "marin-community/marin-eval"
TRAIN_TAG = "delphi-3e18-frontier-phase-fiber"
EVAL_GROUP = "olmo_base_eval_table9_delphi_3e18_frontier_phase_fiber_20260719"
UNCHEATABLE_METRIC = "eval/uncheatable_eval/bpb"
TABLE9_METRIC = "olmo_base_easy/table9_51_component_macro_bpb"
EXPECTED_ROWS = 200
EXPECTED_DIRECTION_PAIRS = 96
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}
TARGETS = {
    "uncheatable": ("uncheatable_bpb", "Uncheatable BPB"),
    "table9": ("table9_macro_bpb", "Table-9 macro BPB"),
}
ANCHOR_PRIMARY_TARGET = {
    "uncheatable_frontier": "uncheatable",
    "table9_frontier": "table9",
}
ANCHOR_LABELS = {
    "uncheatable_frontier": "Uncheatable frontier anchor",
    "table9_frontier": "Table-9 frontier anchor",
}
COLORS = {
    "plus": "#D73027",
    "minus": "#4575B4",
    "domain_vs_rest": "#E76F51",
    "high_mass_pair": "#2A9D8F",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    parser.add_argument("--refresh-wandb", action="store_true")
    return parser.parse_args()


def finite_summary(run: Any, key: str) -> float:
    value = run.summary.get(key)
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"Run {run.name!r} has no finite {key!r}: {value!r}")
    return float(value)


def finished_run(runs: list[Any], *, expected_name: str, allow_hash_suffix: bool) -> Any:
    if allow_hash_suffix:
        matches = [run for run in runs if run.name.startswith(f"{expected_name}-")]
    else:
        matches = [run for run in runs if run.name == expected_name]
    finished = [run for run in matches if run.state == "finished"]
    if len(finished) != 1:
        states = [(run.name, run.id, run.state) for run in matches]
        raise ValueError(f"Expected one finished run for {expected_name!r}, found {states}")
    return finished[0]


def collect_results(manifest: pd.DataFrame, *, timeout: int) -> pd.DataFrame:
    api = wandb.Api(timeout=timeout)
    training_runs = list(api.runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=EXPECTED_ROWS + 20))
    eval_runs = list(api.runs(EVAL_PROJECT, filters={"group": EVAL_GROUP}, per_page=EXPECTED_ROWS + 20))

    rows: list[dict[str, object]] = []
    for record in manifest.to_dict(orient="records"):
        base_name = f"fiber_{int(record['run_order']):03d}_{record['candidate_id']}"
        training_run = finished_run(training_runs, expected_name=base_name, allow_hash_suffix=True)
        eval_run = finished_run(eval_runs, expected_name=f"t9_{base_name}", allow_hash_suffix=False)
        rows.append(
            {
                **record,
                "training_wandb_name": training_run.name,
                "training_wandb_url": training_run.url,
                "eval_wandb_name": eval_run.name,
                "eval_wandb_url": eval_run.url,
                "uncheatable_bpb": finite_summary(training_run, UNCHEATABLE_METRIC),
                "table9_macro_bpb": finite_summary(eval_run, TABLE9_METRIC),
            }
        )
    results = pd.DataFrame(rows).sort_values("run_order").reset_index(drop=True)
    if len(results) != EXPECTED_ROWS or results["candidate_id"].duplicated().any():
        raise ValueError(f"Expected {EXPECTED_ROWS} unique completed rows, found {len(results)}")
    return results


def build_pair_effects(results: pd.DataFrame) -> pd.DataFrame:
    centers = results[results["contrast_family"].eq("center_control")].set_index(["anchor_id", "seed_block"])
    rows: list[dict[str, object]] = []
    contrasts = results[~results["contrast_family"].eq("center_control")]
    for (anchor_id, direction_id), group in contrasts.groupby(["anchor_id", "direction_id"], sort=True):
        if len(group) != 2 or set(group["sign"]) != {"plus", "minus"}:
            raise ValueError(f"Incomplete direction pair {anchor_id}/{direction_id}")
        plus = group[group["sign"].eq("plus")].iloc[0]
        minus = group[group["sign"].eq("minus")].iloc[0]
        if int(plus["seed_block"]) != int(minus["seed_block"]):
            raise ValueError(f"Seed mismatch for {anchor_id}/{direction_id}")
        center = centers.loc[(anchor_id, int(plus["seed_block"]))]
        base = {
            "anchor_id": anchor_id,
            "contrast_family": plus["contrast_family"],
            "direction_id": direction_id,
            "plus_label": plus["direction_label"],
            "minus_label": minus["direction_label"],
            "seed_block": int(plus["seed_block"]),
            "data_seed": int(plus["data_seed"]),
            "phase_tv": float(plus["phase_tv"]),
            "phase_information_kl": float(plus["phase_information_kl"]),
            "plus_candidate_id": plus["candidate_id"],
            "minus_candidate_id": minus["candidate_id"],
        }
        for target, (column, _) in TARGETS.items():
            plus_value = float(plus[column])
            minus_value = float(minus[column])
            center_value = float(center[column])
            rows.append(
                {
                    **base,
                    "target": target,
                    "plus_bpb": plus_value,
                    "minus_bpb": minus_value,
                    "center_bpb": center_value,
                    "odd_effect_plus_minus_over_2": (plus_value - minus_value) / 2.0,
                    "second_difference": plus_value + minus_value - 2.0 * center_value,
                    "mean_contrast_minus_center": (plus_value + minus_value) / 2.0 - center_value,
                    "best_sign_gain_vs_center": center_value - min(plus_value, minus_value),
                    "plus_delta_vs_center": plus_value - center_value,
                    "minus_delta_vs_center": minus_value - center_value,
                }
            )
    frame = pd.DataFrame(rows).sort_values(["anchor_id", "target", "contrast_family", "direction_id"])
    if len(frame) != EXPECTED_DIRECTION_PAIRS * len(TARGETS):
        raise ValueError(f"Expected {EXPECTED_DIRECTION_PAIRS * len(TARGETS)} pair-target rows, found {len(frame)}")
    return frame.reset_index(drop=True)


def attach_seed_center_deltas(results: pd.DataFrame) -> pd.DataFrame:
    frame = results.copy()
    centers = frame[frame["contrast_family"].eq("center_control")].set_index(["anchor_id", "seed_block"])
    for target, (column, _) in TARGETS.items():
        center_values = [float(centers.loc[(row.anchor_id, int(row.seed_block)), column]) for row in frame.itertuples()]
        frame[f"{target}_same_seed_center_bpb"] = center_values
        frame[f"{target}_delta_vs_same_seed_center"] = frame[column].astype(float) - np.asarray(center_values)
    return frame


def center_summary(results: pd.DataFrame, anchor_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    centers = results[results["contrast_family"].eq("center_control")]
    audit = anchor_audit.set_index("anchor_id")
    for anchor_id, group in centers.groupby("anchor_id"):
        for target, (column, _) in TARGETS.items():
            values = group[column].to_numpy(float)
            source_value = float(audit.loc[anchor_id, f"{target}_3e18"])
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "target": target,
                    "source_selected_bpb": source_value,
                    "fresh_center_mean_bpb": float(np.mean(values)),
                    "fresh_center_sd_bpb": float(np.std(values, ddof=1)),
                    "fresh_center_sem_bpb": float(stats.sem(values)),
                    "fresh_minus_source_bpb": float(np.mean(values) - source_value),
                    "fresh_center_min_bpb": float(np.min(values)),
                    "fresh_center_max_bpb": float(np.max(values)),
                    "n_fresh_centers": len(values),
                }
            )
    return pd.DataFrame(rows).sort_values(["anchor_id", "target"]).reset_index(drop=True)


def anchor_metric_summary(results: pd.DataFrame, centers: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    center_lookup = centers.set_index(["anchor_id", "target"])
    contrasts = results[~results["contrast_family"].eq("center_control")]
    for anchor_id, group in contrasts.groupby("anchor_id"):
        for target, (column, _) in TARGETS.items():
            delta_column = f"{target}_delta_vs_same_seed_center"
            deltas = group[delta_column].to_numpy(float)
            best_absolute = group.loc[group[column].astype(float).idxmin()]
            strongest_same_seed = group.loc[group[delta_column].astype(float).idxmin()]
            center = center_lookup.loc[(anchor_id, target)]
            center_sd = float(center["fresh_center_sd_bpb"])
            source_bpb = float(center["source_selected_bpb"])
            fresh_center_mean = float(center["fresh_center_mean_bpb"])
            rows.append(
                {
                    "anchor_id": anchor_id,
                    "target": target,
                    "is_anchor_selection_target": target == ANCHOR_PRIMARY_TARGET[anchor_id],
                    "contrast_rows": len(group),
                    "fraction_better_than_same_seed_center": float(np.mean(deltas < 0.0)),
                    "mean_delta_vs_same_seed_center": float(np.mean(deltas)),
                    "median_delta_vs_same_seed_center": float(np.median(deltas)),
                    "delta_q10": float(np.quantile(deltas, 0.10)),
                    "delta_q90": float(np.quantile(deltas, 0.90)),
                    "best_absolute_candidate_id": best_absolute["candidate_id"],
                    "best_absolute_candidate_bpb": float(best_absolute[column]),
                    "best_absolute_minus_source_bpb": float(best_absolute[column]) - source_bpb,
                    "best_absolute_minus_fresh_center_mean_bpb": float(best_absolute[column]) - fresh_center_mean,
                    "best_absolute_delta_vs_same_seed_center": float(best_absolute[delta_column]),
                    "best_absolute_delta_in_fresh_center_sd": (
                        float(best_absolute[delta_column]) / center_sd if center_sd > 0.0 else np.nan
                    ),
                    "strongest_same_seed_candidate_id": strongest_same_seed["candidate_id"],
                    "strongest_same_seed_candidate_bpb": float(strongest_same_seed[column]),
                    "strongest_same_seed_delta": float(strongest_same_seed[delta_column]),
                    "strongest_same_seed_delta_in_fresh_center_sd": (
                        float(strongest_same_seed[delta_column]) / center_sd if center_sd > 0.0 else np.nan
                    ),
                    "both_targets_improve_count": int(
                        (
                            group["uncheatable_delta_vs_same_seed_center"].lt(0.0)
                            & group["table9_delta_vs_same_seed_center"].lt(0.0)
                        ).sum()
                    ),
                }
            )
    return pd.DataFrame(rows).sort_values(["anchor_id", "target"]).reset_index(drop=True)


def phase_direction_vectors(panel_dir: Path) -> dict[tuple[str, str], np.ndarray]:
    weights = pd.read_csv(panel_dir / "phase_weights.csv")
    domains = list(dict.fromkeys(weights["domain"].astype(str)))
    vectors: dict[tuple[str, str], np.ndarray] = {}
    plus = weights[weights["sign"].eq("plus")]
    for (anchor_id, direction_id), group in plus.groupby(["anchor_id", "direction_id"], sort=True):
        pivot = group.pivot(index="phase", columns="domain", values="weight").reindex(columns=domains)
        if list(pivot.index) != [0, 1]:
            raise ValueError(f"Missing phase weights for {anchor_id}/{direction_id}")
        vector = pivot.loc[1].to_numpy(float) - pivot.loc[0].to_numpy(float)
        if abs(float(vector.sum())) > 1e-10:
            raise ValueError(f"Direction {anchor_id}/{direction_id} is not simplex-tangent")
        vectors[(anchor_id, direction_id)] = vector
    return vectors


def safe_correlation(x: np.ndarray, y: np.ndarray, method: str) -> float:
    if np.std(x) == 0.0 or np.std(y) == 0.0:
        return np.nan
    result = stats.pearsonr(x, y) if method == "pearson" else stats.spearmanr(x, y)
    return float(result.statistic)


def fit_phase_gradients(
    pairs: pd.DataFrame,
    vectors: dict[tuple[str, str], np.ndarray],
    domains: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_rows: list[dict[str, object]] = []
    prediction_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    for (anchor_id, target), group in pairs.groupby(["anchor_id", "target"], sort=True):
        train = group[group["contrast_family"].eq("domain_vs_rest")]
        test = group[group["contrast_family"].eq("high_mass_pair")]
        x_train = np.stack([vectors[(anchor_id, row.direction_id)] for row in train.itertuples()])
        y_train = train["odd_effect_plus_minus_over_2"].to_numpy(float)
        gradient = np.linalg.lstsq(x_train, y_train, rcond=None)[0]
        gradient -= float(np.mean(gradient))
        train_pred = x_train @ gradient
        x_test = np.stack([vectors[(anchor_id, row.direction_id)] for row in test.itertuples()])
        y_test = test["odd_effect_plus_minus_over_2"].to_numpy(float)
        test_pred = x_test @ gradient
        rmse = float(np.sqrt(np.mean(np.square(test_pred - y_test))))
        zero_rmse = float(np.sqrt(np.mean(np.square(y_test))))
        metric_rows.append(
            {
                "anchor_id": anchor_id,
                "target": target,
                "train_direction_count": len(train),
                "train_rank": int(np.linalg.matrix_rank(x_train)),
                "train_rmse": float(np.sqrt(np.mean(np.square(train_pred - y_train)))),
                "heldout_pair_count": len(test),
                "heldout_rmse": rmse,
                "heldout_zero_baseline_rmse": zero_rmse,
                "heldout_rmse_ratio_vs_zero": rmse / zero_rmse if zero_rmse > 0.0 else np.nan,
                "heldout_pearson": safe_correlation(test_pred, y_test, "pearson"),
                "heldout_spearman": safe_correlation(test_pred, y_test, "spearman"),
                "heldout_r2": (
                    1.0
                    - float(np.sum(np.square(test_pred - y_test))) / float(np.sum(np.square(y_test - np.mean(y_test))))
                ),
            }
        )
        for index, row in enumerate(test.itertuples()):
            prediction_rows.append(
                {
                    "anchor_id": anchor_id,
                    "target": target,
                    "direction_id": row.direction_id,
                    "direction_label": row.plus_label,
                    "observed_odd_effect": y_test[index],
                    "predicted_odd_effect": test_pred[index],
                    "residual": test_pred[index] - y_test[index],
                }
            )
        for domain, value in zip(domains, gradient, strict=True):
            gradient_rows.append(
                {
                    "anchor_id": anchor_id,
                    "target": target,
                    "domain": domain,
                    "loss_gradient_per_unit_phase_contrast": float(value),
                }
            )
    return (
        pd.DataFrame(metric_rows).sort_values(["anchor_id", "target"]).reset_index(drop=True),
        pd.DataFrame(prediction_rows).sort_values(["anchor_id", "target", "direction_id"]).reset_index(drop=True),
        pd.DataFrame(gradient_rows).sort_values(["anchor_id", "target", "domain"]).reset_index(drop=True),
    )


def gradient_transfer(gradients: pd.DataFrame) -> pd.DataFrame:
    columns: dict[tuple[str, str], np.ndarray] = {}
    for key, group in gradients.groupby(["anchor_id", "target"], sort=True):
        columns[key] = group.sort_values("domain")["loss_gradient_per_unit_phase_contrast"].to_numpy(float)
    rows: list[dict[str, object]] = []
    keys = sorted(columns)
    for left_index, left in enumerate(keys):
        for right in keys[left_index + 1 :]:
            x = columns[left]
            y = columns[right]
            rows.append(
                {
                    "left_anchor": left[0],
                    "left_target": left[1],
                    "right_anchor": right[0],
                    "right_target": right[1],
                    "same_anchor": left[0] == right[0],
                    "same_target": left[1] == right[1],
                    "cosine_similarity": float(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))),
                    "pearson": safe_correlation(x, y, "pearson"),
                    "spearman": safe_correlation(x, y, "spearman"),
                }
            )
    return pd.DataFrame(rows)


def pair_summary(pairs: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in pairs.groupby(["anchor_id", "target", "contrast_family"], sort=True):
        odd = group["odd_effect_plus_minus_over_2"].to_numpy(float)
        curvature = group["mean_contrast_minus_center"].to_numpy(float)
        gain = group["best_sign_gain_vs_center"].to_numpy(float)
        rows.append(
            {
                "anchor_id": keys[0],
                "target": keys[1],
                "contrast_family": keys[2],
                "direction_count": len(group),
                "odd_effect_rms": float(np.sqrt(np.mean(np.square(odd)))),
                "odd_effect_median_abs": float(np.median(np.abs(odd))),
                "mean_phase_curvature": float(np.mean(curvature)),
                "median_phase_curvature": float(np.median(curvature)),
                "fraction_mean_contrast_better_than_tied": float(np.mean(curvature < 0.0)),
                "median_best_sign_gain_vs_center": float(np.median(gain)),
                "maximum_best_sign_gain_vs_center": float(np.max(gain)),
            }
        )
    return pd.DataFrame(rows).sort_values(["anchor_id", "target", "contrast_family"]).reset_index(drop=True)


def write_delta_scatter(results: pd.DataFrame, output_path: Path) -> None:
    contrasts = results[~results["contrast_family"].eq("center_control")]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[ANCHOR_LABELS[anchor] for anchor in ANCHOR_LABELS],
        horizontal_spacing=0.10,
    )
    for column, anchor_id in enumerate(ANCHOR_LABELS, start=1):
        group = contrasts[contrasts["anchor_id"].eq(anchor_id)]
        for family in ("domain_vs_rest", "high_mass_pair"):
            rows = group[group["contrast_family"].eq(family)]
            figure.add_trace(
                go.Scatter(
                    x=rows["uncheatable_delta_vs_same_seed_center"],
                    y=rows["table9_delta_vs_same_seed_center"],
                    mode="markers",
                    marker={"size": 9, "color": COLORS[family], "opacity": 0.78},
                    name=family.replace("_", " "),
                    legendgroup=family,
                    showlegend=column == 1,
                    customdata=np.stack(
                        [rows["candidate_id"], rows["direction_label"], rows["sign"], rows["phase_tv"]], axis=1
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>%{customdata[1]}<br>sign=%{customdata[2]}"
                        "<br>phase TV=%{customdata[3]:.4f}<br>Uncheatable delta=%{x:.5f}"
                        "<br>Table-9 delta=%{y:.5f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.add_hline(y=0.0, line={"color": "#6C757D", "dash": "dash"}, row=1, col=column)
        figure.add_vline(x=0.0, line={"color": "#6C757D", "dash": "dash"}, row=1, col=column)
    figure.update_xaxes(title_text="Uncheatable delta vs same-seed tied center", zeroline=False)
    figure.update_yaxes(title_text="Table-9 delta vs same-seed tied center", zeroline=False)
    figure.update_layout(
        title="Frontier phase-fiber outcomes at fixed aggregate mixture<br><sup>Lower-left improves both targets</sup>",
        template="plotly_white",
        width=1450,
        height=650,
        legend={"orientation": "h", "y": 1.11, "x": 0.5, "xanchor": "center"},
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def write_tv_paths(results: pd.DataFrame, output_path: Path) -> None:
    contrasts = results[~results["contrast_family"].eq("center_control")]
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{ANCHOR_LABELS[anchor]} / {TARGETS[target][1]}" for anchor in ANCHOR_LABELS for target in TARGETS
        ],
        horizontal_spacing=0.09,
        vertical_spacing=0.14,
    )
    for anchor_index, anchor_id in enumerate(ANCHOR_LABELS):
        for target_index, (target, (_, label)) in enumerate(TARGETS.items()):
            row = anchor_index + 1
            column = target_index + 1
            group = contrasts[contrasts["anchor_id"].eq(anchor_id)]
            for sign in ("plus", "minus"):
                rows = group[group["sign"].eq(sign)]
                figure.add_trace(
                    go.Scatter(
                        x=rows["phase_tv"],
                        y=rows[f"{target}_delta_vs_same_seed_center"],
                        mode="markers",
                        marker={"size": 8, "color": COLORS[sign], "opacity": 0.75},
                        name=sign,
                        legendgroup=sign,
                        showlegend=row == 1 and column == 1,
                        customdata=np.stack([rows["candidate_id"], rows["direction_label"]], axis=1),
                        hovertemplate=(
                            "%{customdata[0]}<br>%{customdata[1]}<br>phase TV=%{x:.4f}<br>delta=%{y:.5f}<extra></extra>"
                        ),
                    ),
                    row=row,
                    col=column,
                )
            figure.add_hline(y=0.0, line={"color": "#6C757D", "dash": "dash"}, row=row, col=column)
            figure.update_xaxes(title_text="Phase TV", row=row, col=column)
            figure.update_yaxes(title_text=f"Delta {label}", row=row, col=column)
    figure.update_layout(
        title="Phase contrast magnitude versus same-seed tied-center performance",
        template="plotly_white",
        width=1500,
        height=1050,
        legend={"orientation": "h", "y": 1.08, "x": 0.5, "xanchor": "center"},
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def write_domain_heatmap(pairs: pd.DataFrame, output_path: Path) -> None:
    domains = pairs[pairs["contrast_family"].eq("domain_vs_rest")]["plus_label"].str.split(" later than").str[0]
    domain_order = list(dict.fromkeys(domains))
    columns = [(anchor, target) for anchor in ANCHOR_LABELS for target in TARGETS]
    z = []
    hover = []
    for domain in domain_order:
        values = []
        labels = []
        for anchor, target in columns:
            row = pairs[
                pairs["anchor_id"].eq(anchor)
                & pairs["target"].eq(target)
                & pairs["contrast_family"].eq("domain_vs_rest")
                & pairs["plus_label"].str.startswith(f"{domain} later than")
            ]
            if len(row) != 1:
                raise ValueError(f"Missing domain contrast for {domain}/{anchor}/{target}")
            record = row.iloc[0]
            value = float(record["odd_effect_plus_minus_over_2"])
            values.append(value)
            labels.append(
                f"{domain}<br>{ANCHOR_LABELS[anchor]} / {TARGETS[target][1]}"
                f"<br>(plus-minus)/2={value:.5f}<br>negative: named domain later is better"
            )
        z.append(values)
        hover.append(labels)
    figure = go.Figure(
        go.Heatmap(
            z=np.asarray(z),
            x=[f"{anchor.replace('_frontier', '')}<br>{target}" for anchor, target in columns],
            y=domain_order,
            colorscale="RdYlGn_r",
            zmid=0.0,
            colorbar={"title": "(plus-minus)/2 BPB"},
            text=np.asarray(hover),
            hovertemplate="%{text}<extra></extra>",
        )
    )
    figure.update_layout(
        title="One-vs-rest odd phase-order effects<br><sup>Negative means the named domain performs better later</sup>",
        template="plotly_white",
        width=1150,
        height=1250,
        margin={"l": 300, "r": 120, "t": 110, "b": 100},
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def write_gradient_holdout(predictions: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=[
            f"{ANCHOR_LABELS[anchor]} / {TARGETS[target][1]}" for anchor in ANCHOR_LABELS for target in TARGETS
        ],
        horizontal_spacing=0.09,
        vertical_spacing=0.15,
    )
    for anchor_index, anchor in enumerate(ANCHOR_LABELS):
        for target_index, target in enumerate(TARGETS):
            row_index = anchor_index + 1
            column_index = target_index + 1
            rows = predictions[predictions["anchor_id"].eq(anchor) & predictions["target"].eq(target)]
            lo = float(min(rows["observed_odd_effect"].min(), rows["predicted_odd_effect"].min()))
            hi = float(max(rows["observed_odd_effect"].max(), rows["predicted_odd_effect"].max()))
            pad = max((hi - lo) * 0.1, 1e-4)
            figure.add_trace(
                go.Scatter(
                    x=rows["observed_odd_effect"],
                    y=rows["predicted_odd_effect"],
                    mode="markers+text",
                    text=rows["direction_id"],
                    textposition="top center",
                    marker={"size": 10, "color": "#264653"},
                    showlegend=False,
                    customdata=rows[["direction_label"]].to_numpy(),
                    hovertemplate="%{customdata[0]}<br>observed=%{x:.5f}<br>predicted=%{y:.5f}<extra></extra>",
                ),
                row=row_index,
                col=column_index,
            )
            figure.add_trace(
                go.Scatter(
                    x=[lo - pad, hi + pad],
                    y=[lo - pad, hi + pad],
                    mode="lines",
                    line={"dash": "dash", "color": "#6C757D"},
                    showlegend=False,
                ),
                row=row_index,
                col=column_index,
            )
            figure.update_xaxes(title_text="Observed odd effect", row=row_index, col=column_index)
            figure.update_yaxes(title_text="Predicted odd effect", row=row_index, col=column_index)
    figure.update_layout(
        title=(
            "Additive phase-gradient test<br>"
            "<sup>Fit 39 one-vs-rest directions; evaluate 9 held-out high-mass pairs</sup>"
        ),
        template="plotly_white",
        width=1450,
        height=1050,
    )
    figure.write_html(output_path, include_plotlyjs=True, config=PLOT_CONFIG)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    return frame[columns].to_markdown(index=False, floatfmt=".6f")


def write_report(
    *,
    output_dir: Path,
    centers: pd.DataFrame,
    anchor_summary: pd.DataFrame,
    pair_metrics: pd.DataFrame,
    gradient_metrics: pd.DataFrame,
    transfer: pd.DataFrame,
) -> None:
    primary = anchor_summary[anchor_summary["is_anchor_selection_target"]].copy()
    uncheatable_primary = primary[
        primary["anchor_id"].eq("uncheatable_frontier") & primary["target"].eq("uncheatable")
    ].iloc[0]
    table9_primary = primary[primary["anchor_id"].eq("table9_frontier") & primary["target"].eq("table9")].iloc[0]
    uncheatable_gradient = gradient_metrics[
        gradient_metrics["anchor_id"].eq("uncheatable_frontier") & gradient_metrics["target"].eq("uncheatable")
    ].iloc[0]
    table9_gradient = gradient_metrics[
        gradient_metrics["anchor_id"].eq("table9_frontier") & gradient_metrics["target"].eq("table9")
    ].iloc[0]
    report = [
        "# Delphi 3e18 frontier phase-fiber DOE: observed results",
        "",
        "## Coverage and interpretation",
        "",
        (
            "All 200 training runs and all 200 native Table-9 evaluations completed. Three crashed Table-9 "
            "attempts were superseded by one finished attempt for the same expected run name and are excluded. "
            "The panel is exploratory development evidence: anchors were selected from prior one-phase outcomes, "
            "and each non-center direction has one data seed."
        ),
        "",
        (
            "Every contrast is compared with the fresh tied center sharing its seed block. Negative BPB deltas "
            "are improvements. The best observed member of 96 contrasts is selection-biased and is not a "
            "confirmed optimum without repeats."
        ),
        "",
        "## Headline conclusion",
        "",
        "The panel detects local phase-order effects but does **not** establish a new frontier. "
        f"For Uncheatable, the lowest absolute contrast is {uncheatable_primary['best_absolute_candidate_bpb']:.6f}, "
        f"which is {uncheatable_primary['best_absolute_minus_source_bpb']:+.6f} BPB relative to the selected "
        "historical source anchor; "
        f"the largest same-seed reduction is {uncheatable_primary['strongest_same_seed_delta']:+.6f} BPB. "
        f"For Table-9, the lowest contrast is {table9_primary['best_absolute_candidate_bpb']:.6f}, "
        f"or {table9_primary['best_absolute_minus_source_bpb']:+.6f} BPB relative to its selected historical source; "
        f"the largest same-seed reduction is {table9_primary['strongest_same_seed_delta']:+.6f} BPB. "
        "These are selected extrema among 96 contrasts and are comparable to fresh-center variation, so "
        "neither is confirmatory.",
        "",
        "The first-order additive phase-gradient hypothesis is target-dependent rather than universal. "
        "On held-out high-mass pair directions it reaches RMSE ratios of "
        f"{uncheatable_gradient['heldout_rmse_ratio_vs_zero']:.3f} for Uncheatable and "
        f"{table9_gradient['heldout_rmse_ratio_vs_zero']:.3f} for Table-9 relative to a zero-effect baseline. "
        "Signed bucket preferences also transfer poorly between anchors. A global additive phase head is therefore "
        "inadequate, especially for Table-9; any successful model needs anchor-dependent nonlinear interactions or "
        "substantially better identification.",
        "",
        "## Fresh tied centers",
        "",
        markdown_table(
            centers,
            [
                "anchor_id",
                "target",
                "source_selected_bpb",
                "fresh_center_mean_bpb",
                "fresh_center_sd_bpb",
                "fresh_minus_source_bpb",
            ],
        ),
        "",
        (
            "The source-to-fresh difference diagnoses winner's curse and ordinary seed drift at the chosen "
            "frontier aggregate. It is descriptive because the source score is a selected historical draw."
        ),
        "",
        "## Primary anchor-target outcomes",
        "",
        markdown_table(
            primary,
            [
                "anchor_id",
                "target",
                "fraction_better_than_same_seed_center",
                "median_delta_vs_same_seed_center",
                "best_absolute_candidate_id",
                "best_absolute_candidate_bpb",
                "best_absolute_minus_source_bpb",
                "best_absolute_delta_vs_same_seed_center",
                "strongest_same_seed_candidate_id",
                "strongest_same_seed_delta",
                "both_targets_improve_count",
            ],
        ),
        "",
        "## Odd ordering and even phase curvature",
        "",
        markdown_table(
            pair_metrics,
            [
                "anchor_id",
                "target",
                "contrast_family",
                "direction_count",
                "odd_effect_rms",
                "mean_phase_curvature",
                "fraction_mean_contrast_better_than_tied",
                "maximum_best_sign_gain_vs_center",
            ],
        ),
        "",
        (
            "The odd effect is `(plus - minus) / 2`; for domain-vs-rest probes, a negative value means placing the "
            "named domain later is better. Mean phase curvature is `(plus + minus) / 2 - tied`; positive values "
            "mean phase asymmetry is harmful on average along that direction."
        ),
        "",
        "## Additive local-gradient falsification",
        "",
        markdown_table(
            gradient_metrics,
            [
                "anchor_id",
                "target",
                "train_rank",
                "train_rmse",
                "heldout_rmse",
                "heldout_zero_baseline_rmse",
                "heldout_rmse_ratio_vs_zero",
                "heldout_pearson",
                "heldout_spearman",
                "heldout_r2",
            ],
        ),
        "",
        (
            "The 39 one-vs-rest odd effects identify a rank-38 simplex-tangent gradient. The nine high-mass "
            "bucket-pair effects are held out from this fit. A ratio below one beats the zero-effect baseline; a "
            "negative heldout R2 rejects a globally additive local phase gradient at this contrast scale."
        ),
        "",
        "## Gradient transfer",
        "",
        markdown_table(
            transfer,
            [
                "left_anchor",
                "left_target",
                "right_anchor",
                "right_target",
                "same_anchor",
                "same_target",
                "cosine_similarity",
                "spearman",
            ],
        ),
        "",
        "## Files",
        "",
        "- `observed_results.csv`: all 200 checkpoints with W&B provenance and same-seed center deltas.",
        "- `paired_phase_effects.csv`: 96 seed-matched sign pairs on both targets.",
        "- `center_control_summary.csv`, `anchor_metric_summary.csv`, and `pair_effect_summary.csv`.",
        (
            "- `phase_gradient_metrics.csv`, `phase_gradient_holdout_predictions.csv`, `phase_gradients.csv`, and "
            "`phase_gradient_transfer.csv`."
        ),
        (
            "- `target_delta_scatter.html`, `delta_vs_phase_tv.html`, `domain_odd_effect_heatmap.html`, and "
            "`phase_gradient_holdout.html`."
        ),
    ]
    (output_dir / "report.md").write_text("\n".join(report) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.panel_dir / "candidate_manifest.csv")
    anchor_audit = pd.read_csv(args.panel_dir / "anchor_audit.csv")
    cached_results = args.output_dir / "observed_results.csv"
    if cached_results.exists() and not args.refresh_wandb:
        results = pd.read_csv(cached_results)
    else:
        results = collect_results(manifest, timeout=args.wandb_timeout)
    results = attach_seed_center_deltas(results)
    pairs = build_pair_effects(results)
    centers = center_summary(results, anchor_audit)
    anchor_summary = anchor_metric_summary(results, centers)
    pair_metrics = pair_summary(pairs)

    weight_frame = pd.read_csv(args.panel_dir / "phase_weights.csv")
    domains = list(dict.fromkeys(weight_frame["domain"].astype(str)))
    vectors = phase_direction_vectors(args.panel_dir)
    gradient_metrics, gradient_predictions, gradients = fit_phase_gradients(pairs, vectors, domains)
    transfer = gradient_transfer(gradients)

    results.to_csv(args.output_dir / "observed_results.csv", index=False)
    pairs.to_csv(args.output_dir / "paired_phase_effects.csv", index=False)
    centers.to_csv(args.output_dir / "center_control_summary.csv", index=False)
    anchor_summary.to_csv(args.output_dir / "anchor_metric_summary.csv", index=False)
    pair_metrics.to_csv(args.output_dir / "pair_effect_summary.csv", index=False)
    gradient_metrics.to_csv(args.output_dir / "phase_gradient_metrics.csv", index=False)
    gradient_predictions.to_csv(args.output_dir / "phase_gradient_holdout_predictions.csv", index=False)
    gradients.to_csv(args.output_dir / "phase_gradients.csv", index=False)
    transfer.to_csv(args.output_dir / "phase_gradient_transfer.csv", index=False)

    write_delta_scatter(results, args.output_dir / "target_delta_scatter.html")
    write_tv_paths(results, args.output_dir / "delta_vs_phase_tv.html")
    write_domain_heatmap(pairs, args.output_dir / "domain_odd_effect_heatmap.html")
    write_gradient_holdout(gradient_predictions, args.output_dir / "phase_gradient_holdout.html")
    write_report(
        output_dir=args.output_dir,
        centers=centers,
        anchor_summary=anchor_summary,
        pair_metrics=pair_metrics,
        gradient_metrics=gradient_metrics,
        transfer=transfer,
    )
    summary = {
        "completed_training_runs": len(results),
        "completed_table9_evals": len(results),
        "direction_pairs": len(pairs) // len(TARGETS),
        "anchor_primary_results": anchor_summary[anchor_summary["is_anchor_selection_target"]].to_dict(orient="records"),
        "gradient_holdout_metrics": gradient_metrics.to_dict(orient="records"),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
