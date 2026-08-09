# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Bootstrap the frozen expanded-300M Pareto baseline.

The 520 observations contain only 280 independent policy correspondences.
This companion analysis leaves the frozen fitting protocol untouched and
resamples ``phase_correspondence_key`` groups within outer folds. Exact
aggregate-matched contrasts remain paired in every draw. Regret is different:
its candidate population is fixed by each outer fold, so its uncertainty
resamples the three fold-level regrets rather than deleting policies through a
correspondence bootstrap.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_expanded_300m_pareto_baseline_20260731 as baseline,
)

DEFAULT_INPUT_DIR = SCRIPT_DIR / "reference_outputs" / "expanded_300m_pareto_baseline_20260731"
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "expanded_300m_pareto_bootstrap_20260731"
PROTOCOL_VERSION = "expanded-300m-pareto-bootstrap-v2"
DEFAULT_DRAWS = 4_000
DEFAULT_SEED = 731_310
INTERVAL_QUANTILES = (0.025, 0.975)
TIE_TOLERANCE = 1e-12
KEY_COLUMNS = (
    "row_index",
    "phase_correspondence_key",
    "policy_family",
    "physical_tied",
    "outer_fold",
    "observed",
)
METRIC_DIRECTIONS = {
    "all_rmse": "lower",
    "tied_rmse": "lower",
    "asymmetric_rmse": "lower",
    "all_low_tail_rmse": "lower",
    "asymmetric_low_tail_rmse": "lower",
    "all_lower_tail_optimism": "lower",
    "asymmetric_lower_tail_optimism": "lower",
    "all_calibration_slope": "unit",
    "asymmetric_calibration_slope": "unit",
    "all_regret_at_1": "lower",
    "asymmetric_regret_at_1": "lower",
    "asymmetric_regret_at_3": "lower",
    "asymmetric_regret_at_5": "lower",
    "pair_delta_rmse": "lower",
    "pair_delta_spearman": "higher",
    "pair_delta_bias": "zero",
    "pair_sign_accuracy": "higher",
}
PLOT_METRICS = (
    "all_rmse",
    "asymmetric_rmse",
    "pair_delta_rmse",
    "asymmetric_regret_at_1",
)


@dataclass(frozen=True)
class TargetData:
    """Aligned OOF predictions and policy-correspondence structure."""

    target: str
    frame: pd.DataFrame
    model_ids: tuple[str, ...]
    predictions: np.ndarray
    group_rows: dict[str, np.ndarray]
    fold_groups: tuple[tuple[str, ...], ...]
    pair_rows: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class BootstrapSample:
    """One paired resample for smooth metrics and fixed-population regret."""

    rows: np.ndarray
    regret_fold_rows: tuple[np.ndarray, ...]
    pair_tied: np.ndarray
    pair_asymmetric: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--draws", type=int, default=DEFAULT_DRAWS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(json_ready(value), indent=2, sort_keys=True) + "\n")


def protocol_payload(input_protocol: dict[str, Any], draws: int, seed: int) -> dict[str, Any]:
    payload = {
        "version": PROTOCOL_VERSION,
        "input_protocol_hash": input_protocol["protocol_hash"],
        "targets": list(baseline.TARGETS),
        "models": list(baseline.MODEL_IDS),
        "reference_only_models": sorted(baseline.REFERENCE_ONLY_MODELS),
        "bootstrap_draws": draws,
        "bootstrap_seed": seed,
        "smooth_metric_bootstrap_unit": "phase_correspondence_key",
        "smooth_metric_bootstrap_strata": "outer_fold",
        "regret_bootstrap_unit": "outer_fold",
        "regret_candidate_population": "fixed full outer-fold test set",
        "pairwise_probability_tie_tolerance": TIE_TOLERANCE,
        "interval_quantiles": INTERVAL_QUANTILES,
        "metric_directions": METRIC_DIRECTIONS,
        "source_hash": file_hash(Path(__file__)),
    }
    encoded = json.dumps(json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "protocol_hash": hashlib.sha256(encoded).hexdigest()}


def _cell_dir(input_dir: Path, target: str, model_id: str) -> Path:
    return input_dir / "cells" / target / model_id


def _validate_complete_cell(path: Path, input_protocol_hash: str) -> None:
    required = (path / "complete.json", path / "predictions.csv", path / "pair_predictions.csv")
    if any(not item.exists() for item in required):
        raise FileNotFoundError(f"incomplete baseline cell: {path}")
    marker = json.loads((path / "complete.json").read_text())
    if marker.get("protocol_hash") != input_protocol_hash:
        raise ValueError(f"stale protocol marker in {path}")


def _aligned_frame(reference: pd.DataFrame, candidate: pd.DataFrame, model_id: str) -> None:
    if len(reference) != len(candidate):
        raise ValueError(f"row-count mismatch for {model_id}")
    for column in KEY_COLUMNS[:-1]:
        if not reference[column].astype(str).equals(candidate[column].astype(str)):
            raise ValueError(f"{column} mismatch for {model_id}")
    if not np.allclose(reference["observed"], candidate["observed"], atol=0.0, rtol=0.0):
        raise ValueError(f"observed-target mismatch for {model_id}")


def load_target(input_dir: Path, input_protocol_hash: str, target: str) -> TargetData:
    frames: list[pd.DataFrame] = []
    model_ids = tuple(baseline.MODEL_IDS)
    for model_id in model_ids:
        path = _cell_dir(input_dir, target, model_id)
        _validate_complete_cell(path, input_protocol_hash)
        frame = pd.read_csv(path / "predictions.csv")
        missing = sorted(set((*KEY_COLUMNS, "predicted")) - set(frame.columns))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        frame = frame.sort_values("row_index").reset_index(drop=True)
        if frames:
            _aligned_frame(frames[0], frame, model_id)
        frames.append(frame)

    frame = frames[0][list(KEY_COLUMNS)].copy()
    predictions = np.stack([candidate["predicted"].to_numpy(dtype=float) for candidate in frames])
    if not np.isfinite(predictions).all():
        raise ValueError(f"non-finite predictions in {target}")

    group_rows = {
        str(key): block["row_index"].to_numpy(dtype=int)
        for key, block in frame.groupby("phase_correspondence_key", sort=True)
    }
    group_folds = frame.groupby("phase_correspondence_key", sort=True)["outer_fold"].nunique()
    if int(group_folds.max()) != 1:
        raise ValueError(f"correspondence group crosses outer folds in {target}")
    fold_groups = tuple(
        tuple(sorted(frame.loc[frame["outer_fold"].eq(fold), "phase_correspondence_key"].astype(str).unique().tolist()))
        for fold in sorted(frame["outer_fold"].unique())
    )

    indexed = frame.set_index(["phase_correspondence_key", "policy_family"])["row_index"]
    family_sets = frame.groupby("phase_correspondence_key")["policy_family"].agg(set)
    pair_rows: dict[str, tuple[int, int]] = {}
    for key, families in family_sets.items():
        key = str(key)
        if not {"single_phase", "two_phase"}.issubset(families):
            continue
        tied = int(indexed.loc[(key, "single_phase")])
        asymmetric = int(indexed.loc[(key, "two_phase")])
        if bool(frame.loc[frame["row_index"].eq(asymmetric), "physical_tied"].iloc[0]):
            continue
        pair_rows[key] = (tied, asymmetric)

    if len(frame) != 520 or len(group_rows) != 280 or len(pair_rows) != 238:
        raise ValueError(
            f"unexpected {target} design: rows={len(frame)}, groups={len(group_rows)}, pairs={len(pair_rows)}"
        )
    return TargetData(
        target=target,
        frame=frame,
        model_ids=model_ids,
        predictions=predictions,
        group_rows=group_rows,
        fold_groups=fold_groups,
        pair_rows=pair_rows,
    )


def full_sample(data: TargetData) -> BootstrapSample:
    regret_fold_rows = tuple(
        data.frame.loc[data.frame["outer_fold"].eq(fold), "row_index"].to_numpy(dtype=int)
        for fold in sorted(data.frame["outer_fold"].unique())
    )
    pair_tied = np.asarray([rows[0] for rows in data.pair_rows.values()], dtype=int)
    pair_asymmetric = np.asarray([rows[1] for rows in data.pair_rows.values()], dtype=int)
    return BootstrapSample(
        rows=np.arange(len(data.frame), dtype=int),
        regret_fold_rows=regret_fold_rows,
        pair_tied=pair_tied,
        pair_asymmetric=pair_asymmetric,
    )


def draw_sample(data: TargetData, rng: np.random.Generator) -> BootstrapSample:
    sampled_groups_by_fold = []
    smooth_metric_rows = []
    for groups in data.fold_groups:
        sampled = rng.choice(groups, size=len(groups), replace=True).tolist()
        sampled_groups_by_fold.append(sampled)
        smooth_metric_rows.append(np.concatenate([data.group_rows[str(group)] for group in sampled]))
    sampled_groups = [str(group) for groups in sampled_groups_by_fold for group in groups]
    paired = [data.pair_rows[group] for group in sampled_groups if group in data.pair_rows]
    full_fold_rows = tuple(
        data.frame.loc[data.frame["outer_fold"].eq(fold), "row_index"].to_numpy(dtype=int)
        for fold in sorted(data.frame["outer_fold"].unique())
    )
    sampled_folds = rng.choice(len(full_fold_rows), size=len(full_fold_rows), replace=True)
    return BootstrapSample(
        rows=np.concatenate(smooth_metric_rows),
        regret_fold_rows=tuple(full_fold_rows[index] for index in sampled_folds),
        pair_tied=np.asarray([row[0] for row in paired], dtype=int),
        pair_asymmetric=np.asarray([row[1] for row in paired], dtype=int),
    )


def safe_spearman(observed: np.ndarray, predicted: np.ndarray) -> float:
    if len(observed) < 2 or np.std(observed) <= 0.0 or np.std(predicted) <= 0.0:
        return float("nan")
    return float(spearmanr(observed, predicted).statistic)


def calibration_slope(observed: np.ndarray, predicted: np.ndarray) -> float:
    centered = predicted - np.mean(predicted)
    denominator = float(centered @ centered)
    if denominator <= 1e-18:
        return float("nan")
    return float(centered @ (observed - np.mean(observed)) / denominator)


def scalar_metrics(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - observed
    count = min(
        len(observed),
        max(baseline.LOWER_TAIL_MIN_COUNT, math.ceil(baseline.LOWER_TAIL_FRACTION * len(observed))),
    )
    tail = np.argsort(predicted)[:count]
    tail_error = error[tail]
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "low_tail_rmse": float(np.sqrt(np.mean(tail_error**2))),
        "lower_tail_optimism": float(np.mean(np.maximum(-tail_error, 0.0))),
        "calibration_slope": calibration_slope(observed, predicted),
    }


def bootstrap_regret(
    observed: np.ndarray,
    predicted: np.ndarray,
    sample: BootstrapSample,
    tied: np.ndarray,
    *,
    asymmetric_only: bool,
    k: int,
) -> float:
    regrets = []
    for candidates in sample.regret_fold_rows:
        if asymmetric_only:
            candidates = candidates[~tied[candidates]]
        if not len(candidates):
            continue
        selected = candidates[np.argsort(predicted[candidates])[: min(k, len(candidates))]]
        regrets.append(float(np.min(observed[selected]) - np.min(observed[candidates])))
    return float(np.mean(regrets)) if regrets else float("nan")


def metric_values(data: TargetData, predicted: np.ndarray, sample: BootstrapSample) -> dict[str, float]:
    observed = data.frame["observed"].to_numpy(dtype=float)
    tied = data.frame["physical_tied"].to_numpy(dtype=bool)
    rows = sample.rows
    tied_rows = rows[tied[rows]]
    asymmetric_rows = rows[~tied[rows]]
    all_metrics = scalar_metrics(observed[rows], predicted[rows])
    tied_metrics = scalar_metrics(observed[tied_rows], predicted[tied_rows])
    asymmetric_metrics = scalar_metrics(observed[asymmetric_rows], predicted[asymmetric_rows])

    observed_delta = observed[sample.pair_asymmetric] - observed[sample.pair_tied]
    predicted_delta = predicted[sample.pair_asymmetric] - predicted[sample.pair_tied]
    return {
        "all_rmse": all_metrics["rmse"],
        "tied_rmse": tied_metrics["rmse"],
        "asymmetric_rmse": asymmetric_metrics["rmse"],
        "all_low_tail_rmse": all_metrics["low_tail_rmse"],
        "asymmetric_low_tail_rmse": asymmetric_metrics["low_tail_rmse"],
        "all_lower_tail_optimism": all_metrics["lower_tail_optimism"],
        "asymmetric_lower_tail_optimism": asymmetric_metrics["lower_tail_optimism"],
        "all_calibration_slope": all_metrics["calibration_slope"],
        "asymmetric_calibration_slope": asymmetric_metrics["calibration_slope"],
        "all_regret_at_1": bootstrap_regret(
            observed,
            predicted,
            sample,
            tied,
            asymmetric_only=False,
            k=1,
        ),
        "asymmetric_regret_at_1": bootstrap_regret(
            observed,
            predicted,
            sample,
            tied,
            asymmetric_only=True,
            k=1,
        ),
        "asymmetric_regret_at_3": bootstrap_regret(
            observed,
            predicted,
            sample,
            tied,
            asymmetric_only=True,
            k=3,
        ),
        "asymmetric_regret_at_5": bootstrap_regret(
            observed,
            predicted,
            sample,
            tied,
            asymmetric_only=True,
            k=5,
        ),
        "pair_delta_rmse": float(np.sqrt(np.mean((predicted_delta - observed_delta) ** 2))),
        "pair_delta_spearman": safe_spearman(observed_delta, predicted_delta),
        "pair_delta_bias": float(np.mean(predicted_delta - observed_delta)),
        "pair_sign_accuracy": float(np.mean(np.sign(predicted_delta) == np.sign(observed_delta))),
    }


def metric_loss(values: np.ndarray, direction: str) -> np.ndarray:
    if direction == "lower":
        return values
    if direction == "higher":
        return -values
    if direction == "unit":
        return np.abs(values - 1.0)
    if direction == "zero":
        return np.abs(values)
    raise ValueError(f"unknown metric direction: {direction}")


def bootstrap_target(
    data: TargetData,
    draws: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_names = tuple(METRIC_DIRECTIONS)
    point = np.empty((len(data.model_ids), len(metric_names)), dtype=float)
    distribution = np.empty((len(data.model_ids), draws, len(metric_names)), dtype=float)
    complete = full_sample(data)
    for model_index, predicted in enumerate(data.predictions):
        values = metric_values(data, predicted, complete)
        point[model_index] = [values[name] for name in metric_names]

    rng = np.random.default_rng(seed)
    for draw in range(draws):
        sample = draw_sample(data, rng)
        for model_index, predicted in enumerate(data.predictions):
            values = metric_values(data, predicted, sample)
            distribution[model_index, draw] = [values[name] for name in metric_names]

    summaries = []
    for model_index, model_id in enumerate(data.model_ids):
        for metric_index, metric in enumerate(metric_names):
            values = distribution[model_index, :, metric_index]
            finite = values[np.isfinite(values)]
            if len(finite):
                lower, upper = np.quantile(finite, INTERVAL_QUANTILES)
                bootstrap_mean = float(np.mean(finite))
                bootstrap_median = float(np.median(finite))
            else:
                lower = upper = bootstrap_mean = bootstrap_median = float("nan")
            summaries.append(
                {
                    "target": data.target,
                    "model": model_id,
                    "reference_only": model_id in baseline.REFERENCE_ONLY_MODELS,
                    "metric": metric,
                    "direction": METRIC_DIRECTIONS[metric],
                    "point_estimate": point[model_index, metric_index],
                    "bootstrap_mean": bootstrap_mean,
                    "bootstrap_median": bootstrap_median,
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "finite_draws": len(finite),
                }
            )

    pairwise = []
    for candidate_index, comparator_index in combinations(range(len(data.model_ids)), 2):
        for metric_index, metric in enumerate(metric_names):
            candidate_values = distribution[candidate_index, :, metric_index]
            comparator_values = distribution[comparator_index, :, metric_index]
            finite = np.isfinite(candidate_values) & np.isfinite(comparator_values)
            if not np.any(finite):
                pairwise.append(
                    {
                        "target": data.target,
                        "candidate": data.model_ids[candidate_index],
                        "comparator": data.model_ids[comparator_index],
                        "metric": metric,
                        "point_loss_difference": float("nan"),
                        "bootstrap_mean_loss_difference": float("nan"),
                        "ci_lower": float("nan"),
                        "ci_upper": float("nan"),
                        "probability_candidate_better": float("nan"),
                        "probability_candidate_tied": float("nan"),
                        "probability_candidate_worse": float("nan"),
                        "finite_draws": 0,
                    }
                )
                continue
            difference = metric_loss(
                candidate_values[finite],
                METRIC_DIRECTIONS[metric],
            ) - metric_loss(
                comparator_values[finite],
                METRIC_DIRECTIONS[metric],
            )
            lower, upper = np.quantile(difference, INTERVAL_QUANTILES)
            tied = np.isclose(difference, 0.0, atol=TIE_TOLERANCE, rtol=0.0)
            better = difference < -TIE_TOLERANCE
            worse = difference > TIE_TOLERANCE
            if not np.all(better | tied | worse):
                raise RuntimeError("pairwise comparison contains an unclassified draw")
            point_difference = float(
                metric_loss(
                    point[candidate_index : candidate_index + 1, metric_index],
                    METRIC_DIRECTIONS[metric],
                )[0]
                - metric_loss(
                    point[comparator_index : comparator_index + 1, metric_index],
                    METRIC_DIRECTIONS[metric],
                )[0]
            )
            pairwise.append(
                {
                    "target": data.target,
                    "candidate": data.model_ids[candidate_index],
                    "comparator": data.model_ids[comparator_index],
                    "metric": metric,
                    "point_loss_difference": point_difference,
                    "bootstrap_mean_loss_difference": float(np.mean(difference)),
                    "ci_lower": float(lower),
                    "ci_upper": float(upper),
                    "probability_candidate_better": float(np.mean(better)),
                    "probability_candidate_tied": float(np.mean(tied)),
                    "probability_candidate_worse": float(np.mean(worse)),
                    "finite_draws": int(finite.sum()),
                }
            )
    return pd.DataFrame(summaries), pd.DataFrame(pairwise)


def build_plot(summary: pd.DataFrame, path: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=tuple(metric.replace("_", " ") for metric in PLOT_METRICS),
        horizontal_spacing=0.12,
        vertical_spacing=0.16,
    )
    for panel, metric in enumerate(PLOT_METRICS):
        row = panel // 2 + 1
        column = panel % 2 + 1
        block = summary.loc[summary["metric"].eq(metric)].copy()
        block["label"] = block["target"].astype(str).str.cat(block["model"].astype(str), sep=" · ")
        figure.add_trace(
            go.Scatter(
                x=block["point_estimate"],
                y=block["label"],
                mode="markers",
                marker={
                    "size": 10,
                    "color": block["point_estimate"],
                    "colorscale": "RdYlGn_r",
                    "showscale": False,
                    "line": {"color": "#183247", "width": 0.7},
                },
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": block["ci_upper"] - block["point_estimate"],
                    "arrayminus": block["point_estimate"] - block["ci_lower"],
                    "thickness": 1.3,
                },
                customdata=np.column_stack(
                    [
                        block["ci_lower"],
                        block["ci_upper"],
                        block["reference_only"],
                    ]
                ),
                hovertemplate=(
                    "%{y}<br>point=%{x:.6f}<br>95% paired-resampling interval="
                    "[%{customdata[0]:.6f}, %{customdata[1]:.6f}]"
                    "<br>reference only=%{customdata[2]}<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=column,
        )
    figure.update_layout(
        title="Expanded 300M Pareto baseline · paired uncertainty",
        template="plotly_white",
        height=1_150,
        width=1_550,
        margin={"l": 300, "r": 50, "t": 110, "b": 70},
        font={"family": "Avenir Next, sans-serif", "color": "#183247"},
    )
    figure.update_xaxes(title_text="BPB loss or regret · lower is better")
    pio.write_html(
        figure,
        path,
        include_plotlyjs=True,
        full_html=True,
        config={"displaylogo": False, "toImageButtonOptions": {"format": "png", "scale": 4}},
    )


def write_report(
    output_dir: Path,
    protocol: dict[str, Any],
    summary: pd.DataFrame,
    pairwise: pd.DataFrame,
) -> None:
    primary = summary.loc[summary["metric"].isin(PLOT_METRICS)].copy()
    primary = primary.sort_values(["target", "metric", "point_estimate"])
    decisive = pairwise.loc[
        pairwise["metric"].isin(PLOT_METRICS) & ((pairwise["ci_upper"] < 0.0) | (pairwise["ci_lower"] > 0.0))
    ].copy()
    report = [
        "# Expanded 300M Pareto Baseline: Paired Bootstrap",
        "",
        f"- Protocol: `{protocol['protocol_hash']}`",
        f"- Input fitting protocol: `{protocol['input_protocol_hash']}`",
        f"- Draws: {protocol['bootstrap_draws']}",
        "- Smooth-error unit: `phase_correspondence_key`, resampled within outer fold.",
        "- Regret unit: outer fold, with each fold's full candidate population fixed.",
        "- Exact aggregate-matched rows remain paired.",
        "- Win probabilities are strict; ties and losses are reported separately.",
        "- HPR-band remains reference-only.",
        "",
        "## Primary Intervals",
        "",
        primary[
            [
                "target",
                "model",
                "metric",
                "point_estimate",
                "ci_lower",
                "ci_upper",
                "reference_only",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Pairwise Differences Excluding Uncertain Ties",
        "",
    ]
    if decisive.empty:
        report.append("No primary pairwise interval excludes zero.")
    else:
        report.append(
            decisive[
                [
                    "target",
                    "candidate",
                    "comparator",
                    "metric",
                    "point_loss_difference",
                    "ci_lower",
                    "ci_upper",
                    "probability_candidate_better",
                    "probability_candidate_tied",
                    "probability_candidate_worse",
                ]
            ].to_markdown(index=False, floatfmt=".6f")
        )
    report.extend(
        [
            "",
            "Selection-regret intervals resample the three fixed outer-fold regret values. "
            "They describe between-fold variability without changing which policies are deployable.",
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report))


def main() -> None:
    args = parse_args()
    if args.draws < 100:
        raise ValueError("at least 100 bootstrap draws are required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    input_protocol = json.loads((args.input_dir / "protocol.json").read_text())
    protocol = protocol_payload(input_protocol, args.draws, args.seed)
    write_json(args.output_dir / "protocol.json", protocol)
    if args.prepare_only:
        print(f"prepared bootstrap protocol {protocol['protocol_hash']}", flush=True)
        return

    marker_path = args.output_dir / "complete.json"
    if not args.force and marker_path.exists():
        marker = json.loads(marker_path.read_text())
        if marker.get("protocol_hash") == protocol["protocol_hash"]:
            print(f"skip complete bootstrap protocol {protocol['protocol_hash']}", flush=True)
            return

    summaries = []
    pairwise = []
    for target_index, target in enumerate(baseline.TARGETS):
        print(f"bootstrap {target}", flush=True)
        data = load_target(args.input_dir, str(input_protocol["protocol_hash"]), target)
        target_summary, target_pairwise = bootstrap_target(
            data,
            args.draws,
            args.seed + target_index,
        )
        summaries.append(target_summary)
        pairwise.append(target_pairwise)

    summary = pd.concat(summaries, ignore_index=True)
    paired = pd.concat(pairwise, ignore_index=True)
    summary.to_csv(args.output_dir / "bootstrap_metric_intervals.csv", index=False)
    paired.to_csv(args.output_dir / "bootstrap_pairwise_differences.csv", index=False)
    build_plot(summary, args.output_dir / "bootstrap_metric_intervals.html")
    write_report(args.output_dir, protocol, summary, paired)
    write_json(
        marker_path,
        {
            "protocol_hash": protocol["protocol_hash"],
            "input_protocol_hash": protocol["input_protocol_hash"],
            "targets": list(baseline.TARGETS),
            "models": list(baseline.MODEL_IDS),
            "draws": args.draws,
        },
    )
    print(f"completed bootstrap protocol {protocol['protocol_hash']}", flush=True)


if __name__ == "__main__":
    main()
