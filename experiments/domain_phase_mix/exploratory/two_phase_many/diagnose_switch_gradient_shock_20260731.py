# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Test whether an observed optimizer shock explains phase-relaxation residuals.

This is an identification diagnostic, not a deployable surrogate. It measures
the asymmetric-minus-tied change in training loss and gradient norm immediately
across the 300M phase boundary. A dynamic switch-cost mechanism is admissible
only if this independently logged shock predicts the common component residual
left by the frozen SUR-068 relaxation law.
"""

from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "switch_gradient_shock_20260731"
RELAXATION_DIR = SCRIPT_DIR / "reference_outputs" / "component_phase_relaxation_20260731"
RUN_MANIFEST_PATH = RELAXATION_DIR / "run_manifest.csv"
PAIR_DIFFERENCES_PATH = RELAXATION_DIR / "pair_differences.csv"
RELAXATION_DECISION_PATH = RELAXATION_DIR / "decision.json"
WANDB_PATH = "marin-community/marin"
HISTORY_CACHE_NAME = "switch_histories.csv"
UNAVAILABLE_CACHE_NAME = "history_unavailable.csv"

PHASE_BOUNDARY_STEP = 18_310
FINAL_STEP = 22_887
PRE_WINDOW = (18_110, 18_300)
POST_WINDOW = (18_310, 18_500)
EVALUATION_STEPS = (19_000, 20_000, 21_000, 22_000, FINAL_STEP)
COMPONENT_KEYS = (
    "eval/uncheatable_eval/ao3_english/bpb",
    "eval/uncheatable_eval/arxiv_computer_science/bpb",
    "eval/uncheatable_eval/arxiv_physics/bpb",
    "eval/uncheatable_eval/bbc_news/bpb",
    "eval/uncheatable_eval/github_cpp/bpb",
    "eval/uncheatable_eval/github_python/bpb",
    "eval/uncheatable_eval/wikipedia_english/bpb",
)
SPLIT_SEED = 20_260_731
N_SPLITS = 5
MIN_WINDOW_ROWS = 10
MIN_COMPLETE_PAIRS = 180
FETCH_ATTEMPTS = 3


@dataclass(frozen=True)
class ShockMetric:
    """One observed boundary-shock coordinate."""

    name: str
    column: str


SHOCK_METRICS = (
    ShockMetric("gradient_log_jump", "grad/norm/total"),
    ShockMetric("training_loss_jump", "train/loss"),
)


class HistoryUnavailableError(RuntimeError):
    """A run lacks the boundary telemetry required by this diagnostic."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-workers", type=int, default=24)
    parser.add_argument("--refresh", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def fetch_history(run_id: str) -> pd.DataFrame:
    keys = ["global_step", *(metric.column for metric in SHOCK_METRICS)]
    records: list[dict[str, object]] | None = None
    last_error: Exception | None = None
    for attempt in range(FETCH_ATTEMPTS):
        try:
            api = wandb.Api(timeout=120)
            run = api.run(f"{WANDB_PATH}/{run_id}")
            records = list(
                run.scan_history(
                    keys=keys,
                    page_size=1_000,
                    min_step=PRE_WINDOW[0],
                    max_step=POST_WINDOW[1],
                )
            )
            break
        except Exception as error:
            last_error = error
            if attempt + 1 < FETCH_ATTEMPTS:
                time.sleep(2**attempt)
    if records is None:
        raise RuntimeError(f"W&B history fetch exhausted retries for {run_id}") from last_error
    history = pd.DataFrame.from_records(records)
    missing = [key for key in keys if key not in history]
    if missing:
        raise HistoryUnavailableError(f"Run {run_id} lacks history keys: {missing}")
    history = history[keys].copy()
    history = history.loc[history[list(metric.column for metric in SHOCK_METRICS)].notna().any(axis=1)].copy()
    history["global_step"] = history["global_step"].astype(int)
    history = history.groupby("global_step", as_index=False, sort=True)[keys[1:]].last()
    history["wandb_run_id"] = run_id
    return history


def collect_histories(output_dir: Path, max_workers: int, refresh: bool) -> pd.DataFrame:
    cache_path = output_dir / HISTORY_CACHE_NAME
    unavailable_path = output_dir / UNAVAILABLE_CACHE_NAME
    manifest = pd.read_csv(RUN_MANIFEST_PATH)
    expected_ids = set(manifest["wandb_run_id"].astype(str))
    if cache_path.exists() and not refresh:
        cached = pd.read_csv(cache_path)
    else:
        cached = pd.DataFrame()
    if unavailable_path.exists() and not refresh:
        unavailable = pd.read_csv(unavailable_path)
    else:
        unavailable = pd.DataFrame(columns=["wandb_run_id", "reason"])
    cached_ids = set(cached.get("wandb_run_id", pd.Series(dtype=str)).astype(str))
    unavailable_ids = set(unavailable["wandb_run_id"].astype(str))
    pending = sorted(expected_ids - cached_ids - unavailable_ids)
    blocks = [cached] if not cached.empty else []
    unavailable_rows = unavailable.to_dict("records")
    if pending:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(fetch_history, run_id): run_id for run_id in pending}
            for index, future in enumerate(as_completed(futures), start=1):
                run_id = futures[future]
                try:
                    blocks.append(future.result())
                except HistoryUnavailableError as error:
                    unavailable_rows.append({"wandb_run_id": run_id, "reason": str(error)})
                except Exception as error:
                    raise RuntimeError(f"Failed to fetch switch history for {run_id}") from error
                if index % 25 == 0:
                    partial = pd.concat(blocks, ignore_index=True)
                    partial.to_csv(cache_path, index=False)
    result = pd.concat(blocks, ignore_index=True)
    result = result.drop_duplicates(["wandb_run_id", "global_step"], keep="last")
    result.to_csv(cache_path, index=False)
    unavailable = pd.DataFrame(unavailable_rows).drop_duplicates("wandb_run_id", keep="last")
    unavailable.to_csv(unavailable_path, index=False)
    found_ids = set(result["wandb_run_id"].astype(str))
    if found_ids | set(unavailable["wandb_run_id"].astype(str)) != expected_ids:
        raise RuntimeError("Switch-history availability does not account for every expected run")
    return result


def run_shocks(histories: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for run_id, block in histories.groupby("wandb_run_id", sort=True):
        pre = block.loc[block["global_step"].between(*PRE_WINDOW)]
        post = block.loc[block["global_step"].between(*POST_WINDOW)]
        if len(pre) < MIN_WINDOW_ROWS or len(post) < MIN_WINDOW_ROWS:
            continue
        row: dict[str, object] = {
            "wandb_run_id": run_id,
            "pre_rows": len(pre),
            "post_rows": len(post),
        }
        for metric in SHOCK_METRICS:
            pre_value = float(pre[metric.column].median())
            post_value = float(post[metric.column].median())
            if metric.name == "gradient_log_jump":
                jump = float(np.log(max(post_value, 1e-12)) - np.log(max(pre_value, 1e-12)))
            else:
                jump = post_value - pre_value
            row[f"{metric.name}_pre"] = pre_value
            row[f"{metric.name}_post"] = post_value
            row[metric.name] = jump
        rows.append(row)
    shocks = pd.DataFrame(rows)
    manifest = pd.read_csv(RUN_MANIFEST_PATH)
    shocks = manifest.merge(shocks, on="wandb_run_id", how="inner", validate="one_to_one")
    value_columns = [metric.name for metric in SHOCK_METRICS]
    pivot = shocks.pivot(index="pair_id", columns="policy_class", values=value_columns)
    pivot = pivot.dropna()
    if len(pivot) < MIN_COMPLETE_PAIRS:
        raise RuntimeError(f"Only {len(pivot)} pairs have complete switch telemetry")
    pair_rows = []
    for pair_id in pivot.index:
        row = {"pair_id": pair_id}
        for metric in SHOCK_METRICS:
            row[metric.name] = float(
                pivot.loc[pair_id, (metric.name, "two_phase")] - pivot.loc[pair_id, (metric.name, "one_phase")]
            )
        pair_rows.append(row)
    return pd.DataFrame(pair_rows)


def boundary_difference(differences: pd.DataFrame, key: str) -> pd.Series:
    pivot = differences.pivot(index="pair_id", columns="global_step", values=key)
    d17 = pivot[17_000]
    d18 = pivot[18_000]
    return d18 + (PHASE_BOUNDARY_STEP - 18_000) / 1_000 * (d18 - d17)


def relaxation_residuals() -> pd.DataFrame:
    differences = pd.read_csv(PAIR_DIFFERENCES_PATH)
    decision = json.loads(RELAXATION_DECISION_PATH.read_text())
    gamma = float(decision["parameters"]["gamma"])
    rate = float(decision["parameters"]["rate"])
    rows: list[dict[str, object]] = []
    for key in COMPONENT_KEYS:
        d0 = boundary_difference(differences, key)
        block = differences.loc[
            differences["global_step"].isin(EVALUATION_STEPS),
            ["pair_id", "global_step", key],
        ].copy()
        block = block.dropna(subset=[key]).merge(d0.rename("d0"), left_on="pair_id", right_index=True, how="inner")
        block = block.dropna(subset=["d0"])
        progress = (block["global_step"].to_numpy(float) - PHASE_BOUNDARY_STEP) / (FINAL_STEP - PHASE_BOUNDARY_STEP)
        factor = -4.0 * gamma + (1.0 + 4.0 * gamma) * np.exp(-rate * progress)
        block["predicted"] = block["d0"].to_numpy(float) * factor
        block["residual"] = block[key].to_numpy(float) - block["predicted"].to_numpy(float)
        block["component"] = key.split("/")[-2]
        rows.append(block.rename(columns={key: "observed"}))
    return pd.concat(rows, ignore_index=True)


def positive_slope(feature: np.ndarray, target: np.ndarray) -> float:
    x = np.maximum(np.asarray(feature, dtype=float), 0.0)
    denominator = float(np.dot(x, x))
    if denominator <= 1e-15:
        return 0.0
    return max(float(np.dot(x, target) / denominator), 0.0)


def grouped_oof(block: pd.DataFrame, feature: str) -> np.ndarray:
    if not np.isfinite(block[feature].to_numpy(float)).all():
        raise RuntimeError(f"Non-finite values in switch feature {feature}")
    if not np.isfinite(block["common_residual"].to_numpy(float)).all():
        raise RuntimeError("Non-finite common residuals")
    pairs = np.asarray(sorted(block["pair_id"].unique()))
    splitter = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SPLIT_SEED)
    prediction = np.full(len(block), np.nan, dtype=float)
    for train_pair_indices, test_pair_indices in splitter.split(pairs):
        train_pairs = set(pairs[train_pair_indices])
        test_pairs = set(pairs[test_pair_indices])
        train = block.loc[block["pair_id"].isin(train_pairs)]
        test_mask = block["pair_id"].isin(test_pairs).to_numpy()
        test = block.loc[test_mask]
        coefficient = positive_slope(train[feature].to_numpy(float), train["common_residual"].to_numpy(float))
        if not np.isfinite(coefficient):
            raise RuntimeError(f"Non-finite {feature} coefficient")
        prediction[test_mask] = coefficient * np.maximum(test[feature].to_numpy(float), 0.0)
    if not np.isfinite(prediction).all():
        raise RuntimeError("Incomplete grouped OOF switch-shock prediction")
    return prediction


def rmse(observed: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(np.asarray(predicted) - np.asarray(observed)))))


def evaluate(pair_shocks: pd.DataFrame, residuals: pd.DataFrame, output_dir: Path) -> None:
    common = (
        residuals.groupby(["pair_id", "global_step"], as_index=False)
        .agg(common_residual=("residual", "mean"), component_residual_sd=("residual", "std"))
        .merge(pair_shocks, on="pair_id", how="inner", validate="many_to_one")
    )
    rows = []
    predictions = []
    for step, block in common.groupby("global_step", sort=True):
        for metric in SHOCK_METRICS:
            local = block.copy()
            observed = local["common_residual"].to_numpy(float)
            predicted = grouped_oof(local, metric.name)
            rho, p_value = spearmanr(local[metric.name], observed)
            candidate_rmse = rmse(observed, predicted)
            zero_rmse = rmse(observed, np.zeros_like(observed))
            rows.append(
                {
                    "global_step": int(step),
                    "shock": metric.name,
                    "pairs": len(local),
                    "spearman": float(rho),
                    "spearman_p": float(p_value),
                    "oof_rmse": candidate_rmse,
                    "zero_rmse": zero_rmse,
                    "zero_improvement": 1.0 - candidate_rmse / zero_rmse,
                    "mean_common_residual": float(observed.mean()),
                    "mean_component_residual_sd": float(local["component_residual_sd"].mean()),
                }
            )
            local["shock"] = metric.name
            local["oof_predicted_common_residual"] = predicted
            predictions.append(local)
    metrics = pd.DataFrame(rows)
    prediction_frame = pd.concat(predictions, ignore_index=True)
    metrics.to_csv(output_dir / "metrics.csv", index=False)
    prediction_frame.to_csv(output_dir / "predictions.csv", index=False)
    pair_shocks.to_csv(output_dir / "pair_shocks.csv", index=False)

    gradient = metrics.loc[metrics["shock"].eq("gradient_log_jump")].set_index("global_step")
    checks = {
        "immediate_spearman": float(gradient.loc[19_000, "spearman"]) >= 0.25,
        "immediate_zero_improvement": float(gradient.loc[19_000, "zero_improvement"]) >= 0.10,
        "step20000_spearman": float(gradient.loc[20_000, "spearman"]) >= 0.20,
        "final_zero_improvement_positive": float(gradient.loc[FINAL_STEP, "zero_improvement"]) > 0.0,
    }
    decision = {
        "candidate_id": "WSD80-SUR-069",
        "decision": (
            "PASS: observed switch shock merits policy-map audit"
            if all(checks.values())
            else "FAIL: observed switch shock rejected"
        ),
        "checks": checks,
        "passed": all(checks.values()),
        "scope": "diagnostic_only_not_deployable",
    }
    write_json(output_dir / "decision.json", decision)
    render_plot(prediction_frame, output_dir / "switch_gradient_shock.html")
    render_report(metrics, decision, output_dir / "report.md")


def render_plot(predictions: pd.DataFrame, path: Path) -> None:
    figure = make_subplots(rows=1, cols=2, subplot_titles=("Immediate phase-1 residual", "Residual decay"))
    immediate = predictions.loc[predictions["global_step"].eq(19_000) & predictions["shock"].eq("gradient_log_jump")]
    figure.add_trace(
        go.Scatter(
            x=immediate["gradient_log_jump"],
            y=immediate["common_residual"],
            mode="markers",
            marker={"color": immediate["common_residual"], "colorscale": "RdYlGn_r", "size": 7},
            text=immediate["pair_id"],
            name="pairs",
        ),
        row=1,
        col=1,
    )
    trajectory = (
        predictions.loc[predictions["shock"].eq("gradient_log_jump")]
        .groupby("global_step", as_index=False)
        .agg(observed=("common_residual", "mean"), predicted=("oof_predicted_common_residual", "mean"))
    )
    for column, label in (("observed", "observed"), ("predicted", "shock-predicted")):
        figure.add_trace(
            go.Scatter(x=trajectory["global_step"], y=trajectory[column], mode="lines+markers", name=label),
            row=1,
            col=2,
        )
    figure.update_xaxes(title_text="Asymmetric-minus-tied log gradient-norm jump", row=1, col=1)
    figure.update_yaxes(title_text="Observed minus SUR-068 common residual", row=1, col=1)
    figure.update_xaxes(title_text="Global step", row=1, col=2)
    figure.update_yaxes(title_text="Common residual (BPB)", row=1, col=2)
    figure.update_layout(title="Observed optimizer shock as a phase-cost state", template="plotly_white")
    figure.write_html(path, include_plotlyjs="cdn")


def render_report(metrics: pd.DataFrame, decision: dict[str, object], path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "# Switch-gradient-shock diagnostic",
                "",
                f"**Decision: {decision['decision']}**",
                "",
                "This diagnostic tests whether an independently logged optimizer shock "
                "explains the common residual left by SUR-068. It does not define a "
                "deployable policy surrogate.",
                "",
                metrics.to_markdown(index=False),
                "",
                "A pass licenses only a policy-to-shock identification audit. It does not "
                "license an endpoint correction or a static divergence penalty.",
                "",
            ]
        )
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories = collect_histories(args.output_dir, args.max_workers, args.refresh)
    shocks = run_shocks(histories)
    residuals = relaxation_residuals()
    evaluate(shocks, residuals, args.output_dir)


if __name__ == "__main__":
    main()
