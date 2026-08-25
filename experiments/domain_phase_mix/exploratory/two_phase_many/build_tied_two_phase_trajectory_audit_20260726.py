# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
#
# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
#   "tabulate",
#   "wandb",
# ]
# ///
"""Audit tied versus aggregate-matched two-phase learning trajectories.

Fixed-distribution Uncheatable evaluation BPB is the primary comparison.
On-policy training loss is included as a descriptive companion: it is not
cross-policy comparable when the sampled mixture changes at the phase boundary.
"""

from __future__ import annotations

import argparse
import concurrent.futures as futures
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "tied_two_phase_trajectory_audit_20260726"

PACKET = REFERENCE_OUTPUTS / "two_phase_surrogate_collaborator_packet_20260721"
CANONICAL = PACKET / "data" / "canonical"
LEGACY_300M = PACKET / "data" / "raw" / "legacy_300m_packet_data"
RAW_300M_REGISTRY = SCRIPT_DIR / "metric_registry/raw_metric_matrix_300m/raw_metric_matrix_300m.csv"
DELPHI_TWO = REFERENCE_OUTPUTS / "delphi_augmented_swarm_3e18_20260714"
DELPHI_ONE = REFERENCE_OUTPUTS / "delphi_one_phase_augmented_swarm_3e18_20260715"
DELPHI_HELDOUT = REFERENCE_OUTPUTS / "delphi_3e18_append_only_heldouts_20260714"

WANDB_PATH = "marin-community/marin"
WANDB_RUN_BASE_URL = f"https://wandb.ai/{WANDB_PATH}/runs"
EVAL_METRIC = "eval/uncheatable_eval/bpb"
TRAIN_METRIC = "train/loss"
EVAL_HISTORY_SAMPLES = 10_000
TRAIN_HISTORY_SAMPLES = 2_000
TRAIN_LOSS_EWMA_SPAN = 31
SIMILAR_ENDPOINT_THRESHOLD = 0.003
PHASE_TV_TOLERANCE = 1e-10
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20260726
COLORS = {
    "one_phase": "#e56b35",
    "two_phase": "#17384d",
    "frontier": "#1a9850",
    "middle": "#fee08b",
    "poor": "#d73027",
}
STRATUM_ORDER = ("frontier", "middle", "poor")


@dataclass(frozen=True)
class ScaleSpec:
    key: str
    label: str
    phase_boundary: float
    expected_pairs: int


SCALES = (
    ScaleSpec("300m", "300M / 6B tokens", 0.80, 238),
    ScaleSpec("delphi_3e18", "Delphi 3e18 / 1.576B tokens", 2400 / 3006, 238),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--max-workers", type=int, default=16)
    return parser.parse_args()


def phase_weight_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    phase0 = [column for column in frame.columns if column.startswith("phase_0_weight::")]
    phase1 = [column for column in frame.columns if column.startswith("phase_1_weight::")]
    if [column.split("::", 1)[1] for column in phase0] != [column.split("::", 1)[1] for column in phase1]:
        raise ValueError("Phase weight columns are misaligned")
    return phase0, phase1


def phase_tv(frame: pd.DataFrame) -> np.ndarray:
    phase0, phase1 = phase_weight_columns(frame)
    return 0.5 * np.abs(frame[phase1].to_numpy(float) - frame[phase0].to_numpy(float)).sum(axis=1)


def endpoint_strata(endpoint_mean: pd.Series) -> pd.Series:
    ranks = endpoint_mean.rank(method="first", ascending=True)
    return pd.qcut(ranks, q=3, labels=STRATUM_ORDER).astype("string")


def load_300m_pairs() -> pd.DataFrame:
    all_rows = pd.read_csv(LEGACY_300M / "all_300m_checkpoint_metrics.csv", low_memory=False)
    two = all_rows.loc[
        all_rows["training_phase_family"].eq("two_phase")
        & all_rows["phase_max_abs_delta"].fillna(0.0).gt(PHASE_TV_TOLERANCE)
    ].copy()
    one = all_rows.loc[all_rows["training_phase_family"].eq("single_phase")].copy()

    registry = pd.read_csv(RAW_300M_REGISTRY, low_memory=False)
    registry = registry.loc[registry["source_experiment"].eq("pinlin_calvin_xu/data_mixture/ngd3dm2_qsplit240_300m_6b")]
    registry = registry[["run_name", "wandb_run_id", "data_seed"]].drop_duplicates("run_name")
    two = two.merge(registry, on="run_name", how="left", validate="one_to_one")

    pairs = two.merge(
        one,
        on="phase_correspondence_key",
        suffixes=("_two", "_one"),
        validate="one_to_one",
    )
    pairs["scale_key"] = "300m"
    pairs["pair_id"] = pairs["phase_correspondence_key"]
    pairs["two_run_id"] = pairs["wandb_run_id"]
    pairs["one_run_id"] = pairs["training_wandb_id_one"]
    pairs["endpoint_two"] = pairs["eval_uncheatable_eval_bpb_two"]
    pairs["endpoint_one"] = pairs["eval_uncheatable_eval_bpb_one"]

    canonical_two = pd.read_csv(CANONICAL / "300m_two_phase_fit.csv")
    canonical_two["phase_tv"] = phase_tv(canonical_two)
    phase_tv_by_id = canonical_two.set_index("row_id")["phase_tv"]
    pairs["phase_tv"] = pairs["run_name_two"].map(phase_tv_by_id)
    pairs["data_seed"] = pairs["data_seed"]
    return pairs[
        [
            "scale_key",
            "pair_id",
            "one_run_id",
            "two_run_id",
            "data_seed",
            "endpoint_one",
            "endpoint_two",
            "phase_tv",
        ]
    ].reset_index(drop=True)


def load_delphi_pairs() -> pd.DataFrame:
    manifest = pd.read_csv(DELPHI_ONE / "training_manifest.csv")
    manifest = manifest.loc[manifest["disposition"].eq("scheduled_new_training")].copy()
    heldout = pd.read_csv(DELPHI_HELDOUT / "heldout_current.csv", low_memory=False)
    one_runs = (
        heldout.loc[
            heldout["training_series"].astype(str).str.contains("one.phase.augmented", case=False, regex=True, na=False)
        ]
        .dropna(subset=["wandb_run_base", "wandb_run_id"])
        .drop_duplicates("wandb_run_base", keep="last")
        .set_index("wandb_run_base")["wandb_run_id"]
    )
    two_wide = pd.read_csv(DELPHI_TWO / "delphi_augmented_swarm_3e18_wide.csv", low_memory=False)
    two_runs = two_wide.set_index("run_name")["training_wandb_run_id"]

    canonical_one = pd.read_csv(CANONICAL / "delphi_3e18_one_phase_fit.csv")
    canonical_two = pd.read_csv(CANONICAL / "delphi_3e18_two_phase_fit.csv")
    canonical_two["phase_tv"] = phase_tv(canonical_two)
    one_endpoint = canonical_one.set_index("row_id")["uncheatable_bpb"]
    two_endpoint = canonical_two.set_index("row_id")["uncheatable_bpb"]
    two_phase_tv = canonical_two.set_index("row_id")["phase_tv"]

    pairs = pd.DataFrame(
        {
            "scale_key": "delphi_3e18",
            "pair_id": manifest["source_run_name"].astype(str),
            "one_run_id": manifest["run_name"].map(one_runs),
            "two_run_id": manifest["source_run_name"].map(two_runs),
            "data_seed": manifest["data_seed"],
            "endpoint_one": manifest["run_name"].map(one_endpoint),
            "endpoint_two": manifest["source_run_name"].map(two_endpoint),
            "phase_tv": manifest["source_run_name"].map(two_phase_tv),
        }
    )
    return pairs.reset_index(drop=True)


def load_pairs() -> pd.DataFrame:
    pairs = pd.concat([load_300m_pairs(), load_delphi_pairs()], ignore_index=True)
    pairs["endpoint_delta"] = pairs["endpoint_two"] - pairs["endpoint_one"]
    pairs["endpoint_mean"] = 0.5 * (pairs["endpoint_two"] + pairs["endpoint_one"])
    pairs["similar_endpoint"] = pairs["endpoint_delta"].abs().le(SIMILAR_ENDPOINT_THRESHOLD)
    pairs["endpoint_stratum"] = pairs.groupby("scale_key", group_keys=False)["endpoint_mean"].apply(endpoint_strata)
    pairs["phase_boundary"] = pairs["scale_key"].map({spec.key: spec.phase_boundary for spec in SCALES})

    for spec in SCALES:
        block = pairs.loc[pairs["scale_key"].eq(spec.key)]
        if len(block) != spec.expected_pairs:
            raise ValueError(f"{spec.key}: expected {spec.expected_pairs} pairs, found {len(block)}")
        required = ["one_run_id", "two_run_id", "endpoint_one", "endpoint_two", "phase_tv", "data_seed"]
        if block[required].isna().any().any():
            missing = block.loc[block[required].isna().any(axis=1), ["pair_id", *required]]
            raise ValueError(f"{spec.key}: incomplete pair rows:\n{missing.head()}")
    return pairs


def fetch_run_history(
    run_id: str,
    pair_id: str,
    scale_key: str,
    policy_class: str,
    metric: str,
    samples: int,
) -> pd.DataFrame:
    api = wandb.Api(timeout=90)
    run = api.run(f"{WANDB_PATH}/{run_id}")
    history = run.history(
        keys=["global_step", "run_progress", metric],
        samples=samples,
        pandas=True,
    )
    if history.empty or metric not in history:
        return pd.DataFrame()
    history = history.loc[history[metric].notna(), ["global_step", "run_progress", metric]].copy()
    history["scale_key"] = scale_key
    history["pair_id"] = pair_id
    history["policy_class"] = policy_class
    history["wandb_run_id"] = run.id
    history["wandb_run_name"] = run.name
    history["wandb_data_seed"] = run.config.get("data_seed")
    return history


def fetch_histories(
    pairs: pd.DataFrame,
    output_path: Path,
    refresh: bool,
    max_workers: int,
    metric: str,
    samples: int,
) -> pd.DataFrame:
    if output_path.exists() and not refresh:
        return pd.read_csv(output_path)

    requests: list[tuple[str, str, str, str]] = []
    for row in pairs.itertuples(index=False):
        requests.append((str(row.one_run_id), str(row.pair_id), str(row.scale_key), "one_phase"))
        requests.append((str(row.two_run_id), str(row.pair_id), str(row.scale_key), "two_phase"))

    frames: list[pd.DataFrame] = []
    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        pending = {
            executor.submit(
                fetch_run_history,
                run_id,
                pair_id,
                scale_key,
                policy_class,
                metric,
                samples,
            ): (
                run_id,
                pair_id,
                policy_class,
            )
            for run_id, pair_id, scale_key, policy_class in requests
        }
        for index, future in enumerate(futures.as_completed(pending), start=1):
            run_id, pair_id, policy_class = pending[future]
            try:
                frame = future.result()
            except Exception as error:
                print(f"warning: {run_id} ({pair_id}/{policy_class}): {type(error).__name__}: {error}", flush=True)
                continue
            if not frame.empty:
                frames.append(frame)
            if index % 100 == 0 or index == len(requests):
                print(f"Fetched {index}/{len(requests)} {metric} histories", flush=True)

    if not frames:
        raise RuntimeError(f"No W&B histories were fetched for {metric}")
    histories = pd.concat(frames, ignore_index=True)
    histories.to_csv(output_path, index=False)
    return histories


def paired_trajectories(pairs: pd.DataFrame, histories: pd.DataFrame) -> pd.DataFrame:
    one = histories.loc[histories["policy_class"].eq("one_phase")].copy()
    two = histories.loc[histories["policy_class"].eq("two_phase")].copy()
    keys = ["scale_key", "pair_id", "global_step"]
    columns = [*keys, "run_progress", EVAL_METRIC, "wandb_data_seed"]
    merged = two[columns].merge(
        one[columns],
        on=keys,
        suffixes=("_two", "_one"),
        validate="one_to_one",
    )
    merged["run_progress"] = 0.5 * (merged["run_progress_two"] + merged["run_progress_one"])
    merged["bpb_two"] = merged[f"{EVAL_METRIC}_two"]
    merged["bpb_one"] = merged[f"{EVAL_METRIC}_one"]
    merged["trajectory_delta"] = merged["bpb_two"] - merged["bpb_one"]
    merged = merged.merge(pairs, on=["scale_key", "pair_id"], how="inner", validate="many_to_one")

    comparable_seeds = merged["wandb_data_seed_one"].notna() & merged["wandb_data_seed_two"].notna()
    if comparable_seeds.any():
        mismatch = merged.loc[
            comparable_seeds
            & (merged["wandb_data_seed_one"].astype(float) != merged["wandb_data_seed_two"].astype(float))
        ]
        if not mismatch.empty:
            raise ValueError(f"W&B data-seed mismatch in {mismatch[['scale_key', 'pair_id']].drop_duplicates().head()}")
    return merged.sort_values(["scale_key", "pair_id", "global_step"]).reset_index(drop=True)


def trapezoid_mean(progress: np.ndarray, values: np.ndarray) -> float:
    if len(progress) < 2 or np.isclose(progress[-1], progress[0]):
        return float("nan")
    return float(np.trapezoid(values, progress) / (progress[-1] - progress[0]))


def pair_summaries(trajectories: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (scale_key, pair_id), block in trajectories.groupby(["scale_key", "pair_id"], sort=True):
        block = block.sort_values("run_progress")
        boundary = float(block["phase_boundary"].iloc[0])
        pre = block.loc[block["run_progress"].lt(boundary)]
        post = block.loc[block["run_progress"].ge(boundary)]
        if pre.empty or post.empty:
            continue
        rows.append(
            {
                "scale_key": scale_key,
                "pair_id": pair_id,
                "endpoint_one": float(block["endpoint_one"].iloc[0]),
                "endpoint_two": float(block["endpoint_two"].iloc[0]),
                "endpoint_delta": float(block["endpoint_delta"].iloc[0]),
                "endpoint_mean": float(block["endpoint_mean"].iloc[0]),
                "endpoint_stratum": str(block["endpoint_stratum"].iloc[0]),
                "similar_endpoint": bool(block["similar_endpoint"].iloc[0]),
                "phase_tv": float(block["phase_tv"].iloc[0]),
                "n_common_evals": len(block),
                "first_progress": float(block["run_progress"].iloc[0]),
                "last_pre_progress": float(pre["run_progress"].iloc[-1]),
                "first_post_progress": float(post["run_progress"].iloc[0]),
                "last_pre_delta": float(pre["trajectory_delta"].iloc[-1]),
                "first_post_delta": float(post["trajectory_delta"].iloc[0]),
                "boundary_catchup": float(post["trajectory_delta"].iloc[0] - pre["trajectory_delta"].iloc[-1]),
                "endpoint_catchup": float(block["endpoint_delta"].iloc[0] - pre["trajectory_delta"].iloc[-1]),
                "mean_pre_delta": trapezoid_mean(
                    pre["run_progress"].to_numpy(float),
                    pre["trajectory_delta"].to_numpy(float),
                ),
                "mean_post_delta": trapezoid_mean(
                    post["run_progress"].to_numpy(float),
                    post["trajectory_delta"].to_numpy(float),
                ),
                "mean_full_delta": trapezoid_mean(
                    block["run_progress"].to_numpy(float),
                    block["trajectory_delta"].to_numpy(float),
                ),
            }
        )
    return pd.DataFrame(rows)


def bootstrap_mean(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float, float]:
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    draws = rng.choice(values, size=(BOOTSTRAP_REPLICATES, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def summary_table(pair_summary: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    rows: list[dict[str, object]] = []
    for scale_key in pair_summary["scale_key"].unique():
        scale = pair_summary.loc[pair_summary["scale_key"].eq(scale_key)]
        for scope, block in (
            ("all asymmetric pairs", scale),
            (f"similar endpoint <= {SIMILAR_ENDPOINT_THRESHOLD:.3f}", scale.loc[scale["similar_endpoint"]]),
        ):
            row: dict[str, object] = {
                "scale_key": scale_key,
                "scope": scope,
                "n": len(block),
                "fraction_two_better_endpoint": float(block["endpoint_delta"].lt(0.0).mean()),
                "fraction_two_worse_before_switch": float(block["last_pre_delta"].gt(0.0).mean()),
                "fraction_catches_up_after_switch": float(block["endpoint_catchup"].lt(0.0).mean()),
            }
            for metric in (
                "last_pre_delta",
                "first_post_delta",
                "endpoint_delta",
                "endpoint_catchup",
                "mean_full_delta",
            ):
                mean, low, high = bootstrap_mean(block[metric].to_numpy(float), rng)
                row[metric] = mean
                row[f"{metric}_ci_low"] = low
                row[f"{metric}_ci_high"] = high
            rows.append(row)
    return pd.DataFrame(rows)


def bootstrap_trajectory_table(trajectories: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    rows: list[dict[str, object]] = []
    similar = trajectories.loc[trajectories["similar_endpoint"]].copy()
    for keys, block in similar.groupby(
        ["scale_key", "endpoint_stratum", "global_step", "run_progress"],
        sort=True,
        observed=True,
    ):
        mean, low, high = bootstrap_mean(block["trajectory_delta"].to_numpy(float), rng)
        rows.append(
            {
                "scale_key": keys[0],
                "endpoint_stratum": keys[1],
                "global_step": keys[2],
                "run_progress": keys[3],
                "n": block["pair_id"].nunique(),
                "mean_delta": mean,
                "ci_low": low,
                "ci_high": high,
            }
        )
    return pd.DataFrame(rows)


def representative_pairs(pair_summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[pd.Series] = []
    for scale_key in pair_summary["scale_key"].unique():
        scale = pair_summary.loc[pair_summary["scale_key"].eq(scale_key) & pair_summary["similar_endpoint"]]
        for stratum in STRATUM_ORDER:
            block = scale.loc[scale["endpoint_stratum"].eq(stratum)].copy()
            if block.empty:
                continue
            closest = block.loc[block["endpoint_delta"].abs().idxmin()]
            rows.append(closest)
            remaining = block.loc[block["pair_id"].ne(closest["pair_id"])]
            if not remaining.empty:
                rows.append(remaining.loc[remaining["phase_tv"].idxmax()])
    if not rows:
        raise RuntimeError("No representative pairs were selected")
    return pd.DataFrame(rows).drop_duplicates(["scale_key", "pair_id"]).reset_index(drop=True)


def common_layout(figure: go.Figure, title: str, height: int) -> None:
    figure.update_layout(
        title=title,
        height=height,
        template="plotly_white",
        font={"family": "IBM Plex Sans, sans-serif", "color": "#17384d"},
        paper_bgcolor="#faf8f1",
        plot_bgcolor="#faf8f1",
        margin={"l": 70, "r": 30, "t": 100, "b": 60},
        hovermode="x unified",
    )


def pair_subplot_title(row: object) -> str:
    return (
        f"{row.pair_id} | {row.endpoint_stratum} | end delta {row.endpoint_delta:+.4f}"
        f"<br><a href='{row.one_run_url}' target='_blank'>Tied W&B</a>"
        f" · <a href='{row.two_run_url}' target='_blank'>Two-phase W&B</a>"
    )


def plot_representative_curves(
    trajectories: pd.DataFrame,
    training_histories: pd.DataFrame,
    selected: pd.DataFrame,
    output_path: Path,
) -> None:
    html_parts = [
        "<!doctype html><html><head><meta charset='utf-8'><title>Tied versus two-phase trajectories</title>",
        "<style>body{margin:0;background:#faf8f1;color:#17384d;font-family:IBM Plex Sans,sans-serif}"
        "main{max-width:1500px;margin:auto;padding:36px}h1,h2{font-family:Georgia,serif}"
        ".note{max-width:1000px;color:#546774;line-height:1.5}</style></head><body><main>",
        "<h1>Aggregate-matched tied versus two-phase learning curves</h1>",
        "<p class='note'>Each pair has the same aggregate mixture and data seed. Curves use a fixed Uncheatable "
        "evaluation distribution. The dashed line marks the 80% phase boundary. Panels were selected before "
        "looking at trajectories: within each endpoint-quality tercile, the closest endpoint match and the "
        "highest phase-TV match under |endpoint difference| <= 0.003 BPB.</p>",
        "<p class='note'><strong>Training-loss caveat:</strong> the companion training curves are on-policy. "
        "Tied and two-phase runs sample different mixtures, and the two-phase sampled-data distribution changes "
        "at the boundary. A loss jump can therefore reflect batch difficulty rather than a change in model "
        "quality. Training loss is shown for diagnosis only; fixed-distribution BPB remains the comparable "
        "trajectory.</p>",
        "<p class='note'>Each panel title links to the exact tied and two-phase W&B runs. W&B does not document "
        "a stable URL parameter for selecting an arbitrary run pair in one workspace, so the page uses two "
        "durable direct run links instead of a fragile saved-workspace URL.</p>",
    ]
    include_plotly = "cdn"
    for spec in SCALES:
        scale_selected = selected.loc[selected["scale_key"].eq(spec.key)].copy()
        rows = max(1, int(np.ceil(len(scale_selected) / 2)))
        titles = [pair_subplot_title(row) for row in scale_selected.itertuples(index=False)]
        figure = make_subplots(rows=rows, cols=2, subplot_titles=titles, horizontal_spacing=0.08)
        for index, row in enumerate(scale_selected.itertuples(index=False)):
            subplot_row = index // 2 + 1
            subplot_col = index % 2 + 1
            block = trajectories.loc[
                trajectories["scale_key"].eq(spec.key) & trajectories["pair_id"].eq(row.pair_id)
            ].sort_values("run_progress")
            for policy_class, label in (("one_phase", "Tied"), ("two_phase", "Two phase")):
                y = block["bpb_one"] if policy_class == "one_phase" else block["bpb_two"]
                figure.add_trace(
                    go.Scatter(
                        x=block["run_progress"],
                        y=y,
                        mode="lines+markers",
                        name=label,
                        legendgroup=policy_class,
                        showlegend=index == 0,
                        line={"color": COLORS[policy_class], "width": 2.5},
                        marker={"size": 6},
                        customdata=np.column_stack(
                            [
                                block["global_step"],
                                np.repeat(row.phase_tv, len(block)),
                            ]
                        ),
                        hovertemplate=(
                            f"{label}<br>progress %{{x:.3f}}<br>BPB %{{y:.6f}}"
                            "<br>step %{customdata[0]:.0f}<br>phase TV %{customdata[1]:.3f}<extra></extra>"
                        ),
                    ),
                    row=subplot_row,
                    col=subplot_col,
                )
            figure.add_vline(
                x=spec.phase_boundary,
                line_dash="dash",
                line_color="#8c9aa2",
                row=subplot_row,
                col=subplot_col,
            )
            figure.update_xaxes(title_text="Training progress", row=subplot_row, col=subplot_col)
            figure.update_yaxes(title_text="Uncheatable BPB", row=subplot_row, col=subplot_col)
        common_layout(figure, f"{spec.label}: fixed-distribution Uncheatable BPB", 360 * rows + 100)
        html_parts.append(f"<h2>{spec.label}</h2>")
        html_parts.append(figure.to_html(full_html=False, include_plotlyjs=include_plotly))
        include_plotly = False

        training_figure = make_subplots(
            rows=rows,
            cols=2,
            subplot_titles=titles,
            horizontal_spacing=0.08,
        )
        for index, row in enumerate(scale_selected.itertuples(index=False)):
            subplot_row = index // 2 + 1
            subplot_col = index % 2 + 1
            for policy_class, label in (("one_phase", "Tied"), ("two_phase", "Two phase")):
                block = training_histories.loc[
                    training_histories["scale_key"].eq(spec.key)
                    & training_histories["pair_id"].eq(row.pair_id)
                    & training_histories["policy_class"].eq(policy_class)
                ].sort_values("run_progress")
                if block.empty:
                    continue
                smoothed = block[TRAIN_METRIC].ewm(span=TRAIN_LOSS_EWMA_SPAN, adjust=False).mean()
                training_figure.add_trace(
                    go.Scatter(
                        x=block["run_progress"],
                        y=block[TRAIN_METRIC],
                        mode="lines",
                        legendgroup=f"{policy_class}_raw",
                        showlegend=False,
                        line={"color": COLORS[policy_class], "width": 0.8},
                        opacity=0.18,
                        hoverinfo="skip",
                    ),
                    row=subplot_row,
                    col=subplot_col,
                )
                training_figure.add_trace(
                    go.Scatter(
                        x=block["run_progress"],
                        y=smoothed,
                        mode="lines",
                        name=f"{label} EWMA",
                        legendgroup=policy_class,
                        showlegend=index == 0,
                        line={"color": COLORS[policy_class], "width": 2.5},
                        customdata=np.column_stack(
                            [
                                block["global_step"],
                                block[TRAIN_METRIC],
                            ]
                        ),
                        hovertemplate=(
                            f"{label}<br>progress %{{x:.3f}}<br>{TRAIN_LOSS_EWMA_SPAN}-point EWMA %{{y:.5f}}"
                            "<br>raw sampled loss %{customdata[1]:.5f}"
                            "<br>step %{customdata[0]:.0f}<extra></extra>"
                        ),
                    ),
                    row=subplot_row,
                    col=subplot_col,
                )
            training_figure.add_vline(
                x=spec.phase_boundary,
                line_dash="dash",
                line_color="#8c9aa2",
                row=subplot_row,
                col=subplot_col,
            )
            training_figure.update_xaxes(title_text="Training progress", row=subplot_row, col=subplot_col)
            training_figure.update_yaxes(title_text="On-policy training loss", row=subplot_row, col=subplot_col)
        common_layout(
            training_figure,
            (
                f"{spec.label}: on-policy training loss "
                f"(raw traces + {TRAIN_LOSS_EWMA_SPAN}-point EWMA; descriptive only)"
            ),
            360 * rows + 100,
        )
        html_parts.append(training_figure.to_html(full_html=False, include_plotlyjs=False))
    html_parts.append("</main></body></html>")
    output_path.write_text("\n".join(html_parts))


def plot_mean_delta(bootstrap: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(
        rows=2,
        cols=3,
        subplot_titles=[f"{spec.label} - {stratum}" for spec in SCALES for stratum in STRATUM_ORDER],
        horizontal_spacing=0.06,
        vertical_spacing=0.16,
    )
    for scale_index, spec in enumerate(SCALES, start=1):
        for stratum_index, stratum in enumerate(STRATUM_ORDER, start=1):
            block = bootstrap.loc[
                bootstrap["scale_key"].eq(spec.key) & bootstrap["endpoint_stratum"].eq(stratum)
            ].sort_values("run_progress")
            if block.empty:
                continue
            x = block["run_progress"].to_numpy(float)
            figure.add_trace(
                go.Scatter(
                    x=np.concatenate([x, x[::-1]]),
                    y=np.concatenate([block["ci_high"], block["ci_low"][::-1]]),
                    fill="toself",
                    fillcolor="rgba(23,56,77,0.14)",
                    line={"color": "rgba(0,0,0,0)"},
                    hoverinfo="skip",
                    showlegend=False,
                ),
                row=scale_index,
                col=stratum_index,
            )
            figure.add_trace(
                go.Scatter(
                    x=x,
                    y=block["mean_delta"],
                    mode="lines+markers",
                    line={"color": COLORS[stratum], "width": 2.5},
                    marker={"size": 6},
                    name=stratum,
                    showlegend=scale_index == 1,
                    customdata=block["n"],
                    hovertemplate=(
                        "progress %{x:.3f}<br>mean two - tied %{y:+.6f} BPB" "<br>pairs %{customdata}<extra></extra>"
                    ),
                ),
                row=scale_index,
                col=stratum_index,
            )
            figure.add_hline(y=0.0, line_color="#17384d", line_width=1, row=scale_index, col=stratum_index)
            figure.add_vline(
                x=spec.phase_boundary,
                line_dash="dash",
                line_color="#8c9aa2",
                row=scale_index,
                col=stratum_index,
            )
            figure.update_xaxes(title_text="Training progress", row=scale_index, col=stratum_index)
            if stratum_index == 1:
                figure.update_yaxes(title_text="Two phase - tied BPB", row=scale_index, col=stratum_index)
    common_layout(
        figure,
        "Mean relative trajectory among endpoint-matched pairs (95% pair-bootstrap intervals)",
        900,
    )
    figure.write_html(output_path, include_plotlyjs="cdn")


def plot_catchup(pair_summary: pd.DataFrame, output_path: Path) -> None:
    figure = make_subplots(rows=1, cols=2, subplot_titles=[spec.label for spec in SCALES])
    for column, spec in enumerate(SCALES, start=1):
        block = pair_summary.loc[pair_summary["scale_key"].eq(spec.key)].copy()
        limit = float(
            np.nanmax(
                np.abs(
                    np.concatenate(
                        [
                            block["last_pre_delta"].to_numpy(float),
                            block["endpoint_delta"].to_numpy(float),
                        ]
                    )
                )
            )
        )
        figure.add_trace(
            go.Scatter(
                x=[-limit, limit],
                y=[-limit, limit],
                mode="lines",
                line={"color": "#8c9aa2", "dash": "dash"},
                showlegend=False,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
        for stratum in STRATUM_ORDER:
            stratum_block = block.loc[block["endpoint_stratum"].eq(stratum)]
            figure.add_trace(
                go.Scatter(
                    x=stratum_block["last_pre_delta"],
                    y=stratum_block["endpoint_delta"],
                    mode="markers",
                    name=stratum,
                    legendgroup=stratum,
                    showlegend=column == 1,
                    marker={
                        "size": np.where(stratum_block["similar_endpoint"], 9, 6),
                        "color": COLORS[stratum],
                        "opacity": np.where(stratum_block["similar_endpoint"], 0.95, 0.42),
                        "line": {
                            "color": np.where(stratum_block["similar_endpoint"], "#17384d", COLORS[stratum]),
                            "width": np.where(stratum_block["similar_endpoint"], 1.2, 0.0),
                        },
                    },
                    customdata=np.column_stack(
                        [
                            stratum_block["pair_id"],
                            stratum_block["phase_tv"],
                            stratum_block["endpoint_catchup"],
                        ]
                    ),
                    hovertemplate=(
                        "%{customdata[0]}<br>last pre-switch delta %{x:+.6f}"
                        "<br>endpoint delta %{y:+.6f}<br>catch-up %{customdata[2]:+.6f}"
                        "<br>phase TV %{customdata[1]:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.add_hline(y=0.0, line_color="#17384d", line_width=1, row=1, col=column)
        figure.add_vline(x=0.0, line_color="#17384d", line_width=1, row=1, col=column)
        figure.update_xaxes(title_text="Two phase - tied BPB at last pre-switch eval", row=1, col=column)
        figure.update_yaxes(title_text="Two phase - tied endpoint BPB", row=1, col=column)
    common_layout(
        figure,
        "Does the two-phase policy catch up after the 80% switch? (below diagonal = catch-up)",
        650,
    )
    figure.write_html(output_path, include_plotlyjs="cdn")


def format_interval(row: pd.Series, metric: str) -> str:
    return f"{row[metric]:+.6f} " f"[{row[f'{metric}_ci_low']:+.6f}, {row[f'{metric}_ci_high']:+.6f}]"


def write_report(
    pairs: pd.DataFrame,
    trajectories: pd.DataFrame,
    pair_summary: pd.DataFrame,
    summary: pd.DataFrame,
    selected: pd.DataFrame,
    output_dir: Path,
) -> None:
    report_table = summary[
        [
            "scale_key",
            "scope",
            "n",
            "fraction_two_worse_before_switch",
            "fraction_two_better_endpoint",
            "fraction_catches_up_after_switch",
        ]
    ].copy()
    for metric in ("last_pre_delta", "endpoint_delta", "endpoint_catchup", "mean_full_delta"):
        report_table[metric] = summary.apply(
            lambda row, metric=metric: format_interval(row, metric),
            axis=1,
        )

    history_counts = (
        trajectories.groupby("scale_key")
        .agg(
            pairs_with_any_overlap=("pair_id", "nunique"),
            common_eval_rows=("global_step", "size"),
            min_common_evals=("pair_id", lambda values: int(values.value_counts().min())),
            max_common_evals=("pair_id", lambda values: int(values.value_counts().max())),
        )
        .reset_index()
    )
    analyzable_counts = pair_summary.groupby("scale_key").size().rename("pairs_with_pre_switch_comparison").reset_index()
    history_counts = history_counts.merge(analyzable_counts, on="scale_key", how="left")
    quality_table = (
        pair_summary.groupby(["scale_key", "endpoint_stratum"], observed=True)
        .agg(
            n=("pair_id", "size"),
            endpoint_delta=("endpoint_delta", "mean"),
            last_pre_delta=("last_pre_delta", "mean"),
            endpoint_catchup=("endpoint_catchup", "mean"),
            mean_full_delta=("mean_full_delta", "mean"),
            fraction_two_better_endpoint=("endpoint_delta", lambda values: float((values < 0).mean())),
        )
        .reset_index()
    )
    quality_table["endpoint_stratum"] = pd.Categorical(
        quality_table["endpoint_stratum"],
        categories=STRATUM_ORDER,
        ordered=True,
    )
    quality_table = quality_table.sort_values(["scale_key", "endpoint_stratum"])
    seed_check = (
        trajectories.groupby(["scale_key", "pair_id"])
        .agg(one_seed=("wandb_data_seed_one", "first"), two_seed=("wandb_data_seed_two", "first"))
        .reset_index()
    )
    known_seeds = seed_check["one_seed"].notna() & seed_check["two_seed"].notna()
    matched_seed_count = int(
        (
            seed_check.loc[known_seeds, "one_seed"].astype(float)
            == seed_check.loc[known_seeds, "two_seed"].astype(float)
        ).sum()
    )

    lines = [
        "# Aggregate-Matched Tied Versus Two-Phase Trajectory Audit",
        "",
        "## Question",
        "",
        "Does a small final gap between the tied policy \\(L(a,0)\\) and an aggregate-matched two-phase policy "
        "\\(L(a,d)\\) conceal a meaningful difference in the optimization trajectory?",
        "",
        "This is a direct empirical check of one limited implication drawn from Hacohen and Weinshall (2019): "
        "their idealized curriculum can steepen an optimization landscape while retaining the same global optimum. "
        "The theorem does **not** assert that Marin's schedules share an asymptotic optimum, and this finite-budget "
        "audit cannot establish that claim.",
        "",
        "## Design",
        "",
        "- Every analyzed pair has the same token-weighted aggregate mixture \\(a\\); only phase contrast "
        "\\(d\\) changes.",
        "- The one- and two-phase runs use the same data seed. The 3e18 panel has 238 independently trained pairs; "
        "42 exact phase-tied aliases are excluded.",
        "- The primary curves use `eval/uncheatable_eval/bpb`, evaluated on a fixed distribution. Companion "
        "`train/loss` curves are included only for diagnosis: they are on-policy, so their sampled-data "
        "distribution differs between policies and changes at the phase boundary.",
        f"- `Similar endpoint` is a descriptive slice with "
        f"\\(|L(a,d)-L(a,0)| \\le {SIMILAR_ENDPOINT_THRESHOLD:.3f}\\) BPB. Conditioning on the endpoint makes this "
        "slice unsuitable for an unqualified causal estimate; all-pair results are reported alongside it.",
        "- The relative curve is \\(\\Delta_t(a,d)=L_t(a,d)-L_t(a,0)\\). This differences out the large loss drop "
        "that both policies receive from the WSD decay.",
        "",
        "## Coverage",
        "",
        history_counts.to_markdown(index=False),
        "",
        f"W&B exposed matched data seeds for {matched_seed_count}/{int(known_seeds.sum())} checked pairs; local "
        "3e18 manifests independently assert all 238 seed matches.",
        "",
        "## Results",
        "",
        report_table.to_markdown(index=False),
        "",
        "Definitions: `last_pre_delta` is two-phase minus tied BPB at the final evaluation before the phase switch; "
        "`endpoint_catchup = endpoint_delta - last_pre_delta`, so a negative value means that the two-phase policy "
        "closed ground after the switch. Brackets are 95% pair-bootstrap intervals.",
        "",
        "Five Delphi tied runs expose only their endpoint in W&B. They remain in the endpoint coverage but are "
        "excluded from statistics requiring a pre-switch comparison, leaving 233 trajectory-analyzable pairs.",
        "",
        "### Endpoint-quality strata",
        "",
        "The following all-pair table uses endpoint-quality terciles within each scale. It is descriptive: a "
        "stratum is defined using the mean endpoint of the same pair whose trajectory is summarized.",
        "",
        quality_table.to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation",
        "",
        "The evidence distinguishes three claims:",
        "",
        "1. **Different trajectories despite similar endpoints:** supported if the endpoint-matched slice has a "
        "nonzero pre-switch delta or systematic post-switch catch-up while ending near zero.",
        "2. **Curriculum accelerates Marin optimization:** supported only if two phase is ahead for a substantial "
        "part of training or reaches a fixed BPB earlier. A late catch-up from an earlier deficit is a phase-order "
        "effect, but it is not the classical easy-to-hard acceleration claim.",
        "3. **The tied and two-phase policies have the same global minimum:** not identified. Both experiments stop "
        "at finite compute under WSD, and neither follows both policies to convergence under a common stationary "
        "objective.",
        "",
        "The 300M curves have roughly 23 common evaluations and resolve the trajectory. Delphi 3e18 logs only two "
        "pre-switch and two post-switch evaluations, with the post-switch points almost coincident at the endpoint; "
        "that panel can show a boundary-scale catch-up but not its detailed shape. Native Table-9 was evaluated only "
        "at the endpoint, so the trajectory claim is tested only on Uncheatable BPB.",
        "",
        "## Representative pairs",
        "",
        selected[
            [
                "scale_key",
                "pair_id",
                "endpoint_stratum",
                "phase_tv",
                "endpoint_one",
                "endpoint_two",
                "endpoint_delta",
                "last_pre_delta",
                "endpoint_catchup",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Artifacts",
        "",
        "- `matched_endpoint_learning_curves.html`: absolute paired fixed-distribution BPB and descriptive "
        "on-policy training-loss curves.",
        "- `mean_delta_trajectories.html`: mean relative trajectory by endpoint-quality stratum.",
        "- `phase_boundary_catchup.html`: last pre-switch difference versus endpoint difference.",
        "- `pair_manifest.csv`, `wandb_histories.csv`, `wandb_training_histories.csv`, "
        "`paired_trajectories.csv`, and `pair_summaries.csv`: re-runnable source tables.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "uv run experiments/domain_phase_mix/exploratory/two_phase_many/"
        "build_tied_two_phase_trajectory_audit_20260726.py",
        "```",
    ]
    (output_dir / "report.md").write_text("\n".join(lines) + "\n")

    provenance = {
        "wandb_path": WANDB_PATH,
        "eval_metric": EVAL_METRIC,
        "training_metric": TRAIN_METRIC,
        "training_history_samples_per_run": TRAIN_HISTORY_SAMPLES,
        "training_loss_ewma_span": TRAIN_LOSS_EWMA_SPAN,
        "similar_endpoint_threshold_bpb": SIMILAR_ENDPOINT_THRESHOLD,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "scales": [
            {
                "key": spec.key,
                "label": spec.label,
                "phase_boundary": spec.phase_boundary,
                "expected_pairs": spec.expected_pairs,
            }
            for spec in SCALES
        ],
        "important_limitations": [
            "Finite WSD trajectories cannot identify equality of asymptotic global optima.",
            "Native Table-9 has endpoint evaluations but no intermediate trajectory.",
            "The endpoint-matched slice is selected on the outcome and is descriptive.",
            "The 3e18 runs have sparse intermediate evaluation cadence.",
            "On-policy training loss is not cross-policy comparable because the sampled mixture differs.",
        ],
    }
    (output_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = load_pairs()
    pairs.to_csv(args.output_dir / "pair_manifest.csv", index=False)
    histories = fetch_histories(
        pairs,
        args.output_dir / "wandb_histories.csv",
        args.refresh,
        args.max_workers,
        EVAL_METRIC,
        EVAL_HISTORY_SAMPLES,
    )
    trajectories = paired_trajectories(pairs, histories)
    trajectories.to_csv(args.output_dir / "paired_trajectories.csv", index=False)
    pair_summary = pair_summaries(trajectories)
    pair_summary.to_csv(args.output_dir / "pair_summaries.csv", index=False)
    summary = summary_table(pair_summary)
    summary.to_csv(args.output_dir / "summary.csv", index=False)
    bootstrap = bootstrap_trajectory_table(trajectories)
    bootstrap.to_csv(args.output_dir / "mean_delta_trajectories.csv", index=False)
    selected = representative_pairs(pair_summary).merge(
        pairs[["scale_key", "pair_id", "one_run_id", "two_run_id"]],
        on=["scale_key", "pair_id"],
        how="left",
        validate="one_to_one",
    )
    selected["one_run_url"] = WANDB_RUN_BASE_URL + "/" + selected["one_run_id"].astype(str)
    selected["two_run_url"] = WANDB_RUN_BASE_URL + "/" + selected["two_run_id"].astype(str)
    selected.to_csv(args.output_dir / "representative_pairs.csv", index=False)
    selected_pairs = pairs.merge(
        selected[["scale_key", "pair_id"]],
        on=["scale_key", "pair_id"],
        how="inner",
        validate="one_to_one",
    )
    training_histories = fetch_histories(
        selected_pairs,
        args.output_dir / "wandb_training_histories.csv",
        args.refresh,
        args.max_workers,
        TRAIN_METRIC,
        TRAIN_HISTORY_SAMPLES,
    )

    plot_representative_curves(
        trajectories,
        training_histories,
        selected,
        args.output_dir / "matched_endpoint_learning_curves.html",
    )
    plot_mean_delta(bootstrap, args.output_dir / "mean_delta_trajectories.html")
    plot_catchup(pair_summary, args.output_dir / "phase_boundary_catchup.html")
    write_report(pairs, trajectories, pair_summary, summary, selected, args.output_dir)

    print(summary.to_string(index=False))
    print(f"Wrote {args.output_dir}")


if __name__ == "__main__":
    main()
