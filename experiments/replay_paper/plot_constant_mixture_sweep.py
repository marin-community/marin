# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "pandas",
#   "plotly",
# ]
# ///

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Refresh replay-paper constant-mixture sweep plots from GCS tracker metrics."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

EXPERIMENT_NAME: Final = "pinlin_calvin_xu/data_mixture/replay_constant_mixture_sweep_20260706_clean_wandb"
GCS_ROOT: Final = f"gs://marin-us-east5/checkpoints/{EXPERIMENT_NAME}"
OUTPUT_DIR: Final = (
    Path(__file__).resolve().parent.parent
    / "domain_phase_mix"
    / "exploratory"
    / "two_phase_many"
    / "reference_outputs"
    / "replay_constant_mixture_sweep_20260706"
)
TARGETS: Final = ("starcoder", "finemath", "flan")
NUM_TRAIN_STEPS: Final = 1024
TRAIN_BATCH_SIZE: Final = 1024
SEQ_LEN: Final = 4096
TRAIN_TOKENS: Final = NUM_TRAIN_STEPS * TRAIN_BATCH_SIZE * SEQ_LEN
RARE_FRACTION_DENOMINATOR: Final = 1024
# True token count from:
# TreeCache.load("gs://marin-us-east5/tokenized/dolma/c4-e0e5ec/train").store.tree["input_ids"].data_size
C4_TOKEN_COUNT: Final = 134_062_553_328
GCS_READ_WORKERS: Final = 16
TARGET_LOSS_METRICS: Final = {
    "starcoder": "eval/starcoder/loss",
    "finemath": "eval/finemath/loss",
    "flan": "eval/flan/loss",
}
TARGET_BPB_METRICS: Final = {
    "starcoder": "eval/starcoder/bpb",
    "finemath": "eval/finemath/bpb",
    "flan": "eval/flan/bpb",
}
PAPER_FIGURE7_BEST: Final = {
    "starcoder": {
        "loss": 3.03,
        "cell": "rho=0.5, alpha=1.0",
        "interpretation": "replay during target-only Stage 2; target data reserved for Stage 2",
    },
    "finemath": {
        "loss": 3.36,
        "cell": "rho=0.25-0.5, alpha=0.25 (rounded tie region)",
        "interpretation": "target introduced early; replay less critical in rounded best region",
    },
    "flan": {
        "loss": 3.29,
        "cell": "rho=0.5, alpha=0.25",
        "interpretation": "target introduced early plus replay",
    },
}
PLOT_CONFIG: Final = {
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


@dataclass(frozen=True)
class TrackerRow:
    target: str
    epochs: int
    status: str
    tracker_uri: str
    summary: dict[str, Any]


def _run_text(command: list[str]) -> str:
    return subprocess.check_output(command, text=True)


def _read_gcs_text(uri: str) -> str:
    return _run_text(["gsutil", "cat", uri])


def _try_read_gcs_text(uri: str) -> str | None:
    try:
        return _read_gcs_text(uri)
    except subprocess.CalledProcessError:
        return None


def _list_epoch_dirs(target: str) -> list[int]:
    output = _run_text(["gsutil", "ls", f"{GCS_ROOT}/{target}/"])
    epochs: list[int] = []
    for line in output.splitlines():
        name = line.rstrip("/").split("/")[-1]
        if name.startswith("epochs_"):
            epochs.append(int(name.removeprefix("epochs_")))
    return sorted(set(epochs))


def _read_tracker_row(target: str, epochs: int) -> TrackerRow | None:
    base = f"{GCS_ROOT}/{target}/epochs_{epochs:02d}"
    status_uri = f"{base}/.executor_status"
    status_text = _try_read_gcs_text(status_uri)
    if status_text is None:
        return None
    status = status_text.strip()
    if status != "SUCCESS":
        return TrackerRow(
            target=target, epochs=epochs, status=status, tracker_uri=f"{base}/tracker_metrics.jsonl", summary={}
        )

    tracker_uri = f"{base}/tracker_metrics.jsonl"
    tracker_text = _try_read_gcs_text(tracker_uri)
    if tracker_text is None:
        return TrackerRow(target=target, epochs=epochs, status="missing_tracker", tracker_uri=tracker_uri, summary={})

    summary: dict[str, Any] = {}
    for line in tracker_text.splitlines():
        record = json.loads(line)
        if isinstance(record.get("summary"), dict):
            summary = record["summary"]
    status = "has_eval" if summary else "missing_summary"
    return TrackerRow(target=target, epochs=epochs, status=status, tracker_uri=tracker_uri, summary=summary)


def _collect_rows() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    all_metric_keys: set[str] = set()
    tracker_rows: list[TrackerRow] = []
    row_keys: list[tuple[str, int]] = []
    for target in TARGETS:
        for epochs in _list_epoch_dirs(target):
            row_keys.append((target, epochs))

    with ThreadPoolExecutor(max_workers=GCS_READ_WORKERS) as executor:
        for row in executor.map(lambda key: _read_tracker_row(*key), row_keys):
            if row is None:
                continue
            tracker_rows.append(row)
            all_metric_keys.update(key for key, value in row.summary.items() if isinstance(value, int | float))

    for row in sorted(tracker_rows, key=lambda tracker: (tracker.target, tracker.epochs)):
        rare_weight = row.epochs / RARE_FRACTION_DENOMINATOR
        common_weight = 1.0 - rare_weight
        broad_tokens = common_weight * TRAIN_TOKENS
        record: dict[str, Any] = {
            "target": row.target,
            "epochs": row.epochs,
            "rare_epochs": row.epochs,
            "rare_weight": rare_weight,
            "common_weight": common_weight,
            "broad_tokens": broad_tokens,
            "broad_epochs": broad_tokens / C4_TOKEN_COUNT,
            "status": row.status,
            "tracker_uri": row.tracker_uri,
        }
        for key in sorted(all_metric_keys):
            if key in row.summary:
                record[key] = row.summary[key]
        rows.append(record)
    frame = pd.DataFrame(rows).sort_values(["target", "epochs"]).reset_index(drop=True)
    return frame


def _target_loss_frame(metrics: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    complete = metrics[metrics["status"] == "has_eval"].copy()
    for _, row in complete.iterrows():
        target = str(row["target"])
        target_loss_metric = TARGET_LOSS_METRICS[target]
        target_bpb_metric = TARGET_BPB_METRICS[target]
        if pd.isna(row.get(target_loss_metric)):
            continue
        paper = PAPER_FIGURE7_BEST[target]
        rows.append(
            {
                "target": target,
                "epochs": int(row["epochs"]),
                "rare_epochs": float(row["rare_epochs"]),
                "rare_weight": float(row["rare_weight"]),
                "common_weight": float(row["common_weight"]),
                "broad_tokens": float(row["broad_tokens"]),
                "broad_epochs": float(row["broad_epochs"]),
                "target_loss": float(row[target_loss_metric]),
                "target_bpb": float(row[target_bpb_metric]),
                "c4_loss": float(row["eval/c4/loss"]),
                "macro_loss": float(row["eval/macro_loss"]),
                "macro_bpb": float(row["eval/macro_bpb"]),
                "paper_figure7_best_schedule_loss_rounded": float(paper["loss"]),
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "epochs"]).reset_index(drop=True)


def _summary_frame(curve: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for target, group in curve.groupby("target", sort=False):
        best = group.loc[group["target_loss"].idxmin()]
        paper = PAPER_FIGURE7_BEST[target]
        gap = float(best["target_loss"] - paper["loss"])
        if abs(gap) <= 0.02:
            verdict = "constant_near_rounded_schedule_precision"
        elif gap > 0.02:
            verdict = "schedule_clearly_better_than_constant"
        else:
            verdict = "constant_better_than_rounded_schedule"
        rows.append(
            {
                "target": target,
                "completed_points": len(group),
                "best_constant_epochs": int(best["epochs"]),
                "best_constant_rare_weight": float(best["rare_weight"]),
                "best_constant_broad_tokens": float(best["broad_tokens"]),
                "best_constant_broad_epochs": float(best["broad_epochs"]),
                "best_constant_target_loss": float(best["target_loss"]),
                "best_constant_target_bpb": float(best["target_bpb"]),
                "best_constant_c4_loss": float(best["c4_loss"]),
                "best_constant_macro_loss": float(best["macro_loss"]),
                "paper_figure7_best_schedule_loss_rounded": float(paper["loss"]),
                "paper_figure7_best_cell": paper["cell"],
                "paper_figure7_interpretation": paper["interpretation"],
                "constant_minus_paper_best": gap,
                "verdict": verdict,
            }
        )
    return pd.DataFrame(rows)


def _hovertemplate() -> str:
    return (
        "rare epochs=%{customdata[0]:.0f}<br>"
        "C4 broad corpus epochs=%{customdata[1]:.4f}<br>"
        "C4 broad tokens=%{customdata[2]:.2e}<br>"
        "rare weight=%{customdata[3]:.4f}<br>"
        "target loss=%{y:.4f}<br>"
        "target BPB=%{customdata[4]:.4f}<br>"
        "C4 loss=%{customdata[5]:.4f}<extra></extra>"
    )


def _build_plot(curve: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=["StarCoder", "FineMath", "FLAN"],
        shared_yaxes=False,
        horizontal_spacing=0.07,
    )
    target_order = ["starcoder", "finemath", "flan"]
    for col, target in enumerate(target_order, start=1):
        subset = curve[curve["target"] == target].sort_values("epochs")
        custom = subset[
            ["rare_epochs", "broad_epochs", "broad_tokens", "rare_weight", "target_bpb", "c4_loss"]
        ].to_numpy()
        fig.add_trace(
            go.Scatter(
                x=subset["epochs"],
                y=subset["target_loss"],
                mode="lines+markers",
                name="constant target loss" if col == 1 else None,
                showlegend=col == 1,
                line={"color": "#2c7bb6", "width": 3},
                marker={"size": 8},
                customdata=custom,
                hovertemplate=_hovertemplate(),
            ),
            row=1,
            col=col,
        )
        paper_loss = PAPER_FIGURE7_BEST[target]["loss"]
        x_min = int(subset["epochs"].min())
        x_max = int(subset["epochs"].max())
        fig.add_trace(
            go.Scatter(
                x=[x_min, x_max],
                y=[paper_loss, paper_loss],
                mode="lines",
                name="Figure 7 best schedule (rounded)" if col == 1 else None,
                showlegend=col == 1,
                line={"color": "#d7191c", "dash": "dash", "width": 2},
                hovertemplate="rounded Fig. 7 best=%{y:.4f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        fig.update_xaxes(title_text="rare-data epochs", row=1, col=col)
        fig.update_yaxes(title_text="target loss" if col == 1 else None, row=1, col=col)

    fig.update_layout(
        title={
            "text": "Constant C4/rare mixture sweep vs Kotha-Liang Figure 7 schedule bests",
            "x": 0.5,
            "xanchor": "center",
        },
        width=1400,
        height=460,
        template="plotly_white",
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.22},
        margin={"l": 70, "r": 30, "t": 75, "b": 95},
        font={"family": "Avenir, Helvetica, Arial, sans-serif", "size": 14, "color": "#263b5e"},
    )
    return fig


def _write_report(summary: pd.DataFrame, curve: pd.DataFrame) -> None:
    lines = [
        "# Constant-mixture vs Figure 7 schedule analysis",
        "",
        "This refresh includes every replay constant-mixture row whose GCS executor status is `SUCCESS`.",
        "Hover text reports rare-data epochs and C4 broad corpus-token epochs. The C4 denominator is 134,062,553,328 tokens from the tokenized C4 cache's Levanter `input_ids.data_size`.",
        "",
        "## Summary",
        "",
    ]
    for _, row in summary.iterrows():
        lines.append(
            "- "
            f"{row['target']}: best constant rare epochs={int(row['best_constant_epochs'])}, "
            f"C4 broad epochs={row['best_constant_broad_epochs']:.4f}, "
            f"target loss={row['best_constant_target_loss']:.4f}, "
            f"rounded Figure 7 best={row['paper_figure7_best_schedule_loss_rounded']:.4f}, "
            f"gap={row['constant_minus_paper_best']:.4f} ({row['verdict']})."
        )
    lines.extend(
        [
            "",
            f"Completed rows in plot: {len(curve)}.",
            "",
        ]
    )
    (OUTPUT_DIR / "constant_vs_replay_figure7_schedule_analysis.md").write_text("\n".join(lines))


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics = _collect_rows()
    curve = _target_loss_frame(metrics)
    summary = _summary_frame(curve)

    metrics.to_csv(OUTPUT_DIR / "constant_mixture_eval_metrics_from_gcs.csv", index=False)
    curve.to_csv(OUTPUT_DIR / "constant_mixture_target_loss_curve.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "constant_vs_replay_figure7_schedule_summary.csv", index=False)
    _write_report(summary, curve)

    fig = _build_plot(curve)
    fig.write_html(
        OUTPUT_DIR / "constant_vs_replay_figure7_target_loss_curves.html",
        include_plotlyjs="cdn",
        config=PLOT_CONFIG,
    )
    print(f"Wrote {len(curve)} completed rows to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
