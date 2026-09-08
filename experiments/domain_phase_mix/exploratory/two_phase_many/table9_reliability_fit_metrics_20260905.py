# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Table 9 reliability table with per-task surrogate fit quality: SNR next to WSPU and OLMix out-of-fold metrics.

Extends ``table9_snr_table_20260905`` with, per Olmix Table 9 task (and per component in the full table),
the out-of-fold Spearman rank correlation, RMSE in repeat-SD units and regret at 1 of both surrogates at the
full 280-run panel, averaged over the ten fold-seed repeats of ``learning_curve_fits_20260905`` (k = 280).
Collapsed tasks average their subtasks' observed and predicted BPB per run before scoring, the same
collapse the SNR table uses for its noise estimate.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_single_phase_observatory_20260902 as benchmark,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_fits_20260905 as fits,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    learning_curve_metrics_20260905 as metrics_module,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import table9_snr_table_20260905 as snr  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "table9_reliability_20260905"
TARGET = "table9"
UNCHEATABLE = "uncheatable"
UNCHEATABLE_PREFIX = "eval/uncheatable_eval/"
UNCHEATABLE_NOISE = OUTPUT_DIR / "proportional_uncheatable_components.csv"
UNCHEATABLE_NOISE_RUN_PREFIX = "proportional_noise_3e18_"
UNCHEATABLE_LABEL = "Uncheatable aggregate (bytes-weighted mean of 7 components)"
UNCHEATABLE_GROUP = "Uncheatable components"
UNCHEATABLE_NAMES = {
    "ao3_english": "AO3 (fiction)",
    "arxiv_computer_science": "arXiv computer science",
    "arxiv_physics": "arXiv physics",
    "bbc_news": "BBC News",
    "github_cpp": "GitHub C++",
    "github_python": "GitHub Python",
    "wikipedia_english": "Wikipedia (English)",
}
BOLD_ROWS = (UNCHEATABLE_LABEL, snr.MACRO_LABEL)
FULL_K = 280
DRAWS = 10
FIT_METRICS = ("spearman", "rmse_over_repeat_sd", "regret_at_1")


def component_key(name: str) -> str:
    """``olmo_base_eval/easy_bpb/<task>/bpb`` or ``eval/uncheatable_eval/<task>/bpb`` -> ``<task>``."""
    for prefix in (snr.PANEL_PREFIX, UNCHEATABLE_PREFIX):
        if name.startswith(prefix):
            name = name[len(prefix) :]
    return name.removesuffix("/bpb")


def load_predictions(panel: benchmark.BenchPanel, target: str = TARGET) -> dict[str, list[pd.DataFrame]]:
    """Per model, the ten out-of-fold prediction matrices at k = 280 (runs x components, keyed by component)."""
    group = panel.group(target)
    keys = [component_key(name) for name in group.components]
    result: dict[str, list[pd.DataFrame]] = {}
    for model in fits.MODEL_KEYS:
        frames = []
        for draw in range(DRAWS):
            # Only the out-of-fold predictions are used here, so a record written under an earlier
            # held-out registry (stale protocol hash) is still valid for this table.
            record, status = metrics_module.load_record_set(
                target, model, FULL_K, draw, len(group.components), fits.RECORD_DIR, strict=False
            )
            if record is None:
                raise FileNotFoundError(f"{model} k={FULL_K} draw {draw}: {status}")
            if not np.array_equal(record.rows, np.arange(panel.rows)):
                raise ValueError("the k = 280 record does not cover the whole panel")
            frames.append(pd.DataFrame(record.oof, columns=keys))
        result[model] = frames
    return result


def score(observed: np.ndarray, predicted: np.ndarray, repeat_sd: float) -> dict[str, float]:
    row = benchmark.metric_row(observed, predicted, np.zeros(len(observed), dtype=bool))
    pearson = float(np.corrcoef(observed, predicted)[0, 1]) if np.ptp(predicted) > 0 else float("nan")
    return {
        "spearman": row["spearman"],
        "pearson": pearson,
        "rmse": row["rmse"],
        "rmse_over_repeat_sd": row["rmse"] / repeat_sd if repeat_sd > 0 else float("nan"),
        "regret_at_1": row["regret_at_1"],
    }


def fit_columns(
    observed: pd.Series,
    predictions: dict[str, list[pd.DataFrame]],
    columns: list[str],
    repeat_sd: float,
    weights: np.ndarray | None = None,
) -> dict[str, float]:
    """Mean and SD over fold repeats of each fit metric, per model, for one task (mean of its columns)."""
    result: dict[str, float] = {}
    for model, frames in predictions.items():
        combined = [
            frame[columns].mean(axis=1) if weights is None else frame[columns].to_numpy(float) @ weights
            for frame in frames
        ]
        scored = pd.DataFrame(
            [score(observed.to_numpy(float), np.asarray(prediction, dtype=float), repeat_sd) for prediction in combined]
        )
        for metric in ("spearman", "pearson", "rmse", "rmse_over_repeat_sd", "regret_at_1"):
            result[f"{model}_{metric}"] = float(scored[metric].mean())
            result[f"{model}_{metric}_sd"] = float(scored[metric].std(ddof=1))
    return result


def uncheatable_rows(panel: benchmark.BenchPanel) -> pd.DataFrame:
    """The Uncheatable aggregate and its seven components in the schema of the Table 9 rows."""
    group = panel.group(UNCHEATABLE)
    keys = [component_key(name) for name in group.components]
    values = pd.DataFrame(group.outcomes, columns=keys)
    noise_frame = pd.read_csv(UNCHEATABLE_NOISE)
    noise_frame = noise_frame[noise_frame["run"].str.startswith(UNCHEATABLE_NOISE_RUN_PREFIX)]
    noise = pd.DataFrame({key: noise_frame[f"{UNCHEATABLE_PREFIX}{key}/bpb"].to_numpy(float) for key in keys})
    weights = np.asarray(group.aggregation_weights, dtype=float)
    predictions = load_predictions(panel, UNCHEATABLE)
    aggregate_noise = pd.Series(noise.to_numpy(float) @ weights)
    aggregate_values = pd.Series(np.asarray(group.aggregate, dtype=float))
    aggregate = {
        "group": "",
        "task": UNCHEATABLE_LABEL,
        "component": "",
        "subtask": UNCHEATABLE_LABEL,
        "subtasks": len(keys),
        **snr.statistics(aggregate_values, aggregate_noise),
        "proportional_mean": float(aggregate_noise.mean()),
        **fit_columns(aggregate_values, predictions, keys, float(aggregate_noise.std(ddof=1)), weights),
    }
    rows = [aggregate]
    for key in keys:
        rows.append(
            {
                "group": UNCHEATABLE_GROUP,
                "task": UNCHEATABLE_NAMES[key],
                "component": key,
                "subtask": UNCHEATABLE_NAMES[key],
                "subtasks": 1,
                **snr.statistics(values[key], noise[key]),
                "proportional_mean": float(noise[key].mean()),
                **fit_columns(values[key], predictions, [key], float(noise[key].std(ddof=1))),
            }
        )
    return pd.DataFrame(rows)


def build_tables(
    values: pd.DataFrame, noise: pd.DataFrame, predictions: dict[str, list[pd.DataFrame]], panel: benchmark.BenchPanel
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del panel
    short, full = snr.build_tables(values, noise)
    all_keys = list(values.columns)
    short_extra = []
    for _, row in short.iterrows():
        if row["task"] == snr.MACRO_LABEL:
            columns = all_keys
        else:
            columns = [
                key
                for group_name, tasks in snr.TABLE9_LAYOUT
                if group_name == row["group"]
                for task_name, subtasks in tasks
                if task_name == row["task"]
                for key, _label in subtasks
            ]
        short_extra.append(
            {"proportional_mean": float(noise[columns].mean(axis=1).mean())}
            | fit_columns(values[columns].mean(axis=1), predictions, columns, float(row["repeat_sd"]))
        )
    full_extra = []
    for _, row in full.iterrows():
        columns = all_keys if row["task"] == snr.MACRO_LABEL else [row["component"]]
        full_extra.append(
            {"proportional_mean": float(noise[columns].mean(axis=1).mean())}
            | fit_columns(values[columns].mean(axis=1), predictions, columns, float(row["repeat_sd"]))
        )
    return pd.concat([short, pd.DataFrame(short_extra)], axis=1), pd.concat([full, pd.DataFrame(full_extra)], axis=1)


def markdown(frame: pd.DataFrame, full: bool, proportional_runs: int) -> str:
    header = (
        f"| Task | Swarm mean | Swarm SD | Proportional mean ({proportional_runs} runs) | Proportional SD | SNR | "
        "WSPU rho | OLMix rho |"
    )
    lines = [header, "|---|---|---|---|---|---|---|---|"]
    current_group = None
    current_task = None

    def cells(row: pd.Series, bold: bool = False) -> str:
        snr_text = f"**{row['snr']:.1f}**" if bold else f"{row['snr']:.1f}"
        return (
            f"{row['panel_mean']:.3f} | {row['panel_sd']:.4f} | {row['proportional_mean']:.3f} | "
            f"{row['repeat_sd']:.4f} | {snr_text} | {row['wspu_spearman']:.2f} | {row['olmix_spearman']:.2f} |"
        )

    for _, row in frame.iterrows():
        if row["task"] in BOLD_ROWS:
            lines.append(f"| **{row['task']}** | {cells(row, bold=True)}")
            continue
        if row["group"] != current_group:
            current_group = row["group"]
            lines.append(f"| *{current_group}* | | | | | | | |")
        if full:
            if row["task"] != current_task:
                current_task = row["task"]
                if row["subtask"] != row["task"]:
                    lines.append(f"| {row['task']} | | | | | | | |")
            label = f"&nbsp;&nbsp;{row['subtask']}" if row["subtask"] != row["task"] else row["task"]
            lines.append(f"| {label} | {cells(row)}")
        else:
            label = f"{row['task']} ({int(row['subtasks'])} subtasks)" if row["subtasks"] > 1 else row["task"]
            lines.append(f"| {label} | {cells(row)}")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    panel = benchmark.load_panel(snr.PANEL)
    values, noise = snr.load_matrices()
    predictions = load_predictions(panel)
    short, full = build_tables(values, noise, predictions, panel)
    uncheatable = uncheatable_rows(panel)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    short.to_csv(args.output_dir / "snr_fit_tasks_delphi.csv", index=False)
    full.to_csv(args.output_dir / "snr_fit_components_delphi.csv", index=False)
    uncheatable.to_csv(args.output_dir / "snr_fit_uncheatable_delphi.csv", index=False)
    runs = int(noise.shape[0])
    (args.output_dir / "snr_fit_tasks_delphi.md").write_text(markdown(short, False, runs) + "\n")
    (args.output_dir / "snr_fit_components_delphi.md").write_text(markdown(full, True, runs) + "\n")
    (args.output_dir / "snr_fit_uncheatable_delphi.md").write_text(markdown(uncheatable, False, runs) + "\n")
    print(markdown(uncheatable, False, runs))
    print(markdown(short, False, runs))


if __name__ == "__main__":
    main()
