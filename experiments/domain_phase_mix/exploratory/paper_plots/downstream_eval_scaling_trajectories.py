# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["fsspec", "gcsfs", "kaleido==0.2.1", "pandas", "plotly"]
# ///
"""Render downstream-eval scaling trajectories for baseline mixtures.

This script reads the central baseline-scaling manifest, collected downstream
eval result CSVs, and checkpoint-attached lm-eval artifacts. The latter matters
for historical rows where MMLU was already present and was intentionally skipped
by the eval launcher.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

import fsspec
import pandas as pd
import plotly.graph_objects as go

from experiments.domain_phase_mix.exploratory.paper_plots.paper_plot_style import (
    configure_interactive_layout,
    configure_static_layout,
    method_color,
    method_dash,
    write_static_images,
)

SCRIPT_DIR = Path(__file__).resolve().parent
IMG_DIR = SCRIPT_DIR / "img"
MANIFEST_CSV = IMG_DIR / "baseline_scaling_trajectories_manifest.csv"
ALL_SOURCES_CSV = IMG_DIR / "baseline_scaling_downstream_eval_metrics_all_sources.csv"
MERGED_METRICS_CSV = IMG_DIR / "baseline_scaling_downstream_eval_metrics_merged.csv"

LM_EVAL_METRIC_SUFFIX = ",none"
RESULT_GLOBS = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_baseline_scaling_downstream_evals*/**/baseline_scaling_downstream_eval_results.csv",
    "gs://marin-us-central1/pinlin_calvin_xu/data_mixture/"
    "ngd3dm2_baseline_scaling_downstream_evals*/**/baseline_scaling_downstream_eval_results.csv",
)


@dataclass(frozen=True)
class EvalPlotSpec:
    """One downstream-eval plot."""

    eval_id: str
    title: str
    default_metric_column: str
    metric_columns: tuple[str, ...]
    output_stem: str


PLOT_SPECS = (
    EvalPlotSpec(
        eval_id="gsm8k",
        title="GSM8K 5-shot scaling trajectories",
        default_metric_column="lm_eval/gsm8k/exact_match,flexible-extract",
        metric_columns=(
            "lm_eval/gsm8k/exact_match,flexible-extract",
            "lm_eval/gsm8k/exact_match,strict-match",
            "lm_eval/gsm8k/exact_match_stderr,flexible-extract",
            "lm_eval/gsm8k/exact_match_stderr,strict-match",
        ),
        output_stem="baseline_scaling_gsm8k_5shot",
    ),
    EvalPlotSpec(
        eval_id="humaneval",
        title="HumanEval 10-shot scaling trajectories",
        default_metric_column="lm_eval/humaneval/pass@1,create_test",
        metric_columns=(
            "lm_eval/humaneval/pass@1,create_test",
            "lm_eval/humaneval/pass@1_stderr,create_test",
        ),
        output_stem="baseline_scaling_humaneval_10shot",
    ),
    EvalPlotSpec(
        eval_id="mmlu",
        title="MMLU 5-shot scaling trajectories",
        default_metric_column="lm_eval/mmlu_5shot/acc",
        metric_columns=(
            "lm_eval/mmlu_5shot/acc",
            "lm_eval/mmlu_5shot/acc_norm",
            "lm_eval/mmlu_5shot/bpb",
            "lm_eval/mmlu_5shot/logprob",
            "lm_eval/mmlu_5shot/choice_logprob",
            "lm_eval/mmlu_5shot/choice_logprob_norm",
            "lm_eval/mmlu_5shot/choice_prob_norm",
            "lm_eval/averages/macro_avg_acc",
            "lm_eval/averages/micro_avg_acc",
            "lm_eval/averages/macro_avg_acc_norm",
            "lm_eval/averages/micro_avg_acc_norm",
            "lm_eval/averages/macro_avg_bpb",
            "lm_eval/averages/micro_avg_bpb",
            "lm_eval/averages/macro_avg_logprob",
            "lm_eval/averages/micro_avg_logprob",
            "lm_eval/averages/macro_avg_choice_logprob",
            "lm_eval/averages/micro_avg_choice_logprob",
            "lm_eval/averages/macro_avg_choice_logprob_norm",
            "lm_eval/averages/micro_avg_choice_logprob_norm",
            "lm_eval/averages/macro_avg_choice_prob_norm",
            "lm_eval/averages/micro_avg_choice_prob_norm",
        ),
        output_stem="baseline_scaling_mmlu_5shot",
    ),
)
PLOT_METRIC_COLUMNS = tuple(dict.fromkeys(metric for spec in PLOT_SPECS for metric in spec.metric_columns))
METRIC_LABELS = {
    "lm_eval/gsm8k/exact_match,flexible-extract": "GSM8K exact match, flexible",
    "lm_eval/gsm8k/exact_match,strict-match": "GSM8K exact match, strict",
    "lm_eval/gsm8k/exact_match_stderr,flexible-extract": "GSM8K exact match stderr, flexible",
    "lm_eval/gsm8k/exact_match_stderr,strict-match": "GSM8K exact match stderr, strict",
    "lm_eval/humaneval/pass@1,create_test": "HumanEval pass@1",
    "lm_eval/humaneval/pass@1_stderr,create_test": "HumanEval pass@1 stderr",
    "lm_eval/mmlu_5shot/acc": "MMLU accuracy",
    "lm_eval/mmlu_5shot/acc_norm": "MMLU normalized accuracy",
    "lm_eval/mmlu_5shot/bpb": "MMLU BPB",
    "lm_eval/mmlu_5shot/logprob": "MMLU logprob",
    "lm_eval/mmlu_5shot/choice_logprob": "MMLU choice logprob",
    "lm_eval/mmlu_5shot/choice_logprob_norm": "MMLU normalized choice logprob",
    "lm_eval/mmlu_5shot/choice_prob_norm": "MMLU normalized choice probability",
    "lm_eval/averages/macro_avg_acc": "Macro average accuracy",
    "lm_eval/averages/micro_avg_acc": "Micro average accuracy",
    "lm_eval/averages/macro_avg_acc_norm": "Macro average normalized accuracy",
    "lm_eval/averages/micro_avg_acc_norm": "Micro average normalized accuracy",
    "lm_eval/averages/macro_avg_bpb": "Macro average BPB",
    "lm_eval/averages/micro_avg_bpb": "Micro average BPB",
    "lm_eval/averages/macro_avg_logprob": "Macro average logprob",
    "lm_eval/averages/micro_avg_logprob": "Micro average logprob",
    "lm_eval/averages/macro_avg_choice_logprob": "Macro average choice logprob",
    "lm_eval/averages/micro_avg_choice_logprob": "Micro average choice logprob",
    "lm_eval/averages/macro_avg_choice_logprob_norm": "Macro average normalized choice logprob",
    "lm_eval/averages/micro_avg_choice_logprob_norm": "Micro average normalized choice logprob",
    "lm_eval/averages/macro_avg_choice_prob_norm": "Macro average normalized choice probability",
    "lm_eval/averages/micro_avg_choice_prob_norm": "Micro average normalized choice probability",
}

METHOD_ORDER = ("grp_no_l2", "proportional", "olmix", "uniform", "unimax")


def _axis_label(row: pd.Series) -> str:
    return str(row["scale_label"])


def _metric_key(metric_column: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", metric_column).strip("_")


def _metric_label(metric_column: str) -> str:
    if metric_column in METRIC_LABELS:
        return METRIC_LABELS[metric_column]
    return metric_column.removeprefix("lm_eval/").replace("/", " ").replace("_", " ")


def _metric_is_percentage(metric_column: str) -> bool:
    metric_name = metric_column.rsplit("/", maxsplit=1)[-1]
    return any(token in metric_name for token in ("acc", "exact_match", "pass@1", "choice_prob"))


def _display_value(metric_column: str, value: object) -> float:
    numeric = float(value)
    return 100.0 * numeric if _metric_is_percentage(metric_column) else numeric


def _y_title(metric_column: str) -> str:
    label = _metric_label(metric_column)
    if metric_column == "lm_eval/humaneval/pass@1,create_test":
        return f"{label} (%; 1 task = 0.61 pp)"
    if _metric_is_percentage(metric_column):
        return f"{label} (%)"
    return label


def _numeric_column(frame: pd.DataFrame, column: str) -> pd.Series:
    """Return a numeric metric column, or all-NA values when absent."""
    if column not in frame.columns:
        return pd.Series([pd.NA] * len(frame), index=frame.index, dtype="Float64")
    return pd.to_numeric(frame[column], errors="coerce")


def _available_metric_columns(metrics: pd.DataFrame, spec: EvalPlotSpec) -> list[str]:
    """Return metrics from this eval that have at least one plotted cell."""
    return [column for column in spec.metric_columns if _numeric_column(metrics, column).notna().any()]


def _result_csv_paths() -> list[str]:
    paths: list[str] = []
    for pattern in RESULT_GLOBS:
        fs, _, _ = fsspec.get_fs_token_paths(pattern)
        for match in fs.glob(pattern):
            path = match if str(match).startswith("gs://") else f"gs://{match}"
            paths.append(path)
    return sorted(set(paths))


def _metric_rows_from_payload(payload: dict[str, object]) -> dict[str, float]:
    metrics: dict[str, float] = {}
    results = payload.get("results", {})
    if isinstance(results, dict):
        for task_key, task_metrics in results.items():
            if not isinstance(task_metrics, dict):
                continue
            for metric_key, value in task_metrics.items():
                metric_name = str(metric_key).removesuffix(LM_EVAL_METRIC_SUFFIX)
                if metric_name.endswith("_stderr"):
                    continue
                if isinstance(value, int | float):
                    metrics[f"lm_eval/{task_key}/{metric_name}"] = float(value)
    averages = payload.get("averages", {})
    if isinstance(averages, dict):
        for metric_key, value in averages.items():
            metric_name = str(metric_key).removesuffix(LM_EVAL_METRIC_SUFFIX)
            if isinstance(value, int | float):
                metrics[f"lm_eval/averages/{metric_name}"] = float(value)
    return metrics


def _checkpoint_artifact_paths(checkpoint_root: str) -> list[str]:
    if not checkpoint_root.startswith("gs://"):
        return []
    pattern = checkpoint_root.rstrip("/") + "/lm_eval_artifacts/lm_eval_harness_results*.json"
    fs, _, _ = fsspec.get_fs_token_paths(pattern)
    return sorted(match if str(match).startswith("gs://") else f"gs://{match}" for match in fs.glob(pattern))


def _checkpoint_metric_records(manifest: pd.DataFrame) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for _, row in manifest.iterrows():
        checkpoint_root = str(row.get("checkpoint_root") or "")
        for path in _checkpoint_artifact_paths(checkpoint_root):
            try:
                with fsspec.open(path, "rt") as handle:
                    payload = json.load(handle)
            except (OSError, json.JSONDecodeError):
                continue
            metrics = _metric_rows_from_payload(payload)
            if not metrics:
                continue
            records.append(
                {
                    "method_id": row["method_id"],
                    "method": row["method"],
                    "scale": row["scale"],
                    "scale_label": row["scale_label"],
                    "x_order": int(row["x_order"]),
                    "non_embedding_params": row["non_embedding_params"],
                    "realized_train_tokens": row["realized_train_tokens"],
                    "run_name": row["run_name"],
                    "checkpoint_root": checkpoint_root,
                    "eval_source": "checkpoint_lm_eval_artifact",
                    "source_path": path,
                    "source_priority": 1,
                    **metrics,
                }
            )
    return records


def _collected_result_records() -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    for path in _result_csv_paths():
        frame = pd.read_csv(path)
        metric_columns = [column for column in frame.columns if column.startswith("lm_eval/")]
        if not metric_columns:
            continue
        for _, row in frame.iterrows():
            if row.get("collection_status") != "collected":
                continue
            metrics = {}
            for column in metric_columns:
                value = pd.to_numeric(pd.Series([row[column]]), errors="coerce").iloc[0]
                if pd.notna(value):
                    metrics[column] = float(value)
            if not metrics:
                continue
            records.append(
                {
                    "method_id": row["method_id"],
                    "scale": row["scale"],
                    "run_name": row.get("run_name"),
                    "eval_source": "collected_result_csv",
                    "source_path": path,
                    "source_priority": 2,
                    **metrics,
                }
            )
    return records


def _attach_manifest_metadata(records: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    metadata_columns = [
        "method_id",
        "scale",
        "method",
        "scale_label",
        "x_order",
        "non_embedding_params",
        "realized_train_tokens",
        "checkpoint_root",
    ]
    metadata = manifest[metadata_columns].drop_duplicates(["method_id", "scale"])
    duplicate_metadata_columns = [
        column for column in metadata_columns if column in records.columns and column not in {"method_id", "scale"}
    ]
    frame = records.drop(columns=duplicate_metadata_columns)
    return frame.merge(metadata, on=["method_id", "scale"], how="left", suffixes=("", "_manifest"))


def build_metric_table() -> pd.DataFrame:
    """Return one merged metric table across collected and checkpoint sources."""
    manifest = pd.read_csv(MANIFEST_CSV)
    records = [*_checkpoint_metric_records(manifest), *_collected_result_records()]
    if not records:
        raise ValueError("No downstream eval metrics found")
    all_sources = _attach_manifest_metadata(pd.DataFrame.from_records(records), manifest)
    all_sources = all_sources.sort_values(
        ["method_id", "scale", "source_priority", "source_path"],
        ascending=[True, True, False, False],
    )
    all_sources.to_csv(ALL_SOURCES_CSV, index=False)

    merged_rows: list[dict[str, object]] = []
    metadata_columns = [
        "method_id",
        "method",
        "scale",
        "scale_label",
        "x_order",
        "non_embedding_params",
        "realized_train_tokens",
        "checkpoint_root",
    ]
    for _, group in all_sources.groupby(["method_id", "scale"], sort=False):
        sorted_group = group.sort_values(["source_priority", "source_path"], ascending=[False, False])
        base = {column: sorted_group.iloc[0][column] for column in metadata_columns}
        source_labels: list[str] = []
        source_paths: list[str] = []
        for metric_column in PLOT_METRIC_COLUMNS:
            value_row = sorted_group.loc[_numeric_column(sorted_group, metric_column).notna()]
            metric_key = _metric_key(metric_column)
            if value_row.empty:
                base[metric_column] = pd.NA
                base[f"{metric_key}__source"] = ""
                base[f"{metric_key}__source_path"] = ""
                continue
            selected = value_row.iloc[0]
            base[metric_column] = float(selected[metric_column])
            base[f"{metric_key}__source"] = selected["eval_source"]
            base[f"{metric_key}__source_path"] = selected["source_path"]
            source_labels.append(str(selected["eval_source"]))
            source_paths.append(str(selected["source_path"]))
        base["eval_source"] = ";".join(sorted(set(source_labels)))
        base["source_path"] = ";".join(sorted(set(source_paths)))
        merged_rows.append(base)

    merged = pd.DataFrame.from_records(merged_rows)
    merged.to_csv(MERGED_METRICS_CSV, index=False)
    return merged


def _hover_text(row: pd.Series, metric_column: str) -> str:
    value = float(row[metric_column])
    metric_key = _metric_key(metric_column)
    source = row.get(f"{metric_key}__source", row.get("eval_source", ""))
    source_path = row.get(f"{metric_key}__source_path", row.get("source_path", ""))
    extra_metric_context = []
    if metric_column == "lm_eval/humaneval/pass@1,create_test":
        extra_metric_context.append(f"Estimated correct: {round(value * 164):d}/164")
    return "<br>".join(
        [
            f"Method: {row['method']}",
            f"Scale: {row['scale_label']}",
            f"Non-embedding params: {float(row['non_embedding_params']) / 1_000_000:.1f}M",
            f"Training tokens: {float(row['realized_train_tokens']) / 1_000_000_000:.1f}B",
            f"{_y_title(metric_column)}: {_display_value(metric_column, value):.4g}",
            *extra_metric_context,
            f"Source: {source}",
            f"Path: {source_path}",
        ]
    )


def render_plot(metrics: pd.DataFrame, spec: EvalPlotSpec) -> go.Figure:
    """Render one downstream-eval trajectory plot."""
    fig = go.Figure()
    metric_columns = _available_metric_columns(metrics, spec)
    if not metric_columns:
        raise ValueError(f"No available metrics for {spec.eval_id}")
    default_metric = spec.default_metric_column if spec.default_metric_column in metric_columns else metric_columns[0]
    trace_metric_columns: list[str] = []
    for metric_column in metric_columns:
        available = metrics.loc[_numeric_column(metrics, metric_column).notna()].copy()
        available[metric_column] = _numeric_column(available, metric_column)
        for method_id in METHOD_ORDER:
            method_rows = available.loc[available["method_id"] == method_id].sort_values("x_order")
            if method_rows.empty:
                continue
            visible = metric_column == default_metric
            fig.add_trace(
                go.Scatter(
                    x=[_axis_label(row) for _, row in method_rows.iterrows()],
                    y=[_display_value(metric_column, value) for value in method_rows[metric_column]],
                    mode="lines+markers",
                    name=str(method_rows.iloc[0]["method"]),
                    line={
                        "color": method_color(method_id),
                        "width": 3.6,
                        "dash": method_dash(method_id),
                    },
                    marker={
                        "color": method_color(method_id),
                        "size": 9,
                        "line": {"color": "white", "width": 1.1},
                    },
                    hovertext=[_hover_text(row, metric_column) for _, row in method_rows.iterrows()],
                    hoverinfo="text",
                    visible=visible,
                    showlegend=visible,
                )
            )
            trace_metric_columns.append(metric_column)

    buttons = []
    for metric_column in metric_columns:
        visible = [trace_metric_column == metric_column for trace_metric_column in trace_metric_columns]
        buttons.append(
            {
                "label": _metric_label(metric_column),
                "method": "update",
                "args": [
                    {"visible": visible, "showlegend": visible},
                    {
                        "title": {
                            "text": f"{spec.title}<br><sup>{_metric_label(metric_column)}</sup>",
                            "x": 0.04,
                            "xanchor": "left",
                            "font": {"size": 24},
                        },
                        "yaxis": {
                            "title": _y_title(metric_column),
                        },
                    },
                ],
            }
        )

    scale_axis_rows = pd.read_csv(MANIFEST_CSV).drop_duplicates("scale").sort_values("x_order")
    configure_interactive_layout(
        fig,
        title=f"{spec.title}<br><sup>{_metric_label(default_metric)}</sup>",
        y_title=_y_title(default_metric),
        x_title="Scale (nominal model size; non-embedding params in parentheses / training tokens)",
    )
    fig.update_layout(
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.22,
                "yanchor": "top",
                "showactive": True,
                "pad": {"r": 0, "t": 0},
            }
        ],
    )
    fig.update_xaxes(
        categoryorder="array",
        categoryarray=[_axis_label(row) for _, row in scale_axis_rows.iterrows()],
    )
    return fig


def main() -> None:
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    metrics = build_metric_table()
    summary_rows: list[dict[str, object]] = []
    for spec in PLOT_SPECS:
        fig = render_plot(metrics, spec)
        html_path = IMG_DIR / f"{spec.output_stem}.html"
        png_path = IMG_DIR / f"{spec.output_stem}.png"
        pdf_path = IMG_DIR / f"{spec.output_stem}.pdf"
        fig.write_html(html_path, include_plotlyjs="cdn")
        available_metric_columns = _available_metric_columns(metrics, spec)
        default_metric = (
            spec.default_metric_column
            if spec.default_metric_column in available_metric_columns
            else available_metric_columns[0]
        )
        static_fig = go.Figure(fig)
        configure_static_layout(
            static_fig,
            y_title=_y_title(default_metric),
            x_title="Scale (nominal model size; non-embedding params in parentheses / training tokens)",
        )
        write_static_images(static_fig, IMG_DIR / spec.output_stem)
        available = _numeric_column(metrics, default_metric).notna()
        summary_rows.append(
            {
                "eval_id": spec.eval_id,
                "default_metric_column": default_metric,
                "available_metrics": ";".join(available_metric_columns),
                "available_cells": int(available.sum()),
                "html_path": str(html_path),
                "png_path": str(png_path),
                "pdf_path": str(pdf_path),
            }
        )
        print(f"Wrote {html_path}")
        print(f"Wrote {png_path}")
        print(f"Wrote {pdf_path}")
    pd.DataFrame.from_records(summary_rows).to_csv(
        IMG_DIR / "baseline_scaling_downstream_eval_plot_summary.csv", index=False
    )
    print(
        metrics[
            ["method", "scale_label", *[column for column in PLOT_METRIC_COLUMNS if column in metrics.columns]]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
