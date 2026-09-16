# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# ruff: noqa: E402, E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "kaleido",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scipy",
#   "wandb",
# ]
# ///
"""Plot completed Delphi scaling datapoints from W&B.

This is intentionally W&B-backed rather than reading local CSVs, because the
scaling runs are live and Fieldbook tracks execution attempts while W&B has the
latest scalar summaries.

The optimization target for issue #6602/#6608 is the uncheatable-eval BPB, not
the top-level eval BPB. We still export both so mistakes are easy to audit, but
the first figure plots only `eval/uncheatable_eval/bpb`.

The issue #6611 Table-9/OLMoBaseEval Easy optimized mixtures now receive native
OLMoBaseEval writebacks for completed scaling checkpoints. The second figure
therefore defaults to the paper-style 51-component Table-9 macro BPB whenever
the component writebacks are available, and keeps training-time `eval/*` proxies
in the dropdown for incomplete rows.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# The repo has a local ./wandb directory containing run files. Remove the current
# working directory from import resolution so `import wandb` loads the package.
_cwd = str(Path.cwd())
sys.path = [path for path in sys.path if path not in {"", _cwd}]

import pandas as pd
import plotly.graph_objects as go
import wandb

_repo_root = Path(__file__).resolve().parents[4]
if str(_repo_root) not in sys.path:
    sys.path.append(str(_repo_root))

from experiments.domain_phase_mix.exploratory.two_phase_many.fit_olmo_base_easy_paper_faithful_olmix_300m import (
    MMLU_CATEGORY_WEIGHTS,
    table9_component_order,
)

OUTPUT_DIR = Path(
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/delphi_scaling_progress_20260625"
)

RUN_BASES = [
    "proportional_3e18",
    "proportional_2e19",
    "proportional_3e20",
    "proportional_1e21",
    "unimax8_3e18",
    "unimax8_2e19",
    "unimax8_3e20",
    "unimax8_1e21",
    "olmix_d001_kl005_cap4_3e18",
    "olmix_d001_kl005_cap4_2e19",
    "olmix_d001_kl005_cap4_3e20",
    "olmix_d001_kl005_cap4_1e21",
    "dsp_effexp_kl01_3e18",
    "dsp_effexp_kl01_2e19",
    "dsp_effexp_kl01_3e20",
    "dsp_effexp_kl01_1e21",
    "olmix_onephase_uncheatable_d001_kl005_cap4_3e18",
    "olmix_onephase_uncheatable_d001_kl005_cap4_2e19",
    "olmix_onephase_uncheatable_d001_kl005_cap4_3e20",
    "olmix_onephase_uncheatable_d001_kl005_cap4_1e21",
    "dsp_onephase_effexp_uncheatable_kl0p1_3e18",
    "dsp_onephase_effexp_uncheatable_kl0p1_2e19",
    "dsp_onephase_effexp_uncheatable_kl0p1_3e20",
    "dsp_onephase_effexp_uncheatable_kl0p1_1e21",
    "olmix_table9_d001_kl005_cap4_3e18",
    "olmix_table9_d001_kl005_cap4_2e19",
    "olmix_table9_d001_kl005_cap4_3e20",
    "olmix_table9_d001_kl005_cap4_1e21",
    "dsp_effexp_table9_kl0025_3e18",
    "dsp_effexp_table9_kl0025_2e19",
    "dsp_effexp_table9_kl0025_3e20",
    "dsp_effexp_table9_kl0025_1e21",
    "olmix_onephase_table9_d001_kl005_cap4_3e18",
    "olmix_onephase_table9_d001_kl005_cap4_2e19",
    "olmix_onephase_table9_d001_kl005_cap4_3e20",
    "olmix_onephase_table9_d001_kl005_cap4_1e21",
    "dsp_onephase_effexp_table9_kl0p1_3e18",
    "olmix_onephase_uncheatable_d001_kl0p1_cap4_3e18",
    "olmix_onephase_uncheatable_d001_kl0p1_cap4_2e19",
    "olmix_onephase_uncheatable_d001_kl0p1_cap4_3e20",
    "olmix_onephase_uncheatable_d001_kl0p1_cap4_1e21",
    "olmix_onephase_table9_d001_kl0p005_cap4_3e18",
    "olmix_onephase_table9_d001_kl0p005_cap4_2e19",
    "olmix_onephase_table9_d001_kl0p005_cap4_3e20",
    "olmix_onephase_table9_d001_kl0p005_cap4_1e21",
]

SCALE_TO_FLOPS = {
    "3e18": 3e18,
    "2e19": 2e19,
    "3e20": 3e20,
    "1e21": 1e21,
}

MIXTURE_DISPLAY = {
    "proportional": "Proportional",
    "unimax8": "UniMax-8",
    "olmix_d001_kl005_cap4": "OLMix d=0.01 KL=0.05 cap4",
    "dsp_effexp_kl01": "DSP effective-exposure KL=0.1",
    "olmix_onephase_uncheatable_d001_kl005_cap4": "1-phase OLMix uncheatable",
    "dsp_onephase_effexp_uncheatable_kl0p1": "1-phase DSP uncheatable",
    "olmix_table9_d001_kl005_cap4": "OLMix Table-9 d=0.01 KL=0.05 cap4",
    "dsp_effexp_table9_kl0025": "DSP Table-9 effective-exposure KL=0.025",
    "olmix_onephase_table9_d001_kl005_cap4": "1-phase OLMix Table-9",
    "dsp_onephase_effexp_table9_kl0p1": "1-phase DSP Table-9 KL=0.1",
    "olmix_onephase_uncheatable_d001_kl0p1_cap4": "1-phase OLMix uncheatable KL=0.1",
    "olmix_onephase_table9_d001_kl0p005_cap4": "1-phase OLMix Table-9 KL=0.005",
}

MIXTURE_COLOR = {
    "Proportional": "#2b6cb0",
    "UniMax-8": "#805ad5",
    "OLMix d=0.01 KL=0.05 cap4": "#dd6b20",
    "DSP effective-exposure KL=0.1": "#2f855a",
    "1-phase OLMix uncheatable": "#f59e0b",
    "1-phase DSP uncheatable": "#10b981",
    "OLMix Table-9 d=0.01 KL=0.05 cap4": "#dd6b20",
    "DSP Table-9 effective-exposure KL=0.025": "#2f855a",
    "1-phase OLMix Table-9": "#f59e0b",
    "1-phase DSP Table-9 KL=0.1": "#10b981",
    "1-phase OLMix uncheatable KL=0.1": "#b45309",
    "1-phase OLMix Table-9 KL=0.005": "#b45309",
}

UNCH_TARGET_MIXTURES = {
    "proportional",
    "unimax8",
    "olmix_d001_kl005_cap4",
    "dsp_effexp_kl01",
    "olmix_onephase_uncheatable_d001_kl005_cap4",
    "olmix_onephase_uncheatable_d001_kl0p1_cap4",
    "dsp_onephase_effexp_uncheatable_kl0p1",
}
TABLE9_PROXY_MIXTURES = {
    "proportional",
    "unimax8",
    "olmix_table9_d001_kl005_cap4",
    "olmix_onephase_table9_d001_kl005_cap4",
    "olmix_onephase_table9_d001_kl0p005_cap4",
    "dsp_onephase_effexp_table9_kl0p1",
}

MIXTURE_LINE_DASH = {
    "1-phase OLMix uncheatable": "dash",
    "1-phase OLMix uncheatable KL=0.1": "dashdot",
    "1-phase DSP uncheatable": "dash",
    "1-phase OLMix Table-9": "dash",
    "1-phase OLMix Table-9 KL=0.005": "dashdot",
    "1-phase DSP Table-9 KL=0.1": "dash",
}


@dataclass(frozen=True)
class RunLookup:
    mixture: str
    scale: str
    flops: float


@dataclass(frozen=True)
class MetricSpec:
    column: str
    label: str
    y_label: str


OLMO_TABLE9_MACRO_COLUMN = "olmo_base_easy_table9_51_component_macro_bpb"
OLMO_PRIMARY_MEAN_COLUMN = "olmo_base_eval_easy_bpb_primary_metric_mean"
NATIVE_OLMO_EVAL_PROJECT = "marin-community/marin-eval"
NATIVE_OLMO_MACRO_KEYS = (
    "olmo_base_easy/table9_51_component_macro_bpb",
    "olmo_base_easy/table9_macro_bpb",
)

PREFERRED_TABLE9_METRICS = [
    OLMO_TABLE9_MACRO_COLUMN,
    OLMO_PRIMARY_MEAN_COLUMN,
    "eval/macro_bpb",
    "eval/bpb",
    "eval/uncheatable_eval/bpb",
    "eval/uncheatable_eval/macro_bpb",
    "eval/loss",
    "eval/macro_loss",
]


def parse_run_base(run_base: str) -> RunLookup:
    scale = run_base.rsplit("_", 1)[-1]
    if scale not in SCALE_TO_FLOPS:
        raise ValueError(f"Unexpected run scale in {run_base!r}")
    mixture = run_base[: -(len(scale) + 1)]
    return RunLookup(mixture=mixture, scale=scale, flops=SCALE_TO_FLOPS[scale])


def scalar(summary: Any, key: str) -> float | None:
    value = summary.get(key)
    if value is None:
        return None
    return float(value)


def scalar_summary_values(summary: Any) -> dict[str, float]:
    values: dict[str, float] = {}
    for key, value in dict(summary).items():
        if isinstance(value, bool):
            continue
        if isinstance(value, int | float):
            values[key] = float(value)
    return values


def olmo_component_value(summary: Any, component: str) -> float | None:
    """Return one Table-9 component BPB from native OLMoBaseEval writeback keys."""
    summary_dict = dict(summary)
    if component in MMLU_CATEGORY_WEIGHTS:
        weighted_values: list[float] = []
        for task, weight in MMLU_CATEGORY_WEIGHTS[component].items():
            key = f"olmo_base_eval/easy_bpb/{task}_rc_bpb/bits_per_byte/bits_per_byte"
            value = summary_dict.get(key)
            if value is None:
                return None
            weighted_values.append(weight * float(value))
        return float(sum(weighted_values))

    if not component.startswith("olmo_base_eval/easy_bpb/"):
        raise ValueError(f"Unexpected Table-9 component key: {component!r}")
    task = component.split("/")[2]
    candidate_keys = (
        f"olmo_base_eval/easy_bpb/{task}/bpb",
        f"olmo_base_eval/easy_bpb/{task}/bits_per_byte/bits_per_byte",
        f"olmo_base_eval/easy_bpb/{task}_olmo3base_bpb/bits_per_byte/bits_per_byte",
        f"olmo_base_eval/easy_bpb/{task}_bpb_olmo3base/bits_per_byte/bits_per_byte",
    )
    for key in candidate_keys:
        value = summary_dict.get(key)
        if value is not None:
            return float(value)
    return None


def olmo_table9_macro_bpb(summary: Any) -> float | None:
    """Compute the OLMix Table-9 51-component macro BPB from W&B summary keys."""
    for key in NATIVE_OLMO_MACRO_KEYS:
        value = summary.get(key)
        if value is not None:
            return float(value)
    values = [olmo_component_value(summary, component) for component in table9_component_order()]
    if any(value is None for value in values):
        return None
    return float(sum(value for value in values if value is not None) / len(values))


def latest_run_for_base(api: wandb.Api, run_base: str):
    runs = api.runs(
        "marin-community/marin",
        filters={"name": {"$regex": f"^{run_base}-"}},
        order="-created_at",
        per_page=5,
    )
    for run in runs:
        if run.name.startswith(f"{run_base}-"):
            return run
    return None


def native_eval_names_for_run_base(run_base: str) -> tuple[str, ...]:
    """Return expected native Table-9 eval W&B display names for one scaling run."""
    lookup = parse_run_base(run_base)
    scale = lookup.scale
    mixture = lookup.mixture
    if mixture == "proportional":
        return (f"t9_{scale}_proportional",)
    if mixture == "unimax8":
        names = [f"t9_{scale}_unimax8"]
        if scale == "1e21":
            names.append("t9_1e21_unimax8_missing")
        return tuple(names)
    if mixture == "olmix_d001_kl005_cap4":
        return (f"t9_{scale}_olmix_uncheatable",)
    if mixture == "dsp_effexp_kl01":
        names = [f"t9_{scale}_dsp_uncheatable"]
        if scale == "1e21":
            names.append("t9_1e21_dsp_effexp_kl01")
        return tuple(names)
    if mixture == "olmix_onephase_uncheatable_d001_kl005_cap4":
        return (f"t9_olmix_onephase_uncheatable_d001_kl005_cap4_{scale}",)
    if mixture == "olmix_onephase_uncheatable_d001_kl0p1_cap4":
        return (f"t9_olmix_onephase_uncheatable_d001_kl0p1_cap4_{scale}",)
    if mixture == "dsp_onephase_effexp_uncheatable_kl0p1":
        return (f"t9_dsp_onephase_effexp_uncheatable_kl0p1_{scale}",)
    if mixture == "olmix_table9_d001_kl005_cap4":
        names = [f"t9_{scale}_olmix_table9"]
        if scale == "1e21":
            names.append("t9_1e21_olmix_table9_missing")
        return tuple(names)
    if mixture == "dsp_effexp_table9_kl0025":
        return (f"t9_{scale}_dsp_table9",)
    if mixture == "olmix_onephase_table9_d001_kl005_cap4":
        return (f"t9_olmix_onephase_table9_d001_kl005_cap4_{scale}",)
    if mixture == "olmix_onephase_table9_d001_kl0p005_cap4":
        return (f"t9_olmix_onephase_table9_d001_kl0p005_cap4_{scale}",)
    if mixture == "dsp_onephase_effexp_table9_kl0p1":
        return (f"t9_dsp_onephase_effexp_table9_kl0p1_{scale}",)
    return ()


def native_olmo_eval_rows(api: wandb.Api) -> dict[str, dict[str, Any]]:
    """Return latest native Table-9 eval metrics keyed by source training run name."""
    rows: dict[str, dict[str, Any]] = {}
    for run_base in RUN_BASES:
        for eval_name in native_eval_names_for_run_base(run_base):
            runs = api.runs(
                NATIVE_OLMO_EVAL_PROJECT,
                filters={"display_name": {"$regex": f"^{re.escape(eval_name)}$"}},
                order="-created_at",
                per_page=20,
            )
            for run in runs:
                if run.name != eval_name:
                    continue
                macro_bpb = olmo_table9_macro_bpb(run.summary)
                if macro_bpb is None:
                    continue
                rows[run_base] = {
                    OLMO_TABLE9_MACRO_COLUMN: macro_bpb,
                    OLMO_PRIMARY_MEAN_COLUMN: scalar(
                        run.summary, "olmo_base_eval/easy_bpb/_summary/primary_metric_mean"
                    ),
                    "olmo_native_eval_state": run.state,
                    "olmo_native_eval_wandb_name": run.name,
                    "olmo_native_eval_wandb_url": run.url,
                    "olmo_native_eval_created_at": run.created_at,
                }
                break
            if run_base in rows:
                break
    return rows


def collect_wandb_rows() -> pd.DataFrame:
    api = wandb.Api()
    native_eval_by_run_base = native_olmo_eval_rows(api)
    rows: list[dict[str, Any]] = []
    for run_base in RUN_BASES:
        lookup = parse_run_base(run_base)
        run = latest_run_for_base(api, run_base)
        if run is None:
            rows.append(
                {
                    "run_base": run_base,
                    "mixture": lookup.mixture,
                    "mixture_display": MIXTURE_DISPLAY[lookup.mixture],
                    "scale": lookup.scale,
                    "flops": lookup.flops,
                    "state": "missing",
                    "is_completed": False,
                }
            )
            continue
        summary_values = scalar_summary_values(run.summary)
        row: dict[str, Any] = {
            "run_base": run_base,
            "mixture": lookup.mixture,
            "mixture_display": MIXTURE_DISPLAY[lookup.mixture],
            "scale": lookup.scale,
            "flops": lookup.flops,
            "state": run.state,
            "is_completed": run.state == "finished",
            "wandb_name": run.name,
            "wandb_url": run.url,
            "created_at": run.created_at,
            "data_seed": run.config.get("data_seed"),
            "eval_bpb": scalar(run.summary, "eval/bpb"),
            "eval_macro_bpb": scalar(run.summary, "eval/macro_bpb"),
            "eval_uncheatable_eval_bpb": scalar(run.summary, "eval/uncheatable_eval/bpb"),
            "eval_uncheatable_eval_macro_bpb": scalar(run.summary, "eval/uncheatable_eval/macro_bpb"),
            "eval_loss": scalar(run.summary, "eval/loss"),
            "eval_macro_loss": scalar(run.summary, "eval/macro_loss"),
            OLMO_TABLE9_MACRO_COLUMN: olmo_table9_macro_bpb(run.summary),
            OLMO_PRIMARY_MEAN_COLUMN: scalar(run.summary, "olmo_base_eval/easy_bpb/_summary/primary_metric_mean"),
            "train_loss": scalar(run.summary, "train/loss") or scalar(run.summary, "loss"),
            "wandb_step": run.summary.get("_step"),
        }
        row.update({key: value for key, value in summary_values.items() if key.startswith("eval/")})
        native_eval = native_eval_by_run_base.get(run_base)
        if native_eval is not None:
            row.update(native_eval)
        rows.append(row)
    return pd.DataFrame(rows)


def add_metric_trace(
    fig: go.Figure, df: pd.DataFrame, metric: str, *, metric_label: str | None = None, visible: bool = True
) -> None:
    displayed_metric = metric_label or metric
    for display_name in df["mixture_display"].drop_duplicates():
        subset = df[df["mixture_display"] == display_name].sort_values("flops")
        completed = subset[(subset["is_completed"]) & subset[metric].notna()]
        partial = subset[(~subset["is_completed"]) & subset[metric].notna()]
        if not completed.empty:
            fig.add_trace(
                go.Scatter(
                    x=completed["flops"],
                    y=completed[metric],
                    mode="lines+markers",
                    name=display_name,
                    legendgroup=display_name,
                    showlegend=True,
                    visible=visible,
                    marker=dict(size=11, color=MIXTURE_COLOR[display_name]),
                    line=dict(
                        width=3,
                        color=MIXTURE_COLOR[display_name],
                        dash=MIXTURE_LINE_DASH.get(display_name, "solid"),
                    ),
                    customdata=completed[["run_base", "state", "wandb_url", "data_seed"]].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        "state=%{customdata[1]}<br>"
                        "data_seed=%{customdata[3]}<br>"
                        f"{displayed_metric}=%{{y:.4f}}<br>"
                        "%{customdata[2]}<extra></extra>"
                    ),
                ),
            )
        if not partial.empty:
            fig.add_trace(
                go.Scatter(
                    x=partial["flops"],
                    y=partial[metric],
                    mode="markers",
                    name=f"{display_name} (latest non-final)",
                    legendgroup=display_name,
                    showlegend=True,
                    visible=visible,
                    marker=dict(
                        size=11,
                        color=MIXTURE_COLOR[display_name],
                        symbol="circle-open",
                        line=dict(width=2),
                        opacity=0.45,
                    ),
                    customdata=partial[["run_base", "state", "wandb_url", "data_seed"]].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        "NOT FINAL: state=%{customdata[1]}<br>"
                        "data_seed=%{customdata[3]}<br>"
                        f"{displayed_metric}=%{{y:.4f}}<br>"
                        "%{customdata[2]}<extra></extra>"
                    ),
                ),
            )


def build_scaling_figure(
    df: pd.DataFrame,
    *,
    metric: str,
    title: str,
    subtitle: str,
    y_label: str = "BPB",
) -> go.Figure:
    fig = go.Figure()
    add_metric_trace(fig, df, metric)
    fig.update_xaxes(type="log", title_text="Training FLOPs")
    fig.update_yaxes(title_text=y_label)
    fig.update_layout(
        title=f"{title}<br><sup>{subtitle}</sup>",
        template="plotly_white",
        width=1050,
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=70, r=40, t=110, b=180),
    )
    return fig


def build_metric_dropdown_figure(
    df: pd.DataFrame,
    *,
    metrics: list[MetricSpec],
    default_metric: str,
    title: str,
    subtitle: str,
) -> go.Figure:
    fig = go.Figure()
    trace_metric_indices: dict[str, list[int]] = {}
    available_metrics = [metric for metric in metrics if metric.column in df and df[metric.column].notna().any()]
    if not available_metrics:
        raise ValueError("No requested metric columns are available to plot")

    for metric in available_metrics:
        before = len(fig.data)
        add_metric_trace(
            fig,
            df,
            metric.column,
            metric_label=metric.label,
            visible=metric.column == default_metric,
        )
        trace_metric_indices[metric.column] = list(range(before, len(fig.data)))

    if default_metric not in trace_metric_indices:
        default_metric = available_metrics[0].column
    default_spec = next(metric for metric in available_metrics if metric.column == default_metric)
    buttons = []
    for metric in available_metrics:
        visible = [False] * len(fig.data)
        for trace_index in trace_metric_indices[metric.column]:
            visible[trace_index] = True
        buttons.append(
            {
                "label": metric.label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title.text": f"{title}: {metric.label}<br><sup>{subtitle}</sup>",
                        "yaxis.title.text": metric.y_label,
                    },
                ],
            }
        )

    fig.update_xaxes(type="log", title_text="Training FLOPs")
    fig.update_yaxes(title_text=default_spec.y_label)
    fig.update_layout(
        title=f"{title}: {default_spec.label}<br><sup>{subtitle}</sup>",
        template="plotly_white",
        width=1050,
        height=760,
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.11,
                "yanchor": "top",
            }
        ],
        annotations=[
            {
                "text": "Metric:",
                "xref": "paper",
                "yref": "paper",
                "x": 0.0,
                "y": 1.15,
                "showarrow": False,
                "font": {"size": 13},
                "xanchor": "left",
            }
        ],
        legend=dict(orientation="h", yanchor="bottom", y=-0.30, xanchor="center", x=0.5),
        margin=dict(l=70, r=40, t=150, b=190),
    )
    return fig


def metric_y_label(metric_key: str) -> str:
    if metric_key.endswith("/bpb") or metric_key.endswith("_bpb") or "bpb" in metric_key:
        return "BPB"
    if metric_key.endswith("/loss") or metric_key.endswith("_loss") or "loss" in metric_key:
        return "loss"
    if metric_key.endswith("/acc") or metric_key.endswith("_acc") or "accuracy" in metric_key:
        return "accuracy"
    return "value"


def table9_metric_specs(df: pd.DataFrame) -> list[MetricSpec]:
    eval_columns = [column for column in df.columns if column.startswith("eval/") and df[column].notna().any()]
    preferred_columns = [
        column for column in PREFERRED_TABLE9_METRICS if column in df.columns and df[column].notna().any()
    ]
    remaining = sorted(column for column in eval_columns if column not in set(preferred_columns))
    labels = {
        OLMO_TABLE9_MACRO_COLUMN: "OLMoBaseEval Table-9 51-component macro BPB",
        OLMO_PRIMARY_MEAN_COLUMN: "OLMoBaseEval raw 109-task primary mean BPB",
    }
    return [
        MetricSpec(column=column, label=labels.get(column, column), y_label=metric_y_label(column))
        for column in [*preferred_columns, *remaining]
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_wandb_rows()
    completed = df[df["is_completed"]].copy()
    all_csv = args.output_dir / "delphi_scaling_latest_wandb.csv"
    completed_csv = args.output_dir / "delphi_scaling_completed_wandb.csv"
    df.to_csv(all_csv, index=False)
    completed.to_csv(completed_csv, index=False)

    uncheatable_df = df[df["mixture"].isin(UNCH_TARGET_MIXTURES)].copy()
    table9_proxy_df = df[df["mixture"].isin(TABLE9_PROXY_MIXTURES)].copy()
    uncheatable_fig = build_scaling_figure(
        uncheatable_df,
        metric="eval_uncheatable_eval_bpb",
        title="Delphi scaling progress: eval/uncheatable_eval/bpb",
        subtitle="Two baselines plus uncheatable-optimized OLMix and DSP effective-exposure. Lower is better. Open markers are latest non-final W&B summaries.",
    )
    table9_proxy_fig = build_metric_dropdown_figure(
        table9_proxy_df,
        metrics=table9_metric_specs(table9_proxy_df),
        default_metric=OLMO_TABLE9_MACRO_COLUMN,
        title="Delphi scaling progress: OLMoBaseEval/Table-9 candidates",
        subtitle="Native Table-9 macro BPB when available; dropdown includes training-time eval proxies. Lower is better. Open markers are latest non-final W&B summaries.",
    )

    outputs = [
        (uncheatable_fig, "delphi_scaling_progress_uncheatable_bpb"),
        (table9_proxy_fig, "delphi_scaling_progress_olmo_base_eval_macro_bpb_proxy"),
    ]
    for fig, stem in outputs:
        html_path = args.output_dir / f"{stem}.html"
        png_path = args.output_dir / f"{stem}.png"
        fig.write_html(
            html_path,
            include_plotlyjs="cdn",
            config={"toImageButtonOptions": {"format": "png", "scale": 4}},
        )
        fig.write_image(png_path, scale=3)
        print(f"Wrote {html_path}")
        print(f"Wrote {png_path}")
    print(f"Wrote {all_csv}")
    print(f"Wrote {completed_csv}")
    print(
        completed[
            [
                "run_base",
                "state",
                "eval_uncheatable_eval_bpb",
                "eval_uncheatable_eval_macro_bpb",
                "eval_bpb",
                "eval_macro_bpb",
                OLMO_TABLE9_MACRO_COLUMN,
                OLMO_PRIMARY_MEAN_COLUMN,
                "wandb_url",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
