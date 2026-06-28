# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido",
#   "pandas",
#   "plotly",
#   "wandb",
# ]
# ///
"""Plot completed Delphi scaling datapoints from W&B.

This is intentionally W&B-backed rather than reading local CSVs, because the
scaling runs are live and Fieldbook tracks execution attempts while W&B has the
latest scalar summaries.

The optimization target for issue #6602/#6608 is the uncheatable-eval BPB, not
the top-level eval BPB. We still export both so mistakes are easy to audit, but
the figure only plots `eval/uncheatable_eval/*`.
"""

from __future__ import annotations

import argparse
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
from plotly.subplots import make_subplots

OUTPUT_DIR = Path(
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/" "delphi_scaling_progress_20260625"
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
    "dsp_canon_kl01_3e18",
    "dsp_canon_kl01_2e19",
    "dsp_canon_kl01_3e20",
    "dsp_canon_kl01_1e21",
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
    "dsp_canon_kl01": "DSP canonical KL=0.1",
}

MIXTURE_COLOR = {
    "Proportional": "#2b6cb0",
    "UniMax-8": "#805ad5",
    "OLMix d=0.01 KL=0.05 cap4": "#dd6b20",
    "DSP effective-exposure KL=0.1": "#2f855a",
    "DSP canonical KL=0.1": "#c53030",
}


@dataclass(frozen=True)
class RunLookup:
    mixture: str
    scale: str
    flops: float


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


def latest_run_for_base(api: wandb.Api, run_base: str):
    runs = api.runs("marin-community/marin", filters={"display_name": {"$regex": f"^{run_base}-"}})
    matching = [run for run in runs if run.name.startswith(f"{run_base}-")]
    if not matching:
        return None
    return sorted(matching, key=lambda run: run.created_at or "")[-1]


def collect_wandb_rows() -> pd.DataFrame:
    api = wandb.Api()
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
        rows.append(
            {
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
                "train_loss": scalar(run.summary, "train/loss") or scalar(run.summary, "loss"),
                "wandb_step": run.summary.get("_step"),
            }
        )
    return pd.DataFrame(rows)


def add_metric_panel(fig: go.Figure, df: pd.DataFrame, metric: str, col: int) -> None:
    for display_name in MIXTURE_DISPLAY.values():
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
                    showlegend=col == 1,
                    marker=dict(size=11, color=MIXTURE_COLOR[display_name]),
                    line=dict(width=3, color=MIXTURE_COLOR[display_name]),
                    customdata=completed[["run_base", "state", "wandb_url", "data_seed"]].to_numpy(),
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        "state=%{customdata[1]}<br>"
                        "data_seed=%{customdata[3]}<br>"
                        f"{metric}=%{{y:.4f}}<br>"
                        "%{customdata[2]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        if not partial.empty:
            fig.add_trace(
                go.Scatter(
                    x=partial["flops"],
                    y=partial[metric],
                    mode="markers",
                    name=f"{display_name} (latest non-final)",
                    legendgroup=display_name,
                    showlegend=col == 1,
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
                        f"{metric}=%{{y:.4f}}<br>"
                        "%{customdata[2]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )


def build_figure(df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            "eval/uncheatable_eval/bpb (target; lower is better)",
            "eval/uncheatable_eval/macro_bpb (lower is better)",
        ),
        horizontal_spacing=0.09,
    )
    add_metric_panel(fig, df, "eval_uncheatable_eval_bpb", 1)
    add_metric_panel(fig, df, "eval_uncheatable_eval_macro_bpb", 2)
    for col in [1, 2]:
        fig.update_xaxes(type="log", title_text="Training FLOPs", row=1, col=col)
        fig.update_yaxes(title_text="BPB", row=1, col=col)
    fig.update_layout(
        title=(
            "Delphi scaling progress: uncheatable-eval BPB target"
            "<br><sup>Solid points are finished W&B runs; open points are latest non-final summaries for context.</sup>"
        ),
        template="plotly_white",
        width=1500,
        height=720,
        legend=dict(orientation="h", yanchor="bottom", y=-0.28, xanchor="center", x=0.5),
        margin=dict(l=70, r=40, t=110, b=180),
    )
    return fig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = collect_wandb_rows()
    completed = df[df["is_completed"]].copy()
    all_csv = args.output_dir / "delphi_scaling_latest_wandb.csv"
    completed_csv = args.output_dir / "delphi_scaling_completed_wandb.csv"
    html_path = args.output_dir / "delphi_scaling_progress_uncheatable_bpb.html"
    png_path = args.output_dir / "delphi_scaling_progress_uncheatable_bpb.png"
    df.to_csv(all_csv, index=False)
    completed.to_csv(completed_csv, index=False)

    fig = build_figure(df)
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
                "wandb_url",
            ]
        ].to_string(index=False)
    )


if __name__ == "__main__":
    main()
