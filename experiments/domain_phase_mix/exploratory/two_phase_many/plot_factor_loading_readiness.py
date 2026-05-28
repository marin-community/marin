# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "numpy",
#   "pandas",
#   "plotly",
# ]
# ///

# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Plot factor loadings with DSP-based task predictability/readiness.

This is a diagnostic for pruning low-readiness metrics from the collaborator
Grug-v4 factor aggregate. The factor basis comes from the reproduced 41-metric
varimax aggregate; readiness comes from the current per-metric canonical DSP
controllability table.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_LOADINGS = (
    REPO_ROOT
    / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "collaborator_grug_v4_aggregate_repro_20260525/sent_raw_metric_matrix_300m_zip/factor_loadings.csv"
)
DEFAULT_DSP_READINESS = (
    REPO_ROOT
    / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "aggregate_metric_clean_slate_20260518/metric_controllability_effective_exposure_dsp.csv"
)
DEFAULT_PROPORTIONAL_NOISE_MATRIX = (
    REPO_ROOT
    / "experiments/domain_phase_mix/exploratory/two_phase_many/metric_registry/"
    "raw_metric_matrix_300m/raw_metric_matrix_300m_with_proportional_noise.csv"
)
DEFAULT_OUTPUT_DIR = (
    REPO_ROOT
    / "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "predictable_metric_factor_pruning_20260527"
)
DEFAULT_PREDICTABLE_R = 0.50
DEFAULT_BORDERLINE_R = 0.35


@dataclass(frozen=True)
class ReadinessMatch:
    """DSP readiness match for one factor metric."""

    requested_metric: str
    readiness_metric: str | None
    match_type: str
    dsp_fit_status: str | None
    dsp_oof_r: float
    dsp_oof_spearman: float
    dsp_oof_r2: float
    dsp_role: str | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loadings-csv", type=Path, default=DEFAULT_LOADINGS)
    parser.add_argument("--dsp-readiness-csv", type=Path, default=DEFAULT_DSP_READINESS)
    parser.add_argument("--proportional-noise-matrix", type=Path, default=DEFAULT_PROPORTIONAL_NOISE_MATRIX)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--predictable-r", type=float, default=DEFAULT_PREDICTABLE_R)
    parser.add_argument("--borderline-r", type=float, default=DEFAULT_BORDERLINE_R)
    return parser.parse_args()


def suffix_aliases(metric: str) -> list[tuple[str, str]]:
    """Return audited readiness aliases for common metric naming drift."""
    if "/" not in metric:
        return [(metric, "exact")]
    base, suffix = metric.rsplit("/", 1)
    aliases = [(metric, "exact")]
    if suffix == "bpb":
        aliases.extend(
            [
                (f"{base}/loss", "bpb_to_loss"),
                (f"{base}/nll", "bpb_to_nll"),
            ]
        )
    elif suffix == "choice_logprob":
        aliases.extend(
            [
                (f"{base}/choice_logprob_norm", "choice_logprob_to_norm"),
                (f"{base}/logprob", "choice_logprob_to_logprob"),
                (f"{base}/bpb", "choice_logprob_to_bpb"),
                (f"{base}/choice_prob_norm", "choice_logprob_to_prob_norm"),
                (f"{base}/choice_prob", "choice_logprob_to_prob"),
            ]
        )
    elif suffix == "logprob":
        aliases.extend(
            [
                (f"{base}/choice_logprob", "logprob_to_choice_logprob"),
                (f"{base}/choice_logprob_norm", "logprob_to_choice_logprob_norm"),
                (f"{base}/bpb", "logprob_to_bpb"),
            ]
        )
    return aliases


def match_readiness(metric: str, dsp: pd.DataFrame) -> ReadinessMatch:
    """Match a factor metric to a DSP readiness row."""
    by_metric = dsp.set_index("metric", drop=False)
    for candidate, match_type in suffix_aliases(metric):
        if candidate not in by_metric.index:
            continue
        row = by_metric.loc[candidate]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        return ReadinessMatch(
            requested_metric=metric,
            readiness_metric=str(row["metric"]),
            match_type=match_type,
            dsp_fit_status=str(row.get("dsp_fit_status", "")),
            dsp_oof_r=float(row.get("dsp_oof_pearson", np.nan)),
            dsp_oof_spearman=float(row.get("dsp_oof_spearman", np.nan)),
            dsp_oof_r2=float(row.get("dsp_oof_r2", np.nan)),
            dsp_role=str(row.get("recommended_role", "")),
        )
    return ReadinessMatch(
        requested_metric=metric,
        readiness_metric=None,
        match_type="missing",
        dsp_fit_status=None,
        dsp_oof_r=float("nan"),
        dsp_oof_spearman=float("nan"),
        dsp_oof_r2=float("nan"),
        dsp_role=None,
    )


def match_proportional_snr(metric: str, matrix: pd.DataFrame) -> dict[str, object]:
    """Estimate proportional-anchor SNR from signal rows and proportional noise rows."""
    signal = matrix[matrix["row_kind"].eq("signal")]
    noise = matrix[matrix["row_kind"].eq("noise_variable_subset_proportional")]
    for candidate, match_type in suffix_aliases(metric):
        if candidate not in matrix.columns:
            continue
        signal_values = signal[candidate].dropna().astype(float)
        noise_values = noise[candidate].dropna().astype(float)
        if len(signal_values) < 2 or len(noise_values) < 2:
            continue
        signal_scale = float(signal_values.std(ddof=1))
        noise_scale = float(noise_values.std(ddof=1))
        signal_to_noise = signal_scale / noise_scale if noise_scale > 0 else float("inf")
        return {
            "proportional_snr_metric": candidate,
            "proportional_snr_match_type": match_type,
            "proportional_signal_to_noise": signal_to_noise,
            "proportional_signal_n": int(len(signal_values)),
            "proportional_noise_n": int(len(noise_values)),
            "proportional_signal_scale": signal_scale,
            "proportional_noise_scale": noise_scale,
        }
    return {
        "proportional_snr_metric": None,
        "proportional_snr_match_type": "missing",
        "proportional_signal_to_noise": float("nan"),
        "proportional_signal_n": 0,
        "proportional_noise_n": 0,
        "proportional_signal_scale": float("nan"),
        "proportional_noise_scale": float("nan"),
    }


def readiness_bucket(row: pd.Series, predictable_r: float, borderline_r: float) -> str:
    """Classify metrics into active factor inputs versus held-out evals."""
    status_ok = row["dsp_fit_status"] == "ok"
    value = float(row["dsp_oof_r"])
    if status_ok and np.isfinite(value) and value >= predictable_r:
        return "predictable"
    if status_ok and np.isfinite(value) and value >= borderline_r:
        return "borderline"
    return "heldout"


def load_joined(loadings_csv: Path, dsp_readiness_csv: Path, proportional_noise_matrix: Path) -> pd.DataFrame:
    """Load factor loadings and join DSP readiness."""
    loadings = pd.read_csv(loadings_csv).rename(columns={"Unnamed: 0": "metric"})
    factor_cols = [column for column in loadings.columns if column != "metric"]
    renamed = {column: f"F{int(column) + 1}" for column in factor_cols if column.isdigit()}
    loadings = loadings.rename(columns=renamed)
    factor_cols = [renamed.get(column, column) for column in factor_cols]
    dsp = pd.read_csv(dsp_readiness_csv)
    matches = pd.DataFrame([match_readiness(metric, dsp).__dict__ for metric in loadings["metric"]])
    joined = loadings.merge(matches, left_on="metric", right_on="requested_metric", how="left")
    joined = joined.drop(columns=["requested_metric"])
    if proportional_noise_matrix.exists():
        matrix = pd.read_csv(proportional_noise_matrix)
        prop_snr = pd.DataFrame([match_proportional_snr(metric, matrix) for metric in loadings["metric"]])
        joined = pd.concat([joined, prop_snr], axis=1)
    else:
        joined["proportional_snr_metric"] = None
        joined["proportional_snr_match_type"] = "missing"
        joined["proportional_signal_to_noise"] = np.nan
        joined["proportional_signal_n"] = 0
        joined["proportional_noise_n"] = 0
        joined["proportional_signal_scale"] = np.nan
        joined["proportional_noise_scale"] = np.nan
    joined["dominant_factor"] = joined[factor_cols].abs().idxmax(axis=1)
    dominant_column_positions = joined.columns.get_indexer(joined["dominant_factor"])
    joined["dominant_loading"] = joined.to_numpy()[np.arange(len(joined)), dominant_column_positions].astype(float)
    joined["dominant_abs_loading"] = joined["dominant_loading"].abs()
    joined["communality"] = (joined[factor_cols] ** 2).sum(axis=1)
    return joined


def sorted_for_plot(joined: pd.DataFrame) -> pd.DataFrame:
    """Sort metrics by factor assignment and readiness for visual inspection."""
    factor_order = {f"F{index}": index for index in range(1, 6)}
    bucket_order = {"predictable": 0, "borderline": 1, "heldout": 2}
    out = joined.copy()
    out["_factor_order"] = out["dominant_factor"].map(factor_order).fillna(99)
    out["_bucket_order"] = out["readiness_bucket"].map(bucket_order).fillna(99)
    out = out.sort_values(
        ["_factor_order", "_bucket_order", "dominant_abs_loading", "metric"],
        ascending=[True, True, False, True],
    )
    return out.drop(columns=["_factor_order", "_bucket_order"]).reset_index(drop=True)


def short_metric_label(metric: str) -> str:
    """Return a readable y-axis label while preserving full names in hover."""
    label = metric
    replacements = (
        ("teacher_forced/", "tf/"),
        ("eval/uncheatable_eval/", "unch/"),
        ("eval/paloma/", "paloma/"),
        ("lm_eval/", "lm/"),
        ("mcq_smooth/", "mcq/"),
        ("humaneval_10shot_canonical_solution", "humaneval10"),
        ("gsm8k_5shot_gold_solution", "gsm8k5"),
        ("truthfulqa_mc1_0shot", "truthfulqa"),
        ("mmlu_sl_verb_5shot", "mmlu_sl_verb"),
        ("arc_challenge_5shot", "arc_chal"),
        ("openbookqa_0shot", "openbookqa"),
        ("choice_logprob", "clp"),
        ("choice_logprob_norm", "clp_norm"),
    )
    for old, new in replacements:
        label = label.replace(old, new)
    if len(label) > 58:
        label = label[:27] + "..." + label[-28:]
    return label


def build_plot(joined: pd.DataFrame, output_html: Path, *, predictable_r: float, borderline_r: float) -> None:
    """Write the loadings/readiness Plotly dashboard."""
    factors = [column for column in joined.columns if column.startswith("F") and column[1:].isdigit()]
    y_labels = [short_metric_label(metric) for metric in joined["metric"]]
    customdata = np.stack(
        [
            joined["metric"].astype(str),
            joined["readiness_bucket"].astype(str),
            joined["readiness_metric"].fillna("").astype(str),
            joined["match_type"].astype(str),
            joined["dsp_oof_r"].map(lambda value: f"{value:.3f}" if np.isfinite(value) else "nan"),
            joined["dsp_oof_spearman"].map(lambda value: f"{value:.3f}" if np.isfinite(value) else "nan"),
            joined["proportional_signal_to_noise"].map(
                lambda value: f"{value:.3f}" if np.isfinite(value) else "missing"
            ),
            joined["proportional_snr_metric"].fillna("").astype(str),
        ],
        axis=-1,
    )
    bucket_color = {"predictable": "#2ca25f", "borderline": "#fec44f", "heldout": "#de2d26"}
    bucket_numeric = joined["readiness_bucket"].map({"heldout": 0, "borderline": 1, "predictable": 2}).to_numpy()
    figure_height = max(900, 22 * len(joined) + 220)
    heatmap = go.Figure()
    heatmap.add_trace(
        go.Heatmap(
            z=joined[factors].to_numpy(),
            x=factors,
            y=y_labels,
            zmid=0.0,
            colorscale="RdBu",
            reversescale=True,
            customdata=customdata,
            hovertemplate=(
                "metric=%{customdata[0]}<br>factor=%{x}<br>loading=%{z:.3f}"
                "<br>bucket=%{customdata[1]}"
                "<br>readiness_metric=%{customdata[2]}"
                "<br>match=%{customdata[3]}"
                "<br>DSP OOF r=%{customdata[4]}"
                "<br>DSP OOF Spearman=%{customdata[5]}"
                "<br>proportional SNR=%{customdata[6]}"
                "<br>proportional SNR metric=%{customdata[7]}<extra></extra>"
            ),
            colorbar={"title": "loading"},
        )
    )
    heatmap.update_yaxes(autorange="reversed", tickfont={"size": 11})
    heatmap.update_xaxes(title_text="factor")
    heatmap.update_layout(
        title={"text": "Varimax factor loadings, sorted by dominant factor/readiness", "x": 0.5},
        height=figure_height,
        width=1450,
        template="plotly_white",
        margin={"l": 350, "r": 70, "t": 80, "b": 60},
    )

    readiness = make_subplots(
        rows=1,
        cols=3,
        column_widths=[0.42, 0.42, 0.16],
        horizontal_spacing=0.055,
        specs=[[{"type": "bar"}, {"type": "bar"}, {"type": "heatmap"}]],
        subplot_titles=("DSP OOF Pearson r", "SNR at proportional", "Split"),
    )
    readiness.add_trace(
        go.Bar(
            x=joined["dsp_oof_r"],
            y=y_labels,
            orientation="h",
            marker={
                "color": [bucket_color[value] for value in joined["readiness_bucket"]],
                "line": {"color": "#24364f", "width": 0.4},
            },
            customdata=customdata,
            hovertemplate=(
                "metric=%{customdata[0]}<br>DSP OOF r=%{x:.3f}"
                "<br>DSP OOF Spearman=%{customdata[5]}"
                "<br>bucket=%{customdata[1]}"
                "<br>readiness_metric=%{customdata[2]}"
                "<br>match=%{customdata[3]}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    prop_snr_values = joined["proportional_signal_to_noise"].copy()
    readiness.add_trace(
        go.Bar(
            x=prop_snr_values,
            y=y_labels,
            orientation="h",
            marker={
                "color": np.where(prop_snr_values.notna(), "#3182bd", "#bdbdbd"),
                "line": {"color": "#24364f", "width": 0.4},
            },
            customdata=customdata,
            hovertemplate=(
                "metric=%{customdata[0]}<br>proportional SNR=%{x:.3f}"
                "<br>SNR metric=%{customdata[7]}"
                "<br>DSP OOF r=%{customdata[4]}"
                "<br>bucket=%{customdata[1]}<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=2,
    )
    readiness.add_trace(
        go.Heatmap(
            z=bucket_numeric[:, None],
            x=["split"],
            y=y_labels,
            colorscale=[
                [0.0, "#de2d26"],
                [0.49, "#de2d26"],
                [0.5, "#fec44f"],
                [0.74, "#fec44f"],
                [0.75, "#2ca25f"],
                [1.0, "#2ca25f"],
            ],
            showscale=False,
            text=joined["readiness_bucket"],
            texttemplate="%{text}",
            customdata=customdata,
            hovertemplate="metric=%{customdata[0]}<br>split=%{text}<extra></extra>",
        ),
        row=1,
        col=3,
    )
    readiness.add_vline(x=predictable_r, line_dash="dash", line_color="#2ca25f", row=1, col=1)
    readiness.add_vline(x=borderline_r, line_dash="dot", line_color="#b8860b", row=1, col=1)
    readiness.add_vline(x=1.0, line_dash="dot", line_color="#636363", row=1, col=2)
    readiness.add_vline(x=2.0, line_dash="dash", line_color="#3182bd", row=1, col=2)
    readiness.update_xaxes(title_text="DSP OOF r", range=[-0.25, 1.0], row=1, col=1)
    readiness.update_xaxes(title_text="signal/noise scale", range=[0.0, 12.0], row=1, col=2)
    readiness.update_yaxes(autorange="reversed", tickfont={"size": 11})
    readiness.update_yaxes(showticklabels=False, row=1, col=2)
    readiness.update_yaxes(showticklabels=False, row=1, col=3)
    readiness.update_layout(
        title={"text": "Metric readiness: DSP fit quality and proportional-anchor noise", "x": 0.5},
        height=figure_height,
        width=1450,
        template="plotly_white",
        margin={"l": 350, "r": 40, "t": 80, "b": 60},
    )
    output_html.parent.mkdir(parents=True, exist_ok=True)
    heatmap_html = pio.to_html(heatmap, include_plotlyjs="cdn", full_html=False)
    readiness_html = pio.to_html(readiness, include_plotlyjs=False, full_html=False)
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Factor Loadings With DSP Readiness</title>
  <style>
    body {{
      margin: 0;
      padding: 24px 32px 48px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2f4a;
      background: #f7f9fc;
    }}
    h1 {{
      margin: 0 0 8px;
      font-size: 24px;
    }}
    p {{
      max-width: 1120px;
      line-height: 1.45;
      color: #4a5872;
    }}
    .plot-card {{
      overflow-x: auto;
      background: white;
      border: 1px solid #d7deea;
      border-radius: 14px;
      padding: 12px 12px 4px;
      margin-top: 18px;
      box-shadow: 0 1px 8px rgba(26, 39, 68, 0.06);
    }}
    .plot-card .plotly-graph-div {{
      margin: 0 auto;
    }}
    code {{
      background: #edf2f8;
      border-radius: 4px;
      padding: 1px 4px;
    }}
  </style>
</head>
<body>
  <h1>Collaborator v4 factor loadings with DSP-based readiness</h1>
  <p>
    Metrics are sorted by dominant factor and split. Axis labels are shortened for readability;
    hover any row to see the full metric name, readiness alias, DSP OOF <code>r</code>,
    DSP OOF Spearman, and proportional-anchor SNR when available.
  </p>
  <div class="plot-card">{heatmap_html}</div>
  <div class="plot-card">{readiness_html}</div>
</body>
</html>
"""
    output_html.write_text(html)


def write_summary(joined: pd.DataFrame, output_dir: Path, args: argparse.Namespace) -> None:
    """Write compact Markdown and JSON summaries."""
    factors = [column for column in joined.columns if column.startswith("F") and column[1:].isdigit()]
    by_factor = []
    for factor, group in joined.groupby("dominant_factor", sort=True):
        counts = group["readiness_bucket"].value_counts().to_dict()
        heldout = group[group["readiness_bucket"].eq("heldout")].nlargest(5, "dominant_abs_loading")
        by_factor.append(
            {
                "factor": factor,
                "metric_count": int(len(group)),
                "predictable": int(counts.get("predictable", 0)),
                "borderline": int(counts.get("borderline", 0)),
                "heldout": int(counts.get("heldout", 0)),
                "top_heldout_metrics": heldout["metric"].tolist(),
            }
        )
    payload = {
        "loadings_csv": str(args.loadings_csv),
        "dsp_readiness_csv": str(args.dsp_readiness_csv),
        "proportional_noise_matrix": str(args.proportional_noise_matrix),
        "predictable_r_threshold": args.predictable_r,
        "borderline_r_threshold": args.borderline_r,
        "metric_count": int(len(joined)),
        "bucket_counts": {str(k): int(v) for k, v in joined["readiness_bucket"].value_counts().items()},
        "factors": by_factor,
        "factor_columns": factors,
        "unmatched_metrics": joined[joined["match_type"].eq("missing")]["metric"].tolist(),
        "proportional_snr_available_count": int(joined["proportional_signal_to_noise"].notna().sum()),
    }
    (output_dir / "factor_readiness_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    lines = [
        "# Factor Loading Readiness Sanity Check",
        "",
        f"- Factor loadings: `{args.loadings_csv}`",
        f"- DSP readiness: `{args.dsp_readiness_csv}`",
        f"- Predictable threshold: DSP OOF Pearson `r` >= `{args.predictable_r:.2f}`",
        f"- Borderline threshold: DSP OOF Pearson `r` >= `{args.borderline_r:.2f}`",
        f"- Proportional-anchor SNR available for `{int(joined['proportional_signal_to_noise'].notna().sum())}/{len(joined)}` metrics",
        f"- Metrics: `{len(joined)}`",
        "",
        "## Split Counts",
        "",
    ]
    for bucket, count in joined["readiness_bucket"].value_counts().items():
        lines.append(f"- `{bucket}`: `{int(count)}`")
    lines.extend(["", "## By Dominant Factor", ""])
    for entry in by_factor:
        lines.append(
            f"- `{entry['factor']}`: {entry['metric_count']} metrics, "
            f"{entry['predictable']} predictable, {entry['borderline']} borderline, {entry['heldout']} heldout"
        )
        if entry["top_heldout_metrics"]:
            joined_names = ", ".join(f"`{metric}`" for metric in entry["top_heldout_metrics"])
            lines.append(f"  - top heldout by loading: {joined_names}")
    unmatched = payload["unmatched_metrics"]
    lines.extend(["", "## Readiness Alias Notes", ""])
    if unmatched:
        lines.append("Unmatched metrics need a fresh DSP readiness fit before they can be classified confidently:")
        lines.extend(f"- `{metric}`" for metric in unmatched)
    else:
        lines.append("All factor metrics matched a DSP readiness row, sometimes via audited metric-kind aliases.")
    (output_dir / "summary.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joined = load_joined(args.loadings_csv, args.dsp_readiness_csv, args.proportional_noise_matrix)
    joined["readiness_bucket"] = joined.apply(
        readiness_bucket,
        axis=1,
        predictable_r=args.predictable_r,
        borderline_r=args.borderline_r,
    )
    joined = sorted_for_plot(joined)
    joined.to_csv(args.output_dir / "factor_loadings_with_dsp_readiness.csv", index=False)
    split_cols = [
        "metric",
        "dominant_factor",
        "dominant_loading",
        "communality",
        "readiness_bucket",
        "readiness_metric",
        "match_type",
        "dsp_fit_status",
        "dsp_oof_r",
        "dsp_oof_spearman",
        "dsp_oof_r2",
        "dsp_role",
        "proportional_signal_to_noise",
        "proportional_signal_n",
        "proportional_noise_n",
        "proportional_snr_metric",
        "proportional_snr_match_type",
    ]
    joined[split_cols].to_csv(args.output_dir / "metric_predictable_vs_heldout.csv", index=False)
    build_plot(
        joined,
        args.output_dir / "factor_loadings_dsp_readiness.html",
        predictable_r=args.predictable_r,
        borderline_r=args.borderline_r,
    )
    write_summary(joined, args.output_dir, args)
    print(f"wrote {args.output_dir / 'factor_loadings_dsp_readiness.html'}")
    print(joined["readiness_bucket"].value_counts().to_string())


if __name__ == "__main__":
    main()
