# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs>=2025.5.1",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scipy>=1.14",
# ]
# ///
"""Analyze exact phase-1 branch coordinates shared by two Delphi prefixes."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from scipy import stats

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/delphi_cross_prefix_branch_correlations_20260827"
RESULT_ROOT = "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_3e18_phase1_common_branches_v6e8_20260825"
PREFIXES = (
    "observed_cap10_best",
    "shared_bounded_ensemble_kl0p05",
)
PREFIX_LABELS = {
    "observed_cap10_best": "Observed cap-10 prefix",
    "shared_bounded_ensemble_kl0p05": "DSP KL=0.05 cap-10 prefix",
}
EXPECTED_BRANCH_CODE_COMMIT = "d016caa0fbd0f1f50e29ffa0c9dea5d40f5438e2"
EXPECTED_CONTINUATION_WEIGHTS_SHA256 = "9305b5c1598c9eb11e7f898f709bfb193f37802efaba40a43fbecd0d52c12355"
EXPECTED_DATA_SEED = 930000
EXPECTED_TRAINER_SEED = 0
EXPECTED_PREFIX_REPEAT_SEED = 0
EXPECTED_TERMINAL_STEP = 3006
EXPECTED_BRANCH_COUNT = 21
MAX_WORKERS = 16

METRICS = {
    "eval/uncheatable_eval/bpb": "Uncheatable micro BPB",
    "eval/uncheatable_eval/macro_bpb": "Uncheatable macro BPB",
    "eval/uncheatable_eval/ao3_english/bpb": "AO3 English BPB",
    "eval/uncheatable_eval/arxiv_computer_science/bpb": "arXiv Computer Science BPB",
    "eval/uncheatable_eval/arxiv_physics/bpb": "arXiv Physics BPB",
    "eval/uncheatable_eval/bbc_news/bpb": "BBC News BPB",
    "eval/uncheatable_eval/github_cpp/bpb": "GitHub C++ BPB",
    "eval/uncheatable_eval/github_python/bpb": "GitHub Python BPB",
    "eval/uncheatable_eval/wikipedia_english/bpb": "Wikipedia English BPB",
}

PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


@dataclass(frozen=True)
class BranchResult:
    """One provenance-verified terminal branch result."""

    prefix: str
    continuation_id: str
    run_root: str
    provenance: dict[str, Any]
    metrics: dict[str, float]


def _read_json(fs: gcsfs.GCSFileSystem, path: str) -> dict[str, Any]:
    with fs.open(path.removeprefix("gs://"), "rb") as handle:
        return json.load(handle)


def _read_final_metrics(fs: gcsfs.GCSFileSystem, run_root: str) -> dict[str, Any]:
    path = f"{run_root}/checkpoints/eval_metrics.jsonl"
    with fs.open(path.removeprefix("gs://"), "rt") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    if not rows:
        raise RuntimeError(f"No evaluation rows found at {path}")
    final = max(rows, key=lambda row: int(row.get("step", -1)))
    if int(final.get("step", -1)) != EXPECTED_TERMINAL_STEP:
        raise RuntimeError(f"Unexpected terminal evaluation step at {path}: {final.get('step')}")
    missing = sorted(set(METRICS) - set(final))
    if missing:
        raise RuntimeError(f"Missing terminal metrics at {path}: {missing}")
    return final


def _run_root(fs: gcsfs.GCSFileSystem, prefix: str, continuation_id: str) -> str:
    pattern = f"{RESULT_ROOT}/branch_{prefix}_seed0_{continuation_id}-*"
    matches = sorted(fs.glob(pattern.removeprefix("gs://")))
    if len(matches) != 1:
        raise RuntimeError(f"Expected one run root for {prefix}/{continuation_id}; found {matches}")
    return f"gs://{matches[0]}"


def _load_result(
    fs: gcsfs.GCSFileSystem,
    prefix: str,
    continuation_id: str,
) -> BranchResult:
    run_root = _run_root(fs, prefix, continuation_id)
    provenance = _read_json(fs, f"{run_root}/branch_provenance.json")
    expected = {
        "branch_code_commit": EXPECTED_BRANCH_CODE_COMMIT,
        "continuation_id": continuation_id,
        "continuation_weights_sha256": EXPECTED_CONTINUATION_WEIGHTS_SHA256,
        "data_seed": EXPECTED_DATA_SEED,
        "prefix_candidate_id": prefix,
        "prefix_repeat_seed": EXPECTED_PREFIX_REPEAT_SEED,
        "terminal_checkpoint_step": EXPECTED_TERMINAL_STEP,
        "trainer_seed": EXPECTED_TRAINER_SEED,
    }
    mismatches = {key: (provenance.get(key), value) for key, value in expected.items() if provenance.get(key) != value}
    if mismatches:
        raise RuntimeError(f"Provenance mismatch for {prefix}/{continuation_id}: {mismatches}")
    hardware = provenance.get("continuation_hardware")
    if hardware != {"region": "us-east5", "tpu_type": "v6e-8", "zone": "us-east5-b"}:
        raise RuntimeError(f"Unexpected continuation hardware for {prefix}/{continuation_id}: {hardware}")
    final = _read_final_metrics(fs, run_root)
    return BranchResult(
        prefix=prefix,
        continuation_id=continuation_id,
        run_root=run_root,
        provenance=provenance,
        metrics={metric: float(final[metric]) for metric in METRICS},
    )


def load_results() -> list[BranchResult]:
    """Load and validate the complete 21-by-2 common-branch crossing."""
    fs = gcsfs.GCSFileSystem()
    continuation_ids = tuple(f"fit_maximin_{index:02d}" for index in range(EXPECTED_BRANCH_COUNT))
    inputs = [(prefix, continuation_id) for prefix in PREFIXES for continuation_id in continuation_ids]
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(lambda item: _load_result(fs, *item), inputs))
    if len(results) != EXPECTED_BRANCH_COUNT * len(PREFIXES):
        raise RuntimeError("Common-branch result inventory is incomplete")
    return results


def paired_frame(results: list[BranchResult]) -> pd.DataFrame:
    """Return one row per branch and prefix with all terminal metrics."""
    rows = []
    for result in results:
        row: dict[str, Any] = {
            "prefix": result.prefix,
            "prefix_label": PREFIX_LABELS[result.prefix],
            "continuation_id": result.continuation_id,
            "run_root": result.run_root,
            "provenance_sha256": (
                hashlib.sha256(json.dumps(result.provenance, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            ),
        }
        row.update(result.metrics)
        rows.append(row)
    frame = pd.DataFrame(rows).sort_values(["continuation_id", "prefix"]).reset_index(drop=True)
    if frame.duplicated(["continuation_id", "prefix"]).any():
        raise RuntimeError("Duplicate branch/prefix results found")
    counts = frame.groupby("continuation_id")["prefix"].nunique()
    if len(counts) != EXPECTED_BRANCH_COUNT or not counts.eq(len(PREFIXES)).all():
        raise RuntimeError("Every shared branch must appear under both prefixes")
    return frame


def correlation_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Compute raw endpoint transfer statistics for each metric."""
    rows = []
    left, right = PREFIXES
    for metric, label in METRICS.items():
        wide = frame.pivot(index="continuation_id", columns="prefix", values=metric).sort_index()
        left_values = wide[left].to_numpy(dtype=float)
        right_values = wide[right].to_numpy(dtype=float)
        slope, intercept = np.polyfit(left_values, right_values, deg=1)
        shift = right_values - left_values
        constant_shift_residual = shift - shift.mean()
        rows.append(
            {
                "metric": metric,
                "metric_label": label,
                "pearson": float(stats.pearsonr(left_values, right_values).statistic),
                "spearman": float(stats.spearmanr(left_values, right_values).statistic),
                "mean_shift_kl_minus_observed": float(shift.mean()),
                "constant_shift_residual_sd": float(constant_shift_residual.std(ddof=1)),
                "constant_shift_rmse": float(np.sqrt(np.mean(constant_shift_residual**2))),
                "observed_range": float(np.ptp(left_values)),
                "kl0p05_range": float(np.ptp(right_values)),
                "linear_slope": float(slope),
                "linear_intercept": float(intercept),
            }
        )
    return pd.DataFrame(rows)


def correlation_figure(summary: pd.DataFrame) -> go.Figure:
    """Plot Pearson and Spearman correlations by metric."""
    ordered = summary.sort_values("spearman")
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=ordered["pearson"],
            y=ordered["metric_label"],
            name="Pearson",
            orientation="h",
            marker_color="#d95f38",
            hovertemplate="%{y}<br>Pearson %{x:.4f}<extra></extra>",
        )
    )
    figure.add_trace(
        go.Bar(
            x=ordered["spearman"],
            y=ordered["metric_label"],
            name="Spearman",
            orientation="h",
            marker_color="#147d78",
            hovertemplate="%{y}<br>Spearman %{x:.4f}<extra></extra>",
        )
    )
    figure.update_layout(
        barmode="group",
        height=560,
        margin={"l": 220, "r": 30, "t": 70, "b": 70},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, Avenir, sans-serif", "color": "#17324d", "size": 15},
        title={"text": "Raw endpoint correlation across the two prefixes", "x": 0.02, "xanchor": "left"},
        legend={"orientation": "h", "y": 1.08, "x": 0.68},
        xaxis={"title": "Correlation", "range": [0.90, 1.002], "gridcolor": "#e3ddcf"},
        yaxis={"title": ""},
    )
    return figure


def scatter_figure(frame: pd.DataFrame, summary: pd.DataFrame) -> go.Figure:
    """Build a metric-selectable cross-prefix scatter plot."""
    left, right = PREFIXES
    figure = go.Figure()
    traces_per_metric = 3
    metric_items = list(METRICS.items())
    for metric_index, (metric, _label) in enumerate(metric_items):
        wide = frame.pivot(index="continuation_id", columns="prefix", values=metric).sort_index()
        stats_row = summary.loc[summary["metric"].eq(metric)].iloc[0]
        x = wide[left].to_numpy(dtype=float)
        y = wide[right].to_numpy(dtype=float)
        domain = np.linspace(float(x.min()), float(x.max()), 100)
        visible = metric_index == 0
        figure.add_trace(
            go.Scatter(
                x=x,
                y=y,
                mode="markers+text",
                text=[value.removeprefix("fit_maximin_") for value in wide.index],
                textposition="top center",
                textfont={"size": 9, "color": "#6b7280"},
                marker={"size": 10, "color": "#147d78", "line": {"color": "#0f2e46", "width": 1}},
                customdata=np.asarray(wide.index)[:, None],
                name="Shared branch",
                visible=visible,
                hovertemplate=(
                    "Branch %{customdata[0]}<br>Observed prefix %{x:.6f}<br>KL=0.05 prefix %{y:.6f}<extra></extra>"
                ),
            )
        )
        figure.add_trace(
            go.Scatter(
                x=domain,
                y=stats_row["linear_slope"] * domain + stats_row["linear_intercept"],
                mode="lines",
                line={"color": "#d95f38", "width": 3},
                name="Linear fit",
                visible=visible,
                hoverinfo="skip",
            )
        )
        figure.add_trace(
            go.Scatter(
                x=domain,
                y=domain + stats_row["mean_shift_kl_minus_observed"],
                mode="lines",
                line={"color": "#6b7280", "width": 2, "dash": "dash"},
                name="Constant prefix shift",
                visible=visible,
                hoverinfo="skip",
            )
        )
    buttons = []
    for metric_index, (_, label) in enumerate(metric_items):
        visible = [False] * (len(metric_items) * traces_per_metric)
        start = metric_index * traces_per_metric
        visible[start : start + traces_per_metric] = [True] * traces_per_metric
        stats_row = summary.iloc[metric_index]
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [
                    {"visible": visible},
                    {
                        "title": {
                            "text": (
                                f"{label}: Pearson {stats_row['pearson']:.3f}, " f"Spearman {stats_row['spearman']:.3f}"
                            ),
                            "x": 0.02,
                            "xanchor": "left",
                        },
                        "xaxis.title": f"{PREFIX_LABELS[left]} — {label}",
                        "yaxis.title": f"{PREFIX_LABELS[right]} — {label}",
                    },
                ],
            }
        )
    first = summary.iloc[0]
    figure.update_layout(
        height=690,
        margin={"l": 100, "r": 40, "t": 130, "b": 90},
        paper_bgcolor="#fbf7ee",
        plot_bgcolor="#fbf7ee",
        font={"family": "Avenir Next, Avenir, sans-serif", "color": "#17324d", "size": 15},
        title={
            "text": f"{metric_items[0][1]}: Pearson {first['pearson']:.3f}, Spearman {first['spearman']:.3f}",
            "x": 0.02,
            "xanchor": "left",
        },
        legend={"orientation": "h", "y": 1.08, "x": 0.48},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "showactive": True,
                "x": 0.0,
                "xanchor": "left",
                "y": 1.17,
                "yanchor": "top",
                "bgcolor": "#fffdf8",
                "bordercolor": "#b9c7d8",
                "font": {"size": 14},
            }
        ],
        xaxis={"title": f"{PREFIX_LABELS[left]} — {metric_items[0][1]}", "gridcolor": "#e3ddcf"},
        yaxis={"title": f"{PREFIX_LABELS[right]} — {metric_items[0][1]}", "gridcolor": "#e3ddcf"},
    )
    return figure


def _summary_table(summary: pd.DataFrame) -> str:
    rows = []
    for row in summary.itertuples(index=False):
        rows.append(
            "<tr>"
            f"<td>{row.metric_label}</td>"
            f"<td>{row.pearson:.4f}</td>"
            f"<td>{row.spearman:.4f}</td>"
            f"<td>{row.mean_shift_kl_minus_observed:+.6f}</td>"
            f"<td>{row.constant_shift_residual_sd:.6f}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def write_artifacts(frame: pd.DataFrame, summary: pd.DataFrame) -> None:
    """Write data tables, report, and a self-contained interactive HTML artifact."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT_DIR / "paired_terminal_metrics.csv", index=False)
    summary.to_csv(OUTPUT_DIR / "correlation_summary.csv", index=False)

    correlation_plot = pio.to_html(
        correlation_figure(summary),
        include_plotlyjs=True,
        full_html=False,
        config=PLOT_CONFIG,
        div_id="correlation-summary-plot",
    )
    scatter_plot = pio.to_html(
        scatter_figure(frame, summary),
        include_plotlyjs=False,
        full_html=False,
        config=PLOT_CONFIG,
        div_id="metric-scatter-plot",
    )
    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Delphi cross-prefix branch correlations</title>
  <style>
    :root {{ --ink:#17324d; --muted:#5d6d7e; --paper:#fbf7ee; --card:#fffdf8; --teal:#147d78; --orange:#d95f38; --rule:#ddd5c5; }}
    * {{ box-sizing: border-box; }}
    body {{ margin:0; background:var(--paper); color:var(--ink); font-family:"Avenir Next",Avenir,sans-serif; }}
    main {{ max-width:1320px; margin:0 auto; padding:48px 34px 80px; }}
    h1,h2 {{ font-family:Georgia,"Times New Roman",serif; letter-spacing:-0.025em; }}
    h1 {{ font-size:54px; line-height:1.02; margin:0 0 18px; }}
    h2 {{ font-size:31px; margin:54px 0 12px; }}
    p {{ font-size:18px; line-height:1.62; max-width:980px; color:var(--muted); }}
    .lede {{ font-size:21px; color:var(--ink); }}
    .cards {{ display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:14px; margin:30px 0 36px; }}
    .card {{ background:var(--card); border:1px solid var(--rule); border-top:5px solid var(--teal); padding:20px 22px; }}
    .card strong {{ display:block; font-family:Georgia,"Times New Roman",serif; font-size:34px; margin-bottom:5px; }}
    .card span {{ color:var(--muted); font-size:14px; line-height:1.35; }}
    .callout {{ background:#eef5f3; border-left:6px solid var(--teal); padding:18px 24px; margin:28px 0; }}
    .callout strong {{ color:var(--ink); }}
    .plot {{ background:var(--card); border:1px solid var(--rule); padding:8px 10px; margin:18px 0 26px; }}
    table {{ width:100%; border-collapse:collapse; background:var(--card); font-variant-numeric:tabular-nums; }}
    th,td {{ padding:12px 14px; border-bottom:1px solid var(--rule); text-align:right; }}
    th:first-child,td:first-child {{ text-align:left; }}
    th {{ color:var(--muted); font-size:13px; text-transform:uppercase; letter-spacing:.055em; }}
    .limits {{ border-top:1px solid var(--rule); margin-top:46px; padding-top:20px; }}
    code {{ background:#eee7da; border-radius:4px; padding:2px 5px; }}
    @media (max-width:850px) {{ main {{ padding:30px 18px 60px; }} h1 {{ font-size:39px; }} .cards {{ grid-template-columns:1fr 1fr; }} }}
  </style>
</head>
<body>
<main>
  <h1>How much does branch quality transfer across prefix states?</h1>
  <p class="lede">Twenty-one identical phase-1 mixtures were continued from two different cap-10 phase-boundary checkpoints. Every continuation used the same data seed, trainer seed, v6e-8 hardware, and frozen branch-design manifest. This page compares their terminal Uncheatable metrics.</p>
  <div class="cards">
    <div class="card"><strong>21</strong><span>exactly shared phase-1 mixtures</span></div>
    <div class="card"><strong>2</strong><span>different prefix checkpoints</span></div>
    <div class="card"><strong>{summary['spearman'].min():.3f}</strong><span>lowest component Spearman correlation</span></div>
    <div class="card"><strong>{summary['pearson'].min():.3f}</strong><span>lowest component Pearson correlation</span></div>
  </div>
  <div class="callout"><strong>Main result.</strong> Gross branch ordering transfers extremely strongly between these two nearby cap-10 prefixes across every measured component. Absolute performance does not: prefix-dependent shifts range from improving GitHub C++ by about 0.046 BPB to worsening BBC News by about 0.009 BPB.</div>

  <h2>Correlation across objectives</h2>
  <p>Each bar uses the same 21 branch coordinates. Pearson measures linear level transfer; Spearman measures rank transfer. Higher is stronger.</p>
  <div class="plot">{correlation_plot}</div>

  <h2>Inspect each objective</h2>
  <p>Select an objective in the dropdown. Point labels are branch indices. The orange line is an unconstrained linear fit; the dashed line assumes the prefix only adds a constant metric-specific shift.</p>
  <div class="plot">{scatter_plot}</div>

  <h2>Numerical summary</h2>
  <table>
    <thead><tr><th>Metric</th><th>Pearson</th><th>Spearman</th><th>KL - observed BPB</th><th>Shift SD</th></tr></thead>
    <tbody>{_summary_table(summary)}</tbody>
  </table>

  <div class="limits">
    <h2>What this does and does not establish</h2>
    <p><strong>Established descriptively:</strong> these two prefix states induce almost the same broad ordering over a deliberately wide set of branches. This is direct evidence for a low-dimensional shared branch-response component.</p>
    <p><strong>Not established:</strong> these are raw terminal endpoints, not branch effects relative to matched tied continuations. There is one prefix seed and one common continuation-data seed per branch, so the plots have no seed-level confidence intervals. The result covers two related cap-10 prefixes and cannot establish transfer to proportional, cap-4, or arbitrary canonical-swarm states.</p>
    <p>Frozen provenance: branch commit <code>{EXPECTED_BRANCH_CODE_COMMIT[:12]}</code>, continuation manifest <code>{EXPECTED_CONTINUATION_WEIGHTS_SHA256[:12]}</code>, data seed <code>{EXPECTED_DATA_SEED}</code>, terminal step <code>{EXPECTED_TERMINAL_STEP}</code>.</p>
  </div>
</main>
</body>
</html>
"""
    (OUTPUT_DIR / "cross_prefix_branch_correlations.html").write_text(html)

    report = f"""# Delphi cross-prefix branch correlations

## Result

The largest exact non-control branch crossing currently materialized contains 21 identical phase-1 mixtures under two cap-10 prefixes: `observed_cap10_best` and `shared_bounded_ensemble_kl0p05`. Every continuation uses prefix seed 0, data seed {EXPECTED_DATA_SEED}, trainer seed {EXPECTED_TRAINER_SEED}, and v6e-8 in `us-east5-b`.

Across the nine terminal Uncheatable metrics, Pearson correlation ranges from {summary['pearson'].min():.6f} to {summary['pearson'].max():.6f}; Spearman correlation ranges from {summary['spearman'].min():.6f} to {summary['spearman'].max():.6f}. Uncheatable micro BPB has Pearson {summary.iloc[0]['pearson']:.6f} and Spearman {summary.iloc[0]['spearman']:.6f}.

This is strong descriptive evidence for shared low-dimensional branch ordering across these two nearby prefix states. It is not a branch-gain comparison: the observed prefix lacks a matched tied anchor, and every branch/prefix cell has only one continuation seed.

## Artifacts

- `cross_prefix_branch_correlations.html`: self-contained interactive report.
- `paired_terminal_metrics.csv`: provenance-verified terminal metrics.
- `correlation_summary.csv`: metric-wise correlations, shifts, and residual scale.
"""
    (OUTPUT_DIR / "report.md").write_text(report)


def main() -> None:
    results = load_results()
    frame = paired_frame(results)
    summary = correlation_summary(frame)
    write_artifacts(frame, summary)


if __name__ == "__main__":
    main()
