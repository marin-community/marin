# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "kaleido==0.2.1",
#   "numpy",
#   "pandas",
#   "plotly",
# ]
# ///
"""Visualize phase divergence for the four Uncheatable epsilon=0.005 policies."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_PANEL_DIR = SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_validation_panel_20260712"
DEFAULT_RESULTS_DIR = SCRIPT_DIR / "reference_outputs" / "decoupled_phase_information_validation_results_20260712"
DEFAULT_OBSERVED_RESULTS = DEFAULT_RESULTS_DIR / "observed_results.csv"
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}

CANDIDATES = {
    "Canonical DSP": "dphase_unch05_can_e0p005",
    "Effective-exposure DSP": "dphase_unch05_eff_e0p005",
    "Effective-exposure + geometry": "dphase_unch05_geo_e0p005",
    "Separate heads": "dphase_unch05_sep_e0p005",
}
MODEL_COLORS = ["#D73027", "#FC8D59", "#1A9850", "#4575B4"]
PHASE_FRACTIONS = np.array([0.8, 0.2])
LOG_RATIO_CLIP = 6.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--panel-dir", type=Path, default=DEFAULT_PANEL_DIR)
    parser.add_argument("--observed-results", type=Path, default=DEFAULT_OBSERVED_RESULTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser.parse_args()


def short_domain(domain: str) -> str:
    if domain.startswith("dolma3_cc/"):
        return "cc/" + domain.removeprefix("dolma3_cc/")
    if domain.startswith("dolma3_"):
        return domain.removeprefix("dolma3_")
    if domain.startswith("dolmino_"):
        return domain.removeprefix("dolmino_")
    return domain


def load_candidates(panel_dir: Path, observed_results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    observed = pd.read_csv(observed_results).set_index("candidate")
    long_rows: list[pd.DataFrame] = []
    aggregate_reference: np.ndarray | None = None
    domains_reference: list[str] | None = None
    for model, candidate in CANDIDATES.items():
        frame = pd.read_csv(panel_dir / "mixtures" / f"{candidate}.csv")
        domains = frame["domain"].astype(str).tolist()
        aggregate = frame["aggregate_weight"].to_numpy(float)
        if domains_reference is None:
            domains_reference = domains
            aggregate_reference = aggregate
        elif domains != domains_reference or not np.allclose(aggregate, aggregate_reference, atol=1e-12):
            raise ValueError(f"Candidate {candidate} does not preserve the common aggregate policy")

        phase_0 = frame["phase_0_weight"].to_numpy(float)
        phase_1 = frame["phase_1_weight"].to_numpy(float)
        reconstructed = PHASE_FRACTIONS[0] * phase_0 + PHASE_FRACTIONS[1] * phase_1
        if not np.allclose(reconstructed, aggregate, atol=1e-12):
            raise ValueError(f"Aggregate reconstruction failed for {candidate}")
        row = observed.loc[candidate]
        candidate_rows = pd.DataFrame(
            {
                "model": model,
                "candidate": candidate,
                "domain": domains,
                "short_domain": [short_domain(domain) for domain in domains],
                "phase_0_weight": phase_0,
                "phase_1_weight": phase_1,
                "aggregate_weight": aggregate,
                "phase_delta": phase_1 - phase_0,
                "phase_delta_pp": 100.0 * (phase_1 - phase_0),
                "absolute_tv_contribution": 0.5 * np.abs(phase_1 - phase_0),
                "log2_phase_ratio": np.log2(np.clip(phase_1, 1e-10, None) / np.clip(phase_0, 1e-10, None)),
                "observed_uncheatable_bpb": float(row["observed_uncheatable_bpb"]),
                "phase_information": float(row["phase_information"]),
                "phase_tv": float(row["phase_tv"]),
            }
        )
        long_rows.append(candidate_rows)
    long = pd.concat(long_rows, ignore_index=True)
    model_summary = (
        long.groupby(["model", "candidate"], sort=False)
        .agg(
            phase_tv=("phase_tv", "first"),
            phase_information=("phase_information", "first"),
            observed_uncheatable_bpb=("observed_uncheatable_bpb", "first"),
            max_bucket_shift_pp=("phase_delta_pp", lambda values: float(np.abs(values).max())),
        )
        .reset_index()
    )
    return long, model_summary


def ordered_domains(long: pd.DataFrame) -> list[str]:
    order = (
        long.groupby("short_domain")
        .agg(max_abs_shift=("phase_delta_pp", lambda values: float(np.abs(values).max())))
        .sort_values("max_abs_shift", ascending=False)
    )
    return order.index.tolist()


def matrix(long: pd.DataFrame, domains: list[str], column: str) -> np.ndarray:
    return (
        long.pivot(index="short_domain", columns="model", values=column)
        .reindex(index=domains, columns=list(CANDIDATES))
        .to_numpy(float)
    )


def render(long: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    models = list(CANDIDATES)
    domains = ordered_domains(long)
    delta = matrix(long, domains, "phase_delta_pp")
    ratio = np.clip(matrix(long, domains, "log2_phase_ratio"), -LOG_RATIO_CLIP, LOG_RATIO_CLIP)
    phase_0 = 100.0 * matrix(long, domains, "phase_0_weight")
    phase_1 = 100.0 * matrix(long, domains, "phase_1_weight")
    aggregate = 100.0 * matrix(long, domains, "aggregate_weight")
    contribution = 100.0 * matrix(long, domains, "absolute_tv_contribution")

    customdata = np.stack([phase_0, phase_1, aggregate, delta, ratio, contribution], axis=-1)
    max_delta = float(np.abs(delta).max())
    max_ratio = float(np.abs(ratio).max())
    max_weight = float(max(phase_0.max(), phase_1.max()))

    fig = make_subplots(
        rows=2,
        cols=1,
        row_heights=[0.18, 0.82],
        vertical_spacing=0.08,
        subplot_titles=["Global phase divergence", ""],
    )
    fig.add_trace(
        go.Bar(
            x=summary["model"],
            y=summary["phase_tv"],
            marker_color=MODEL_COLORS,
            text=[f"TV={row.phase_tv:.3f}<br>BPB={row.observed_uncheatable_bpb:.6f}" for row in summary.itertuples()],
            textposition="outside",
            customdata=np.stack(
                [summary["phase_information"], summary["max_bucket_shift_pp"], summary["candidate"]], axis=-1
            ),
            hovertemplate=(
                "%{customdata[2]}<br>phase TV=%{y:.4f}<br>phase information=%{customdata[0]:.4f}"
                "<br>largest bucket shift=%{customdata[1]:.2f} pp<extra></extra>"
            ),
            showlegend=False,
        ),
        row=1,
        col=1,
    )
    heatmap = go.Heatmap(
        z=delta,
        x=models,
        y=domains,
        colorscale="RdYlGn_r",
        zmid=0.0,
        zmin=-max_delta,
        zmax=max_delta,
        colorbar={"title": "Late - early<br>weight (pp)"},
        customdata=customdata,
        hovertemplate=(
            "%{y}<br>%{x}<br>phase 0=%{customdata[0]:.3f}%<br>phase 1=%{customdata[1]:.3f}%"
            "<br>aggregate=%{customdata[2]:.3f}%<br>late - early=%{customdata[3]:+.3f} pp"
            "<br>log2(late/early)=%{customdata[4]:+.3f}<br>TV contribution=%{customdata[5]:.3f} pp"
            "<extra></extra>"
        ),
    )
    fig.add_trace(heatmap, row=2, col=1)

    metric_specs = [
        (delta, "Per-bucket phase shift: late minus early", "Late - early<br>weight (pp)", -max_delta, max_delta, 0.0),
        (ratio, "Per-bucket log2 phase ratio", "log2(late/early)", -max_ratio, max_ratio, 0.0),
        (phase_0, "Phase 0 (early) mixture weights", "Phase 0<br>weight (%)", 0.0, max_weight, None),
        (phase_1, "Phase 1 (late) mixture weights", "Phase 1<br>weight (%)", 0.0, max_weight, None),
    ]
    buttons = []
    for values, title, colorbar_title, zmin, zmax, zmid in metric_specs:
        buttons.append(
            {
                "label": title,
                "method": "update",
                "args": [
                    {
                        "z": [values],
                        "zmin": [zmin],
                        "zmax": [zmax],
                        "zmid": [zmid],
                        "colorbar.title.text": [colorbar_title],
                    },
                    {},
                    [1],
                ],
            }
        )

    fig.update_yaxes(title_text="Phase TV", range=[0, float(summary["phase_tv"].max()) * 1.35], row=1, col=1)
    fig.update_xaxes(side="top", row=2, col=1)
    fig.update_yaxes(title_text="Bucket (ordered by maximum absolute shift)", autorange="reversed", row=2, col=1)
    fig.update_layout(
        title={
            "text": "Uncheatable fixed-aggregate policies at epsilon_phase=0.005",
            "x": 0.5,
        },
        template="plotly_white",
        width=1500,
        height=1550,
        margin={"l": 285, "r": 150, "t": 165, "b": 80},
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
        paper_bgcolor="white",
        plot_bgcolor="white",
    )
    fig.write_html(
        output_dir / "uncheatable_epsilon005_phase_divergence.html", include_plotlyjs=True, config=EXPORT_CONFIG
    )
    fig.write_image(output_dir / "uncheatable_epsilon005_phase_divergence.png", scale=2)


def write_report(long: pd.DataFrame, summary: pd.DataFrame, output_dir: Path) -> None:
    top_rows = []
    for model, model_rows in long.groupby("model", sort=False):
        for row in model_rows.nlargest(5, "absolute_tv_contribution").itertuples():
            top_rows.append(
                {
                    "model": model,
                    "domain": row.domain,
                    "phase_0_weight": row.phase_0_weight,
                    "phase_1_weight": row.phase_1_weight,
                    "phase_delta": row.phase_delta,
                    "absolute_tv_contribution": row.absolute_tv_contribution,
                }
            )
    top = pd.DataFrame(top_rows)
    top.to_csv(output_dir / "uncheatable_epsilon005_top_phase_shifts.csv", index=False)
    summary.to_csv(output_dir / "uncheatable_epsilon005_phase_divergence_summary.csv", index=False)
    summary_rows = "\n".join(
        f"- **{row.model}:** phase TV {row.phase_tv:.4f}; largest bucket shift "
        f"{row.max_bucket_shift_pp:.2f} pp; observed BPB {row.observed_uncheatable_bpb:.6f}."
        for row in summary.itertuples()
    )
    report = f"""# Uncheatable epsilon=0.005 phase divergence

All four policies preserve the same aggregate mixture to numerical precision and use phase-information budget 0.005.

{summary_rows}

The interactive heatmap defaults to absolute phase-weight differences and can switch to log2 phase ratios or either
phase's raw weights. Buckets are ordered by their largest absolute phase shift across the four models.
"""
    (output_dir / "uncheatable_epsilon005_phase_divergence.md").write_text(report)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    long, summary = load_candidates(args.panel_dir, args.observed_results)
    long.to_csv(args.output_dir / "uncheatable_epsilon005_phase_divergence.csv", index=False)
    render(long, summary, args.output_dir)
    write_report(long, summary, args.output_dir)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
