# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib", "numpy", "pandas", "plotly"]
# ///
"""Plot smooth benchmark x deleted-domain p-value matrices.

The input table is produced by ``analyze_domain_ablation_pvalue_histograms.py``.
It contains one row per selected smooth benchmark and deleted domain, with
one-sided p-values computed against the proportional repeat-noise baseline.

The main matrix visualizes ``-log10(p_harm)``. Larger red values mean that
deleting the domain gives a high-confidence degradation of the smooth
benchmark, under the proportional-noise diagnostic.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go

SCRIPT_DIR = Path(__file__).resolve().parent
INPUT_CSV = (
    SCRIPT_DIR / "reference_outputs" / "domain_ablation_pvalue_histograms_20260623" / "domain_ablation_cell_pvalues.csv"
)
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "domain_ablation_pvalue_matrix_20260623"

TO_IMAGE_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


@dataclass(frozen=True)
class MatrixData:
    frame: pd.DataFrame
    benchmark_order: list[str]
    domain_order: list[str]
    display_labels: dict[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=INPUT_CSV)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    return parser.parse_args()


def benjamini_hochberg_q(values: pd.Series) -> pd.Series:
    """Return Benjamini-Hochberg q-values in the original row order."""
    p = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    q = np.full_like(p, np.nan, dtype=float)
    finite = np.isfinite(p)
    if finite.sum() == 0:
        return pd.Series(q, index=values.index)
    finite_indices = np.flatnonzero(finite)
    order = finite_indices[np.argsort(p[finite])]
    ranked_p = p[order]
    n = len(ranked_p)
    adjusted = ranked_p * n / np.arange(1, n + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    q[order] = np.clip(adjusted, 0.0, 1.0)
    return pd.Series(q, index=values.index)


def short_benchmark_name(name: str) -> str:
    replacements = [
        ("raw_ppl/", "raw/"),
        ("lm_eval/", "lm/"),
        ("teacher_forced/", "tf/"),
        ("mcq_smooth/", "mcq/"),
        ("eval/agentic_coding/", "agent/"),
    ]
    out = name
    for old, new in replacements:
        if out.startswith(old):
            out = new + out.removeprefix(old)
    return out


def short_domain_name(name: str) -> str:
    """Return compact axis labels while keeping full domain names in hover text."""
    replacements = [
        ("dolma3_cc/", "cc/"),
        ("dolma3_", "d3_"),
        ("dolmino_", "dm_"),
        ("science_math_and_technology", "sci_math_tech"),
        ("education_and_jobs", "edu_jobs"),
        ("electronics_and_hardware", "electronics"),
        ("finance_and_business", "finance"),
        ("food_and_dining", "food"),
        ("history_and_geography", "history_geo"),
        ("common_crawl", "cc"),
        ("stack_edu_fim", "stack_fim"),
        ("stem_heavy_crawl", "stem_crawl"),
        ("synth_instruction", "synth_instr"),
        ("synth_thinking", "synth_think"),
        ("synth_math", "synth_math"),
        ("synth_code", "synth_code"),
        ("synth_qa", "synth_qa"),
    ]
    out = name
    for old, new in replacements:
        out = out.replace(old, new)
    return out


def prepare_matrix(path: Path) -> MatrixData:
    frame = pd.read_csv(path)
    required = {
        "benchmark_key",
        "metric",
        "metric_family",
        "metric_kind",
        "target_domain",
        "base_mass",
        "domain_deletion_utility_delta",
        "noise_sd",
        "t_statistic",
        "p_harm",
        "p_improve",
        "p_two_sided",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    frame = frame.copy()
    frame["p_harm_clipped"] = frame["p_harm"].clip(lower=1e-300, upper=1.0)
    frame["neg_log10_p_harm"] = -np.log10(frame["p_harm_clipped"])
    frame["bonferroni_p_harm_by_benchmark"] = frame.groupby("benchmark_key")["p_harm"].transform(
        lambda s: np.minimum(1.0, s * len(s))
    )
    frame["neg_log10_bonferroni_p_harm"] = -np.log10(
        frame["bonferroni_p_harm_by_benchmark"].clip(lower=1e-300, upper=1.0)
    )
    frame["bh_q_harm_all_cells"] = benjamini_hochberg_q(frame["p_harm"])
    frame["deletion_hurts"] = frame["domain_deletion_utility_delta"] < 0.0

    domain_summary = (
        frame.groupby("target_domain", as_index=True)
        .agg(
            burden=("neg_log10_p_harm", "sum"),
            significant_bonferroni=("bonferroni_p_harm_by_benchmark", lambda s: int((s < 0.05).sum())),
            significant_raw=("p_harm", lambda s: int((s < 0.05).sum())),
            base_mass=("base_mass", "first"),
        )
        .sort_values(["significant_bonferroni", "burden", "base_mass"], ascending=[False, False, False])
    )
    domain_order = domain_summary.index.tolist()

    benchmark_summary = (
        frame.groupby("benchmark_key", as_index=True)
        .agg(
            metric_family=("metric_family", "first"),
            max_neglog=("neg_log10_p_harm", "max"),
            min_p=("p_harm", "min"),
            significant_bonferroni=("bonferroni_p_harm_by_benchmark", lambda s: int((s < 0.05).sum())),
        )
        .sort_values(
            ["metric_family", "significant_bonferroni", "max_neglog", "min_p"], ascending=[True, False, False, True]
        )
    )
    benchmark_order = benchmark_summary.index.tolist()
    display_labels = {b: short_benchmark_name(b) for b in benchmark_order}

    return MatrixData(
        frame=frame, benchmark_order=benchmark_order, domain_order=domain_order, display_labels=display_labels
    )


def pivot_values(data: MatrixData, column: str) -> pd.DataFrame:
    pivot = data.frame.pivot(index="benchmark_key", columns="target_domain", values=column)
    return pivot.reindex(index=data.benchmark_order, columns=data.domain_order)


def hover_text(data: MatrixData) -> pd.DataFrame:
    rows = []
    for row in data.frame.itertuples(index=False):
        rows.append(
            {
                "benchmark_key": row.benchmark_key,
                "target_domain": row.target_domain,
                "hover": (
                    f"<b>{row.benchmark_key}</b><br>"
                    f"metric: {row.metric}<br>"
                    f"deleted domain: {row.target_domain}<br>"
                    f"utility delta: {row.domain_deletion_utility_delta:.5g}<br>"
                    f"t statistic: {row.t_statistic:.3g}<br>"
                    f"p_harm: {row.p_harm:.3g}<br>"
                    f"p_improve: {row.p_improve:.3g}<br>"
                    f"p_two_sided: {row.p_two_sided:.3g}<br>"
                    f"Bonferroni p_harm over domains: {row.bonferroni_p_harm_by_benchmark:.3g}<br>"
                    f"BH q_harm over all cells: {row.bh_q_harm_all_cells:.3g}<br>"
                    f"proportional noise sd: {row.noise_sd:.3g}<br>"
                    f"deleted domain mass: {row.base_mass:.3g}"
                ),
            }
        )
    hover = pd.DataFrame(rows).pivot(index="benchmark_key", columns="target_domain", values="hover")
    return hover.reindex(index=data.benchmark_order, columns=data.domain_order)


def write_interactive_heatmap(data: MatrixData, output_dir: Path) -> Path:
    z = pivot_values(data, "neg_log10_p_harm")
    text = hover_text(data)
    y_labels = [data.display_labels[b] for b in z.index]
    x_ticktext = [short_domain_name(domain) for domain in z.columns]
    x_values = list(range(len(z.columns)))
    y_values = list(range(len(z.index)))
    domain_to_x = {domain: index for index, domain in enumerate(z.columns)}
    benchmark_to_y = {benchmark: index for index, benchmark in enumerate(z.index)}

    fig = go.Figure()
    fig.add_trace(
        go.Heatmap(
            z=z.to_numpy(),
            x=x_values,
            y=y_values,
            text=text.to_numpy(),
            hovertemplate="%{text}<extra></extra>",
            colorscale="RdYlGn_r",
            zmin=0.0,
            zmax=max(8.0, float(np.nanquantile(z.to_numpy(), 0.995))),
            colorbar={
                "title": "-log10 p_harm<br>(deletion hurts)",
                "ticks": "outside",
            },
        )
    )

    raw = pivot_values(data, "p_harm")
    bonf = pivot_values(data, "bonferroni_p_harm_by_benchmark")
    raw_marker_y: list[int] = []
    raw_marker_x: list[int] = []
    bonf_marker_y: list[int] = []
    bonf_marker_x: list[int] = []
    for benchmark in raw.index:
        for domain in raw.columns[raw.loc[benchmark].lt(0.05)]:
            raw_marker_x.append(domain_to_x[domain])
            raw_marker_y.append(benchmark_to_y[benchmark])
        for domain in bonf.columns[bonf.loc[benchmark].lt(0.05)]:
            bonf_marker_x.append(domain_to_x[domain])
            bonf_marker_y.append(benchmark_to_y[benchmark])
    fig.add_trace(
        go.Scatter(
            x=raw_marker_x,
            y=raw_marker_y,
            mode="markers",
            marker={
                "symbol": "circle-open",
                "size": 4,
                "color": "rgba(0,0,0,0.38)",
                "line": {"width": 0.8},
            },
            name="Raw p_harm < 0.05",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bonf_marker_x,
            y=bonf_marker_y,
            mode="markers",
            marker={"symbol": "circle-open", "size": 9, "color": "black", "line": {"width": 1.4}},
            name="Bonferroni p_harm < 0.05 within benchmark",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=bonf_marker_x,
            y=bonf_marker_y,
            mode="markers",
            marker={"symbol": "circle-open", "size": 5, "color": "black", "line": {"width": 1.0}},
            showlegend=False,
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        title="Smooth benchmark × deleted domain p-value matrix",
        template="plotly_white",
        width=1500,
        height=max(1800, 18 * len(y_labels) + 340),
        margin={"l": 360, "r": 70, "t": 165, "b": 185},
        xaxis={
            "title": None,
            "tickangle": 48,
            "side": "top",
            "tickmode": "array",
            "tickvals": x_values,
            "ticktext": x_ticktext,
            "tickfont": {"size": 10},
        },
        yaxis={
            "title": "Selected smooth benchmark",
            "tickmode": "array",
            "tickvals": y_values,
            "ticktext": y_labels,
            "range": [len(y_values) - 0.5, -0.5],
        },
        legend={"orientation": "h", "yanchor": "top", "y": -0.06, "xanchor": "center", "x": 0.5},
    )

    out = output_dir / "smooth_benchmark_deleted_domain_pvalue_matrix.html"
    fig.write_html(out, include_plotlyjs="cdn", config=TO_IMAGE_CONFIG)
    return out


def write_top_png(data: MatrixData, output_dir: Path, top_n: int = 70) -> Path:
    top_benchmarks = (
        data.frame.groupby("benchmark_key")["neg_log10_p_harm"]
        .max()
        .sort_values(ascending=False)
        .head(top_n)
        .index.tolist()
    )
    subset = MatrixData(
        frame=data.frame[data.frame["benchmark_key"].isin(top_benchmarks)].copy(),
        benchmark_order=top_benchmarks,
        domain_order=data.domain_order,
        display_labels={b: data.display_labels[b] for b in top_benchmarks},
    )
    z = pivot_values(subset, "neg_log10_p_harm")
    raw = pivot_values(subset, "p_harm")
    bonf = pivot_values(subset, "bonferroni_p_harm_by_benchmark")

    fig, ax = plt.subplots(figsize=(15.5, max(10.0, 0.22 * len(top_benchmarks) + 3.0)), constrained_layout=True)
    image = ax.imshow(
        z.to_numpy(), aspect="auto", cmap="RdYlGn_r", vmin=0.0, vmax=max(8.0, np.nanquantile(z.to_numpy(), 0.995))
    )
    ax.set_title("Top smooth benchmark × deleted domain p-values", fontsize=15, pad=14)
    ax.set_xlabel("Deleted domain")
    ax.set_ylabel("Smooth benchmark")
    ax.set_xticks(np.arange(len(z.columns)), labels=z.columns, rotation=60, ha="left", fontsize=7)
    ax.xaxis.tick_top()
    ax.xaxis.set_label_position("top")
    ax.set_yticks(np.arange(len(z.index)), labels=[subset.display_labels[b] for b in z.index], fontsize=7)

    raw_significant = np.argwhere(raw.to_numpy() < 0.05)
    if len(raw_significant):
        ax.scatter(
            raw_significant[:, 1],
            raw_significant[:, 0],
            s=8,
            facecolors="none",
            edgecolors=(0.0, 0.0, 0.0, 0.42),
            linewidths=0.45,
        )

    bonferroni_significant = np.argwhere(bonf.to_numpy() < 0.05)
    if len(bonferroni_significant):
        ax.scatter(
            bonferroni_significant[:, 1],
            bonferroni_significant[:, 0],
            s=24,
            facecolors="none",
            edgecolors="black",
            linewidths=0.8,
        )
        ax.scatter(
            bonferroni_significant[:, 1],
            bonferroni_significant[:, 0],
            s=10,
            facecolors="none",
            edgecolors="black",
            linewidths=0.6,
        )

    cbar = fig.colorbar(image, ax=ax, fraction=0.018, pad=0.012)
    cbar.set_label("-log10 p_harm (deletion hurts)")
    fig.text(
        0.01,
        0.01,
        "Small rings: raw p_harm < 0.05. Double rings: Bonferroni p_harm < 0.05 across 39 deletions. "
        "P-values use the pooled proportional reference, not a full heteroskedastic noise model.",
        fontsize=8,
    )
    out = output_dir / "smooth_benchmark_deleted_domain_pvalue_matrix_top70.png"
    fig.savefig(out, dpi=220)
    plt.close(fig)
    return out


def write_summaries(data: MatrixData, output_dir: Path) -> tuple[Path, Path]:
    enriched = data.frame.sort_values(["p_harm", "benchmark_key", "target_domain"]).copy()
    enriched_path = output_dir / "smooth_benchmark_deleted_domain_pvalue_matrix_cells.csv"
    enriched.to_csv(enriched_path, index=False)

    domain_summary = (
        data.frame.groupby("target_domain", as_index=False)
        .agg(
            base_mass=("base_mass", "first"),
            min_p_harm=("p_harm", "min"),
            min_bonferroni_p_harm=("bonferroni_p_harm_by_benchmark", "min"),
            n_raw_p_lt_0p05=("p_harm", lambda s: int((s < 0.05).sum())),
            n_bonferroni_p_lt_0p05=("bonferroni_p_harm_by_benchmark", lambda s: int((s < 0.05).sum())),
            mean_delta=("domain_deletion_utility_delta", "mean"),
            median_delta=("domain_deletion_utility_delta", "median"),
            fraction_harm=("domain_deletion_utility_delta", lambda s: float((s < 0.0).mean())),
            sum_neg_log10_p_harm=("neg_log10_p_harm", "sum"),
        )
        .sort_values(["n_bonferroni_p_lt_0p05", "sum_neg_log10_p_harm"], ascending=[False, False])
    )
    domain_summary_path = output_dir / "smooth_benchmark_deleted_domain_pvalue_matrix_domain_summary.csv"
    domain_summary.to_csv(domain_summary_path, index=False)
    return enriched_path, domain_summary_path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    data = prepare_matrix(args.input_csv)
    html = write_interactive_heatmap(data, args.output_dir)
    png = write_top_png(data, args.output_dir)
    enriched_path, domain_summary_path = write_summaries(data, args.output_dir)

    print(f"Wrote {html}")
    print(f"Wrote {png}")
    print(f"Wrote {enriched_path}")
    print(f"Wrote {domain_summary_path}")
    print()
    print("Top deleted domains by Bonferroni-significant harmful benchmark count:")
    domain_summary = pd.read_csv(domain_summary_path)
    print(
        domain_summary[
            [
                "target_domain",
                "base_mass",
                "n_bonferroni_p_lt_0p05",
                "n_raw_p_lt_0p05",
                "fraction_harm",
                "min_p_harm",
            ]
        ]
        .head(15)
        .to_string(index=False)
    )
    print()
    print("Top benchmark-domain harmful deletion cells:")
    print(
        data.frame.sort_values("p_harm")[
            [
                "benchmark_key",
                "target_domain",
                "domain_deletion_utility_delta",
                "noise_sd",
                "t_statistic",
                "p_harm",
                "bonferroni_p_harm_by_benchmark",
                "bh_q_harm_all_cells",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
