# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "plotly>=6.0", "scipy>=1.14", "tabulate>=0.9"]
# ///
"""Audit paired phase effects near one-phase frontiers without fitting a surrogate."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

from audit_cross_scale_variance_decomposition_round61 import component_frame

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round66_frontier_phase_benefit"
MATCHED = OUTPUT_ROOT / "round1_cross_scale_matched_policy" / "matched_targets.csv"
BOOTSTRAP_SAMPLES = 20_000
BOOTSTRAP_SEED = 20260719
FRONTIER_SIZES = (10, 25, 50)
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def bootstrap_mean_interval(values: np.ndarray, seed: int) -> tuple[float, float]:
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, len(values), size=(BOOTSTRAP_SAMPLES, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return float(low), float(high)


def summarize_slice(
    frame: pd.DataFrame,
    target: str,
    ranking_scale: str,
    evaluation_scale: str,
    slice_name: str,
    seed: int,
) -> dict[str, object]:
    delta = frame[f"phase_delta_{evaluation_scale}"].to_numpy(dtype=float)
    one_phase = frame[f"aggregate_{evaluation_scale}"].to_numpy(dtype=float)
    two_phase = frame[f"two_phase_{evaluation_scale}"].to_numpy(dtype=float)
    low, high = bootstrap_mean_interval(delta, seed)
    return {
        "target": target,
        "ranking_scale": ranking_scale,
        "evaluation_scale": evaluation_scale,
        "same_scale_selection": ranking_scale == evaluation_scale,
        "slice": slice_name,
        "n": len(frame),
        "mean_one_phase_bpb": float(one_phase.mean()),
        "mean_two_phase_bpb": float(two_phase.mean()),
        "mean_phase_delta": float(delta.mean()),
        "median_phase_delta": float(np.median(delta)),
        "standard_deviation_phase_delta": float(delta.std(ddof=1)),
        "bootstrap_mean_delta_low": low,
        "bootstrap_mean_delta_high": high,
        "fraction_two_phase_better": float(np.mean(delta < 0.0)),
        "best_phase_delta": float(delta.min()),
        "worst_phase_delta": float(delta.max()),
    }


def frontier_summaries(components: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    seed = BOOTSTRAP_SEED
    for target, frame in components.items():
        for ranking_scale in ("300m", "delphi"):
            ranked = frame.sort_values(f"aggregate_{ranking_scale}", kind="stable").reset_index(drop=True)
            ranked["one_phase_rank"] = np.arange(1, len(ranked) + 1)
            ranked["one_phase_rank_percentile"] = ranked["one_phase_rank"] / len(ranked)
            quartiles = pd.qcut(ranked["one_phase_rank"], 4, labels=["Q1 best", "Q2", "Q3", "Q4 worst"])
            for evaluation_scale in ("300m", "delphi"):
                slices: list[tuple[str, pd.DataFrame]] = [("all", ranked)]
                slices.extend((f"top_{count}", ranked.head(count)) for count in FRONTIER_SIZES)
                slices.extend(
                    (str(label), ranked.loc[quartiles.eq(label)]) for label in ("Q1 best", "Q2", "Q3", "Q4 worst")
                )
                for slice_name, selected in slices:
                    rows.append(
                        summarize_slice(
                            selected,
                            target,
                            ranking_scale,
                            evaluation_scale,
                            slice_name,
                            seed,
                        )
                    )
                    seed += 1
    return pd.DataFrame(rows)


def rank_correlations(components: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for target, frame in components.items():
        for ranking_scale in ("300m", "delphi"):
            rank = frame[f"aggregate_{ranking_scale}"].rank(method="average", ascending=True).to_numpy(dtype=float)
            for evaluation_scale in ("300m", "delphi"):
                delta = frame[f"phase_delta_{evaluation_scale}"].to_numpy(dtype=float)
                rows.append(
                    {
                        "target": target,
                        "ranking_scale": ranking_scale,
                        "evaluation_scale": evaluation_scale,
                        "same_scale_selection": ranking_scale == evaluation_scale,
                        "spearman_one_phase_rank_vs_phase_delta": float(spearmanr(rank, delta).statistic),
                    }
                )
    return pd.DataFrame(rows)


def sign_transition_table(components: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for target, frame in components.items():
        improves_300m = frame["phase_delta_300m"].lt(0.0)
        improves_delphi = frame["phase_delta_delphi"].lt(0.0)
        categories = np.select(
            [
                improves_300m & improves_delphi,
                improves_300m & ~improves_delphi,
                ~improves_300m & improves_delphi,
            ],
            ["improves_both", "improves_300m_only", "improves_delphi_only"],
            default="improves_neither",
        )
        counts = pd.Series(categories).value_counts()
        for category in ("improves_both", "improves_300m_only", "improves_delphi_only", "improves_neither"):
            count = int(counts.get(category, 0))
            rows.append({"target": target, "category": category, "count": count, "fraction": count / len(frame)})
    return pd.DataFrame(rows)


def write_plots(summaries: pd.DataFrame, components: dict[str, pd.DataFrame]) -> None:
    frontier = summaries.loc[summaries["slice"].isin(["top_10", "top_25", "top_50"])].copy()
    frontier["selection"] = np.where(frontier["same_scale_selection"], "same-scale rank", "cross-scale rank")
    frontier["line"] = frontier["ranking_scale"] + " rank -> " + frontier["evaluation_scale"] + " effect"
    frontier["error_plus"] = frontier["bootstrap_mean_delta_high"] - frontier["mean_phase_delta"]
    frontier["error_minus"] = frontier["mean_phase_delta"] - frontier["bootstrap_mean_delta_low"]
    figure = px.scatter(
        frontier,
        x="slice",
        y="mean_phase_delta",
        error_y="error_plus",
        error_y_minus="error_minus",
        color="line",
        symbol="selection",
        facet_row="target",
        facet_col="evaluation_scale",
        title="Paired phase effect near fixed one-phase frontiers (negative favors two phases)",
        labels={"mean_phase_delta": "Mean two-minus-one-phase BPB", "slice": "Frontier selected by one-phase BPB"},
        color_discrete_sequence=px.colors.qualitative.Safe,
    )
    figure.add_hline(y=0.0, line_dash="dash", line_color="#48545e")
    figure.update_layout(template="plotly_white", height=820, width=1320)
    figure.write_html(ROUND_DIR / "frontier_phase_benefit.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: rank at 300M",
            "Uncheatable: rank at Delphi",
            "Table-9: rank at 300M",
            "Table-9: rank at Delphi",
        ),
        vertical_spacing=0.12,
    )
    for row_index, target in enumerate(("uncheatable", "table9"), start=1):
        frame = components[target]
        for column_index, ranking_scale in enumerate(("300m", "delphi"), start=1):
            rank_percentile = frame[f"aggregate_{ranking_scale}"].rank(method="average", pct=True)
            for evaluation_scale, color, symbol in (
                ("300m", "#2b6777", "circle"),
                ("delphi", "#d95f02", "diamond"),
            ):
                figure.add_trace(
                    go.Scatter(
                        x=rank_percentile,
                        y=frame[f"phase_delta_{evaluation_scale}"],
                        mode="markers",
                        name=f"effect at {evaluation_scale}",
                        legendgroup=evaluation_scale,
                        showlegend=row_index == 1 and column_index == 1,
                        marker={"color": color, "symbol": symbol, "size": 6, "opacity": 0.58},
                        customdata=np.column_stack([frame["source_index"], frame[f"aggregate_{ranking_scale}"]]),
                        hovertemplate=(
                            "one-phase rank pct=%{x:.3f}<br>phase delta=%{y:.5f}<br>"
                            "source=%{customdata[0]}<br>ranking BPB=%{customdata[1]:.5f}<extra></extra>"
                        ),
                    ),
                    row=row_index,
                    col=column_index,
                )
    figure.add_hline(y=0.0, line_dash="dash", line_color="#48545e")
    figure.update_xaxes(title_text="One-phase rank percentile (lower is better)")
    figure.update_yaxes(title_text="Two-minus-one-phase BPB")
    figure.update_layout(
        template="plotly_white",
        height=820,
        width=1200,
        title="Matched phase effects are heterogeneous and scale dependent",
    )
    figure.write_html(ROUND_DIR / "phase_delta_by_one_phase_rank.html", include_plotlyjs="cdn", config=PLOT_CONFIG)


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    matched = pd.read_csv(MATCHED)
    components = {target: component_frame(matched, target) for target in sorted(matched["target"].unique())}
    if any(len(frame) != 238 for frame in components.values()):
        raise ValueError("Expected 238 exact matched policies for every target")

    summaries = frontier_summaries(components)
    correlations = rank_correlations(components)
    transitions = sign_transition_table(components)
    summaries.to_csv(ROUND_DIR / "frontier_phase_delta_summaries.csv", index=False)
    correlations.to_csv(ROUND_DIR / "one_phase_rank_phase_delta_correlations.csv", index=False)
    transitions.to_csv(ROUND_DIR / "phase_benefit_sign_transitions.csv", index=False)
    write_plots(summaries, components)

    cross_scale_frontier = summaries.loc[
        ~summaries["same_scale_selection"] & summaries["slice"].isin(["top_10", "top_25", "top_50"])
    ]
    same_scale_frontier = summaries.loc[
        summaries["same_scale_selection"] & summaries["slice"].isin(["top_10", "top_25", "top_50"])
    ]
    report = "\n".join(
        [
            "# Round 66: paired phase benefit near one-phase frontiers",
            "",
            "This diagnostic uses the 238 exact matched one-phase/two-phase policies at 300M and Delphi 3e18. It fits no surrogate, changes no model choice, and reads no sealed confirmation outcome.",
            "",
            "For each policy coordinate, the paired phase effect is",
            "",
            "$$\\Delta_{phase}=Y_{2p}-Y_{1p},$$",
            "",
            "so negative values favor the two-phase realization. Frontier sizes $k\\in\\{10,25,50\\}$ and 20,000 paired bootstrap resamples were fixed before inspecting this diagnostic.",
            "",
            "## Cross-scale frontier audit",
            "",
            "Ranking and evaluating at the same scale mechanically couples the frontier definition to $\\Delta_{phase}$ through $-Y_{1p}$. The primary descriptive safeguard therefore ranks policies by one-phase BPB at one scale and evaluates their phase effect at the other scale.",
            "",
            cross_scale_frontier.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Same-scale descriptive slices",
            "",
            same_scale_frontier.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Rank association",
            "",
            correlations.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Cross-scale sign stability",
            "",
            transitions.to_markdown(index=False, floatfmt=".5f"),
            "",
            "## Interpretation",
            "",
            "A one-phase policy does not define a symmetric empirical distribution of phase effects in these paired swarms. The matched phase contrasts were not randomized conditionally around each aggregate anchor, and phase effect magnitude contracts substantially from 300M to Delphi. These results therefore do not estimate a global two-phase advantage. They do show why random two-phase sampling is an inefficient way to identify it: aggregate quality, phase contrast, and scale-dependent phase response remain entangled.",
            "",
            "The appropriate next experiment remains the preregistered two-stage phase-fiber design: identify a small set of one-phase anchors, then apply signed aggregate-preserving contrasts around each anchor. This diagnostic supports that design argument; it does not rescue any rejected surrogate or activate the sealed confirmation panel.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
