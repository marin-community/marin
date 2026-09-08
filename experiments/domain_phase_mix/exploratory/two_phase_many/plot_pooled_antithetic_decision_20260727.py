# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["kaleido==0.2.1", "numpy", "pandas", "plotly"]
# ///
"""Pool every 3e18 antithetic pair and plot the two-phase decision boundary directly.

The contrast-magnitude figure is built from one ladder along one direction from one anchor, which is
24 pairs. It shows the mechanism cleanly but cannot speak to generality, and the obvious objection --
that a single ray was searched -- is answerable from data already collected. Four panels have trained
both orientations against a shared control at 3e18, spanning roughly 300 pairs, two anchors, several
direction families, and contrast magnitudes from 0.002 to 0.50.

Pooling them needs one care. Averaging the *signed* ordering effect across directions would cancel
real effects that point opposite ways, so the ordering term enters as a magnitude. That is legitimate
here because the quantity being tested is a comparison of magnitudes: a two-phase policy in its better
orientation beats its tied control exactly when ``|o| > c``, since
``min(L+, L-) = L0 + c - |o|``. Plotting ``|o|`` against ``c`` therefore turns the criterion into a
diagonal, and every pair is a point that either clears it or does not.

Both terms carry run noise, and near the origin that noise dominates: the ordering magnitude has
standard deviation about ``sigma/sqrt(2)`` and the cost about ``1.22 sigma`` when the two orientations
and the control are independent draws. The region where neither term is resolved is drawn so those
points are not read as evidence either way.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "pooled_antithetic_decision_20260727"

RUN_SIGMA = {"uncheatable": 0.000913, "table9": 0.003772}
PANEL_TITLES = {"uncheatable": "Uncheatable", "table9": "Table-9 macro"}
ANCHOR_COLOR = {"uncheatable_frontier": "#1A6FB5", "table9_frontier": "#C1443C", "proportional": "#6A8D3A"}
SOURCE_SYMBOL = {
    "phase fiber (Jul 19)": "circle",
    "aggressive asymmetry (Jul 23)": "diamond",
    "TV ladder (Jul 27)": "square",
    "composite proposal (Jul 26)": "star",
}
# Standard deviation of each decomposed term when the two orientations and the control are independent
# draws at one run sigma: the ordering magnitude averages two runs, the cost also subtracts a control.
ODD_NOISE = 1.0 / np.sqrt(2.0)
COST_NOISE = np.sqrt(1.5)
# Standard deviation of the margin |o| - c for a single pair, and the 95 percent resolution it implies.
# The odd and cost terms are uncorrelated when the three runs are independent, so variances add.
MARGIN_NOISE = np.sqrt(0.5 + 1.5)
MARGIN_RESOLUTION = 1.96 * MARGIN_NOISE
# Measured margin noise, from the only treatments that were replicated across seed blocks at 3e18:
# the four ladder levels and the composite point, nine degrees of freedom per objective. It comes out
# roughly half the independence figure above, because the two orientations in a pair share data and
# trainer seeds and their noise is correlated. Estimated on one anchor and one contrast direction, so
# applying it to the other panels is an assumption -- which is why both bands are drawn.
EMPIRICAL_MARGIN_NOISE = {"uncheatable": 0.679, "table9": 0.900}
EMPIRICAL_DOF = 9
BOOTSTRAP_DRAWS = 4000
BOOTSTRAP_SEED = 20260727
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def load_phase_fiber() -> pd.DataFrame:
    path = REFERENCE_OUTPUTS / "delphi_3e18_frontier_phase_fiber_results_20260719" / "paired_phase_effects.csv"
    frame = pd.read_csv(path)
    # The panel stores the decomposition already; recompute from the raw losses so a schema change
    # cannot silently substitute a differently defined column.
    odd = 0.5 * (frame["plus_bpb"] - frame["minus_bpb"])
    cost = 0.5 * (frame["plus_bpb"] + frame["minus_bpb"]) - frame["center_bpb"]
    assert np.allclose(odd, frame["odd_effect_plus_minus_over_2"], atol=1e-9), "phase fiber odd term disagrees"
    assert np.allclose(cost, frame["mean_contrast_minus_center"], atol=1e-9), "phase fiber cost term disagrees"
    return pd.DataFrame(
        {
            "source": "phase fiber (Jul 19)",
            "target": frame["target"],
            "anchor_id": frame["anchor_id"],
            "phase_tv": frame["phase_tv"],
            "odd_effect": odd,
            "asymmetry_cost": cost,
        }
    )


def load_aggressive() -> pd.DataFrame:
    path = (
        REFERENCE_OUTPUTS / "delphi_3e18_aggressive_phase_asymmetry_results_20260723" / "balanced_antithetic_pairs.csv"
    )
    frame = pd.read_csv(path)
    # Stored wide, one row per pair with both objectives, and already expressed against the control.
    blocks = []
    for target in ("uncheatable", "table9"):
        blocks.append(
            pd.DataFrame(
                {
                    "source": "aggressive asymmetry (Jul 23)",
                    "target": target,
                    "anchor_id": frame["anchor_id"],
                    "phase_tv": frame["target_phase_tv"],
                    "odd_effect": frame[f"{target}_odd_effect"],
                    "asymmetry_cost": 0.5 * (frame[f"{target}_plus_delta"] + frame[f"{target}_minus_delta"]),
                }
            )
        )
    return pd.concat(blocks, ignore_index=True)


def load_ladder() -> pd.DataFrame:
    path = REFERENCE_OUTPUTS / "delphi_3e18_uncheatable_phase_tv_ladder_results_20260727" / "level_decomposition.csv"
    frame = pd.read_csv(path)
    return pd.DataFrame(
        {
            "source": "TV ladder (Jul 27)",
            "target": frame["target"],
            "anchor_id": "uncheatable_frontier",
            "phase_tv": frame["phase_tv"],
            "odd_effect": frame["odd_effect"],
            "asymmetry_cost": frame["asymmetry_cost"],
        }
    )


def load_composite() -> pd.DataFrame:
    path = REFERENCE_OUTPUTS / "delphi_3e18_composite_proposal_validation_results_20260727" / "pair_decomposition.csv"
    frame = pd.read_csv(path)
    return pd.DataFrame(
        {
            "source": "composite proposal (Jul 26)",
            "target": frame["target"],
            "anchor_id": "uncheatable_frontier",
            "phase_tv": 0.24,
            "odd_effect": frame["odd_effect"],
            "asymmetry_cost": frame["asymmetry_cost"],
        }
    )


def build_figure(pairs: pd.DataFrame) -> go.Figure:
    targets = [target for target in ("uncheatable", "table9") if target in set(pairs["target"])]
    figure = make_subplots(
        rows=1,
        cols=len(targets),
        subplot_titles=[PANEL_TITLES[target] for target in targets],
        horizontal_spacing=0.10,
    )
    limit = float(np.nanmax([pairs["abs_odd_sigma"].max(), pairs["cost_sigma"].max()])) * 1.05
    span = [min(-0.6, float(pairs["cost_sigma"].min()) * 1.05), limit]

    seen: set[str] = set()
    for column, target in enumerate(targets, start=1):
        block = pairs[pairs["target"] == target]
        # The decision boundary: above it the better orientation beats the tied control.
        figure.add_trace(
            go.Scatter(
                x=span,
                y=span,
                mode="lines",
                line={"color": "#222", "width": 1.6, "dash": "dash"},
                name="|o| = c  (two-phase breaks even)",
                showlegend=column == 1,
                hoverinfo="skip",
            ),
            row=1,
            col=column,
        )
        # Band around the diagonal where a single pair cannot resolve which side it is on. The margin
        # |o| - c combines three runs, so its standard deviation is sqrt(0.5 + 1.5) = 1.41 run sigma
        # and a 95 percent call needs 2.77. Almost every pair falls inside this, which is the point:
        # no individual experiment settles the question and the claim rests on the ensemble.
        measured = 1.96 * EMPIRICAL_MARGIN_NOISE[target]
        for half_width, shade, label in (
            (MARGIN_RESOLUTION, "rgba(120,120,120,0.10)", "unresolved, independent-run assumption"),
            (measured, "rgba(120,120,120,0.20)", "unresolved, measured seed-matched noise"),
        ):
            figure.add_trace(
                go.Scatter(
                    x=span + span[::-1],
                    y=[value - half_width for value in span] + [value + half_width for value in span[::-1]],
                    mode="lines",
                    fill="toself",
                    fillcolor=shade,
                    line={"width": 0},
                    hoverinfo="skip",
                    showlegend=column == 1,
                    name=f"{label} (±{half_width:.2f})",
                ),
                row=1,
                col=column,
            )
        for (source, anchor), group in block.groupby(["source", "anchor_id"]):
            label = f"{source} · {anchor}"
            figure.add_trace(
                go.Scatter(
                    x=group["cost_sigma"],
                    y=group["abs_odd_sigma"],
                    mode="markers",
                    marker={
                        "color": ANCHOR_COLOR.get(anchor, "#777"),
                        "symbol": SOURCE_SYMBOL.get(source, "circle"),
                        "size": 7,
                        "opacity": 0.6,
                        "line": {"width": 0.5, "color": "white"},
                    },
                    name=label,
                    legendgroup=label,
                    showlegend=label not in seen,
                    customdata=group[["phase_tv"]],
                    hovertemplate=(
                        f"{label}<br>contrast TV %{{customdata[0]:.3f}}"
                        "<br>cost %{x:+.2f} sigma<br>|ordering| %{y:.2f} sigma<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            seen.add(label)

        # The ensemble mean, which is where the question actually resolves: individual pairs scatter
        # across the whole band, but averaging hundreds of them pins the margin to a fraction of a
        # sigma. Intervals are bootstrapped over pairs rather than assumed, so they do not rely on the
        # independence used for the band above.
        generator = np.random.default_rng(BOOTSTRAP_SEED)
        for anchor, group in block.groupby("anchor_id"):
            margin = (group["abs_odd_sigma"] - group["cost_sigma"]).to_numpy()
            draws = np.array(
                [margin[generator.integers(0, margin.size, margin.size)].mean() for _ in range(BOOTSTRAP_DRAWS)]
            )
            low, high = np.quantile(draws, [0.025, 0.975])
            resolves = "cancels" if low <= 0.0 <= high else ("two-phase wins" if low > 0 else "two-phase loses")
            figure.add_trace(
                go.Scatter(
                    x=[group["cost_sigma"].mean()],
                    y=[group["abs_odd_sigma"].mean()],
                    mode="markers",
                    marker={
                        "color": ANCHOR_COLOR.get(anchor, "#777"),
                        "symbol": "diamond-wide",
                        "size": 18,
                        "line": {"width": 2, "color": "#222"},
                    },
                    name=f"mean, {anchor}",
                    showlegend=False,
                    hovertemplate=(
                        f"<b>ensemble mean · {anchor}</b><br>n = {len(group)}"
                        f"<br>mean margin {margin.mean():+.3f} sigma"
                        f"<br>95% CI [{low:+.3f}, {high:+.3f}]<br>{resolves}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            figure.add_annotation(
                x=group["cost_sigma"].mean(),
                y=group["abs_odd_sigma"].mean(),
                text=f"{margin.mean():+.2f} [{low:+.2f}, {high:+.2f}]",
                showarrow=True,
                arrowhead=0,
                arrowwidth=1,
                ax=34,
                ay=-26,
                font={"size": 10, "color": ANCHOR_COLOR.get(anchor, "#777")},
                bgcolor="rgba(255,255,255,0.78)",
                row=1,
                col=column,
            )
        figure.update_xaxes(title_text="asymmetry cost c (run sigma)", range=span, row=1, col=column)
        figure.update_yaxes(
            title_text="ordering effect |o| (run sigma)" if column == 1 else None,
            range=[0.0, limit],
            row=1,
            col=column,
        )

    figure.update_layout(
        template="simple_white",
        height=520,
        width=1040,
        title={
            "text": (
                "Every 3e18 antithetic pair against the two-phase decision boundary<br>"
                "<sub>A pair sits above the dashed line exactly when its better orientation beats its own "
                "tied control. Shaded corner is where neither term is resolved against run noise.</sub>"
            )
        },
        legend={"orientation": "v", "yanchor": "top", "y": 1.0, "xanchor": "left", "x": 1.02, "font": {"size": 10}},
        margin={"t": 96, "r": 260},
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = pd.concat(
        [load_phase_fiber(), load_aggressive(), load_ladder(), load_composite()], ignore_index=True
    ).dropna(subset=["odd_effect", "asymmetry_cost"])
    sigma = pairs["target"].map(RUN_SIGMA)
    pairs["abs_odd_sigma"] = pairs["odd_effect"].abs() / sigma
    pairs["cost_sigma"] = pairs["asymmetry_cost"] / sigma
    pairs["two_phase_wins"] = pairs["abs_odd_sigma"] > pairs["cost_sigma"]
    pairs["resolved"] = (pairs["abs_odd_sigma"] > ODD_NOISE) | (pairs["cost_sigma"] > COST_NOISE)
    pairs.to_csv(args.output_dir / "pooled_pairs.csv", index=False)

    print(f"pooled {len(pairs)} antithetic pair-observations at 3e18")
    print(pairs.groupby(["source", "target"]).size().to_string())
    print("\nfraction where the better orientation beats its tied control (|o| > c):")
    for keys, group in pairs.groupby(["target", "anchor_id"]):
        wins = int(group["two_phase_wins"].sum())
        resolved = group[group["resolved"]]
        wins_resolved = int(resolved["two_phase_wins"].sum())
        print(
            f"  {keys[0]:<12} {keys[1]:<22} {wins:>3}/{len(group):<4} ({wins / len(group):.0%})   "
            f"resolved only: {wins_resolved}/{len(resolved)}"
            + (f" ({wins_resolved / len(resolved):.0%})" if len(resolved) else "")
        )
    print("\nmedian magnitudes in run sigma:")
    print(pairs.groupby("target")[["abs_odd_sigma", "cost_sigma"]].median().round(3).to_string())

    print("\nhow many individual pairs resolve which side of the diagonal they are on:")
    margin = (pairs["abs_odd_sigma"] - pairs["cost_sigma"]).abs()
    for target, group in pairs.groupby("target"):
        block = margin[group.index]
        measured = 1.96 * EMPIRICAL_MARGIN_NOISE[target]
        print(
            f"  {target:<12} measured band ±{measured:.2f}: {int((block > measured).sum()):>3}/{len(block)}   "
            f"independent-run band ±{MARGIN_RESOLUTION:.2f}: {int((block > MARGIN_RESOLUTION).sum()):>3}/{len(block)}"
        )
    print(
        f"  measured noise rests on {EMPIRICAL_DOF} degrees of freedom from one anchor and one "
        "contrast direction, so the true band lies between the two"
    )

    figure = build_figure(pairs)
    figure.write_html(args.output_dir / "pooled_antithetic_decision.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    figure.write_image(args.output_dir / "pooled_antithetic_decision.png", scale=4)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
