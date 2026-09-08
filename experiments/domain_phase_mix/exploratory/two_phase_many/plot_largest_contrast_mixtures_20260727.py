# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["kaleido==0.2.1", "numpy", "pandas", "plotly"]
# ///
"""Show what the largest-contrast antithetic pair actually looks like as mixtures.

The decomposition figures argue about magnitudes and never show a policy, which leaves the natural
question of how different these two-phase mixtures really are. This draws the pair at the ladder's
largest contrast, total variation 0.24, against the aggregate mixture they both preserve.

Two panels, because the answer has two halves. The aggregate is what the runs train on in total and is
identical across the pair to machine precision, so it is drawn once. The phase mixtures are then drawn
as ratios to it, which is the readable form: a value of 1.5 means that bucket appears half again as
often in that phase as it does across the run overall, and it is the ratio rather than the absolute
share that governs how many times the model sees a bucket's tokens.

The contrast direction here scales each group proportionally, so the ratios collapse to two levels --
one for the technical-specialization group and one for everything else. That flatness is a property of
this direction rather than of two-phase policies in general, and it is worth showing precisely because
it makes the geometry legible: at this contrast the policy is simply "technical content up by half in
one phase, everything else down by a third", with the aggregate unchanged.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Both constants are re-exported by the launcher, but importing them from there would pull in the whole
# training stack -- jax, jmp, levanter -- to draw a bar chart. They originate here.
from experiments.domain_phase_mix.dolma3_dolmino_top_level_domains import (  # noqa: E402
    TARGET_BUDGET_DOLMA3_COMMON_CRAWL,
    TOP_LEVEL_DOMAIN_TOKEN_COUNTS,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SIMULATED_EPOCH_TARGET_BUDGET = TARGET_BUDGET_DOLMA3_COMMON_CRAWL
# Realized token fractions, the same ones the policy geometry uses. The launcher's own epoch diagnostic
# uses the *nominal* schedule fractions 0.8/0.2 instead, which for a two-phase policy does not recover
# the aggregate it was built to preserve -- so its logged epoch figure is very slightly off for tilted
# policies and exact only for tied ones. Using realized fractions here keeps the two phase labels
# summing to the aggregate, which is what the figure claims; the divergence from the logged value is
# reported at the end.
NOMINAL_PHASE_FRACTIONS = {"phase_0": 0.8, "phase_1": 0.2}
PHASE_NAMES = ("phase_0", "phase_1")
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
PANEL_DIR = REFERENCE_OUTPUTS / "delphi_3e18_uncheatable_phase_tv_ladder_20260727"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "largest_contrast_mixtures_20260727"

TARGET_PHASE_TV = 0.24
ALPHA_0 = 0.7981376787495837
ALPHA_1 = 1.0 - ALPHA_0
# The contrast direction moves a technical-specialization group against everything else. Membership is
# taken verbatim from the materializer so the figure cannot drift from the policy it draws.
TECHNICAL_TOPICS = ("science_math", "education_and_jobs", "electronics_and_hardware")
TECHNICAL_EXPLICIT = (
    "dolma3_stack_edu",
    "dolma3_arxiv",
    "dolma3_finemath_3plus",
    "dolmino_stack_edu_fim",
    "dolmino_stem_heavy_crawl",
    "dolmino_synth_code",
    "dolmino_synth_math",
    "dolmino_synth_thinking",
)
GROUP_COLOR = {"technical": "#1A6FB5", "other": "#B0752F"}
PHASE_STYLE = {
    ("plus", "phase_1"): ("technical late — late phase", "#1A6FB5", "solid"),
    ("plus", "phase_0"): ("technical late — early phase", "#1A6FB5", "dot"),
    ("minus", "phase_1"): ("technical early — late phase", "#C1443C", "solid"),
    ("minus", "phase_0"): ("technical early — early phase", "#C1443C", "dot"),
}
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def is_technical(domain: str) -> bool:
    return domain in TECHNICAL_EXPLICIT or any(topic in domain for topic in TECHNICAL_TOPICS)


def short_label(domain: str) -> str:
    """Bucket names are long and repetitive; keep what distinguishes them."""
    return domain.replace("dolma3_cc/", "cc:").replace("dolma3_", "").replace("dolmino_", "d:")


def simulated_epochs(weights: np.ndarray, domains: tuple[str, ...], phase: str) -> np.ndarray:
    """Simulated epochs each bucket would see from one phase, at the full target token budget.

    This is the launcher's own repetition measure, decomposed by phase: the aggregate figure in
    ``_weight_diagnostics`` is the sum of the two phases computed here. It is *simulated* -- it asks how
    many times a bucket would be revisited if this mixture ran at the target budget, not how many times
    it is revisited in these 1.58e9-token runs, where every count would be far below one. Repetition
    pressure is the thing the phase split actually changes, so it is the number worth putting on a bar.
    """
    fraction = ALPHA_0 if phase == PHASE_NAMES[0] else ALPHA_1
    budget = SIMULATED_EPOCH_TARGET_BUDGET * fraction
    available = np.array([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain] for domain in domains], dtype=float)
    return budget * weights / available


def epoch_label(value: float) -> str:
    """Compact epoch text: enough digits to read, few enough to fit 39 stacked rows."""
    if value >= 10.0:
        return f"{value:.0f}"
    if value >= 1.0:
        return f"{value:.1f}"
    return f"{value:.2f}"


def build_side_by_side(
    policies: dict[tuple[str, str], np.ndarray],
    aggregate: np.ndarray,
    order: np.ndarray,
    labels: list[str],
    phase_tv: float,
    domains: tuple[str, ...],
) -> go.Figure:
    """The two phase mixtures as adjacent panels, in the Observatory's policy-diagnostic style.

    Horizontal grouped bars with the buckets on a shared axis, each policy drawn against the aggregate
    it preserves. Splitting by phase rather than by policy is what makes the asymmetry visible: the
    early panel is nearly three identical bars because the long phase barely moves, while the late
    panel fans out, since holding the aggregate fixed forces the short phase to absorb four times the
    displacement.
    """
    figure = make_subplots(
        rows=1,
        cols=2,
        shared_yaxes=True,
        horizontal_spacing=0.05,
        subplot_titles=(
            f"Early phase — {ALPHA_0:.0%} of tokens",
            f"Late phase — {ALPHA_1:.0%} of tokens",
        ),
    )
    # The aggregate is phase-independent; the two policies are looked up per phase.
    series = (
        ("aggregate", None, "#8A8A8A", 0.75),
        ("technical late", "plus", "#1A6FB5", 0.92),
        ("technical early", "minus", "#C1443C", 0.92),
    )
    for column, phase in enumerate(("phase_0", "phase_1"), start=1):
        for label, sign, color, opacity in series:
            resolved = aggregate if sign is None else policies[(sign, phase)]
            epochs = simulated_epochs(resolved, domains, phase)
            figure.add_trace(
                go.Bar(
                    x=resolved[order],
                    y=labels,
                    orientation="h",
                    name=label,
                    legendgroup=label,
                    showlegend=column == 1,
                    marker_color=color,
                    opacity=opacity,
                    text=[epoch_label(value) for value in epochs[order]],
                    textposition="outside",
                    textfont={"size": 7, "color": color},
                    cliponaxis=False,
                    customdata=np.stack([resolved[order] / aggregate[order], epochs[order]], axis=-1),
                    hovertemplate=(
                        f"<b>%{{y}}</b><br>{label}<br>weight %{{x:.5f}}"
                        "<br>%{customdata[0]:.3f}x aggregate"
                        "<br>%{customdata[1]:.2f} simulated epochs this phase<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
        figure.update_xaxes(title_text="mixture weight", row=1, col=column)
    figure.update_yaxes(categoryorder="array", categoryarray=labels[::-1], tickfont={"size": 9}, row=1, col=1)
    figure.update_layout(
        title={
            "text": f"Antithetic pair at phase total variation {phase_tv:g} — both preserve the same aggregate",
            "x": 0.5,
            "xanchor": "center",
        },
        barmode="group",
        bargap=0.25,
        template="plotly_white",
        width=1500,
        height=1150,
        margin={"l": 210, "r": 60, "t": 130, "b": 90},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": -0.07},
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--phase-tv", type=float, default=TARGET_PHASE_TV)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel_files = sorted(PANEL_DIR.glob("ladder_panel-*.csv"))
    assert len(panel_files) == 1, f"expected one ladder panel, found {panel_files}"
    panel = pd.read_csv(panel_files[0])
    domains = tuple(column[len("phase_0_") :] for column in panel.columns if column.startswith("phase_0_"))

    def weights(row: pd.Series, phase: str) -> np.ndarray:
        return np.array([row[f"{phase}_{domain}"] for domain in domains], dtype=float)

    tied = panel[panel["sign"] == "center"].iloc[0]
    aggregate = weights(tied, "phase_0")
    assert np.abs(aggregate - weights(tied, "phase_1")).max() == 0.0, "tied control is not tied"

    policies = {}
    for sign in ("plus", "minus"):
        match = panel[(panel["sign"] == sign) & np.isclose(panel["phase_tv"], args.phase_tv)]
        assert len(match), f"no {sign} row at phase TV {args.phase_tv}"
        row = match.iloc[0]
        for phase in ("phase_0", "phase_1"):
            policies[(sign, phase)] = weights(row, phase)
        recovered = ALPHA_0 * policies[(sign, "phase_0")] + ALPHA_1 * policies[(sign, "phase_1")]
        assert np.abs(recovered - aggregate).max() < 2e-12, f"{sign} does not preserve the aggregate"

    order = np.argsort(-aggregate)
    labels = [short_label(domains[index]) for index in order]
    groups = ["technical" if is_technical(domains[index]) else "other" for index in order]

    side_by_side = build_side_by_side(policies, aggregate, order, labels, args.phase_tv, domains)
    side_by_side.write_html(args.output_dir / "side_by_side_mixtures.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    side_by_side.write_image(args.output_dir / "side_by_side_mixtures.png", scale=4)

    figure = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.09,
        row_heights=[0.42, 0.58],
        subplot_titles=(
            "Aggregate mixture — identical for both policies and the tied control",
            "Phase mixtures as a ratio to the aggregate — 1.0 means unchanged",
        ),
    )
    for group in ("technical", "other"):
        mask = [index for index, value in enumerate(groups) if value == group]
        figure.add_trace(
            go.Bar(
                x=[labels[index] for index in mask],
                y=[aggregate[order][index] for index in mask],
                marker_color=GROUP_COLOR[group],
                name="technical group" if group == "technical" else "all other buckets",
                legendgroup=group,
                hovertemplate="%{x}<br>aggregate share %{y:.4f}<extra></extra>",
            ),
            row=1,
            col=1,
        )

    for (sign, phase), (label, color, dash) in PHASE_STYLE.items():
        ratio = policies[(sign, phase)][order] / aggregate[order]
        figure.add_trace(
            go.Scatter(
                x=labels,
                y=ratio,
                mode="lines+markers",
                line={"color": color, "width": 2, "dash": dash},
                marker={"color": color, "size": 5},
                name=label,
                hovertemplate=f"{label}<br>%{{x}}<br>%{{y:.3f}}x aggregate<extra></extra>",
            ),
            row=2,
            col=1,
        )
    figure.add_hline(y=1.0, line={"color": "#444", "width": 1.2, "dash": "dot"}, row=2, col=1)

    figure.update_yaxes(title_text="share of tokens", type="log", row=1, col=1)
    figure.update_yaxes(title_text="phase share / aggregate share", row=2, col=1)
    figure.update_xaxes(tickangle=-60, tickfont={"size": 8}, row=2, col=1)
    figure.update_layout(
        template="simple_white",
        height=760,
        width=1180,
        barmode="stack",
        title={
            "text": (
                f"The antithetic pair at phase total variation {args.phase_tv:g}, 3e18<br>"
                "<sub>Both policies train on exactly the same data overall; they differ only in when "
                "each bucket appears.</sub>"
            )
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.34, "xanchor": "center", "x": 0.5},
        margin={"t": 90, "b": 190},
    )
    figure.write_html(args.output_dir / "largest_contrast_mixtures.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    figure.write_image(args.output_dir / "largest_contrast_mixtures.png", scale=4)

    technical = np.array([is_technical(domain) for domain in domains])
    late_ratio = policies[("plus", "phase_1")] / aggregate
    early_ratio = policies[("plus", "phase_0")] / aggregate
    print(f"phase TV {args.phase_tv:g}, technical-late orientation:")
    print(f"  technical group ({technical.sum()} buckets): early {early_ratio[technical].mean():.3f}x, ")
    print(f"    late {late_ratio[technical].mean():.3f}x  (spread {np.ptp(late_ratio[technical]):.2e})")
    print(f"  other buckets ({(~technical).sum()}): early {early_ratio[~technical].mean():.3f}x, ")
    print(f"    late {late_ratio[~technical].mean():.3f}x  (spread {np.ptp(late_ratio[~technical]):.2e})")
    print(f"  technical share of the aggregate: {aggregate[technical].sum():.3f}")
    # The phase decomposition must reconstruct the launcher's aggregate epoch figure exactly, or the
    # labels on the bars are measuring something other than what the run specs recorded.
    for sign in ("plus", "minus"):
        total = sum(simulated_epochs(policies[(sign, phase)], domains, phase) for phase in ("phase_0", "phase_1"))
        tied_total = sum(simulated_epochs(aggregate, domains, phase) for phase in ("phase_0", "phase_1"))
        assert np.abs(total - tied_total).max() < 1e-9, f"{sign} phase epochs do not sum to the aggregate"
    aggregate_epochs = sum(simulated_epochs(aggregate, domains, phase) for phase in ("phase_0", "phase_1"))
    print(f"  aggregate simulated epochs: max {aggregate_epochs.max():.2f}, median {np.median(aggregate_epochs):.2f}")
    for sign, phase in (("plus", "phase_1"), ("minus", "phase_1")):
        values = simulated_epochs(policies[(sign, phase)], domains, phase)
        print(f"  {sign:<6} late-phase epochs: max {values.max():.2f} on {domains[int(np.argmax(values))]}")
    available = np.array([TOP_LEVEL_DOMAIN_TOKEN_COUNTS[domain] for domain in domains], dtype=float)
    nominal = sum(
        SIMULATED_EPOCH_TARGET_BUDGET * NOMINAL_PHASE_FRACTIONS[phase] * policies[("plus", phase)] / available
        for phase in PHASE_NAMES
    )
    drift = float(np.max(np.abs(nominal - aggregate_epochs) / aggregate_epochs))
    print(f"  launcher's nominal-fraction epochs differ from these by at most {drift:.2%} on tilted policies")
    print(f"  smallest bucket weight anywhere in the pair: {min(v.min() for v in policies.values()):.5f}")
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
