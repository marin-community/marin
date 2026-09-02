# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy>=2.0", "pandas>=2.2", "plotly>=6.0", "scipy>=1.14", "tabulate>=0.9"]
# ///
"""Decompose matched two-phase target variation into aggregate and phase effects."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_ROOT = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719"
ROUND_DIR = OUTPUT_ROOT / "round61_cross_scale_variance_decomposition"
MATCHED = OUTPUT_ROOT / "round1_cross_scale_matched_policy" / "matched_targets.csv"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}


def regression_slope(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    design = np.column_stack([np.ones(len(x)), x])
    intercept, slope = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = design @ np.array([intercept, slope])
    total = float(np.sum((y - y.mean()) ** 2))
    r_squared = 1.0 - float(np.sum((y - predicted) ** 2)) / total
    return float(slope), float(intercept), r_squared


def component_frame(data: pd.DataFrame, target: str) -> pd.DataFrame:
    one_phase = data.loc[data["target"].eq(target) & data["policy_class"].eq("one_phase")].set_index("source_index")
    two_phase = (
        data.loc[data["target"].eq(target) & data["policy_class"].eq("two_phase")]
        .set_index("source_index")
        .loc[one_phase.index]
    )
    phase_delta = (
        data.loc[data["target"].eq(target) & data["policy_class"].eq("phase_delta")]
        .set_index("source_index")
        .loc[one_phase.index]
    )
    frame = pd.DataFrame(index=one_phase.index)
    for scale in ("300m", "delphi"):
        frame[f"aggregate_{scale}"] = one_phase[f"value_{scale}"]
        frame[f"phase_delta_{scale}"] = phase_delta[f"value_{scale}"]
        frame[f"two_phase_{scale}"] = two_phase[f"value_{scale}"]
        error = (frame[f"aggregate_{scale}"] + frame[f"phase_delta_{scale}"] - frame[f"two_phase_{scale}"]).abs().max()
        if error > 1e-12:
            raise ValueError(f"Phase decomposition identity failed for {target}/{scale}: {error}")
    return frame.reset_index()


def main() -> None:
    ROUND_DIR.mkdir(parents=True, exist_ok=True)
    matched = pd.read_csv(MATCHED)
    targets = sorted(matched["target"].unique())
    components = {target: component_frame(matched, target) for target in targets}

    variance_rows = []
    transfer_rows = []
    for target, frame in components.items():
        for scale in ("300m", "delphi"):
            aggregate = frame[f"aggregate_{scale}"].to_numpy(dtype=float)
            phase_delta = frame[f"phase_delta_{scale}"].to_numpy(dtype=float)
            total = frame[f"two_phase_{scale}"].to_numpy(dtype=float)
            variance_total = float(np.var(total, ddof=1))
            variance_aggregate = float(np.var(aggregate, ddof=1))
            variance_delta = float(np.var(phase_delta, ddof=1))
            covariance_term = float(2.0 * np.cov(aggregate, phase_delta, ddof=1)[0, 1])
            variance_rows.append(
                {
                    "target": target,
                    "scale": scale,
                    "n": len(frame),
                    "standard_deviation_aggregate": float(np.std(aggregate, ddof=1)),
                    "standard_deviation_phase_delta": float(np.std(phase_delta, ddof=1)),
                    "standard_deviation_two_phase": float(np.std(total, ddof=1)),
                    "phase_to_aggregate_sd_ratio": float(np.std(phase_delta, ddof=1) / np.std(aggregate, ddof=1)),
                    "aggregate_phase_delta_correlation": float(np.corrcoef(aggregate, phase_delta)[0, 1]),
                    "aggregate_variance_share": variance_aggregate / variance_total,
                    "phase_delta_variance_share": variance_delta / variance_total,
                    "covariance_variance_share": covariance_term / variance_total,
                    "variance_identity_error": abs(
                        variance_total - variance_aggregate - variance_delta - covariance_term
                    ),
                }
            )
        for component in ("aggregate", "phase_delta", "two_phase"):
            source = frame[f"{component}_300m"].to_numpy(dtype=float)
            destination = frame[f"{component}_delphi"].to_numpy(dtype=float)
            slope, intercept, r_squared = regression_slope(source, destination)
            transfer_rows.append(
                {
                    "target": target,
                    "component": component,
                    "n": len(frame),
                    "pearson": float(np.corrcoef(source, destination)[0, 1]),
                    "spearman": float(spearmanr(source, destination).statistic),
                    "delphi_on_300m_slope": slope,
                    "delphi_on_300m_intercept": intercept,
                    "r_squared": r_squared,
                    "standard_deviation_ratio_delphi_over_300m": float(
                        np.std(destination, ddof=1) / np.std(source, ddof=1)
                    ),
                }
            )

    variance = pd.DataFrame(variance_rows)
    transfer = pd.DataFrame(transfer_rows)
    if variance["variance_identity_error"].max() > 1e-12:
        raise ValueError("Variance decomposition failed")
    variance.to_csv(ROUND_DIR / "variance_decomposition.csv", index=False)
    transfer.to_csv(ROUND_DIR / "component_scale_transfer.csv", index=False)

    scatter_rows = []
    for target, frame in components.items():
        for component in ("aggregate", "phase_delta", "two_phase"):
            for row in frame.itertuples(index=False):
                scatter_rows.append(
                    {
                        "target": target,
                        "component": component,
                        "source_index": row.source_index,
                        "value_300m": getattr(row, f"{component}_300m"),
                        "value_delphi": getattr(row, f"{component}_delphi"),
                    }
                )
    scatter = pd.DataFrame(scatter_rows)
    figure = px.scatter(
        scatter,
        x="value_300m",
        y="value_delphi",
        color="component",
        facet_row="target",
        facet_col="component",
        hover_data=["source_index"],
        color_discrete_map={
            "aggregate": "#2b6777",
            "phase_delta": "#d95f02",
            "two_phase": "#4b8f29",
        },
        title="Matched-policy cross-scale transfer: aggregate response transfers more reliably than phase correction",
        labels={"value_300m": "300M BPB component", "value_delphi": "Delphi 3e18 BPB component"},
    )
    figure.update_layout(template="plotly_white", height=780, width=1300)
    figure.write_html(ROUND_DIR / "component_scale_transfer.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    figure = make_subplots(rows=1, cols=2, subplot_titles=["Uncheatable", "Table-9"])
    for column, target in enumerate(("uncheatable", "table9"), start=1):
        target_rows = variance.loc[variance["target"].eq(target)]
        for name, field, color in (
            ("Aggregate variance", "aggregate_variance_share", "#2b6777"),
            ("Phase-delta variance", "phase_delta_variance_share", "#d95f02"),
            ("2 x covariance", "covariance_variance_share", "#7a7a7a"),
        ):
            figure.add_trace(
                go.Bar(
                    x=target_rows["scale"],
                    y=target_rows[field],
                    name=name,
                    marker_color=color,
                    legendgroup=name,
                    showlegend=column == 1,
                    customdata=np.column_stack(
                        [
                            target_rows["phase_to_aggregate_sd_ratio"],
                            target_rows["aggregate_phase_delta_correlation"],
                        ]
                    ),
                    hovertemplate=(
                        "%{x}<br>variance contribution=%{y:.3f}<br>"
                        "sd(delta)/sd(aggregate)=%{customdata[0]:.3f}<br>"
                        "corr(aggregate, delta)=%{customdata[1]:.3f}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
    figure.add_hline(y=0.0, line_color="#333333")
    figure.update_layout(
        barmode="relative",
        template="plotly_white",
        height=540,
        width=1120,
        title="Two-phase variance decomposition: phase effects attenuate from 300M to Delphi",
        yaxis_title="Contribution divided by total two-phase variance",
    )
    figure.write_html(ROUND_DIR / "variance_decomposition.html", include_plotlyjs="cdn", config=PLOT_CONFIG)

    report = "\n".join(
        [
            "# Round 61: matched-policy cross-scale variance decomposition",
            "",
            "This diagnostic uses only the 238 coordinates with matched one-phase and two-phase observations at both 300M and Delphi 3e18. It fits no response surrogate and reads no sealed confirmation outcome.",
            "",
            "For each coordinate and scale, the identity is",
            "",
            "$$Y_{2p}=Y_{1p}+\\Delta_{phase},$$",
            "",
            "where $Y_{1p}$ is the independently trained phase-tied outcome and $\\Delta_{phase}$ is the paired two-minus-one-phase effect. Therefore",
            "",
            "$$\\operatorname{Var}(Y_{2p})=\\operatorname{Var}(Y_{1p})+\\operatorname{Var}(\\Delta_{phase})+2\\operatorname{Cov}(Y_{1p},\\Delta_{phase}).$$",
            "",
            "## Variance decomposition",
            "",
            variance.to_markdown(index=False, floatfmt=".5f"),
            "",
            "At 300M, the phase effect has nearly the same standard deviation as the aggregate response: the delta/aggregate ratio is 0.994 for Uncheatable and 1.082 for Table-9. At Delphi it falls to 0.480 and 0.540. Aggregate response and phase effect are anticorrelated at both scales (correlations from -0.446 to -0.529), so the phase correction cancels substantial aggregate variation rather than adding independent spread.",
            "",
            "## Cross-scale transfer",
            "",
            transfer.to_markdown(index=False, floatfmt=".5f"),
            "",
            "The one-phase aggregate transfers better than the two-phase outcome on both targets. The phase effect itself transfers moderately for Uncheatable (Pearson 0.671, slope 0.417) but poorly for Table-9 (Pearson 0.385, slope 0.286). The phase-effect amplitude is therefore not scale invariant: the 300M correction is substantially attenuated at Delphi even on identical policies.",
            "",
            "## Interpretation",
            "",
            "The weaker two-phase correlation is not explained by a simple claim that two-phase runs have more stochastic variance. It contains a deterministic scale-transfer problem: the phase correction is large and anticorrelated with aggregate quality at 300M, then contracts at 3e18. A transferable surrogate needs a dimensionless transition law that predicts this attenuation from model size, token budget, optimizer time, or another declared state. A scale-specific intercept cannot repair it, and none of the tested transition laws passed the shape and transfer gates.",
        ]
    )
    (ROUND_DIR / "report.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
