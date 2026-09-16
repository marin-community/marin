# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["kaleido==0.2.1", "numpy", "pandas", "plotly", "wandb"]
# ///
"""Decompose the phase-TV ladder per Uncheatable component, not just on the macro.

The redistribution result so far rests on one contrast magnitude with two seed blocks: four deferred
components improved, three others degraded, and the macro average hid the trade. The ladder logs all
seven components for all 27 runs, so the same decomposition can be run at four magnitudes with three
seed blocks -- which turns a single observation into a response curve.

Three questions the macro cannot answer.

Does the redistribution grow with contrast? If the ordering effect is the odd part of a smooth
response, each component's own ordering term should grow roughly linearly in the contrast magnitude,
with deferred and non-deferred components moving in opposite directions.

Does each component pay a quadratic cost? The macro's asymmetry cost grows faster than its ordering
effect, which is what makes two-phase lose. If that structure holds per component, the cancellation is
a property of the response surface rather than of the averaging.

And is any single component a genuine two-phase win? A component with ``|o| > c`` at every magnitude
would be a real gain that the macro averages away -- which would matter, because it would mean
two-phase policies are useful for narrow targets even where they are useless for broad ones.

Noise is estimated per component from the spread across seed blocks within each treatment, pooled over
all nine treatments, giving eighteen degrees of freedom per component. Components differ substantially
in absolute BPB, so everything is reported in each component's own run-sigma units.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import wandb
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "component_redistribution_ladder_20260727"

TRAIN_PROJECT = "marin-community/marin"
TRAIN_TAG = "delphi-3e18-uncheatable-phase-tv-ladder"
EXPECTED_RUNS = 27
COMPONENT_PATTERN = re.compile(r"^eval/uncheatable_eval/([^/]+)/bpb$")
RUN_NAME_PATTERN = re.compile(r"^tvladder_(\d+)_(plus|minus|center)_tv([0-9.]+)_s(\d+)")
# The contrast direction defers technical content into the late phase, so these four components are the
# ones the intervention was designed to help. Taken from the panel's own direction definition.
DEFERRED = ("arxiv_computer_science", "arxiv_physics", "github_cpp", "github_python")
DEFERRED_COLOR = "#1A6FB5"
OTHER_COLOR = "#B0752F"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def collect(timeout: int) -> pd.DataFrame:
    """Per-run, per-component Uncheatable BPB, with the ladder coordinates parsed from the run name."""
    runs = list(wandb.Api(timeout=timeout).runs(TRAIN_PROJECT, filters={"tags": {"$in": [TRAIN_TAG]}}, per_page=60))
    assert len(runs) == EXPECTED_RUNS, f"expected {EXPECTED_RUNS} ladder runs, found {len(runs)}"
    rows = []
    for run in runs:
        match = RUN_NAME_PATTERN.match(run.name)
        assert match is not None, f"cannot parse ladder coordinates from {run.name!r}"
        _index, sign, phase_tv, seed_block = match.groups()
        summary = dict(run.summary)
        for key, value in summary.items():
            component = COMPONENT_PATTERN.match(key)
            if component is None:
                continue
            rows.append(
                {
                    "component": component.group(1),
                    "sign": sign,
                    "phase_tv": 0.0 if sign == "center" else float(phase_tv),
                    "seed_block": int(seed_block),
                    "bpb": float(value),
                }
            )
    frame = pd.DataFrame(rows)
    assert frame["component"].nunique() == 7, f"expected 7 components, found {sorted(frame['component'].unique())}"
    return frame


def component_sigma(frame: pd.DataFrame) -> pd.Series:
    """Run-to-run standard deviation per component, from seed-block spread within each treatment.

    Every treatment was run in three seed blocks with the policy held fixed, so the spread across
    blocks is noise by construction. Pooling the nine treatments gives eighteen degrees of freedom.
    """
    pooled = {}
    for component, block in frame.groupby("component"):
        variances, weights = [], []
        for _key, treatment in block.groupby(["sign", "phase_tv"]):
            if len(treatment) < 2:
                continue
            variances.append(treatment["bpb"].var(ddof=1))
            weights.append(len(treatment) - 1)
        pooled[component] = float(np.sqrt(np.average(variances, weights=weights)))
    return pd.Series(pooled, name="sigma")


def decompose(frame: pd.DataFrame, sigma: pd.Series) -> pd.DataFrame:
    """Ordering effect and asymmetry cost per component, magnitude and seed block."""
    controls = frame[frame["sign"] == "center"].set_index(["component", "seed_block"])["bpb"]
    rows = []
    treated = frame[frame["sign"] != "center"]
    for (component, phase_tv, seed_block), block in treated.groupby(["component", "phase_tv", "seed_block"]):
        signs = set(block["sign"])
        if signs != {"plus", "minus"}:
            continue
        plus = float(block.loc[block["sign"] == "plus", "bpb"].iloc[0])
        minus = float(block.loc[block["sign"] == "minus", "bpb"].iloc[0])
        tied = float(controls.loc[(component, seed_block)])
        scale = sigma[component]
        rows.append(
            {
                "component": component,
                "deferred": component in DEFERRED,
                "phase_tv": phase_tv,
                "seed_block": seed_block,
                "ordering_effect": (0.5 * (plus - minus)) / scale,
                "asymmetry_cost": (0.5 * (plus + minus) - tied) / scale,
                "best_gain": (min(plus, minus) - tied) / scale,
                "plus_gain": (plus - tied) / scale,
            }
        )
    return pd.DataFrame(rows)


def build_figure(levels: pd.DataFrame) -> go.Figure:
    panels = (
        ("ordering_effect", "ordering effect o (component sigma)"),
        ("asymmetry_cost", "asymmetry cost c (component sigma)"),
        ("plus_gain", "preregistered orientation vs tied (component sigma)"),
    )
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[title for _column, title in panels],
        horizontal_spacing=0.07,
    )
    seen: set[str] = set()
    for column, (value_column, _title) in enumerate(panels, start=1):
        for component, block in levels.groupby("component"):
            deferred = bool(block["deferred"].iloc[0])
            mean = block.groupby("phase_tv")[value_column].mean().reset_index()
            group = "deferred (technical)" if deferred else "not deferred"
            figure.add_trace(
                go.Scatter(
                    x=mean["phase_tv"],
                    y=mean[value_column],
                    mode="lines+markers",
                    line={"color": DEFERRED_COLOR if deferred else OTHER_COLOR, "width": 1.8},
                    marker={"size": 6},
                    opacity=0.85,
                    name=group,
                    legendgroup=group,
                    showlegend=column == 1 and group not in seen,
                    hovertemplate=f"{component}<br>TV %{{x:.2f}}<br>%{{y:+.2f}} sigma<extra></extra>",
                ),
                row=1,
                col=column,
            )
            seen.add(group)
        figure.add_hline(y=0.0, line={"color": "#444", "width": 1.1, "dash": "dot"}, row=1, col=column)
        figure.update_xaxes(title_text="phase contrast, total variation", row=1, col=column)
    figure.update_yaxes(title_text="component sigma", row=1, col=1)
    figure.update_layout(
        template="simple_white",
        height=440,
        width=1320,
        title={
            "text": (
                "Phase ordering per Uncheatable component, 3e18 ladder<br>"
                "<sub>Each line is one of the seven components, averaged over three seed blocks. "
                "The macro average is the mean of these curves.</sub>"
            )
        },
        legend={"orientation": "h", "yanchor": "bottom", "y": -0.30, "xanchor": "center", "x": 0.5},
        margin={"t": 96, "b": 96},
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--wandb-timeout", type=int, default=180)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    frame = collect(args.wandb_timeout)
    sigma = component_sigma(frame)
    levels = decompose(frame, sigma)
    frame.to_csv(args.output_dir / "component_bpb.csv", index=False)
    levels.to_csv(args.output_dir / "component_decomposition.csv", index=False)

    print("per-component run sigma, from seed-block spread (18 dof each):")
    for component, value in sigma.sort_values().items():
        marker = "deferred" if component in DEFERRED else ""
        print(f"  {component:<26} {value:.6f}   {marker}")

    print("\nfitted response per component, o = kappa*t and c = rho/2*t^2 through the origin:")
    fits = []
    for component, block in levels.groupby("component"):
        tilt = block["phase_tv"].to_numpy()
        kappa = float(np.linalg.lstsq(tilt[:, None], block["ordering_effect"].to_numpy(), rcond=None)[0][0])
        rho = float(
            2.0 * np.linalg.lstsq((0.5 * tilt**2)[:, None], block["asymmetry_cost"].to_numpy(), rcond=None)[0][0]
        )
        # The preregistered orientation is `plus`, technical content late. Counting wins with
        # min(plus, minus) instead would select the better arm after seeing both outcomes, which is
        # biased negative by roughly 0.56 sigma even when neither orientation truly helps.
        wins = int((block.groupby("phase_tv")["plus_gain"].mean() < 0).sum())
        posthoc_wins = int((block.groupby("phase_tv")["best_gain"].mean() < 0).sum())
        fits.append(
            {
                "component": component,
                "deferred": component in DEFERRED,
                "kappa": kappa,
                "rho": rho,
                "optimum_tv": abs(kappa) / rho if rho > 0 else float("nan"),
                "best_gain_sigma": -(kappa**2) / (2.0 * rho) if rho > 0 else float("nan"),
                "levels_won_prereg": wins,
                "levels_won_posthoc": posthoc_wins,
            }
        )
    fit_table = pd.DataFrame(fits).sort_values("kappa")
    fit_table.to_csv(args.output_dir / "component_fits.csv", index=False)
    print(fit_table.to_string(index=False, float_format=lambda value: f"{value:+.3f}"))

    print("\nis any component a genuine two-phase win at every magnitude?")
    print("  (counted on the preregistered `plus` orientation; the post-hoc best-arm count is")
    print("   reported alongside and is biased negative by construction)")
    always = fit_table[fit_table["levels_won_prereg"] == levels["phase_tv"].nunique()]
    if len(always):
        for _, row in always.iterrows():
            print(
                f"  {row['component']}: preregistered orientation beats tied at all "
                f"{int(row['levels_won_prereg'])} magnitudes"
            )
    else:
        print("  none -- no component wins at every magnitude")

    figure = build_figure(levels)
    figure.write_html(args.output_dir / "component_redistribution.html", include_plotlyjs="cdn", config=PLOT_CONFIG)
    figure.write_image(args.output_dir / "component_redistribution.png", scale=4)
    print(f"\nwrote {args.output_dir}")


if __name__ == "__main__":
    main()
