# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.5",
# ]
# ///
"""Render the frozen gradient-mechanism repair tables without changing their estimands."""

import argparse
import html
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.basedatatypes import BaseTraceType

PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}
STATE_ORDER = ["fraction_0p10", "fraction_0p25", "fraction_0p70", "decay_onset", "final"]
STATE_LABELS = {
    "fraction_0p10": "0.10T",
    "fraction_0p25": "0.25T",
    "fraction_0p40": "0.40T",
    "fraction_0p55": "0.55T",
    "fraction_0p70": "0.70T",
    "decay_minus_256": "decay - 256",
    "decay_minus_64": "decay - 64",
    "decay_onset": "decay onset",
    "decay_plus_64": "decay + 64",
    "decay_plus_256": "decay + 256",
    "optimizer_decay_minus_256": "decay - 256",
    "optimizer_decay_minus_64": "decay - 64",
    "optimizer_decay_onset": "decay onset",
    "optimizer_decay_plus_64": "decay + 64",
    "final": "final",
}
TARGET_LABELS = {
    "paloma_programming_languages": "Programming Languages",
    "paloma_c4_en": "C4",
    "uncheatable_github_python": "GitHub Python",
    "uncheatable_wikipedia_english": "Wikipedia",
}
SOURCE_LABELS = {
    "starcoder_excluded_global": "StarCoder heldout",
    "starcoder_support_reference": "StarCoder support",
    "nemotron_aggregate": "Nemotron",
}
STATISTIC_LABELS = {"gradient": "Raw gradient", "optimizer_update": "Optimizer update"}
REQUIRED_FILES = (
    "source_source_geometry.csv",
    "target_source_utilities.csv",
    "target_source_choice_alignment.csv",
    "h2_h3_summary.csv",
    "h3_repetition_mechanism_summary.csv",
    "h5_profile_summary.csv",
)


def _require_columns(frame: pd.DataFrame, columns: set[str], *, source: Path) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _read_table(input_dir: Path, name: str, columns: set[str]) -> pd.DataFrame:
    path = input_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing analyzer output: {path}")
    frame = pd.read_csv(path)
    _require_columns(frame, columns, source=path)
    return frame


def _friendly(value: Any, labels: dict[str, str]) -> str:
    text = str(value)
    return labels.get(text, text.replace("_", " "))


def _cohort_label(row: pd.Series, *, include_checkpoint: bool) -> str:
    parts = [
        _friendly(row["analysis_role"], {}),
        _friendly(row["policy_role"], {}),
        _friendly(row["support_id"], {}),
    ]
    if include_checkpoint:
        parts.append(_friendly(row["checkpoint_label"], STATE_LABELS))
    return " | ".join(parts)


def _dropdown_figure(traces: list[BaseTraceType], labels: list[str], *, title: str) -> go.Figure:
    if not traces:
        raise ValueError(f"No traces available for {title}")
    for index, trace in enumerate(traces):
        trace.visible = index == 0
    buttons = []
    for index, label in enumerate(labels):
        visibility = [position == index for position in range(len(traces))]
        buttons.append(
            {
                "label": label,
                "method": "update",
                "args": [{"visible": visibility}, {"title.text": f"{title}<br><sup>{label}</sup>"}],
            }
        )
    figure = go.Figure(traces)
    figure.update_layout(
        title={"text": f"{title}<br><sup>{labels[0]}</sup>", "x": 0.03},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 0.0,
                "xanchor": "left",
                "y": 1.15,
                "yanchor": "top",
                "showactive": True,
            }
        ],
        margin={"l": 130, "r": 50, "t": 145, "b": 90},
        paper_bgcolor="#fbf8ef",
        plot_bgcolor="#fbf8ef",
        font={"family": "Avenir Next, Avenir, sans-serif", "color": "#183149"},
    )
    return figure


def source_conflict_figure(frame: pd.DataFrame) -> go.Figure:
    selected = frame[frame["geometry"].eq("projected") & frame["component"].eq("trunk")].copy()
    selected["cohort"] = selected.apply(_cohort_label, axis=1, include_checkpoint=False)
    summary = (
        selected.groupby(["cohort", "checkpoint_label", "statistic"], as_index=False, dropna=False)["cosine"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    traces: list[BaseTraceType] = []
    labels: list[str] = []
    for cohort, group in summary.groupby("cohort", sort=True):
        mean = group.pivot(index="statistic", columns="checkpoint_label", values="mean").reindex(
            index=["gradient", "optimizer_update"], columns=STATE_ORDER
        )
        std = group.pivot(index="statistic", columns="checkpoint_label", values="std").reindex(
            index=mean.index, columns=mean.columns
        )
        count = group.pivot(index="statistic", columns="checkpoint_label", values="count").reindex(
            index=mean.index, columns=mean.columns
        )
        text = np.empty(mean.shape, dtype=object)
        custom = np.empty((*mean.shape, 3), dtype=object)
        for row_index in range(mean.shape[0]):
            for column_index in range(mean.shape[1]):
                value = mean.iat[row_index, column_index]
                if pd.isna(value):
                    text[row_index, column_index] = "N/A"
                    custom[row_index, column_index] = [np.nan, np.nan, 0]
                else:
                    sd = std.iat[row_index, column_index]
                    n = int(count.iat[row_index, column_index])
                    text[row_index, column_index] = f"{value:+.3f}"
                    custom[row_index, column_index] = [value, sd, n]
        traces.append(
            go.Heatmap(
                z=-mean.to_numpy(dtype=float),
                x=[STATE_LABELS[state] for state in mean.columns],
                y=[STATISTIC_LABELS[statistic] for statistic in mean.index],
                text=text,
                texttemplate="%{text}",
                customdata=custom,
                colorscale="RdYlGn_r",
                zmin=-1,
                zmax=1,
                colorbar={"title": "Conflict score<br>-cosine"},
                hovertemplate=(
                    "%{y} at %{x}<br>mean cosine=%{customdata[0]:+.4f}"
                    "<br>seed SD=%{customdata[1]:.4f}<br>defined n=%{customdata[2]}<extra></extra>"
                ),
            )
        )
        labels.append(str(cohort))
    figure = _dropdown_figure(
        traces,
        labels,
        title="StarCoder-Nemotron source conflict over training",
    )
    figure.update_layout(
        height=510,
        xaxis_title="Restored training state",
        yaxis_title="Measured vector",
        annotations=[
            {
                "text": (
                    "Larger/red means more conflict. Final optimizer update is undefined because LR=0; "
                    "it is not imputed."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.22,
                "showarrow": False,
                "xanchor": "left",
            }
        ],
    )
    return figure


def target_utility_figure(frame: pd.DataFrame) -> go.Figure:
    selected = frame[frame["geometry"].eq("projected") & frame["component"].eq("trunk")].copy()
    selected["cohort"] = selected.apply(_cohort_label, axis=1, include_checkpoint=True)
    summary = selected.groupby(["cohort", "target", "source"], as_index=False)["cosine"].agg(
        mean="mean", std="std", count="count"
    )
    target_order = list(TARGET_LABELS)
    source_order = list(SOURCE_LABELS)
    traces: list[BaseTraceType] = []
    labels: list[str] = []
    for cohort, group in summary.groupby("cohort", sort=True):
        mean = group.pivot(index="target", columns="source", values="mean").reindex(
            index=target_order, columns=source_order
        )
        std = group.pivot(index="target", columns="source", values="std").reindex(index=mean.index, columns=mean.columns)
        count = group.pivot(index="target", columns="source", values="count").reindex(
            index=mean.index, columns=mean.columns
        )
        text = mean.map(lambda value: "" if pd.isna(value) else f"{value:+.3f}").to_numpy()
        custom = np.stack(
            [mean.to_numpy(dtype=float), std.to_numpy(dtype=float), count.fillna(0).to_numpy(dtype=int)], axis=-1
        )
        traces.append(
            go.Heatmap(
                z=-mean.to_numpy(dtype=float),
                x=[SOURCE_LABELS[source] for source in mean.columns],
                y=[TARGET_LABELS[target] for target in mean.index],
                text=text,
                texttemplate="%{text}",
                customdata=custom,
                colorscale="RdYlGn_r",
                zmid=0,
                colorbar={"title": "Misalignment<br>-utility cosine"},
                hovertemplate=(
                    "%{y} target / %{x} update<br>mean utility cosine=%{customdata[0]:+.4f}"
                    "<br>seed SD=%{customdata[1]:.4f}<br>n=%{customdata[2]}<extra></extra>"
                ),
            )
        )
        labels.append(str(cohort))
    figure = _dropdown_figure(traces, labels, title="Target-source optimizer-update alignment matrix")
    figure.update_layout(
        height=610,
        xaxis_title="Counterfactual source update",
        yaxis_title="Evaluation target gradient",
        annotations=[
            {
                "text": (
                    "Cell labels are utility cosines: positive means the source update locally reduces the target loss."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.18,
                "showarrow": False,
                "xanchor": "left",
            }
        ],
    )
    return figure


def source_choice_figure(frame: pd.DataFrame) -> go.Figure:
    selected = frame[frame["geometry"].eq("projected") & frame["component"].eq("trunk")].copy()
    selected["cohort"] = selected.apply(_cohort_label, axis=1, include_checkpoint=False) + " | " + selected["contrast"]
    summary = selected.groupby(["cohort", "target", "checkpoint_label"], as_index=False)["A_y"].agg(
        mean="mean", std="std", count="count"
    )
    target_order = list(TARGET_LABELS)
    traces: list[BaseTraceType] = []
    labels: list[str] = []
    for cohort, group in summary.groupby("cohort", sort=True):
        observed_states = [state for state in STATE_LABELS if state in set(group["checkpoint_label"])]
        mean = group.pivot(index="target", columns="checkpoint_label", values="mean").reindex(
            index=target_order, columns=observed_states
        )
        std = group.pivot(index="target", columns="checkpoint_label", values="std").reindex(
            index=mean.index, columns=mean.columns
        )
        count = group.pivot(index="target", columns="checkpoint_label", values="count").reindex(
            index=mean.index, columns=mean.columns
        )
        text = mean.map(lambda value: "" if pd.isna(value) else f"{value:+.3f}").to_numpy()
        custom = np.stack(
            [mean.to_numpy(dtype=float), std.to_numpy(dtype=float), count.fillna(0).to_numpy(dtype=int)], axis=-1
        )
        traces.append(
            go.Heatmap(
                z=-mean.to_numpy(dtype=float),
                x=[STATE_LABELS.get(state, state) for state in mean.columns],
                y=[TARGET_LABELS[target] for target in mean.index],
                text=text,
                texttemplate="%{text}",
                customdata=custom,
                colorscale="RdYlGn_r",
                zmid=0,
                colorbar={"title": "Choice penalty<br>-A_y"},
                hovertemplate=(
                    "%{y} at %{x}<br>mean A_y=%{customdata[0]:+.4f}"
                    "<br>seed SD=%{customdata[1]:.4f}<br>n=%{customdata[2]}<extra></extra>"
                ),
            )
        )
        labels.append(str(cohort))
    figure = _dropdown_figure(traces, labels, title="Target-conditioned source-choice alignment")
    figure.update_layout(
        height=620,
        xaxis_title="Restored training state",
        yaxis_title="Evaluation target",
        annotations=[
            {
                "text": (
                    "Positive A_y favors the left source in the selected contrast; inspect the dropdown contrast name."
                ),
                "xref": "paper",
                "yref": "paper",
                "x": 0,
                "y": -0.18,
                "showarrow": False,
                "xanchor": "left",
            }
        ],
    )
    return figure


def mechanism_forest_figure(frames: list[pd.DataFrame]) -> go.Figure:
    combined = pd.concat(frames, ignore_index=True)
    _require_columns(
        combined,
        {"contrast", "mean", "bootstrap_ci95_low", "bootstrap_ci95_high", "n_paired_seeds", "evidence_role"},
        source=Path("summary tables"),
    )
    combined = combined.iloc[::-1].reset_index(drop=True)
    colors = np.where(combined["contrast"].str.startswith("H2"), "#1b7f79", "#d65a31")
    colors = np.where(combined["contrast"].str.startswith("H5"), "#d9a21b", colors)
    figure = go.Figure(
        go.Scatter(
            x=combined["mean"],
            y=combined["contrast"],
            mode="markers",
            marker={"size": 10, "color": colors, "line": {"color": "#183149", "width": 1}},
            error_x={
                "type": "data",
                "symmetric": False,
                "array": combined["bootstrap_ci95_high"] - combined["mean"],
                "arrayminus": combined["mean"] - combined["bootstrap_ci95_low"],
                "color": "#183149",
                "thickness": 1.5,
            },
            customdata=np.stack([combined["n_paired_seeds"], combined["evidence_role"]], axis=-1),
            hovertemplate=(
                "%{y}<br>mean=%{x:+.5f}<br>paired seeds=%{customdata[0]}<br>role=%{customdata[1]}<extra></extra>"
            ),
        )
    )
    figure.add_vline(x=0, line_color="#6f7f87", line_dash="dash")
    figure.update_layout(
        title={"text": "H2/H3/H5 mechanism effects with seed-bootstrap 95% intervals", "x": 0.03},
        xaxis_title="Frozen contrast estimate",
        yaxis_title="",
        height=max(720, 32 * len(combined) + 180),
        margin={"l": 430, "r": 40, "t": 90, "b": 80},
        paper_bgcolor="#fbf8ef",
        plot_bgcolor="#fbf8ef",
        font={"family": "Avenir Next, Avenir, sans-serif", "color": "#183149"},
    )
    return figure


def _write_figure(figure: go.Figure, path: Path) -> None:
    path.write_text(pio.to_html(figure, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG))


def _write_index(output_dir: Path) -> None:
    cards = [
        (
            "Source conflict",
            "source_source_conflict_matrix.html",
            "StarCoder-Nemotron gradient and optimizer-update cosines over time.",
        ),
        (
            "Target-source utility",
            "target_source_utility_matrix.html",
            "Which source update locally helps each evaluation target.",
        ),
        (
            "Source-choice alignment",
            "target_source_choice_alignment.html",
            "Target-specific preference for one source update over another.",
        ),
        (
            "Mechanism effects",
            "mechanism_effect_forest.html",
            "H2, H3, and H5 effect estimates with seed-bootstrap intervals.",
        ),
    ]
    card_html = "".join(
        f'<a class="card" href="{html.escape(href)}"><h2>{html.escape(title)}</h2><p>{html.escape(body)}</p></a>'
        for title, href, body in cards
    )
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>StarCoder WSD80 gradient-mechanism repair</title>
<style>
:root{{--ink:#183149;--paper:#fbf8ef;--teal:#1b7f79;--orange:#d65a31;--line:#d8cfbd}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Avenir Next",Avenir,sans-serif}}
main{{max-width:1180px;margin:0 auto;padding:72px 32px 96px}}
.eyebrow{{text-transform:uppercase;letter-spacing:.16em;color:var(--orange);font-weight:700}}
h1{{font-family:Georgia,serif;font-size:clamp(42px,7vw,78px);line-height:.98;margin:12px 0 24px;max-width:900px}}
.scope{{font-size:19px;line-height:1.6;max-width:900px;border-left:5px solid var(--teal);padding:8px 0 8px 22px}}
.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:18px;margin-top:44px}}
.card{{display:block;color:inherit;text-decoration:none;border:1px solid var(--line);background:#fffdf7;
padding:28px;min-height:180px;transition:transform .16s ease,border-color .16s ease}}
.card:hover{{transform:translateY(-4px);border-color:var(--teal)}}
.card h2{{font-family:Georgia,serif;font-size:28px;margin:0 0 12px}}
.card p{{line-height:1.55;margin:0}}
footer{{margin-top:48px;padding-top:20px;border-top:1px solid var(--line);line-height:1.5;color:#536674}}
@media(max-width:720px){{main{{padding:44px 18px 70px}}.grid{{grid-template-columns:1fr}}}}
</style></head><body><main>
<p class="eyebrow">Post-outcome development evidence</p>
<h1>Gradient conflict and target alignment through training</h1>
<p class="scope">These plots visualize the frozen v10 repair tables. They do not alter the estimands or restore
untouched-confirmatory status. H1 is descriptive; H2, H3, and the H5 profile are development/falsification evidence.
H4 is excluded because its calibration rule was not frozen before outcomes.</p>
<section class="grid">{card_html}</section>
<footer>Final optimizer-update cosine is intentionally missing at the zero-learning-rate checkpoint. All other matrix
cells display seed means; hover exposes SD and sample count.</footer>
</main></body></html>"""
    (output_dir / "index.html").write_text(document)


def render(input_dir: Path, output_dir: Path) -> None:
    missing = [name for name in REQUIRED_FILES if not (input_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Analyzer output is incomplete; missing {missing} in {input_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    h1 = _read_table(
        input_dir,
        "source_source_geometry.csv",
        {
            "analysis_role",
            "policy_role",
            "support_id",
            "checkpoint_label",
            "statistic",
            "geometry",
            "component",
            "cosine",
        },
    )
    utilities = _read_table(
        input_dir,
        "target_source_utilities.csv",
        {
            "analysis_role",
            "policy_role",
            "support_id",
            "checkpoint_label",
            "target",
            "source",
            "geometry",
            "component",
            "cosine",
        },
    )
    alignment = _read_table(
        input_dir,
        "target_source_choice_alignment.csv",
        {
            "analysis_role",
            "policy_role",
            "support_id",
            "checkpoint_label",
            "target",
            "contrast",
            "geometry",
            "component",
            "A_y",
        },
    )
    summaries = [
        _read_table(
            input_dir,
            name,
            {"contrast", "mean", "bootstrap_ci95_low", "bootstrap_ci95_high", "n_paired_seeds", "evidence_role"},
        )
        for name in ("h2_h3_summary.csv", "h3_repetition_mechanism_summary.csv", "h5_profile_summary.csv")
    ]
    _write_figure(source_conflict_figure(h1), output_dir / "source_source_conflict_matrix.html")
    _write_figure(target_utility_figure(utilities), output_dir / "target_source_utility_matrix.html")
    _write_figure(source_choice_figure(alignment), output_dir / "target_source_choice_alignment.html")
    _write_figure(mechanism_forest_figure(summaries), output_dir / "mechanism_effect_forest.html")
    _write_index(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory produced by the frozen analyzer")
    parser.add_argument("--output-dir", type=Path, required=True, help="Destination for self-contained HTML plots")
    args = parser.parse_args()
    render(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
