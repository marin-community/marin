# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build the "case file" edition of the Delphi midtraining contamination report.

This is a re-skin of ``build_delphi_midtraining_unified_story_report.py``: it
reuses that script's data loaders and Plotly figure builders verbatim (single
source of truth for the numbers) and wraps them in a distinct visual design --
a forensic dossier with a hand-built hero plot of "the miss", numbered case
steps, and figures framed as labelled exhibits.

It writes a *new* file and never touches the original report.

Run:
    uv run --with plotly --with pandas --with numpy \\
      python scripts/analysis/build_delphi_midtraining_dossier_report.py
"""

from __future__ import annotations

import argparse
import html
import math
from pathlib import Path

# Sibling-script reuse: running `python scripts/analysis/<file>.py` puts this
# directory on sys.path[0], so the unified-story module imports directly. We keep
# all figure/data logic there and only restyle the presentation here.
import build_delphi_midtraining_unified_story_report as story
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio

DEFAULT_OUTPUT = Path("sk_midtrain_analysis_fable/delphi_midtraining_dossier_report.html")

# Design tokens, mirrored from the CSS so the Plotly figures share the palette.
INK = "#15233f"
MUTED = "#5f6b83"
GRID = "#e8edf5"
LINE = "#dde4ef"
PAGE_FONT = "IBM Plex Sans, -apple-system, BlinkMacSystemFont, Segoe UI, sans-serif"
DISPLAY_FONT = "Space Grotesk, IBM Plex Sans, sans-serif"
MONO_FONT = "IBM Plex Mono, ui-monospace, SFMono-Regular, Menlo, monospace"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--contamination-root", type=Path, default=story.DEFAULT_CONTAMINATION_ROOT)
    return parser.parse_args()


def restyle_figure(fig: go.Figure) -> go.Figure:
    """Apply the dossier theme to a figure built by the unified-story module.

    The figures keep their semantic trace colors (red = the K=0.20 outlier, etc.)
    so the charts stay consistent with the narrative. We only restyle the chrome:
    fonts, gridlines, hover, legend, and we drop the in-chart title because each
    figure gets a labelled exhibit caption instead.
    """
    fig.update_layout(
        title_text=None,
        template="plotly_white",
        paper_bgcolor="white",
        plot_bgcolor="white",
        margin=dict(l=64, r=28, t=46, b=56),
        font=dict(family=PAGE_FONT, size=13, color=INK),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.27,
            xanchor="left",
            x=0,
            font=dict(family=MONO_FONT, size=11, color=MUTED),
        ),
        hoverlabel=dict(bgcolor=INK, bordercolor=INK, font=dict(family=MONO_FONT, size=12, color="white")),
    )
    fig.update_xaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=LINE,
        tickfont=dict(family=MONO_FONT, size=11, color=MUTED),
        title_font=dict(family=PAGE_FONT, size=12, color=MUTED),
    )
    fig.update_yaxes(
        gridcolor=GRID,
        zerolinecolor=GRID,
        linecolor=LINE,
        tickfont=dict(family=MONO_FONT, size=11, color=MUTED),
        title_font=dict(family=PAGE_FONT, size=12, color=MUTED),
    )
    for annotation in fig.layout.annotations:  # subplot titles
        annotation.font.family = DISPLAY_FONT
        annotation.font.size = 13
        annotation.font.color = INK
    return fig


def figure_card(fig: go.Figure, div_id: str, exhibit_no: int, caption: str) -> str:
    body = pio.to_html(
        restyle_figure(fig),
        include_plotlyjs=False,
        full_html=False,
        div_id=div_id,
        config={"responsive": True, "displaylogo": False},
    )
    return f"""
    <figure class="exhibit">
      <div class="exhibit-frame">{body}</div>
      <figcaption><span class="exhibit-tag">Exhibit {exhibit_no:02d}</span>{html.escape(caption)}</figcaption>
    </figure>
    """


def build_hero_svg() -> str:
    """Hand-built SVG of the central finding: the fitted law, and the 10^22 miss.

    Solid curve = fit through 3e20; dashed = extrapolation. The 10^22 point lands
    well below its own prediction (the model "beat the prophecy" by ~18.6%).
    """
    width, height = 860, 430
    left, right, top, bottom = 66, 150, 34, 366
    px0, px1 = left, width - right
    py0, py1 = top, bottom
    lx0, lx1 = math.log10(3e18), math.log10(1e22)
    vtop, vbot = 0.98, 0.52
    floor, amplitude, exponent = 0.5788, 0.4523, 0.18

    def sx(log10x: float) -> float:
        return px0 + (log10x - lx0) / (lx1 - lx0) * (px1 - px0)

    def sv(value: float) -> float:
        return py0 + (vtop - value) / (vtop - vbot) * (py1 - py0)

    def curve(log10x: float) -> float:
        return float(story.floor_power(10**log10x, floor, amplitude, exponent))

    def path(a: float, b: float, samples: int = 48) -> str:
        pts = []
        for i in range(samples + 1):
            log10x = a + (b - a) * i / samples
            pts.append((sx(log10x), sv(curve(log10x))))
        return "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)

    cutoff = math.log10(3e20)
    solid_path = path(lx0, cutoff)
    dashed_path = path(cutoff, lx1)

    parts: list[str] = [
        f'<svg viewBox="0 0 {width} {height}" class="missplot" role="img" aria-label="Fitted math-validation loss versus base compute; the 10^22 endpoint falls far below the extrapolated fit.">'
    ]

    # Horizontal gridlines + y labels.
    for value in (0.6, 0.7, 0.8, 0.9):
        y = sv(value)
        parts.append(f'<line class="hp-grid" x1="{left}" y1="{y:.1f}" x2="{width - right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="hp-tick hp-tick-y" x="{left - 10}" y="{y + 3.5:.1f}">{value:.1f}</text>')

    # Vertical decade ticks + x labels.
    for label, log10x in (("3e18", lx0), ("1e20", 20.0), ("1e22", 22.0)):
        x = sx(log10x)
        parts.append(f'<line class="hp-grid hp-grid-v" x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{bottom}"/>')
        parts.append(f'<text class="hp-tick hp-tick-x" x="{x:.1f}" y="{bottom + 22}">{label}</text>')

    # Fit cutoff marker.
    xc = sx(cutoff)
    parts.append(f'<line class="hp-cutoff" x1="{xc:.1f}" y1="{top}" x2="{xc:.1f}" y2="{bottom}"/>')
    parts.append(f'<text class="hp-cutoff-label" x="{xc - 8:.1f}" y="{top + 14}">fit cutoff · 3e20</text>')

    # Axis baseline.
    parts.append(f'<line class="hp-axis" x1="{left}" y1="{bottom}" x2="{width - right}" y2="{bottom}"/>')

    # The law: solid through 3e20, dashed extrapolation beyond.
    parts.append(f'<path class="hp-curve hp-curve-solid" d="{solid_path}" pathLength="1"/>')
    parts.append(f'<path class="hp-curve hp-curve-dash" d="{dashed_path}" pathLength="1"/>')

    # Observed points on the fit (through 3e20) + the 1e21 endpoint that landed close.
    fit_scales = ["3e18", "9e18", "2e19", "3e19", "9e19", "2e20", "3e20"]
    for index, name in enumerate(fit_scales):
        log10x = math.log10(story.SCALE_FLOPS[name])
        cx, cy = sx(log10x), sv(curve(log10x))
        parts.append(
            f'<circle class="hp-obs" style="--d:{0.9 + index * 0.05:.2f}s" cx="{cx:.1f}" cy="{cy:.1f}" r="5.2"/>'
        )
    x21 = sx(21.0)
    y21 = sv(curve(21.0) + 0.004)
    parts.append(f'<circle class="hp-obs hp-obs-held" style="--d:1.35s" cx="{x21:.1f}" cy="{y21:.1f}" r="5.2"/>')

    # The headline: predicted vs actual at 1e22.
    x22 = sx(22.0)
    y_pred = sv(curve(22.0))  # 0.665
    y_act = sv(0.561)
    parts.append('<g class="hp-miss">')
    parts.append(f'<line class="hp-delta" x1="{x22:.1f}" y1="{y_pred:.1f}" x2="{x22:.1f}" y2="{y_act:.1f}"/>')
    parts.append(f'<circle class="hp-pred" cx="{x22:.1f}" cy="{y_pred:.1f}" r="7"/>')
    parts.append(f'<circle class="hp-act" cx="{x22:.1f}" cy="{y_act:.1f}" r="7.5"/>')
    lx = x22 + 16
    parts.append(f'<text class="hp-lab hp-lab-pred" x="{lx}" y="{y_pred + 1:.1f}">predicted 0.665</text>')
    parts.append(f'<text class="hp-lab hp-lab-act" x="{lx}" y="{y_act + 4:.1f}">actual 0.561</text>')
    parts.append(f'<text class="hp-lab hp-lab-delta" x="{lx}" y="{(y_pred + y_act) / 2 + 4:.1f}">+18.6%</text>')
    parts.append("</g>")

    # Axis titles.
    parts.append(
        f'<text class="hp-axis-title" x="{(left + width - right) / 2:.1f}" y="{height - 6}">base pretraining compute (FLOPs, log scale)</text>'
    )
    parts.append(
        f'<text class="hp-axis-title hp-axis-title-y" transform="translate(16,{(top + bottom) / 2:.1f}) rotate(-90)" x="0" y="0">old 4plus math val loss</text>'
    )

    parts.append("</svg>")
    return "\n".join(parts)


CSS = """
:root {
  --abyss: #0b1830;
  --abyss-2: #122038;
  --ink: #15233f;
  --ink-soft: #38456180;
  --muted: #5f6b83;
  --faint: #8b97ac;
  --paper: #f4f6fb;
  --surface: #ffffff;
  --line: #dde4ef;
  --grid: #e8edf5;
  --signal: #1f5fd6;
  --signal-deep: #1850b8;
  --miss: #d62728;
  --amber: #dd8a1a;
}
* { box-sizing: border-box; }
html { scroll-behavior: smooth; }
body {
  margin: 0;
  background: var(--paper);
  color: var(--ink);
  font-family: "IBM Plex Sans", -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  font-size: 17px;
  line-height: 1.65;
  -webkit-font-smoothing: antialiased;
}
a { color: var(--signal-deep); text-decoration: none; }
a:hover { text-decoration: underline; }
code {
  font-family: "IBM Plex Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
  font-size: 0.86em;
  background: #eef2f9;
  border: 1px solid var(--line);
  border-radius: 5px;
  padding: 0.05em 0.34em;
  word-break: break-word;
}
.mono { font-family: "IBM Plex Mono", ui-monospace, monospace; }

/* ---------- hero ---------- */
.hero {
  position: relative;
  overflow: hidden;
  background:
    radial-gradient(1100px 520px at 78% -8%, rgba(31,95,214,0.30), transparent 60%),
    radial-gradient(720px 420px at 8% 110%, rgba(214,39,40,0.14), transparent 60%),
    linear-gradient(165deg, #0a1530 0%, #0d1a38 55%, #0b1830 100%);
  color: #eaf0fb;
  border-bottom: 1px solid #0a1430;
}
.hero-inner {
  max-width: 1180px;
  margin: 0 auto;
  padding: clamp(40px, 6vw, 86px) clamp(20px, 5vw, 56px) clamp(34px, 4vw, 56px);
}
.kicker {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
  letter-spacing: 0.26em;
  text-transform: uppercase;
  color: #8fb4ff;
  display: flex;
  align-items: center;
  gap: 12px;
}
.kicker::after {
  content: "";
  flex: 1;
  height: 1px;
  background: linear-gradient(90deg, rgba(143,180,255,0.5), transparent);
}
.hero-title {
  font-family: "Space Grotesk", sans-serif;
  font-weight: 600;
  font-size: clamp(2.3rem, 5.6vw, 4.5rem);
  line-height: 1.02;
  letter-spacing: -0.025em;
  margin: 22px 0 0;
  max-width: 16ch;
}
.hero-title .accent { color: #ff6b6c; }
.hero-title sup { font-size: 0.5em; font-weight: 500; top: -0.7em; }
.hero-lead {
  max-width: 60ch;
  margin: 22px 0 0;
  font-size: clamp(1.02rem, 1.5vw, 1.2rem);
  color: #bcc8e0;
  line-height: 1.6;
}
.hero-grid {
  display: grid;
  grid-template-columns: 1.35fr 1fr;
  gap: clamp(22px, 4vw, 56px);
  align-items: center;
  margin-top: clamp(30px, 4vw, 48px);
}
.hero-plotwrap {
  background: rgba(255,255,255,0.04);
  border: 1px solid rgba(143,180,255,0.18);
  border-radius: 14px;
  padding: 16px 16px 10px;
  box-shadow: 0 30px 60px -30px rgba(0,0,0,0.7);
}
.hero-plot-cap {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #8fb4ff;
  margin: 4px 2px 10px;
}
.hero-legend {
  display: flex;
  flex-wrap: wrap;
  gap: 7px 16px;
  margin: 8px 2px 2px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  color: #aab8d6;
}
.hero-legend span { display: inline-flex; align-items: center; gap: 7px; }
.lg-swatch { width: 16px; height: 0; border-top: 3px solid var(--signal); display: inline-block; }
.lg-dash { border-top: 3px dashed #6f9bf0; }
.lg-dot { width: 10px; height: 10px; border-radius: 50%; border-top: 0; }
.lg-pred { background: transparent; border: 2px solid #9fc0ff; }
.lg-act { background: #ff6b6c; }

/* verdict */
.verdict {
  margin-top: clamp(26px, 3.5vw, 40px);
  display: flex;
  flex-wrap: wrap;
  align-items: stretch;
  gap: 14px;
}
.vbox {
  border: 1px solid rgba(143,180,255,0.20);
  background: rgba(255,255,255,0.035);
  border-radius: 12px;
  padding: 14px 18px;
  min-width: 168px;
}
.vbox-label {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.68rem;
  letter-spacing: 0.16em;
  text-transform: uppercase;
  color: #8b9bbd;
}
.vbox-num {
  font-family: "Space Grotesk", sans-serif;
  font-weight: 600;
  font-size: 2rem;
  line-height: 1.1;
  margin-top: 4px;
}
.vbox-num.bad { color: #ff6b6c; }
.vbox-num.good { color: #6fd3a8; }
.vbox-sub { font-size: 0.82rem; color: #aab8d6; margin-top: 2px; }
.varrow {
  align-self: center;
  font-family: "Space Grotesk", sans-serif;
  font-size: 1.6rem;
  color: #8fb4ff;
  padding: 0 2px;
}

/* ---------- sticky nav ---------- */
.rail {
  position: sticky;
  top: 0;
  z-index: 40;
  background: rgba(244,246,251,0.86);
  backdrop-filter: blur(10px);
  border-bottom: 1px solid var(--line);
}
.rail-inner {
  max-width: 1180px;
  margin: 0 auto;
  padding: 0 clamp(20px, 5vw, 56px);
  display: flex;
  align-items: center;
  gap: 18px;
  min-height: 52px;
  overflow-x: auto;
}
.rail-brand {
  font-family: "Space Grotesk", sans-serif;
  font-weight: 600;
  color: var(--ink);
  white-space: nowrap;
  letter-spacing: -0.01em;
}
.rail-brand .delta { color: var(--signal); }
.rail-links { display: flex; gap: 4px; }
.rail-links a {
  display: inline-flex;
  align-items: baseline;
  gap: 6px;
  white-space: nowrap;
  color: var(--muted);
  font-size: 0.86rem;
  font-weight: 500;
  padding: 6px 11px;
  border-radius: 7px;
  text-decoration: none;
}
.rail-links a .n {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.74rem;
  color: var(--faint);
}
.rail-links a:hover { background: #e7edf8; text-decoration: none; }
.rail-links a.active { background: var(--signal); color: #fff; }
.rail-links a.active .n { color: #cfe0ff; }

/* ---------- body ---------- */
.wrap {
  max-width: 1180px;
  margin: 0 auto;
  padding: 0 clamp(20px, 5vw, 56px) 80px;
}
.prose { max-width: 70ch; }
.prose p { margin: 0 0 1.05em; }
.prose p:first-child { margin-top: 0; }

/* glossary */
.glossary {
  margin: 30px 0 6px;
  border: 1px solid var(--line);
  border-radius: 12px;
  background: var(--surface);
  overflow: hidden;
}
.glossary > summary {
  cursor: pointer;
  list-style: none;
  padding: 16px 20px;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.78rem;
  letter-spacing: 0.14em;
  text-transform: uppercase;
  color: var(--signal-deep);
  display: flex;
  align-items: center;
  gap: 10px;
}
.glossary > summary::-webkit-details-marker { display: none; }
.glossary > summary::before { content: "+"; font-size: 1.1rem; color: var(--faint); }
.glossary[open] > summary::before { content: "-"; }
.glossary-body { padding: 0 20px 12px; }

/* steps */
.step {
  padding-top: clamp(48px, 6vw, 84px);
  scroll-margin-top: 64px;
}
.step-head {
  display: flex;
  align-items: flex-start;
  gap: clamp(16px, 2.4vw, 28px);
  border-top: 1px solid var(--line);
  padding-top: 26px;
}
.step-no {
  font-family: "Space Grotesk", sans-serif;
  font-weight: 600;
  font-size: clamp(2.4rem, 5vw, 3.6rem);
  line-height: 0.9;
  color: var(--signal);
  opacity: 0.32;
  letter-spacing: -0.03em;
}
.step-eyebrow {
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.76rem;
  letter-spacing: 0.18em;
  text-transform: uppercase;
  color: var(--signal-deep);
}
.step h2 {
  font-family: "Space Grotesk", sans-serif;
  font-weight: 600;
  font-size: clamp(1.55rem, 3vw, 2.35rem);
  line-height: 1.08;
  letter-spacing: -0.02em;
  margin: 6px 0 0;
  max-width: 22ch;
}
.step-body { margin-top: 22px; }
.step h3 {
  font-family: "Space Grotesk", sans-serif;
  font-weight: 600;
  font-size: 1.12rem;
  margin: 34px 0 8px;
  letter-spacing: -0.01em;
}

/* exhibits */
.exhibit {
  margin: 26px 0 30px;
  background: var(--surface);
  border: 1px solid var(--line);
  border-radius: 14px;
  box-shadow: 0 24px 50px -34px rgba(20,40,90,0.30);
  overflow: hidden;
}
.exhibit-frame { padding: 12px 14px 6px; }
.exhibit .plotly-graph-div { width: 100% !important; }
figcaption {
  display: flex;
  gap: 12px;
  align-items: baseline;
  padding: 12px 18px 16px;
  border-top: 1px solid var(--grid);
  color: var(--muted);
  font-size: 0.92rem;
  line-height: 1.5;
}
.exhibit-tag {
  flex: none;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  letter-spacing: 0.1em;
  text-transform: uppercase;
  color: #fff;
  background: var(--signal);
  border-radius: 6px;
  padding: 3px 8px;
  position: relative;
  top: -1px;
}

/* tables */
.data-table {
  width: 100%;
  border-collapse: collapse;
  margin: 14px 0 8px;
  font-size: 0.9rem;
  font-variant-numeric: tabular-nums;
}
.data-table th {
  text-align: left;
  font-family: "IBM Plex Mono", monospace;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  color: var(--muted);
  border-bottom: 1.5px solid var(--line);
  padding: 9px 12px;
  vertical-align: bottom;
  white-space: nowrap;
}
.data-table td {
  border-bottom: 1px solid var(--grid);
  padding: 9px 12px;
  vertical-align: top;
}
.data-table tr:last-child td { border-bottom: none; }
.data-table tr:hover td { background: #f7f9fd; }
.glossary-body .data-table td:first-child {
  font-weight: 600;
  color: var(--ink);
  white-space: nowrap;
}

/* footer */
.foot {
  background: var(--abyss);
  color: #aab8d6;
  margin-top: 70px;
}
.foot-inner {
  max-width: 1180px;
  margin: 0 auto;
  padding: 40px clamp(20px, 5vw, 56px) 56px;
  font-size: 0.9rem;
  line-height: 1.6;
}
.foot a { color: #9fc0ff; }
.foot code { background: rgba(255,255,255,0.08); border-color: rgba(255,255,255,0.12); color: #d7e2f7; }

/* ---------- the hero plot ---------- */
.missplot { width: 100%; height: auto; display: block; }
.hp-grid { stroke: rgba(143,180,255,0.14); stroke-width: 1; }
.hp-grid-v { stroke-dasharray: 2 5; }
.hp-axis { stroke: rgba(143,180,255,0.4); stroke-width: 1.2; }
.hp-tick { fill: #9fb0d0; font-family: "IBM Plex Mono", monospace; font-size: 11px; }
.hp-tick-y { text-anchor: end; }
.hp-tick-x { text-anchor: middle; }
.hp-axis-title { fill: #8b9bbd; font-family: "IBM Plex Mono", monospace; font-size: 11px; letter-spacing: 0.08em; text-anchor: middle; text-transform: uppercase; }
.hp-cutoff { stroke: rgba(255,255,255,0.28); stroke-width: 1; stroke-dasharray: 4 4; }
.hp-cutoff-label { fill: #aab8d6; font-family: "IBM Plex Mono", monospace; font-size: 10px; text-anchor: end; letter-spacing: 0.06em; }
.hp-curve { fill: none; stroke: #6f9bf0; stroke-width: 3; stroke-linecap: round; }
.hp-curve-solid { stroke: #9fc0ff; }
.hp-curve-dash { stroke-dasharray: 7 6; }
.hp-obs { fill: #9fc0ff; stroke: #0d1a38; stroke-width: 1.5; }
.hp-obs-held { fill: #d7e6ff; }
.hp-pred { fill: #0d1a38; stroke: #9fc0ff; stroke-width: 2.5; }
.hp-act { fill: #ff6b6c; stroke: #0d1a38; stroke-width: 1.5; }
.hp-delta { stroke: #ff6b6c; stroke-width: 1.6; stroke-dasharray: 3 3; }
.hp-lab { font-family: "IBM Plex Mono", monospace; font-size: 12px; }
.hp-lab-pred { fill: #9fc0ff; }
.hp-lab-act { fill: #ffb0b0; }
.hp-lab-delta { fill: #ff6b6c; font-weight: 700; font-size: 15px; font-family: "Space Grotesk", sans-serif; }

/* hero plot entrance animation */
.hp-curve { stroke-dashoffset: 1; animation: draw 1.25s ease forwards; }
.hp-curve-solid { stroke-dasharray: 1; }
.hp-curve-dash { animation-delay: 0.9s; opacity: 0; animation: draw 0.9s ease 0.9s forwards, fade 0.01s linear 0.9s forwards; }
.hp-obs { opacity: 0; animation: pop 0.4s ease forwards; animation-delay: var(--d, 1s); }
.hp-miss { opacity: 0; animation: rise 0.6s ease 1.6s forwards; }
@keyframes draw { to { stroke-dashoffset: 0; } }
@keyframes fade { to { opacity: 1; } }
@keyframes pop { from { opacity: 0; transform: scale(0.4); } to { opacity: 1; transform: scale(1); } }
@keyframes rise { from { opacity: 0; transform: translateY(10px); } to { opacity: 1; transform: translateY(0); } }

@media (max-width: 860px) {
  .hero-grid { grid-template-columns: 1fr; }
  body { font-size: 16px; }
  .step-head { gap: 14px; }
}
@media (prefers-reduced-motion: reduce) {
  html { scroll-behavior: auto; }
  .hp-curve, .hp-curve-dash, .hp-obs, .hp-miss {
    animation: none !important;
    opacity: 1 !important;
    stroke-dashoffset: 0 !important;
  }
}
"""

SCROLLSPY_JS = """
(function () {
  var links = Array.prototype.slice.call(document.querySelectorAll('.rail-links a'));
  var byId = {};
  links.forEach(function (a) {
    var id = a.getAttribute('href').slice(1);
    if (id) byId[id] = a;
  });
  var targets = Object.keys(byId).map(function (id) { return document.getElementById(id); }).filter(Boolean);
  if (!('IntersectionObserver' in window) || !targets.length) return;
  var observer = new IntersectionObserver(function (entries) {
    entries.forEach(function (entry) {
      if (!entry.isIntersecting) return;
      links.forEach(function (a) { a.classList.remove('active'); });
      var active = byId[entry.target.id];
      if (active) active.classList.add('active');
    });
  }, { rootMargin: '-45% 0px -50% 0px', threshold: 0 });
  targets.forEach(function (t) { observer.observe(t); });
})();
"""


def render_report(paths: story.Paths) -> str:
    # Build figures (data + numbers come entirely from the unified-story module).
    figures = {
        "original_error": story.make_original_error_figure(),
        "original_curve": story.make_original_curve_figure(),
        "base_step0": story.make_base_step0_figure(),
        "endpoint_forms": story.make_endpoint_form_figure(),
        "fit_family": story.make_fit_family_figure(),
        "old_isotoken": story.make_old_isotoken_figure(),
        "jaccard": story.make_jaccard_histogram_figure(paths),
        "exposure": story.make_exposure_figure(),
        "ppl_gap": story.make_ppl_gap_figure(paths),
        "clean_unified": story.make_clean_unified_figure(),
        "seen_partition": story.make_seen_partition_figure(),
        "final_error": story.make_final_error_figure(),
    }

    original_table = pd.DataFrame(
        story.OLD_ERROR_TARGETS,
        columns=["series", "old_1e22_actual", "prediction", "prediction_error_pct", "loss_error"],
    )
    seen_summary = story.read_csv("sk_midtrain_analysis_fable/delphi_k020_seen_partition_scaling_fit_summary.csv")
    iso_summary = story.read_csv("sk_midtrain_analysis_fable/delphi_isotoken_clean_seen_unified_report_fit_summary.csv")
    fit_rows = pd.DataFrame(
        [
            {"fact": "old K=0.20 lr0.50 1e22 error", "value": "+18.56%", "source": "old 4plus target"},
            {"fact": "clean-seen K=0.20 lr0.50 1e22 error", "value": "+2.83%", "source": "clean-seen target"},
            {
                "fact": "dropped contaminated 1e22 absolute loss error",
                "value": "+0.0999",
                "source": "seen-partition complement",
            },
            {"fact": "retained clean 1e22 absolute loss error", "value": "+0.0233", "source": "seen partition"},
            {
                "fact": "iso-token clean-seen 1e22 errors",
                "value": "-2.31% to -2.82%",
                "source": "1B/2B/4B/8B fixed-token ladders",
            },
        ]
    )

    steps = [
        f"""
        <section id="symptom" class="step">
          <div class="step-head">
            <div class="step-no">01</div>
            <div>
              <div class="step-eyebrow">The symptom</div>
              <h2>The old 4plus target made 10<sup>22</sup> look too good</h2>
            </div>
          </div>
          <div class="step-body">
            <div class="prose">
              <p>The frozen original report fit endpoint laws through 3e20 and held out 1e21/1e22. The p33m67 K=0.20 ladder was close at 1e21 but badly high at 1e22: the fit predicted loss around 0.665 for lr0.50 while the measured value was 0.561.</p>
              <p>The target was <code>eval/nemotron_cc_math_v1/4plus/loss_anchor</code>. Throughout, the sign convention is <em>prediction minus actual</em>: positive error means the model did better than the fit expected.</p>
            </div>
            {figure_card(figures["original_error"], "fig-original-error", 1, "At 1e22 the old-target fit predicted ~0.66 loss for every p33m67 learning rate, while the runs measured ~0.56 — a uniform ~18% overshoot.")}
            {figure_card(figures["original_curve"], "fig-original-curve", 2, "Fit through the 3e20 cutoff: 1e21 lands on the curve; 1e22 falls far below it. The law looked sound right up to the largest scale.")}
            <h3>Original p33m67 K=0.20 old-target 1e22 numbers</h3>
            {story.table_html(original_table)}
          </div>
        </section>
        """,
        f"""
        <section id="failed-fits" class="step">
          <div class="step-head">
            <div class="step-no">02</div>
            <div>
              <div class="step-eyebrow">Ruling out the easy answers</div>
              <h2>The base models were smooth, and no fit form rescued the old target</h2>
            </div>
          </div>
          <div class="step-body">
            <div class="prose">
              <p>The step-0 base loss showed no such failure. A Chinchilla-style fit through 3e20 predicted base step-0 math loss at 1e22 within about +2.4%, while the endpoint p33m67 old-target fit missed by +18.6%. The break is in the post-midtraining endpoint, not the base.</p>
            </div>
            {figure_card(figures["base_step0"], "fig-base-step0", 3, "The base model's step-0 math loss extrapolates cleanly; only the post-midtraining endpoint loss misses. The functional form is not the culprit.")}
            <div class="prose">
              <p>We then tried per-recipe power laws, Chinchilla floor-plus-power fits, pooled LR-aware fits, log-log fits, parameter/data axes, base rows at D=0, and separate base/improvement components. These fits described the fixed-token series well, but the old K=0.20 target stayed an outlier.</p>
            </div>
            {figure_card(figures["endpoint_forms"], "fig-endpoint-forms", 4, "Swapping endpoint functional forms barely moves the 1e22 error — the miss survives every reasonable law we tried.")}
            {figure_card(figures["fit_family"], "fig-fit-family", 5, "The same fit families that fail on the old target behave on iso-token and clean-seen views. The forms work; the target was the problem.")}
          </div>
        </section>
        """,
        f"""
        <section id="token-budget" class="step">
          <div class="step-head">
            <div class="step-no">03</div>
            <div>
              <div class="step-eyebrow">Confound one</div>
              <h2>K=0.20 was never a fixed-token ladder</h2>
            </div>
          </div>
          <div class="step-body">
            <div class="prose">
              <p>K=0.20 spends 20% of the base model's pretraining token budget on midtraining. In p33m67 the total midtraining budget grows from about 0.245B tokens at 3e18 to about 32B tokens at 1e22 — and about 67% of that budget is math.</p>
              <p>The iso-token controls held the midtraining token budget fixed while sweeping base scale. On the old target, those fixed-token ladders had small 1e22 errors around -3% to -4%; only K=0.20 carried the large positive error.</p>
            </div>
            {figure_card(figures["old_isotoken"], "fig-old-isotoken", 6, "Hold midtraining tokens fixed and the ladders stay smooth; only K=0.20 (red, dashed) — whose token budget grows with scale — breaks away.")}
          </div>
        </section>
        """,
        f"""
        <section id="contamination" class="step">
          <div class="step-head">
            <div class="step-no">04</div>
            <div>
              <div class="step-eyebrow">Confound two</div>
              <h2>The old validation split had fuzzy and same-source leakage</h2>
            </div>
          </div>
          <div class="step-body">
            <div class="prose">
              <p>The exact-duplicate scan found zero duplicate document hashes across the 45.1M-doc corpus — and that result was not enough. Fuzzy MinHash/LSH plus exact 5-character-shingle Jaccard verification found substantial near-duplicate overlap between train and validation documents.</p>
              <p>At verified Jaccard ≥ 0.75, 9,757 / 57,243 validation docs were implicated, touching 6,839 / 12,500 validation windows and 9.53M / 51.20M validation tokens.</p>
            </div>
            {figure_card(figures["jaccard"], "fig-jaccard", 7, "Exact dedup found zero duplicates, but verified Jaccard overlap exposes thousands of near-duplicate validation docs and pairs.")}
            <div class="prose">
              <p>An actual-exposure replay made the mechanism scale-dependent. For p33m67 K=0.20, combined exposed validation tokens grew from 0.635M at 3e18 to 20.165M at 1e22; at 1e22 the exposure also tracked math fraction across mixes.</p>
            </div>
            {figure_card(figures["exposure"], "fig-exposure", 8, "Replaying the actual stream, exposed validation tokens climb with both compute and math fraction — the leak grows exactly where the miss grows.")}
            <div class="prose">
              <p>The curated perplexity-gap study showed the same mechanism at the document level. High-Jaccard documents improved far more at 1e22 than clean documents — consistent with memorization or near-twin exposure rather than a generic base-scaling effect.</p>
            </div>
            {figure_card(figures["ppl_gap"], "fig-ppl-gap", 9, "High-Jaccard documents drop in loss far faster than clean ones by 1e22 — the fingerprint of near-duplicate exposure, not generic scaling.")}
          </div>
        </section>
        """,
        f"""
        <section id="clean-seen" class="step">
          <div class="step-head">
            <div class="step-no">05</div>
            <div>
              <div class="step-eyebrow">The correction</div>
              <h2>The endpoint fits go smooth on the actual-seen clean target</h2>
            </div>
          </div>
          <div class="step-body">
            <div class="prose">
              <p>The final clean-seen set was built against documents actually seen by the 1e22 p33m67 K=0.20 math midtraining stream. It kept 3,367 docs, 2,265,243 tokens, and 553 eval sequences.</p>
              <p>The K=0.20 lr0.50 1e22 error moved from +18.56% on old 4plus to +2.83% on clean-seen. The dropped contaminated complement retained a large absolute miss: +0.0999 loss at 1e22, nearly the old target's +0.1042.</p>
            </div>
            {figure_card(figures["clean_unified"], "fig-clean-unified", 10, "On the actual-seen clean target the K=0.20 outlier collapses onto the fit, while the iso-token ladders stay smooth.")}
            {figure_card(figures["seen_partition"], "fig-seen-partition", 11, "Splitting the seen set: retained-clean docs fix the miss; the dropped contaminated complement keeps almost all of it.")}
            {figure_card(figures["final_error"], "fig-final-error", 12, "After removing the target and token-budget confounds, every 1e22 error sits in the low single digits.")}
            <h3>Final compact facts</h3>
            {story.table_html(fit_rows)}
            <h3>Seen-partition summary</h3>
            {story.table_html(seen_summary[["target_label", "actual_1e22", "pred_1e22", "error_1e22_pct", "abs_error_1e22", "heldout_mae_pct"]])}
            <h3>Clean-seen iso-token summary</h3>
            {story.table_html(iso_summary[iso_summary["target_key"].eq("clean_seen")][["series_label", "actual_1e22", "pred_1e22", "error_1e22_pct", "heldout_mae_pct"]])}
          </div>
        </section>
        """,
        f"""
        <section id="verdict" class="step">
          <div class="step-head">
            <div class="step-no">06</div>
            <div>
              <div class="step-eyebrow">Current interpretation</div>
              <h2>An eval-target confound stacked on a token-budget confound</h2>
            </div>
          </div>
          <div class="step-body">
            <div class="prose">
              <p>The evidence supports a validation/measurement confound rather than a broken law of midtraining. The old K=0.20 target mixed base scale, midtraining token budget, math exposure, and near-duplicate validation exposure. Fix the token budget or move to actual-seen clean validation, and the 1e22 endpoint errors return to low single digits.</p>
              <p>This is not a claim that every old-target artifact is fully explained. The clean-seen target is built against the 1e22 p33m67 K=0.20 seen set; per-mix actual-seen clean sets would be the stricter follow-up for p50m50 and p67m33.</p>
            </div>
            <h3>Artifacts &amp; provenance</h3>
            {story.make_artifact_table()}
          </div>
        </section>
        """,
    ]

    nav_items = [
        ("glossary", "", "Terms"),
        ("symptom", "01", "Symptom"),
        ("failed-fits", "02", "Failed fits"),
        ("token-budget", "03", "Token budget"),
        ("contamination", "04", "Contamination"),
        ("clean-seen", "05", "Clean-seen"),
        ("verdict", "06", "Verdict"),
    ]
    nav_links = []
    for sid, num, label in nav_items:
        num_html = f'<span class="n">{num}</span>' if num else ""
        nav_links.append(f'<a href="#{sid}">{num_html}{html.escape(label)}</a>')
    nav = "\n".join(nav_links)

    hero_svg = build_hero_svg()

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Delphi Midtraining · Case File: the 10^22 miss</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600&family=IBM+Plex+Mono:wght@400;500&display=swap" rel="stylesheet">
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>{CSS}</style>
</head>
<body>
  <header class="hero" id="top">
    <div class="hero-inner">
      <div class="kicker">Delphi midtraining · case file</div>
      <h1 class="hero-title">The law didn't break at 10<sup>22</sup>.<br>The <span class="accent">validation</span> did.</h1>
      <p class="hero-lead">The old 4plus math-validation target stacked a midtraining token budget that grew with scale on top of scale-dependent exposure to near-duplicate math documents. Fixed-token controls and actual-seen clean validation make the endpoint fits smooth again.</p>
      <div class="hero-grid">
        <div class="hero-plotwrap">
          <div class="hero-plot-cap">Exhibit A · the miss, reconstructed</div>
          {hero_svg}
          <div class="hero-legend">
            <span><i class="lg-swatch"></i> fit through 3e20</span>
            <span><i class="lg-swatch lg-dash"></i> extrapolation</span>
            <span><i class="lg-dot lg-pred"></i> predicted at 10<sup>22</sup></span>
            <span><i class="lg-dot lg-act"></i> actual at 10<sup>22</sup></span>
          </div>
        </div>
        <div>
          <div class="verdict">
            <div class="vbox">
              <div class="vbox-label">Old 4plus target</div>
              <div class="vbox-num bad">+18.6%</div>
              <div class="vbox-sub">K=0.20 lr0.50 at 1e22</div>
            </div>
            <div class="varrow">→</div>
            <div class="vbox">
              <div class="vbox-label">Clean-seen target</div>
              <div class="vbox-num good">+2.83%</div>
              <div class="vbox-sub">same fit, re-measured</div>
            </div>
          </div>
          <div class="verdict">
            <div class="vbox" style="min-width:0;flex:1">
              <div class="vbox-label">The leak, isolated</div>
              <div class="vbox-num" style="font-size:1.5rem;color:#ffb0b0">+0.0999 loss</div>
              <div class="vbox-sub">dropped-contaminated complement keeps almost the entire original miss</div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </header>

  <nav class="rail" aria-label="Case sections">
    <div class="rail-inner">
      <a class="rail-brand" href="#top"><span class="delta">&#916;</span> Delphi&nbsp;midtraining</a>
      <div class="rail-links">{nav}</div>
    </div>
  </nav>

  <main class="wrap">
    <details class="glossary" id="glossary">
      <summary>Glossary — terms used in this case file</summary>
      <div class="glossary-body">{story.glossary_html()}</div>
    </details>
    {''.join(steps)}
  </main>

  <footer class="foot">
    <div class="foot-inner">
      <p>Generated by <code>scripts/analysis/build_delphi_midtraining_dossier_report.py</code>, which reuses the figure and data logic of <code>build_delphi_midtraining_unified_story_report.py</code>. Plotly loads from the public CDN; all report data is embedded into this HTML at generation time.</p>
      <p>Sign convention is prediction minus actual: positive error means the model did better than the fit expected.</p>
    </div>
  </footer>
  <script>{SCROLLSPY_JS}</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    paths = story.Paths(output=args.output, contamination_root=args.contamination_root)
    if not paths.contamination_root.exists():
        raise FileNotFoundError(f"Contamination worktree not found: {paths.contamination_root}")
    paths.output.parent.mkdir(parents=True, exist_ok=True)
    paths.output.write_text(render_report(paths), encoding="utf-8")
    print(paths.output)


if __name__ == "__main__":
    main()
