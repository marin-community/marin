# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Standalone Plotly click inspector for two-phase mixture artifacts."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

MIXTURE_COLUMNS = [
    "domain",
    "proportional",
    "phase_0_weight",
    "phase_1_weight",
    "aggregate_weight",
    "available_tokens",
    "simulated_epochs",
]


def mixture_inspector_payload(
    mixture_paths: Mapping[str, Path],
    labels: Mapping[str, str],
) -> dict[str, dict[str, Any]]:
    """Load mixture weights into a JSON-compatible inspector payload."""
    if mixture_paths.keys() != labels.keys():
        raise ValueError("Mixture paths and labels must have identical keys")

    payload: dict[str, dict[str, Any]] = {}
    for key, mixture_path in mixture_paths.items():
        mixture = pd.read_csv(mixture_path)
        missing = set(MIXTURE_COLUMNS) - set(mixture.columns)
        if missing:
            raise ValueError(f"Mixture {mixture_path} lacks columns {sorted(missing)}")
        payload[key] = {
            "label": labels[key],
            "rows": json.loads(mixture[MIXTURE_COLUMNS].to_json(orient="records")),
        }
    return payload


def mixture_inspector_script(payload: Mapping[str, Any], *, parameter_count: int) -> str:
    """Return a Plotly post-script that links points to mixture bar charts.

    The parent figure's traces must put the payload key and tokens per parameter
    in customdata positions zero and one. A null TPP displays the fit-panel
    simulated epochs stored in the mixture CSV.
    """
    serialized_payload = json.dumps(payload, separators=(",", ":"))
    script = r"""
const plot = document.getElementById('{plot_id}');
const mixtures = __MIXTURE_PAYLOAD__;
const parameterCount = __PARAMETER_COUNT__;

const style = document.createElement('style');
style.textContent = `
  .mixture-inspector { max-width: 1580px; margin: 12px auto 42px; padding: 24px 28px 30px;
    border: 1px solid #c9c2b6; background: #faf7ef; color: #243b64; font-family: Arial, sans-serif; }
  .mixture-inspector h2 { margin: 0 0 6px; font-size: 25px; }
  .mixture-inspector .hint { margin: 0 0 18px; color: #5b6674; }
  .mixture-inspector .summary { display: flex; gap: 28px; flex-wrap: wrap; margin: 4px 0 12px;
    padding: 11px 14px; background: #fff; border: 1px solid #ddd6ca; }
  .mixture-inspector .summary span { min-width: 170px; }
  .mixture-inspector .selection-title { margin: 22px 0 2px; text-align: center; }
  .mixture-inspector .selection-title h3 { margin: 0; font-size: 27px; font-weight: 500; }
  .mixture-inspector .selection-title p { margin: 5px 0 0; font-size: 17px; color: #425777; }
  .mixture-inspector .chart { width: 100%; min-height: 680px; }
`;
document.head.appendChild(style);

const inspector = document.createElement('section');
inspector.className = 'mixture-inspector';
inspector.innerHTML = `
  <h2>Mixture inspector</h2>
  <p class="hint">Click an observed or predicted datapoint above to inspect its policy.</p>
  <div class="summary"><span>No mixture selected.</span></div>
  <div class="selection-title"></div>
  <div class="chart"></div>`;
plot.insertAdjacentElement('afterend', inspector);

const summary = inspector.querySelector('.summary');
const selectionTitle = inspector.querySelector('.selection-title');
const chart = inspector.querySelector('.chart');

function formatEpoch(value) {
  if (!Number.isFinite(value)) return 'n/a';
  if (value < 0.01) return value.toExponential(1);
  return value.toFixed(value < 10 ? 2 : 1);
}

function renderMixture(key, tpp) {
  const selected = mixtures[key];
  if (!selected) return;
  const trainTokens = Number.isFinite(tpp) ? tpp * parameterCount : null;
  const rows = [...selected.rows].sort((a, b) => b.aggregate_weight - a.aggregate_weight);
  const domains = rows.map(row => row.domain);
  const phaseTv = 0.5 * rows.reduce((total, row) => total + Math.abs(row.phase_0_weight - row.phase_1_weight), 0);
  const totalEpochs = rows.map(row => trainTokens === null
    ? row.simulated_epochs
    : trainTokens * (0.8 * row.phase_0_weight + 0.2 * row.phase_1_weight) / row.available_tokens);
  const maxEpochs = Math.max(...totalEpochs);
  const epochBasis = trainTokens === null
    ? '300M fit-panel simulated epochs'
    : `realized epochs at TPP ${tpp.toFixed(1)}`;
  summary.innerHTML = `
    <span><b>${selected.label}</b></span>
    <span><b>Phase TV</b> ${phaseTv.toFixed(3)}</span>
    <span><b>Max ${epochBasis}</b> ${formatEpoch(maxEpochs)}</span>
    <span><b>Ordering</b> aggregate weight, descending</span>`;
  selectionTitle.innerHTML = `
    <h3>${selected.label}</h3>
    <p>Bar-end labels show ${epochBasis}</p>`;

  const phase0Epochs = rows.map(row => trainTokens === null
    ? null
    : trainTokens * 0.8 * row.phase_0_weight / row.available_tokens);
  const phase1Epochs = rows.map(row => trainTokens === null
    ? null
    : trainTokens * 0.2 * row.phase_1_weight / row.available_tokens);
  const traces = [
    {name: 'Proportional', x: rows.map(row => row.proportional), color: '#b8b0a3', epochs: rows.map(() => null)},
    {name: 'Phase 0', x: rows.map(row => row.phase_0_weight), color: '#e76f2e', epochs: phase0Epochs},
    {name: 'Phase 1', x: rows.map(row => row.phase_1_weight), color: '#278a82', epochs: phase1Epochs},
    {name: 'Aggregate', x: rows.map(row => row.aggregate_weight), color: '#263b5b', epochs: totalEpochs},
  ].map(spec => ({
    type: 'bar', orientation: 'h', name: spec.name, x: spec.x, y: domains,
    marker: {color: spec.color},
    customdata: spec.epochs,
    text: spec.epochs.map(formatEpoch),
    textposition: spec.name === 'Proportional' ? 'none' : 'outside',
    cliponaxis: false,
    hovertemplate: `${spec.name}<br>%{y}<br>weight=%{x:.6f}<br>epochs=%{customdata}<extra></extra>`,
  }));
  Plotly.react(chart, traces, {
    barmode: 'group', height: Math.max(700, rows.length * 29),
    margin: {l: 310, r: 110, t: 72, b: 65},
    xaxis: {title: 'Mixture weight', gridcolor: '#dfe4ea', zeroline: false},
    yaxis: {autorange: 'reversed', automargin: true},
    legend: {orientation: 'h', x: 0.5, xanchor: 'center', y: 1.03, yanchor: 'bottom'},
    paper_bgcolor: '#faf7ef', plot_bgcolor: '#ffffff', font: {color: '#243b64'},
  }, {responsive: true, displaylogo: false, toImageButtonOptions: {format: 'png', scale: 4}});
}

plot.on('plotly_click', event => {
  const point = event.points[0];
  const custom = point.customdata;
  if (!Array.isArray(custom) || !custom[0]) return;
  const parsedTpp = Number(custom[1]);
  renderMixture(custom[0], Number.isFinite(parsedTpp) && parsedTpp > 0 ? parsedTpp : null);
  inspector.scrollIntoView({behavior: 'smooth', block: 'start'});
});
"""
    return script.replace("__MIXTURE_PAYLOAD__", serialized_payload).replace("__PARAMETER_COUNT__", str(parameter_count))
