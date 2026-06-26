# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build a standalone HTML page for the Endpoint Scaling Law (Compute Vs Final Loss) section.

This is the single-section extract of the scaling-law component from
``build_delphi_midtraining_interactive_report.py``. It embeds only the endpoint
data and precomputed per-(mix, LR) fits, so the page is a few hundred KB instead
of the full report's ~17MB.

Run:
    uv run python scripts/analysis/build_delphi_endpoint_scaling_report.py
"""

import argparse
import json
from pathlib import Path
from typing import Any

from build_delphi_midtraining_interactive_report import (
    CUTOFF_SCALES,
    HELD_OUT_SCALES,
    LR_ORDER,
    MIX_ORDER,
    OUT_DIR,
    SMALL_LADDER_SCALES,
    endpoint_scaling_data,
    finite_or_none,
    records_for_json,
)

DEFAULT_OUTPUT_PATH = OUT_DIR / "endpoint_scaling_law.html"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def payload() -> dict[str, Any]:
    endpoints, targets, scaling_fits = endpoint_scaling_data()
    return {
        "endpoints": records_for_json(endpoints),
        "heldoutTargets": records_for_json(targets),
        "scalingFits": [{key: finite_or_none(value) for key, value in row.items()} for row in scaling_fits],
        "smallLadderScales": list(SMALL_LADDER_SCALES),
        "heldOutScales": list(HELD_OUT_SCALES),
        "cutoffScales": list(CUTOFF_SCALES),
        "mixOrder": list(MIX_ORDER),
        "lrOrder": list(LR_ORDER),
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Delphi Endpoint Scaling Law</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <script>
    window.MathJax = { tex: { inlineMath: [["\\(", "\\)"], ["$", "$"]] } };
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js"></script>
  <style>
    :root {
      color-scheme: light;
      --ink: #172033;
      --muted: #667085;
      --line: #e5e7eb;
      --bg: #ffffff;
      --soft: #f8fafc;
      --accent: #2563eb;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background: var(--bg);
    }
    header {
      padding: 24px 32px 14px;
      border-bottom: 1px solid var(--line);
    }
    h1 { margin: 0 0 8px; font-size: 30px; }
    h3 { margin: 0 0 12px; font-size: 18px; }
    p { color: var(--muted); line-height: 1.45; }
    main { padding: 18px 32px 42px; max-width: 1540px; }
    .target-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 16px 22px;
      align-items: center;
      padding: 14px 16px;
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
      margin: 14px 0 20px;
    }
    label {
      font-size: 13px;
      color: var(--muted);
      display: flex;
      flex-direction: column;
      gap: 5px;
    }
    select, input[type=range] { min-width: 150px; }
    .checks { display: flex; gap: 10px; align-items: center; }
    .checks label {
      color: var(--ink);
      flex-direction: row;
      gap: 4px;
      align-items: center;
    }
    .value {
      color: var(--ink);
      font-weight: 650;
      font-variant-numeric: tabular-nums;
    }
    .chart { width: 100%; height: 650px; }
    table {
      border-collapse: collapse;
      margin: 10px 0 26px;
      min-width: 520px;
    }
    th, td {
      border: 1px solid var(--line);
      padding: 8px 10px;
      text-align: left;
      font-variant-numeric: tabular-nums;
    }
    th { background: #f3f4f6; }
    .tables-stacked {
      display: grid;
      grid-template-columns: 1fr;
      gap: 22px;
      align-items: start;
    }
    .tables-stacked > div { overflow-x: auto; }
    .note { font-size: 13px; color: var(--muted); }
  </style>
</head>
<body>
  <header>
    <h1>Endpoint Scaling Law (Compute Vs Final Loss)</h1>
    <p>
      Chinchilla-style 3-parameter fit: \(L_\infty(C) = E + A\,(C/10^{18})^{-\alpha}\), where \(E\) is the
      irreducible-loss floor, \(C\) is base-model FLOPs, and \(L_\infty\) is final
      \(\texttt{math\_val\_loss}\). Fit per \((\textrm{mix}, \textrm{LR})\) on the small ladder
      (3e18 → 3e20) by \(\texttt{scipy.optimize.curve\_fit}\) with \(E < \min y\), \(A,\alpha \ge 0\).
      The 1e21 and 1e22 cells are never used by the fit; their actuals are plotted as triangles for the
      extrapolation check. The two-parameter log-log fit \(\log L = a + b\,\log C\) is available as a
      toggle for comparison — it lacks the asymptote so it under-predicts loss at very large compute.
    </p>
  </header>
  <main>
    <section>
      <div class="target-controls">
        <label>fit type <select id="scalingFitType">
          <option value="floor_power" selected>floor + power (Chinchilla)</option>
          <option value="log_linear">log-log linear (2-param)</option>
        </select></label>
        <label>mix <select id="scalingMix"></select></label>
        <div>
          <div class="note">learning rates</div>
          <div id="scalingLrChecks" class="checks"></div>
        </div>
        <label>fit through
          <input id="scalingCutoff" type="range" min="1" max="6" step="1" value="6" />
          <span class="value" id="scalingCutoffValue">3e20</span>
        </label>
      </div>
      <p class="note">
        Slider sets the upper compute bound used to fit. Drop it to 2e20 to see how the held-out predictions
        degrade when the largest small-ladder cell is unavailable; drop further to see the fit collapse as
        the lever shrinks. Open circles mark training cells dropped by the current cutoff.
      </p>
      <div id="scalingPlot" class="chart"></div>
      <div class="tables-stacked">
        <div>
          <h3>Held-Out Predictions (1e21, 1e22)</h3>
          <div id="scalingPredTable"></div>
        </div>
        <div>
          <h3>Per-Recipe Fit Quality</h3>
          <div id="scalingFitTable"></div>
        </div>
      </div>
    </section>
  </main>
  <script id="payload" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("payload").textContent);
    const smallLadderScales = DATA.smallLadderScales;
    const heldOutScales = DATA.heldOutScales;
    const cutoffScales = DATA.cutoffScales || DATA.smallLadderScales;
    const mixOrder = DATA.mixOrder;
    const lrOrder = DATA.lrOrder;
    const endpoints = DATA.endpoints || [];
    const heldoutTargets = DATA.heldoutTargets || [];
    const scalingFits = DATA.scalingFits || [];
    const colors = {"33": "#4C78A8", "50": "#F58518", "67": "#54A24B", "83": "#E45756"};
    const scaleFlops = {"3e18": 3e18, "9e18": 9e18, "2e19": 2e19, "3e19": 3e19, "9e19": 9e19, "2e20": 2e20, "3e20": 3e20, "1e21": 1e21, "1e22": 1e22};

    function tableHtml(rows, columns) {
      if (!rows.length) {
        return '<p class="note"><em>No rows for this selection.</em></p>';
      }
      return `<table><thead><tr>${columns.map((column) => `<th>${column.label}</th>`).join("")}</tr></thead><tbody>${
        rows.map((row) => `<tr>${columns.map((column) => `<td>${column.format ? column.format(row[column.key], row) : row[column.key]}</td>`).join("")}</tr>`).join("")
      }</tbody></table>`;
    }

    function lookupScalingFit(mix, lr, cutoffIndex) {
      return scalingFits.find((row) => row.mix === mix
        && String(row.lr) === String(lr)
        && Number(row.cutoff_index) === Number(cutoffIndex));
    }

    function activeFitOk(fit, fitType) {
      if (!fit) {
        return false;
      }
      if (fitType === "floor_power") {
        return fit.fp_floor != null && fit.fp_amplitude != null && fit.fp_alpha != null;
      }
      return fit.ll_slope != null && fit.ll_intercept != null;
    }

    function predictScaling(fit, fitType, flops) {
      if (!activeFitOk(fit, fitType)) {
        return null;
      }
      if (fitType === "floor_power") {
        const xn = flops / 1e18;
        return Number(fit.fp_floor) + Number(fit.fp_amplitude) * Math.pow(xn, -Number(fit.fp_alpha));
      }
      return Math.exp(Number(fit.ll_intercept) + Number(fit.ll_slope) * Math.log(flops));
    }

    function selectedScalingLrs() {
      return [...document.querySelectorAll("#scalingLrChecks input:checked")].map((input) => input.value);
    }

    function scalingPalette(mix, lr) {
      const mixIndex = mixOrder.indexOf(mix);
      const lrIndex = lrOrder.indexOf(String(lr));
      const palette = [
        ["#4C78A8", "#5B9BD5", "#73B3DB", "#9CCFE2"],
        ["#F58518", "#FF9F44", "#FFB976", "#FFD2A6"],
        ["#54A24B", "#76B96E", "#9ED099", "#C2E0BD"],
      ];
      const row = palette[Math.max(0, mixIndex)] || palette[0];
      return row[Math.max(0, lrIndex) % row.length] || colors[String(lr)] || "#666";
    }

    function renderControls() {
      document.getElementById("scalingMix").innerHTML = [
        '<option value="all">all mixes</option>',
        ...mixOrder.map((mix) => `<option value="${mix}">${mix}</option>`),
      ].join("");
      document.getElementById("scalingMix").value = "p33m67";
      document.getElementById("scalingLrChecks").innerHTML = lrOrder.map(
        (lr) => `<label><input type="checkbox" name="scalingLr" value="${lr}" checked />lr${lr}</label>`,
      ).join("");
      document.getElementById("scalingCutoff").max = String(cutoffScales.length - 1);
      document.getElementById("scalingCutoff").value = String(smallLadderScales.length - 1);
      document.querySelectorAll("select,input").forEach((element) => element.addEventListener("input", renderScalingSection));
    }

    function renderScalingSection() {
      const mix = document.getElementById("scalingMix").value;
      const lrs = selectedScalingLrs();
      const cutoffIndex = Number(document.getElementById("scalingCutoff").value);
      const cutoffScale = cutoffScales[cutoffIndex];
      const cutoffFlops = scaleFlops[cutoffScale];
      const fitType = document.getElementById("scalingFitType").value;
      document.getElementById("scalingCutoffValue").textContent = cutoffScale;

      const filteredEndpoints = endpoints.filter(
        (row) => (mix === "all" || row.mix === mix) && lrs.includes(String(row.lr)),
      );
      const filteredTargets = heldoutTargets.filter(
        (row) => (mix === "all" || row.mix === mix) && lrs.includes(String(row.lr)),
      );

      const recipes = new Map();
      // Small-ladder endpoints: <= cutoff -> training (filled), > cutoff -> dropped (open).
      for (const row of filteredEndpoints) {
        const key = `${row.mix}|${row.lr}`;
        if (!recipes.has(key)) {
          recipes.set(key, {mix: row.mix, lr: String(row.lr), trainRows: [], dropRows: [], heldRows: []});
        }
        const bucket = recipes.get(key);
        if (Number(row.scale_flops) <= cutoffFlops + 1e6) {
          bucket.trainRows.push(row);
        } else {
          bucket.dropRows.push(row);
        }
      }
      // Held-out targets: <= cutoff -> they entered the training pool (filled); > cutoff -> heldout (triangles).
      for (const row of filteredTargets) {
        const key = `${row.mix}|${row.lr}`;
        if (!recipes.has(key)) {
          recipes.set(key, {mix: row.mix, lr: String(row.lr), trainRows: [], dropRows: [], heldRows: []});
        }
        const bucket = recipes.get(key);
        if (Number(row.scale_flops) <= cutoffFlops + 1e6) {
          bucket.trainRows.push(row);
        } else {
          bucket.heldRows.push(row);
        }
      }

      const recipeKeys = [...recipes.keys()].sort((left, right) => {
        const [lmix, llr] = left.split("|");
        const [rmix, rlr] = right.split("|");
        return mixOrder.indexOf(lmix) - mixOrder.indexOf(rmix)
          || lrOrder.indexOf(llr) - lrOrder.indexOf(rlr);
      });

      const traces = [];
      const fitTableRows = [];
      const predTableRows = [];

      const minFlop = Math.min(...smallLadderScales.map((scale) => scaleFlops[scale]));
      const maxFlop = scaleFlops[heldOutScales[heldOutScales.length - 1]];

      for (const key of recipeKeys) {
        const recipe = recipes.get(key);
        const color = scalingPalette(recipe.mix, recipe.lr);
        const labelBase = `${recipe.mix} lr${recipe.lr}`;

        if (recipe.trainRows.length) {
          traces.push({
            x: recipe.trainRows.map((row) => row.scale_flops),
            y: recipe.trainRows.map((row) => row.value),
            mode: "markers",
            type: "scatter",
            name: `${labelBase} train`,
            legendgroup: labelBase,
            marker: {symbol: "circle", size: 9, color},
            hovertext: recipe.trainRows.map((row) => `${row.scale} ${row.run_name}`),
            hovertemplate: "%{hovertext}<br>flops=%{x:.3e}<br>loss=%{y:.5f}<extra></extra>",
          });
        }
        if (recipe.dropRows.length) {
          traces.push({
            x: recipe.dropRows.map((row) => row.scale_flops),
            y: recipe.dropRows.map((row) => row.value),
            mode: "markers",
            type: "scatter",
            name: `${labelBase} dropped`,
            legendgroup: labelBase,
            showlegend: false,
            marker: {symbol: "circle-open", size: 10, color, line: {width: 2, color}},
            hovertext: recipe.dropRows.map((row) => `${row.scale} (excluded by cutoff)<br>${row.run_name}`),
            hovertemplate: "%{hovertext}<br>flops=%{x:.3e}<br>loss=%{y:.5f}<extra></extra>",
          });
        }

        const fit = lookupScalingFit(recipe.mix, recipe.lr, cutoffIndex);
        if (activeFitOk(fit, fitType)) {
          const lineX = [];
          const lineY = [];
          const logLo = Math.log10(minFlop);
          const logHi = Math.log10(maxFlop);
          for (let index = 0; index <= 60; index += 1) {
            const lf = logLo + ((logHi - logLo) * index) / 60;
            const flops = Math.pow(10, lf);
            lineX.push(flops);
            lineY.push(predictScaling(fit, fitType, flops));
          }
          traces.push({
            x: lineX,
            y: lineY,
            mode: "lines",
            type: "scatter",
            name: `${labelBase} fit`,
            legendgroup: labelBase,
            showlegend: false,
            line: {color, dash: "dot", width: 2},
            hoverinfo: "skip",
          });
          if (fitType === "floor_power") {
            fitTableRows.push({
              recipe: labelBase,
              n: fit.n,
              p1: Number(fit.fp_floor),
              p2: Number(fit.fp_amplitude),
              p3: Number(fit.fp_alpha),
              r2: fit.fp_r2 == null ? null : Number(fit.fp_r2),
              rmse: fit.fp_rmse == null ? null : Number(fit.fp_rmse),
            });
          } else {
            fitTableRows.push({
              recipe: labelBase,
              n: fit.n,
              p1: Number(fit.ll_slope),
              p2: Number(fit.ll_intercept),
              p3: null,
              r2: fit.ll_r2 == null ? null : Number(fit.ll_r2),
              rmse: fit.ll_rmse_log == null ? null : Number(fit.ll_rmse_log),
            });
          }
        }

        const heldRows = recipe.heldRows;
        if (heldRows.length) {
          traces.push({
            x: heldRows.map((row) => row.scale_flops),
            y: heldRows.map((row) => row.value),
            mode: "markers",
            type: "scatter",
            name: `${labelBase} held-out`,
            legendgroup: labelBase,
            showlegend: false,
            marker: {symbol: "triangle-up", size: 13, color, line: {color: "#111", width: 1}},
            hovertext: heldRows.map((row) => `${row.scale} held-out actual<br>${row.run_name}`),
            hovertemplate: "%{hovertext}<br>flops=%{x:.3e}<br>actual=%{y:.5f}<extra></extra>",
          });
        }
        if (activeFitOk(fit, fitType)) {
          for (const row of heldRows) {
            const predicted = predictScaling(fit, fitType, Number(row.scale_flops));
            traces.push({
              x: [row.scale_flops],
              y: [predicted],
              mode: "markers",
              type: "scatter",
              name: `${labelBase} pred`,
              legendgroup: labelBase,
              showlegend: false,
              marker: {symbol: "x", size: 12, color, line: {width: 2}},
              hovertext: [`${row.scale} predicted (cutoff ${cutoffScale})<br>${labelBase}`],
              hovertemplate: "%{hovertext}<br>flops=%{x:.3e}<br>predicted=%{y:.5f}<extra></extra>",
            });
            const actual = Number(row.value);
            const error = predicted - actual;
            const pctError = actual !== 0 ? (error / actual) * 100 : null;
            predTableRows.push({
              recipe: labelBase,
              scale: row.scale,
              actual,
              predicted,
              error,
              absError: Math.abs(error),
              pctError,
              absPctError: pctError == null ? null : Math.abs(pctError),
            });
          }
        }
      }

      const layout = {
        height: 620,
        margin: {l: 80, r: 20, t: 30, b: 60},
        xaxis: {
          title: "training compute (FLOPs, log scale)",
          type: "log",
          showgrid: true,
        },
        yaxis: {
          title: "final math_val_loss (log scale)",
          type: "log",
          showgrid: true,
        },
        legend: {orientation: "v"},
        shapes: [{
          type: "line",
          xref: "x",
          yref: "paper",
          x0: cutoffFlops,
          x1: cutoffFlops,
          y0: 0,
          y1: 1,
          line: {color: "#94a3b8", dash: "dash", width: 1},
        }],
        annotations: [{
          x: Math.log10(cutoffFlops),
          xref: "x",
          y: 1.04,
          yref: "paper",
          showarrow: false,
          text: `fit cutoff (${cutoffScale})`,
          font: {size: 12, color: "#475569"},
        }],
      };
      Plotly.react("scalingPlot", traces, layout, {responsive: true});

      const fitColumns = fitType === "floor_power"
        ? [
          {key: "recipe", label: "recipe"},
          {key: "n", label: "n"},
          {key: "p1", label: "floor E", format: (value) => value.toFixed(4)},
          {key: "p2", label: "amplitude A", format: (value) => value.toFixed(4)},
          {key: "p3", label: "alpha", format: (value) => value.toFixed(4)},
          {key: "r2", label: "R^2", format: (value) => value == null ? "—" : value.toFixed(4)},
          {key: "rmse", label: "RMSE", format: (value) => value == null ? "—" : value.toFixed(4)},
        ]
        : [
          {key: "recipe", label: "recipe"},
          {key: "n", label: "n"},
          {key: "p1", label: "slope b", format: (value) => value.toFixed(4)},
          {key: "p2", label: "intercept", format: (value) => value.toFixed(4)},
          {key: "r2", label: "R^2", format: (value) => value == null ? "—" : value.toFixed(4)},
          {key: "rmse", label: "RMSE (log)", format: (value) => value == null ? "—" : value.toFixed(4)},
        ];
      const predColumns = [
        {key: "recipe", label: "recipe"},
        {key: "scale", label: "scale"},
        {key: "actual", label: "actual", format: (value) => value.toFixed(5)},
        {key: "predicted", label: "predicted", format: (value) => value.toFixed(5)},
        {key: "error", label: "error", format: (value) => value.toFixed(5)},
        {key: "absError", label: "abs error", format: (value) => value.toFixed(5)},
        {key: "pctError", label: "% error", format: (value) => value == null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`},
        {key: "absPctError", label: "abs % error", format: (value) => value == null ? "—" : `${value.toFixed(2)}%`},
      ];
      document.getElementById("scalingFitTable").innerHTML = tableHtml(fitTableRows, fitColumns);
      document.getElementById("scalingPredTable").innerHTML = tableHtml(
        predTableRows.sort((left, right) => left.recipe.localeCompare(right.recipe)
          || left.scale.localeCompare(right.scale)),
        predColumns,
      );
    }

    renderControls();
    renderScalingSection();
  </script>
</body>
</html>
"""


def build_html(data: dict[str, Any]) -> str:
    payload_json = json.dumps(data, allow_nan=False, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__PAYLOAD_JSON__", payload_json)


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(payload()), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
