# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build the decontaminated-val endpoint scaling law HTML.

Same component as ``build_delphi_endpoint_scaling_report.py`` but with one
curve family per validation set: the original (contaminated) math val anchor
plus the paranoid decon val sets at Jaccard cutoffs 0.90 / 0.75 / 0.50.
Losses come from the decon eval sweep downloaded to ``scratch/decon_evals/``
(see .agents/logbooks/nemotron_math_data.md); fits reuse the exact
``fit_floor_power`` / ``fit_log_linear`` code from the main report builder so
the Chinchilla-style fits are directly comparable.

Run:
    uv run python scripts/analysis/build_decon_scaling_report.py
"""

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
from build_delphi_midtraining_interactive_report import (
    CUTOFF_SCALES,
    HELD_OUT_SCALES,
    LR_ORDER,
    MIX_ORDER,
    OUT_DIR,
    SMALL_LADDER_SCALES,
    finite_or_none,
    fit_floor_power,
    fit_log_linear,
)

EVALS_DIR = Path("scratch/decon_evals")
DEFAULT_OUTPUT_PATH = OUT_DIR / "decon_endpoint_scaling_law.html"

SCALE_FLOPS = {
    "3e18": 3e18,
    "9e18": 9e18,
    "2e19": 2e19,
    "3e19": 3e19,
    "9e19": 9e19,
    "2e20": 2e20,
    "3e20": 3e20,
    "1e21": 1e21,
    "1e22": 1e22,
}
DATASET_METRICS = {
    "anchor": "nemotron_cc_math_v1/4plus/loss",
    "j090": "decon_j090/loss",
    "j075": "decon_j075/loss",
    "j050": "decon_j050/loss",
}
DATASET_ORDER = ("anchor", "j090", "j075", "j050")
DATASET_LABELS = {
    "anchor": "original val (anchor)",
    "j090": "decon J≥0.90 dropped",
    "j075": "decon J≥0.75 dropped",
    "j050": "decon J≥0.50 dropped (strictest)",
}
# Small-ladder runs tag the lr factor as an integer (``lr50``); the 1e21/1e22
# base runs use the decimal factor (``lr0.5``). Normalize both to the integer
# tag (0.5 -> "50", 0.33 -> "33") so a (mix, lr) cell unifies across scales.
RUN_NAME_RE = re.compile(r"delphi-(?P<scale>\de\d\d)-(?P<mix>p\d\dm\d\d)-.*lr(?P<lr>0?\.?\d+)")


def normalize_lr(raw: str) -> str:
    return f"{round(float(raw) * 100):02d}" if "." in raw else raw


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evals-dir", type=Path, default=EVALS_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def load_points(evals_dir: Path) -> list[dict[str, Any]]:
    """One row per (mix, lr, scale, dataset) from the downloaded eval results."""
    points = []
    for path in sorted(evals_dir.glob("delphi-*/step-*/metrics.jsonl/eval_results.json")):
        run = path.parts[-4]
        match = RUN_NAME_RE.match(run)
        if not match:
            raise ValueError(f"Unparseable run name: {run}")
        step = int(re.search(r"step-(\d+)", path.parts[-3]).group(1))
        data = json.loads(path.read_text())
        metrics = data if isinstance(data, dict) else data[-1]
        flat = metrics.get("metrics", metrics)
        for dataset, metric_key in DATASET_METRICS.items():
            value = flat.get(metric_key, flat.get(f"eval/{metric_key}"))
            if value is None:
                raise ValueError(f"{run}: missing metric {metric_key}")
            points.append(
                {
                    "run_name": run,
                    "step": step,
                    "scale": match["scale"],
                    "scale_flops": SCALE_FLOPS[match["scale"]],
                    "mix": match["mix"],
                    "lr": normalize_lr(match["lr"]),
                    "dataset": dataset,
                    "value": float(value),
                }
            )
    if not points:
        raise FileNotFoundError(f"No eval results under {evals_dir}")
    return points


def scaling_fits(points: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Per-(mix, lr, dataset, cutoff) fits — same pooling rules as endpoint_scaling_data()."""
    cells: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for point in points:
        # 1e22 never enters the training pool; 1e21 may when the slider reaches it.
        if point["scale"] not in CUTOFF_SCALES:
            continue
        cells.setdefault((point["mix"], point["lr"], point["dataset"]), []).append(point)

    fits: list[dict[str, Any]] = []
    for (mix, lr, dataset), cell_points in cells.items():
        cell_points.sort(key=lambda p: p["scale_flops"])
        for cutoff_index, cutoff_scale in enumerate(CUTOFF_SCALES):
            cutoff_flops = SCALE_FLOPS[cutoff_scale]
            included = [p for p in cell_points if p["scale_flops"] <= cutoff_flops + 1.0]
            if not included:
                continue
            xs = np.array([p["scale_flops"] for p in included], dtype=float)
            ys = np.array([p["value"] for p in included], dtype=float)
            row: dict[str, Any] = {
                "mix": mix,
                "lr": lr,
                "dataset": dataset,
                "cutoff_index": cutoff_index,
                "cutoff_scale": cutoff_scale,
                "n": int(xs.size),
                "min_scale": included[0]["scale"],
                "max_scale": included[-1]["scale"],
            }
            fp = fit_floor_power(xs, ys)
            if fp:
                row.update(
                    {
                        "fp_floor": fp["floor"],
                        "fp_amplitude": fp["amplitude"],
                        "fp_alpha": fp["alpha"],
                        "fp_r2": fp["r2"],
                        "fp_rmse": fp["rmse"],
                    }
                )
            ll = fit_log_linear(xs, ys)
            if ll:
                row.update(
                    {
                        "ll_slope": ll["slope"],
                        "ll_intercept": ll["intercept"],
                        "ll_r2": ll["r2"],
                        "ll_rmse_log": ll["rmse_log"],
                    }
                )
            fits.append(row)
    return fits


def payload(evals_dir: Path) -> dict[str, Any]:
    points = load_points(evals_dir)
    fits = scaling_fits(points)
    return {
        "points": [{key: finite_or_none(value) for key, value in row.items()} for row in points],
        "scalingFits": [{key: finite_or_none(value) for key, value in row.items()} for row in fits],
        "smallLadderScales": list(SMALL_LADDER_SCALES),
        "heldOutScales": list(HELD_OUT_SCALES),
        "cutoffScales": list(CUTOFF_SCALES),
        "mixOrder": list(MIX_ORDER),
        "lrOrder": list(LR_ORDER),
        "datasetOrder": list(DATASET_ORDER),
        "datasetLabels": DATASET_LABELS,
        "scaleFlops": {scale: finite_or_none(math.log10(flops)) for scale, flops in SCALE_FLOPS.items()},
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Delphi Decon Endpoint Scaling Law</title>
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
    h4.subhead { margin: 16px 0 4px; font-size: 14px; color: var(--muted); font-variant-numeric: tabular-nums; }
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
    .checks { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; }
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
    /* mix x lr tab grid: rows are mixes, the row label + four lr cells per row. */
    .cell-tabs {
      display: grid;
      grid-template-columns: max-content repeat(4, minmax(72px, 1fr));
      gap: 6px;
      max-width: 620px;
      margin: 0 0 18px;
    }
    .cell-tabs .row-label {
      align-self: center;
      font-size: 13px;
      font-weight: 650;
      color: var(--ink);
      padding-right: 8px;
      font-variant-numeric: tabular-nums;
    }
    .cell-tabs .col-head {
      font-size: 12px;
      color: var(--muted);
      text-align: center;
      padding-bottom: 2px;
    }
    .cell-tab {
      border: 1px solid var(--line);
      background: #fff;
      border-radius: 7px;
      padding: 8px 6px;
      font-size: 13px;
      cursor: pointer;
      color: var(--ink);
      font-variant-numeric: tabular-nums;
      transition: background 0.1s, border-color 0.1s;
    }
    .cell-tab:hover { background: var(--soft); }
    .cell-tab.active {
      background: var(--accent);
      border-color: var(--accent);
      color: #fff;
      font-weight: 650;
    }
    .cell-tab.partial::after { content: " *"; color: #f59e0b; }
    .cell-tab.active.partial::after { color: #fde68a; }
  </style>
</head>
<body>
  <header>
    <h1>Decontaminated Endpoint Scaling Law (Compute Vs Final Loss)</h1>
    <p>
      Same Chinchilla-style 3-parameter fit as the main report:
      \(L_\infty(C) = E + A\,(C/10^{18})^{-\alpha}\), fit per
      \((\textrm{mix}, \textrm{LR}, \textrm{val set})\) on the small ladder
      (3e18 → 3e20) by \(\texttt{scipy.optimize.curve\_fit}\) with \(E < \min y\), \(A,\alpha \ge 0\)
      — identical code path (\(\texttt{fit\_floor\_power}\)). The four val sets: the original
      (contaminated) 12,500-window math val as anchor, and the paranoid decon sets dropping val docs
      with any verified train near-duplicate at Jaccard ≥ 0.90 / 0.75 / 0.50. Losses come from the
      decon eval sweep (one v6e-4 job per checkpoint, all four sets evaluated in-harness together).
      1e21 and 1e22 actuals are triangles — never used by the fit below the cutoff. The log-log
      2-parameter fit is available as a toggle.
    </p>
  </header>
  <main>
    <section>
      <div class="note" style="margin-bottom:6px">mix &times; learning rate</div>
      <div id="scalingCellTabs" class="cell-tabs"></div>
      <div class="target-controls">
        <label>fit type <select id="scalingFitType">
          <option value="floor_power" selected>floor + power (Chinchilla)</option>
          <option value="log_linear">log-log linear (2-param)</option>
        </select></label>
        <div>
          <div class="note">val sets</div>
          <div id="scalingDatasetChecks" class="checks"></div>
        </div>
        <label>fit through
          <input id="scalingCutoff" type="range" min="1" max="6" step="1" value="6" />
          <span class="value" id="scalingCutoffValue">3e20</span>
        </label>
      </div>
      <p class="note">
        Pick a (mix, learning rate) cell; the plot shows one line per val set — original anchor vs
        the three decon cutoffs. The contamination signature: the anchor's fit bends down harder at
        large compute (memorization credit), while the J≥0.50 set is the honest curve. Open circles
        mark training cells dropped by the current cutoff.
      </p>
      <div id="scalingPlot" class="chart"></div>
      <div class="tables-stacked">
        <div>
          <h3>Held-Out Predictions</h3>
          <h4 class="subhead">1e21</h4>
          <div id="scalingPredTable1e21"></div>
          <h4 class="subhead">1e22</h4>
          <div id="scalingPredTable1e22"></div>
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
    const cutoffScales = DATA.cutoffScales;
    const mixOrder = DATA.mixOrder;
    const lrOrder = DATA.lrOrder;
    const datasetOrder = DATA.datasetOrder;
    const datasetLabels = DATA.datasetLabels;
    const points = DATA.points || [];
    const scalingFits = DATA.scalingFits || [];
    const scaleFlops = {"3e18": 3e18, "9e18": 9e18, "2e19": 2e19, "3e19": 3e19, "9e19": 9e19, "2e20": 2e20, "3e20": 3e20, "1e21": 1e21, "1e22": 1e22};

    // One color per val set — a single (mix, lr) cell is shown at a time.
    const datasetColors = {
      anchor: "#1f2937",
      j090: "#F58518",
      j075: "#54A24B",
      j050: "#4C78A8",
    };

    function tableHtml(rows, columns) {
      if (!rows.length) {
        return '<p class="note"><em>No rows for this selection.</em></p>';
      }
      return `<table><thead><tr>${columns.map((column) => `<th>${column.label}</th>`).join("")}</tr></thead><tbody>${
        rows.map((row) => `<tr>${columns.map((column) => `<td>${column.format ? column.format(row[column.key], row) : row[column.key]}</td>`).join("")}</tr>`).join("")
      }</tbody></table>`;
    }

    function lookupScalingFit(mix, lr, dataset, cutoffIndex) {
      return scalingFits.find((row) => row.mix === mix
        && String(row.lr) === String(lr)
        && row.dataset === dataset
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

    function checkedValues(containerId) {
      return [...document.querySelectorAll(`#${containerId} input:checked`)].map((input) => input.value);
    }

    // Which (mix, lr) cells exist, and how many scales each has.
    function cellScaleCounts() {
      const counts = new Map();
      for (const p of points) {
        const key = `${p.mix}|${p.lr}`;
        if (!counts.has(key)) {
          counts.set(key, new Set());
        }
        counts.get(key).add(p.scale);
      }
      return counts;
    }

    let activeCell = null;

    function renderControls() {
      const cellScales = cellScaleCounts();
      const totalScales = smallLadderScales.length + heldOutScales.length;
      const mixes = [...new Set(points.map((p) => p.mix))].sort((a, b) => mixOrder.indexOf(a) - mixOrder.indexOf(b));
      const lrs = [...new Set(points.map((p) => String(p.lr)))].sort((a, b) => lrOrder.indexOf(a) - lrOrder.indexOf(b));

      // Build a (mix rows) x (lr columns) grid of tab buttons.
      const parts = ['<div class="col-head"></div>', ...lrs.map((lr) => `<div class="col-head">lr${lr}</div>`)];
      for (const mix of mixes) {
        parts.push(`<div class="row-label">${mix}</div>`);
        for (const lr of lrs) {
          const key = `${mix}|${lr}`;
          if (!cellScales.has(key)) {
            parts.push('<div></div>');
            continue;
          }
          const n = cellScales.get(key).size;
          const partial = n < totalScales ? " partial" : "";
          const title = n < totalScales ? ` title="partial: ${n}/${totalScales} scales"` : "";
          parts.push(`<button class="cell-tab${partial}" data-cell="${key}"${title}>lr${lr}</button>`);
        }
      }
      const tabs = document.getElementById("scalingCellTabs");
      tabs.innerHTML = parts.join("");

      // Default to the first fully-complete cell.
      const allKeys = [...cellScales.keys()];
      activeCell = allKeys.find((key) => cellScales.get(key).size === totalScales) || allKeys[0];

      tabs.querySelectorAll(".cell-tab").forEach((button) => {
        button.addEventListener("click", () => {
          activeCell = button.dataset.cell;
          renderScalingSection();
        });
      });

      document.getElementById("scalingDatasetChecks").innerHTML = datasetOrder.map(
        (dataset) => `<label><input type="checkbox" name="scalingDataset" value="${dataset}" checked />${datasetLabels[dataset]}</label>`,
      ).join("");
      document.getElementById("scalingCutoff").max = String(cutoffScales.length - 1);
      document.getElementById("scalingCutoff").value = String(smallLadderScales.length - 1);
      document.querySelectorAll("select,input").forEach((element) => element.addEventListener("input", renderScalingSection));
    }

    function renderScalingSection() {
      const [mix, lr] = activeCell.split("|");
      document.querySelectorAll("#scalingCellTabs .cell-tab").forEach((button) => {
        button.classList.toggle("active", button.dataset.cell === activeCell);
      });
      const datasets = checkedValues("scalingDatasetChecks");
      const cutoffIndex = Number(document.getElementById("scalingCutoff").value);
      const cutoffScale = cutoffScales[cutoffIndex];
      const cutoffFlops = scaleFlops[cutoffScale];
      const fitType = document.getElementById("scalingFitType").value;
      document.getElementById("scalingCutoffValue").textContent = cutoffScale;

      const filtered = points.filter(
        (row) => row.mix === mix
          && String(row.lr) === lr
          && datasets.includes(row.dataset),
      );

      // Group by (mix, lr, dataset); split small-ladder points into train/dropped
      // by the cutoff and held-out scales into trained-on vs held-out, exactly as
      // in the anchor-only report.
      const recipes = new Map();
      for (const row of filtered) {
        const key = `${row.mix}|${row.lr}|${row.dataset}`;
        if (!recipes.has(key)) {
          recipes.set(key, {mix: row.mix, lr: String(row.lr), dataset: row.dataset, trainRows: [], dropRows: [], heldRows: []});
        }
        const bucket = recipes.get(key);
        const heldOut = heldOutScales.includes(row.scale);
        if (Number(row.scale_flops) <= cutoffFlops + 1e6) {
          bucket.trainRows.push(row);
        } else if (heldOut) {
          bucket.heldRows.push(row);
        } else {
          bucket.dropRows.push(row);
        }
      }

      const recipeKeys = [...recipes.keys()].sort((left, right) => {
        const [lmix, llr, lds] = left.split("|");
        const [rmix, rlr, rds] = right.split("|");
        return mixOrder.indexOf(lmix) - mixOrder.indexOf(rmix)
          || lrOrder.indexOf(llr) - lrOrder.indexOf(rlr)
          || datasetOrder.indexOf(lds) - datasetOrder.indexOf(rds);
      });

      const traces = [];
      const fitTableRows = [];
      const predTableRows = [];

      const minFlop = Math.min(...smallLadderScales.map((scale) => scaleFlops[scale]));
      const maxFlop = scaleFlops[heldOutScales[heldOutScales.length - 1]];

      for (const key of recipeKeys) {
        const recipe = recipes.get(key);
        const color = datasetColors[recipe.dataset] || "#666";
        const labelBase = `${recipe.mix} lr${recipe.lr} ${datasetLabels[recipe.dataset]}`;

        if (recipe.trainRows.length) {
          traces.push({
            x: recipe.trainRows.map((row) => row.scale_flops),
            y: recipe.trainRows.map((row) => row.value),
            mode: "markers",
            type: "scatter",
            name: labelBase,
            legendgroup: key,
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
            legendgroup: key,
            showlegend: false,
            marker: {symbol: "circle-open", size: 10, color, line: {width: 2, color}},
            hovertext: recipe.dropRows.map((row) => `${row.scale} (excluded by cutoff)<br>${row.run_name}`),
            hovertemplate: "%{hovertext}<br>flops=%{x:.3e}<br>loss=%{y:.5f}<extra></extra>",
          });
        }

        const fit = lookupScalingFit(recipe.mix, recipe.lr, recipe.dataset, cutoffIndex);
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
            legendgroup: key,
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
            legendgroup: key,
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
              legendgroup: key,
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
          title: "final math val loss (log scale)",
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
      // Scale is the subsection now, so it drops out of the columns.
      const predColumns = [
        {key: "recipe", label: "recipe"},
        {key: "actual", label: "actual", format: (value) => value.toFixed(5)},
        {key: "predicted", label: "predicted", format: (value) => value.toFixed(5)},
        {key: "error", label: "error", format: (value) => value.toFixed(5)},
        {key: "absError", label: "abs error", format: (value) => value.toFixed(5)},
        {key: "pctError", label: "% error", format: (value) => value == null ? "—" : `${value >= 0 ? "+" : ""}${value.toFixed(2)}%`},
        {key: "absPctError", label: "abs % error", format: (value) => value == null ? "—" : `${value.toFixed(2)}%`},
      ];
      document.getElementById("scalingFitTable").innerHTML = tableHtml(fitTableRows, fitColumns);
      for (const scale of heldOutScales) {
        const scaleRows = predTableRows
          .filter((row) => row.scale === scale)
          .sort((left, right) => left.recipe.localeCompare(right.recipe));
        document.getElementById(`scalingPredTable${scale}`).innerHTML = tableHtml(scaleRows, predColumns);
      }
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
    data = payload(args.evals_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(data), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
