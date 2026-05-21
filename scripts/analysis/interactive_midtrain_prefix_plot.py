# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Build an interactive prefix-sensitivity plot for historical v5-isoflop runs.

These runs were previously labeled "Delphi 1e20", but the base checkpoint was
the deprecated v5 isoflop ablation, not the canonical Delphi 3e20 v6 bucket
winner.

Run: ``uv run python scripts/analysis/interactive_midtrain_prefix_plot.py``.
The script writes and opens ``interactive_midtrain_prefix_plot.html``.
"""

import json
import logging
import os
import subprocess
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from midtrain_loss_predictor import (
    RUNS_1E20,
    TARGET_WINDOW,
    TOTAL_STEPS,
    WARMUP_STEPS,
    compute_cumulative_lr,
    evaluate_final,
    fit_cumlr_fixed_c,
    fit_cumlr_power,
    fit_cumlr_regularized_c,
    fit_last_value,
    fit_sqrt_t,
    load_run,
    normalized_remaining_lr,
    power_prediction,
    smooth_train_loss,
)
from plotly.offline import get_plotlyjs

logger = logging.getLogger(__name__)

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
OUT_PATH = os.path.join(OUT_DIR, "interactive_midtrain_prefix_plot.html")
MIN_PREFIX_PERCENT = int(np.ceil(100 * WARMUP_STEPS / TOTAL_STEPS))
PREFIX_GRID = tuple(round(float(v), 2) for v in np.arange(MIN_PREFIX_PERCENT / 100, 0.901, 0.01))
PLOT_STEPS = np.linspace(WARMUP_STEPS, TOTAL_STEPS, 420)
RUN_COLORS = {
    "lr=0.5": "#2563eb",
    "lr=0.67": "#16a34a",
    "lr=0.83": "#dc2626",
}


@dataclass(frozen=True)
class MetricSpec:
    key: str
    label: str
    column: str
    detail: str
    observed_mode: str


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    formula: str
    detail: str
    color: str
    dash: str
    fit: Callable


METRICS = (
    MetricSpec(
        "train_loss",
        "Train loss: math data",
        "train/loss_smooth",
        "Dense every-step training loss, EMA-smoothed with halflife=100.",
        "lines",
    ),
    MetricSpec(
        "eval_c4_en",
        "Eval loss: Paloma c4_en",
        "eval/paloma/c4_en/loss",
        "Sparse retention eval. Higher means worse c4_en retention; late cooldown can make it move down again.",
        "lines+markers",
    ),
)


def _fit_last_value(df, prefix_frac: float, metric: str):
    pred = fit_last_value(df, prefix_frac, metric)
    return pred, ()


METHODS = (
    MethodSpec(
        "B0_last_value",
        "B0 carry-forward",
        "final = last prefix value",
        "No curve fit; uses the last observed prefix loss as the endpoint prediction.",
        "#7f7f7f",
        "dot",
        _fit_last_value,
    ),
    MethodSpec(
        "B1_sqrt_t",
        "B1 step power law",
        "L(t)=a+b/sqrt(t)",
        "Schedule-unaware baseline fit on raw training step.",
        "#ff7f0e",
        "dash",
        fit_sqrt_t,
    ),
    MethodSpec(
        "B2_profiled_c",
        "B2 fitted exponent",
        "L(x)=L_inf+A*x^c",
        "Chooses c by prefix-tail validation, then refits L_inf and A.",
        "#2ca02c",
        "dash",
        fit_cumlr_power,
    ),
    MethodSpec(
        "B2r_profiled_c_logprior1",
        "B2r fitted exponent + c=1 prior",
        "L(x)=L_inf+A*x^c",
        "Same as B2, with a weak log-c prior centered at c=1.",
        "#17becf",
        "dashdot",
        fit_cumlr_regularized_c,
    ),
    MethodSpec(
        "B3_cumlr_c=1",
        "B3 linear remaining-LR",
        "L(x)=L_inf+A*x",
        "Schedule-aware model with c fixed to 1 before seeing data.",
        "#9467bd",
        "solid",
        lambda d, f, m: fit_cumlr_fixed_c(d, f, m, 1.0),
    ),
    MethodSpec(
        "B3_cumlr_c=0.5",
        "B3 sqrt remaining-LR",
        "L(x)=L_inf+A*sqrt(x)",
        "Schedule-aware model with c fixed to 0.5 before seeing data.",
        "#8c564b",
        "longdash",
        lambda d, f, m: fit_cumlr_fixed_c(d, f, m, 0.5),
    ),
)


def finite_or_none(value: float) -> float | None:
    if np.isfinite(value):
        return float(value)
    return None


def values_or_none(values: np.ndarray) -> list[float | None]:
    return [finite_or_none(float(v)) for v in values]


def line_for_method(df, method_key: str, params: tuple[float, ...], pred: float) -> tuple[list[float | None], dict]:
    if method_key == "B0_last_value":
        return [finite_or_none(pred)] * len(PLOT_STEPS), {}

    if method_key == "B1_sqrt_t":
        a, b = params
        return values_or_none(a + b / np.sqrt(PLOT_STEPS)), {"a": float(a), "b": float(b)}

    L_inf, A, c, fit_start_u = params[:4]
    u_full = compute_cumulative_lr(df).to_numpy(dtype=float)
    u_plot = np.interp(PLOT_STEPS, df["_step"].to_numpy(dtype=float), u_full)
    U = float(u_full[-1])
    x_plot = normalized_remaining_lr(u_plot, U, fit_start_u)
    extra = {"L_inf": float(L_inf), "A": float(A), "c": float(c)}
    if len(params) >= 5:
        extra["validation_rmse"] = float(params[4])
    return values_or_none(power_prediction(x_plot, L_inf, A, c)), extra


def build_payload() -> dict:
    runs = {}
    for spec in RUNS_1E20:
        logger.info("Loading %s", spec.name)
        df = smooth_train_loss(load_run(spec))
        steps = df["_step"].to_numpy(dtype=float)
        u_full = compute_cumulative_lr(df).to_numpy(dtype=float)
        U = float(u_full[-1])
        plot_u_raw = np.interp(PLOT_STEPS, steps, u_full)
        target_window_steps = np.asarray(TARGET_WINDOW, dtype=float)
        target_window_u_raw = np.interp(target_window_steps, steps, u_full)
        warmup_u_raw = float(np.interp(WARMUP_STEPS, steps, u_full))
        run_payload = {
            "label": spec.name,
            "lr_factor": spec.lr_factor,
            "color": RUN_COLORS[spec.name],
            "total_u": U,
            "plot_u_raw": values_or_none(plot_u_raw),
            "plot_u_norm": values_or_none(plot_u_raw / U),
            "warmup_u_raw": warmup_u_raw,
            "warmup_u_norm": warmup_u_raw / U,
            "target_window_u_raw": values_or_none(target_window_u_raw),
            "target_window_u_norm": values_or_none(target_window_u_raw / U),
            "metrics": {},
        }

        for metric in METRICS:
            target = evaluate_final(df, metric.column)
            observed = df[["_step", metric.column]].dropna()
            observed_steps = observed["_step"].to_numpy(dtype=float)
            observed_u_raw = np.interp(observed_steps, steps, u_full)
            metric_payload = {
                "target": target,
                "steps": values_or_none(observed_steps),
                "u_raw": values_or_none(observed_u_raw),
                "u_norm": values_or_none(observed_u_raw / U),
                "loss": values_or_none(observed[metric.column].to_numpy(dtype=float)),
                "prefixes": {},
            }

            for prefix in PREFIX_GRID:
                prefix_step = prefix * TOTAL_STEPS
                prefix_u_raw = float(np.interp(prefix_step, steps, u_full))
                method_payload = {}
                for method in METHODS:
                    try:
                        pred, params = method.fit(df, prefix, metric.column)
                        line, extra = line_for_method(df, method.key, tuple(params), pred)
                        method_payload[method.key] = {
                            "prediction": finite_or_none(pred),
                            "abs_error": finite_or_none(abs(pred - target)),
                            "line": line,
                            "extra": extra,
                        }
                    except Exception as e:  # pragma: no cover
                        logger.debug(
                            "fit failed for %s/%s/%s@%.2f: %s",
                            spec.name,
                            metric.key,
                            method.key,
                            prefix,
                            e,
                        )
                        method_payload[method.key] = {
                            "prediction": None,
                            "abs_error": None,
                            "line": [None] * len(PLOT_STEPS),
                            "extra": {"error": str(e)},
                        }
                metric_payload["prefixes"][f"{prefix:.2f}"] = {
                    "prefix_frac": prefix,
                    "prefix_step": prefix_step,
                    "prefix_u_raw": prefix_u_raw,
                    "prefix_u_norm": prefix_u_raw / U,
                    "methods": method_payload,
                }
            run_payload["metrics"][metric.key] = metric_payload
        runs[spec.name] = run_payload

    return {
        "metrics": [
            {
                "key": metric.key,
                "label": metric.label,
                "column": metric.column,
                "detail": metric.detail,
                "observed_mode": metric.observed_mode,
            }
            for metric in METRICS
        ],
        "total_steps": TOTAL_STEPS,
        "warmup_steps": WARMUP_STEPS,
        "min_prefix_percent": MIN_PREFIX_PERCENT,
        "target_window": TARGET_WINDOW,
        "prefixes": list(PREFIX_GRID),
        "plot_steps": values_or_none(PLOT_STEPS),
        "methods": [
            {
                "key": method.key,
                "label": method.label,
                "formula": method.formula,
                "detail": method.detail,
                "color": method.color,
                "dash": method.dash,
            }
            for method in METHODS
        ],
        "runs": runs,
    }


def html_template(payload: dict) -> str:
    payload_json = json.dumps(payload, allow_nan=False)
    plotly_js = get_plotlyjs()
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Midtrain Prefix Predictor</title>
  <style>
    :root {{
      color-scheme: light;
      --text: #1f2933;
      --muted: #52606d;
      --line: #d9e2ec;
      --panel: #f8fafc;
      --accent: #2563eb;
    }}
    body {{
      margin: 0;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--text);
      background: #ffffff;
    }}
    header {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 18px 24px 14px;
      border-bottom: 1px solid var(--line);
    }}
    h1 {{
      margin: 0;
      font-size: 20px;
      font-weight: 650;
      letter-spacing: 0;
    }}
    .metric {{
      color: var(--muted);
      font-size: 13px;
      white-space: nowrap;
    }}
    .metric-toggle {{
      display: inline-flex;
      gap: 4px;
      padding: 4px;
      border: 1px solid var(--line);
      border-radius: 7px;
      background: var(--panel);
    }}
    .metric-button {{
      border: 0;
      border-radius: 5px;
      padding: 7px 10px;
      background: transparent;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
      cursor: pointer;
    }}
    .metric-button.active {{
      background: var(--accent);
      color: #ffffff;
    }}
    .controls {{
      display: grid;
      grid-template-columns: minmax(190px, 240px) minmax(190px, 250px) minmax(190px, 280px) 1fr 88px;
      gap: 16px;
      align-items: end;
      padding: 16px 24px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }}
    label {{
      display: grid;
      gap: 6px;
      font-size: 12px;
      font-weight: 650;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    select, input[type="range"] {{
      width: 100%;
    }}
    select {{
      appearance: none;
      height: 36px;
      padding: 0 12px;
      border: 1px solid #bcccdc;
      border-radius: 6px;
      background: #ffffff;
      color: var(--text);
      font-size: 14px;
    }}
    input[type="range"] {{
      accent-color: var(--accent);
    }}
    #prefixReadout {{
      font-variant-numeric: tabular-nums;
      font-size: 24px;
      font-weight: 700;
      color: var(--accent);
      text-align: right;
    }}
    .notes {{
      display: grid;
      grid-template-columns: repeat(3, minmax(0, 1fr));
      gap: 10px;
      padding: 12px 24px;
      border-bottom: 1px solid var(--line);
      background: #ffffff;
    }}
    .note {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 12px;
      font-size: 13px;
      color: var(--muted);
      line-height: 1.35;
    }}
    .note strong {{
      color: var(--text);
    }}
    main {{
      padding: 18px 24px 28px;
    }}
    .plots {{
      display: grid;
      grid-template-columns: minmax(0, 1.45fr) minmax(380px, 0.75fr);
      gap: 16px;
    }}
    #fitPlot, #barPlot, #errorPlot {{
      min-height: 440px;
      border: 1px solid var(--line);
      border-radius: 6px;
    }}
    #errorPlot {{
      margin-top: 16px;
      min-height: 330px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin-top: 16px;
      font-size: 13px;
      font-variant-numeric: tabular-nums;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 8px 10px;
      text-align: right;
    }}
    th:first-child, td:first-child {{
      text-align: left;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
      background: #f8fafc;
    }}
    td.detail {{
      text-align: left;
      color: var(--muted);
      max-width: 520px;
      line-height: 1.35;
    }}
    @media (max-width: 980px) {{
      .controls, .plots, .notes {{
        grid-template-columns: 1fr;
      }}
      #prefixReadout {{
        text-align: left;
      }}
    }}
  </style>
  <script>{plotly_js}</script>
</head>
<body>
  <header>
    <h1>v5-isoflop 3e20 Midtrain Prefix Predictor</h1>
    <div class="metric-toggle" id="metricToggle"></div>
  </header>
  <section class="controls">
    <label>
      Compare
      <select id="runSelect"></select>
    </label>
    <label>
      X Axis
      <select id="axisSelect">
        <option value="u_norm" selected>Normalized cumulative LR u/U</option>
        <option value="u_raw">Raw cumulative LR u</option>
        <option value="step">Training step</option>
      </select>
    </label>
    <label>
      Fit Overlay
      <select id="methodSelect"></select>
    </label>
    <label>
      Prefix
      <input id="prefixSlider" type="range" min="{payload["min_prefix_percent"]}" max="90" step="1" value="50">
    </label>
    <div id="prefixReadout">50%</div>
  </section>
  <section class="notes">
    <div class="note">
      <strong>Selected prefix</strong> is the part of the run each predictor is allowed to see.
      The slider starts just after warmup, because the fits skip warmup-only points.
    </div>
    <div class="note">
      <strong>Main x-axis</strong> defaults to normalized cumulative LR <code>u/U</code>, so all three LR-factor
      runs can be compared on the same progress axis.
    </div>
    <div class="note">
      <strong>Metric toggle</strong> switches all plots and errors between train loss and sparse Paloma c4_en eval loss.
      Eval loss can rise or recover; read it as a retention metric, not a monotone training objective.
    </div>
  </section>
  <main>
    <section class="plots">
      <div id="fitPlot"></div>
      <div id="barPlot"></div>
    </section>
    <div id="errorPlot"></div>
    <table>
      <thead>
        <tr id="methodHead"></tr>
      </thead>
      <tbody id="methodRows"></tbody>
    </table>
  </main>
  <script>
    const DATA = {payload_json};
    const METHOD_BY_KEY = Object.fromEntries(DATA.methods.map((m) => [m.key, m]));
    const METRIC_BY_KEY = Object.fromEntries(DATA.metrics.map((m) => [m.key, m]));
    const ALL_RUNS = "__all_runs__";
    let currentMetricKey = DATA.metrics[0].key;
    const metricToggle = document.getElementById("metricToggle");
    const runSelect = document.getElementById("runSelect");
    const axisSelect = document.getElementById("axisSelect");
    const methodSelect = document.getElementById("methodSelect");
    const prefixSlider = document.getElementById("prefixSlider");
    const prefixReadout = document.getElementById("prefixReadout");
    const methodHead = document.getElementById("methodHead");
    const methodRows = document.getElementById("methodRows");

    for (const metric of DATA.metrics) {{
      const button = document.createElement("button");
      button.type = "button";
      button.className = "metric-button";
      button.dataset.metricKey = metric.key;
      button.textContent = metric.key === "train_loss" ? "Train loss" : "Eval loss";
      button.title = `${{metric.label}}: ${{metric.detail}}`;
      button.addEventListener("click", () => {{
        currentMetricKey = metric.key;
        updateAll();
      }});
      metricToggle.appendChild(button);
    }}

    const allOption = document.createElement("option");
    allOption.value = ALL_RUNS;
    allOption.textContent = "All LR runs";
    runSelect.appendChild(allOption);
    for (const runName of Object.keys(DATA.runs)) {{
      const option = document.createElement("option");
      option.value = runName;
      option.textContent = runName;
      runSelect.appendChild(option);
    }}

    for (const method of DATA.methods) {{
      const option = document.createElement("option");
      option.value = method.key;
      option.textContent = method.label;
      methodSelect.appendChild(option);
    }}
    methodSelect.value = "B3_cumlr_c=1";

    const AXES = {{
      step: {{
        label: "training step",
        dataKey: "steps",
        plotKey: "plot_steps",
        prefixKey: "prefix_step",
        warmupKey: "warmup_steps",
        targetWindowKey: "target_window",
      }},
      u_raw: {{
        label: "cumulative learning rate u(t)",
        dataKey: "u_raw",
        plotKey: "plot_u_raw",
        prefixKey: "prefix_u_raw",
        warmupKey: "warmup_u_raw",
        targetWindowKey: "target_window_u_raw",
      }},
      u_norm: {{
        label: "normalized cumulative learning rate u(t)/U",
        dataKey: "u_norm",
        plotKey: "plot_u_norm",
        prefixKey: "prefix_u_norm",
        warmupKey: "warmup_u_norm",
        targetWindowKey: "target_window_u_norm",
      }},
    }};

    function prefixKey() {{
      return (Number(prefixSlider.value) / 100).toFixed(2);
    }}

    function fmt(value, digits = 4) {{
      if (value === null || value === undefined || Number.isNaN(value)) return "-";
      return Number(value).toFixed(digits);
    }}

    function currentRunNames() {{
      if (runSelect.value === ALL_RUNS) return Object.keys(DATA.runs);
      return [runSelect.value];
    }}

    function currentAxis() {{
      return AXES[axisSelect.value];
    }}

    function currentMetric() {{
      return METRIC_BY_KEY[currentMetricKey];
    }}

    function selectedMethod() {{
      return METHOD_BY_KEY[methodSelect.value];
    }}

    function runPrefix(run) {{
      return run.metrics[currentMetricKey].prefixes[prefixKey()];
    }}

    function methodValue(run, methodKey) {{
      return runPrefix(run).methods[methodKey];
    }}

    function mean(values) {{
      const clean = values.filter((v) => v !== null && Number.isFinite(v));
      if (clean.length === 0) return null;
      return clean.reduce((a, b) => a + b, 0) / clean.length;
    }}

    function axisValues(run, plot = false) {{
      const axis = currentAxis();
      if (axisSelect.value === "step" && plot) return DATA.plot_steps;
      if (!plot) return run.metrics[currentMetricKey][axis.dataKey];
      return run[plot ? axis.plotKey : axis.dataKey];
    }}

    function prefixX(run) {{
      return runPrefix(run)[currentAxis().prefixKey];
    }}

    function warmupX(run) {{
      const axis = currentAxis();
      return axisSelect.value === "step" ? DATA.warmup_steps : run[axis.warmupKey];
    }}

    function targetWindowX(run) {{
      const axis = currentAxis();
      return axisSelect.value === "step" ? DATA.target_window : run[axis.targetWindowKey];
    }}

    function xRange(runNames) {{
      if (axisSelect.value === "u_norm") return [0, 1.02];
      if (axisSelect.value === "step") return [0, DATA.total_steps + 80];
      const maxU = Math.max(...runNames.map((name) => DATA.runs[name].total_u));
      return [0, maxU * 1.02];
    }}

    function lossRange(runNames) {{
      const fitValues = [];
      const methodKey = methodSelect.value;
      const allValues = [];
      for (const runName of runNames) {{
        const run = DATA.runs[runName];
        const metric = run.metrics[currentMetricKey];
        allValues.push(...metric.loss.filter((v) => v !== null && Number.isFinite(v)), metric.target);
        for (const value of methodValue(run, methodKey).line) {{
          if (value !== null && Number.isFinite(value)) fitValues.push(value);
        }}
      }}
      allValues.push(...fitValues);
      const yMin = Math.min(...allValues);
      const yMax = Math.max(...allValues);
      const pad = Math.max((yMax - yMin) * 0.08, 0.02);
      return [Math.max(0, yMin - pad), yMax + pad];
    }}

    function updateFitPlot() {{
      const runNames = currentRunNames();
      const axis = currentAxis();
      const metric = currentMetric();
      const method = selectedMethod();
      const yRange = lossRange(runNames);
      const traces = [];
      const shapes = [];
      const annotations = [];

      for (const runName of runNames) {{
        const run = DATA.runs[runName];
        const runMetric = run.metrics[currentMetricKey];
        const item = methodValue(run, method.key);
        traces.push({{
          x: axisValues(run),
          y: runMetric.loss,
          mode: metric.observed_mode,
          name: `${{run.label}} observed ${{metric.label}}`,
          line: {{color: run.color, width: 1.4}},
          marker: {{color: run.color, size: currentMetricKey === "eval_c4_en" ? 6 : 0}},
          opacity: 0.72,
          hovertemplate: `${{run.label}} observed<br>${{metric.label}}` +
            `<br>${{axis.label}}=%{{x:.4g}}<br>loss=%{{y:.4f}}<extra></extra>`,
        }});
        traces.push({{
          x: axisValues(run, true),
          y: item.line,
          mode: "lines",
          name: `${{run.label}} fit: ${{method.label}}`,
          line: {{color: run.color, width: 2.4, dash: method.dash}},
          hovertemplate: `${{run.label}}<br>${{method.label}}<br>${{method.formula}}` +
            `<br>${{axis.label}}=%{{x:.4g}}<br>fit=%{{y:.4f}}<extra></extra>`,
        }});
        traces.push({{
          x: [axisValues(run, true).at(-1)],
          y: [item.prediction],
          mode: "markers",
          name: `${{run.label}} endpoint prediction`,
          showlegend: false,
          marker: {{color: run.color, size: 9, symbol: "circle"}},
          hovertemplate: `${{run.label}}<br>${{method.label}} endpoint prediction=%{{y:.4f}}<extra></extra>`,
        }});

        const targetWindow = targetWindowX(run);
        shapes.push(
          {{
            type: "rect",
            xref: "x",
            yref: "paper",
            x0: targetWindow[0],
            x1: targetWindow[1],
            y0: 0,
            y1: 1,
            fillcolor: run.color,
            opacity: 0.04,
            line: {{width: 0}},
          }},
          {{
            type: "line",
            xref: "x",
            yref: "paper",
            x0: warmupX(run),
            x1: warmupX(run),
            y0: 0,
            y1: 1,
            line: {{color: run.color, width: 1, dash: "dot"}},
            opacity: 0.35,
          }},
          {{
            type: "line",
            xref: "x",
            yref: "paper",
            x0: prefixX(run),
            x1: prefixX(run),
            y0: 0,
            y1: 1,
            line: {{color: run.color, width: 1.2, dash: "dash"}},
            opacity: 0.55,
          }},
          {{
            type: "line",
            xref: "paper",
            yref: "y",
            x0: 0,
            x1: 1,
            y0: runMetric.target,
            y1: runMetric.target,
            line: {{color: run.color, width: 1, dash: "dot"}},
            opacity: 0.6,
          }},
        );
      }}

      const firstRun = DATA.runs[runNames[0]];
      annotations.push(
        {{
          x: warmupX(firstRun),
          y: 1,
          xref: "x",
          yref: "paper",
          text: "warmup ends",
          showarrow: false,
          xanchor: "left",
          yanchor: "bottom",
          font: {{size: 11, color: "#6b7280"}},
        }},
        {{
          x: prefixX(firstRun),
          y: 1,
          xref: "x",
          yref: "paper",
          text: "selected prefix",
          showarrow: false,
          xanchor: "left",
          yanchor: "bottom",
          font: {{size: 11, color: "#6b7280"}},
        }},
      );

      const scopeLabel = runSelect.value === ALL_RUNS ? "all LR runs" : firstRun.label;
      const layout = {{
        title: {{
          text: `${{scopeLabel}}: ${{metric.label}} with ${{method.label}} on ${{axis.label}} ` +
            `at ${{prefixSlider.value}}% prefix` +
            `<br><sup>Solid lines are observed loss; dashed/solid fit lines use only data left of each run's ` +
            `selected-prefix line. Endpoint markers are predictions.</sup>`,
          x: 0.02,
          xanchor: "left",
        }},
        margin: {{l: 76, r: 26, t: 78, b: 84}},
        xaxis: {{title: axis.label, range: xRange(runNames), zeroline: false}},
        yaxis: {{title: metric.label, range: yRange, zeroline: false}},
        hovermode: "x unified",
        legend: {{
          orientation: "h",
          y: -0.26,
          x: 0,
          font: {{size: 11}},
        }},
        shapes,
        annotations,
      }};
      Plotly.react("fitPlot", traces, layout, {{responsive: true, displaylogo: false}});
    }}

    function updateBarPlot() {{
      const runNames = currentRunNames();
      const x = DATA.methods.map((m) => m.label);
      const y = DATA.methods.map((m) => mean(runNames.map((name) => methodValue(DATA.runs[name], m.key).abs_error)));
      const colors = DATA.methods.map((m) => m.color);
      const perRunText = DATA.methods.map((m) =>
        runNames
          .map((name) => {{
            const item = methodValue(DATA.runs[name], m.key);
            return `${{name}}: pred ${{fmt(item.prediction)}} / err ${{fmt(item.abs_error)}}`;
          }})
          .join("<br>")
      );
      const trace = {{
        x,
        y,
        type: "bar",
        marker: {{color: colors}},
        customdata: perRunText,
        hovertemplate: "%{{x}}<br>mean abs error=%{{y:.4f}}<br>%{{customdata}}<extra></extra>",
      }};
      const scopeLabel = runSelect.value === ALL_RUNS ? "mean across all LR runs" : runNames[0];
      const metric = currentMetric();
      const layout = {{
        title: {{
          text: "Endpoint prediction error at selected prefix" +
            `<br><sup>${{metric.label}}; ${{scopeLabel}}; absolute error vs each run's final target mean. ` +
            `Lower is better.</sup>`,
          x: 0.02,
          xanchor: "left",
        }},
        margin: {{l: 74, r: 24, t: 76, b: 132}},
        yaxis: {{title: "absolute error"}},
        xaxis: {{title: "predictor", tickangle: -35}},
        shapes: [
          {{
            type: "rect",
            xref: "paper",
            yref: "y",
            x0: 0,
            x1: 1,
            y0: 0.005,
            y1: 0.010,
            fillcolor: "#22c55e",
            opacity: 0.12,
            line: {{width: 0}},
          }},
        ],
        annotations: [
          {{
            x: 0.98,
            y: 0.010,
            xref: "paper",
            yref: "y",
            text: "noise floor band",
            showarrow: false,
            xanchor: "right",
            yanchor: "bottom",
            font: {{size: 11, color: "#15803d"}},
          }},
        ],
      }};
      Plotly.react("barPlot", [trace], layout, {{responsive: true, displaylogo: false}});
    }}

    function updateErrorPlot() {{
      const runNames = currentRunNames();
      const currentPrefix = Number(prefixSlider.value);
      const metric = currentMetric();
      const traces = DATA.methods.map((method) => {{
        const y = DATA.prefixes.map((p) =>
          mean(runNames.map((name) => DATA.runs[name].prefixes[p.toFixed(2)].methods[method.key].abs_error))
        );
        return {{
          x: DATA.prefixes.map((p) => p * 100),
          y,
          mode: "lines+markers",
          name: method.label,
          line: {{color: method.color, width: 2, dash: method.dash}},
          marker: {{size: 5}},
          hovertemplate: "%{{x:.0f}}% prefix<br>abs error=%{{y:.4f}}<extra></extra>",
        }};
      }});
      const scopeLabel = runSelect.value === ALL_RUNS ? "all LR runs" : runNames[0];
      const layout = {{
        title: {{
          text: `${{scopeLabel}}: ${{metric.label}} endpoint error as the prefix grows` +
            `<br><sup>The vertical line matches the slider. Lower curves mean better final-loss prediction.</sup>`,
          x: 0.02,
          xanchor: "left",
        }},
        margin: {{l: 76, r: 26, t: 76, b: 90}},
        xaxis: {{title: "observed prefix of run (%)", range: [19, 91]}},
        yaxis: {{title: "absolute endpoint error"}},
        hovermode: "x unified",
        legend: {{
          orientation: "h",
          y: -0.28,
          x: 0,
          font: {{size: 11}},
        }},
        shapes: [
          {{
            type: "line",
            xref: "x",
            yref: "paper",
            x0: currentPrefix,
            x1: currentPrefix,
            y0: 0,
            y1: 1,
            line: {{color: "#6b7280", width: 1, dash: "dash"}},
          }},
          {{
            type: "rect",
            xref: "paper",
            yref: "y",
            x0: 0,
            x1: 1,
            y0: 0.005,
            y1: 0.010,
            fillcolor: "#22c55e",
            opacity: 0.12,
            line: {{width: 0}},
          }},
        ],
        annotations: [
          {{
            x: currentPrefix,
            y: 1,
            xref: "x",
            yref: "paper",
            text: "selected prefix",
            showarrow: false,
            xanchor: "left",
            yanchor: "bottom",
            font: {{size: 11, color: "#6b7280"}},
          }},
          {{
            x: 0.99,
            y: 0.010,
            xref: "paper",
            yref: "y",
            text: "noise floor band",
            showarrow: false,
            xanchor: "right",
            yanchor: "bottom",
            font: {{size: 11, color: "#15803d"}},
          }},
        ],
      }};
      Plotly.react("errorPlot", traces, layout, {{responsive: true, displaylogo: false}});
    }}

    function updateTable() {{
      const runNames = currentRunNames();
      const metric = currentMetric();
      if (runSelect.value === ALL_RUNS) {{
        methodHead.innerHTML = `
          <th>Method</th>
          <th>Formula / Constraint</th>
          <th>Mean Abs Error</th>
          ${{runNames.map((name) => `<th>${{name}} Pred / Err</th>`).join("")}}
          <th>Selected c values</th>
        `;
        methodRows.innerHTML = DATA.methods.map((method) => {{
          const items = runNames.map((name) => methodValue(DATA.runs[name], method.key));
          const meanErr = mean(items.map((item) => item.abs_error));
          const runCells = items
            .map((item) => `<td>${{fmt(item.prediction)}} / ${{fmt(item.abs_error)}}</td>`)
            .join("");
          const cValues = items.map((item) => fmt(item.extra.c, 3)).join(" / ");
          return `<tr>
            <td style="color:${{method.color}};font-weight:650">${{method.label}}</td>
            <td class="detail"><strong>${{method.formula}}</strong><br>${{method.detail}}</td>
            <td>${{fmt(meanErr)}}</td>
            ${{runCells}}
            <td>${{cValues}}</td>
          </tr>`;
        }}).join("");
        return;
      }}

      const run = DATA.runs[runNames[0]];
      methodHead.innerHTML = `
        <th>Method</th>
        <th>Formula / Constraint</th>
        <th>Endpoint Prediction</th>
        <th>Abs Error</th>
        <th>Selected c</th>
        <th>Prefix Val RMSE</th>
      `;
      const rows = DATA.methods.map((method) => {{
        const item = methodValue(run, method.key);
        return `<tr>
          <td style="color:${{method.color}};font-weight:650">${{method.label}}</td>
          <td class="detail"><strong>${{method.formula}}</strong><br>${{method.detail}}</td>
          <td>${{fmt(item.prediction)}}</td>
          <td>${{fmt(item.abs_error)}}</td>
          <td>${{fmt(item.extra.c, 3)}}</td>
          <td>${{fmt(item.extra.validation_rmse, 5)}}</td>
        </tr>`;
      }});
      methodRows.innerHTML = rows.join("");
    }}

    function updateAll() {{
      prefixReadout.textContent = `${{prefixSlider.value}}%`;
      for (const button of metricToggle.querySelectorAll(".metric-button")) {{
        button.classList.toggle("active", button.dataset.metricKey === currentMetricKey);
      }}
      updateFitPlot();
      updateBarPlot();
      updateErrorPlot();
      updateTable();
    }}

    runSelect.addEventListener("change", updateAll);
    axisSelect.addEventListener("change", updateAll);
    methodSelect.addEventListener("change", updateAll);
    prefixSlider.addEventListener("input", updateAll);
    updateAll();
  </script>
</body>
</html>
"""


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    payload = build_payload()
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_template(payload))
    logger.info("Wrote %s", OUT_PATH)

    subprocess.run(["open", "-a", "Google Chrome", OUT_PATH], check=False)


if __name__ == "__main__":
    main()
