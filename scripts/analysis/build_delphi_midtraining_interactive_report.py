# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Build the standalone Delphi midtraining interactive report.

The report is a single HTML file backed by the cached outputs from
``delphi_within_run_prediction.py``. It is suitable for GitHub Pages because all
data needed for interactivity is embedded in the page.

Run:
    uv run python scripts/analysis/build_delphi_midtraining_interactive_report.py
"""

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

OUT_DIR = Path("midtrain_analysis_outputs/small_final_loss_scaling")
TRAJECTORY_POINTS_PATH = OUT_DIR / "trajectory_points.csv"
TRAJECTORY_PREDICTIONS_PATH = OUT_DIR / "trajectory_prefix_predictions.csv"
JOINT_PREDICTIONS_PATH = OUT_DIR / "trajectory_joint_prefix_predictions.csv"
JOINT_MODELS_PATH = OUT_DIR / "trajectory_joint_prefix_models.csv"
ENDPOINTS_PATH = OUT_DIR / "endpoints.csv"
EXTRAPOLATION_TARGETS_PATH = OUT_DIR / "extrapolation_targets.csv"
DEFAULT_OUTPUT_PATH = OUT_DIR / "delphi_midtraining_interactive.html"

SMALL_LADDER_SCALES = ("3e18", "9e18", "2e19", "3e19", "9e19", "2e20", "3e20")
HELD_OUT_SCALES = ("1e21", "1e22")
CUTOFF_SCALES = (*SMALL_LADDER_SCALES, "1e21")

METRIC_LABEL = "math_val_loss"
SCALE_ORDER = ("3e18", "9e18", "2e19", "3e19", "9e19", "2e20", "3e20", "1e21", "1e22")
MIX_ORDER = ("p33m67", "p50m50", "p67m33")
LR_ORDER = ("33", "50", "67", "83")
METHOD_ORDER = (
    "template_by_recipe",
    "template_by_mix",
    "template_global",
    "last_value",
    "linear_tau",
    "curve_log_mae",
    "curve_log_huber",
    "curve_exp_mae",
    "curve_exp_huber",
    "curve_power_mae",
    "curve_power_huber",
    "curve_rational_mae",
    "curve_rational_huber",
)
METHOD_LABELS = {
    "template_by_recipe": "template recipe",
    "template_by_mix": "template mix",
    "template_global": "template global",
    "last_value": "last",
    "linear_tau": "linear",
    "curve_log_mae": "log MAE",
    "curve_log_huber": "log Huber",
    "curve_exp_mae": "exp MAE",
    "curve_exp_huber": "exp Huber",
    "curve_power_mae": "power MAE",
    "curve_power_huber": "power Huber",
    "curve_rational_mae": "rational MAE",
    "curve_rational_huber": "rational Huber",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--points", type=Path, default=TRAJECTORY_POINTS_PATH)
    parser.add_argument("--predictions", type=Path, default=TRAJECTORY_PREDICTIONS_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def finite_or_none(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, bool | str):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isfinite(value):
            rounded = round(value, 6)
            if rounded == 0:
                return 0
            return rounded
        return None
    return value


def records_for_json(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records = frame.to_dict(orient="records")
    return [{key: finite_or_none(value) for key, value in record.items()} for record in records]


def payload(points_path: Path, predictions_path: Path) -> dict[str, Any]:
    points = pd.read_csv(points_path, dtype={"scale": str, "lr": str})
    points = points[points["metric_label"].eq(METRIC_LABEL)].copy()
    point_columns = [
        "run_id",
        "run_name",
        "scale",
        "mix",
        "lr",
        "eval_split",
        "step",
        "final_step",
        "tau",
        "value",
    ]
    points = points[point_columns].sort_values(["scale", "mix", "lr", "eval_split", "run_id", "tau"])

    predictions = pd.read_csv(predictions_path, dtype={"scale": str, "lr": str}, low_memory=False)
    predictions = predictions[predictions["metric_label"].eq(METRIC_LABEL)].copy()
    prediction_columns = [
        "run_id",
        "run_name",
        "scale",
        "mix",
        "lr",
        "eval_split",
        "target_kind",
        "complete",
        "prefix",
        "prefix_actual_tau",
        "method",
        "target",
        "predicted",
        "error",
        "abs_error",
        "param_floor",
        "param_amplitude",
        "param_shape_1",
        "param_shape_2",
        "prefix_fit_mae",
    ]
    predictions = predictions[prediction_columns].sort_values(["scale", "mix", "lr", "prefix", "method", "run_id"])
    if JOINT_PREDICTIONS_PATH.exists():
        joint_predictions = pd.read_csv(JOINT_PREDICTIONS_PATH, dtype={"scale": str, "lr": str}, low_memory=False)
        joint_columns = [
            "run_id",
            "run_name",
            "scale",
            "mix",
            "lr",
            "eval_split",
            "complete",
            "prefix",
            "scope",
            "form",
            "method",
            "method_label",
            "target",
            "predicted",
            "error",
            "abs_error",
            "fit_n",
            "optimizer",
            "optimizer_success",
        ]
        joint_predictions = joint_predictions[joint_columns].sort_values(["prefix", "method", "scale", "mix", "lr"])
    else:
        joint_predictions = pd.DataFrame()
    if JOINT_MODELS_PATH.exists():
        joint_models = pd.read_csv(JOINT_MODELS_PATH, dtype={"scale": str}, low_memory=False)
        joint_model_columns = [
            "scale",
            "prefix",
            "scope",
            "form",
            "method",
            "method_label",
            "include_flops",
            "beta_json",
            "theta_json",
            "fit_n",
            "optimizer",
            "optimizer_success",
        ]
        joint_models = joint_models[joint_model_columns].sort_values(["prefix", "method", "scale"])
    else:
        joint_models = pd.DataFrame()

    def _floor_power_model(x: np.ndarray, floor: float, amplitude: float, alpha: float) -> np.ndarray:
        return floor + amplitude * np.power(x, -alpha)

    def _fit_floor_power(xs: np.ndarray, ys: np.ndarray) -> dict[str, float] | None:
        if xs.size < 3:
            return None
        xs_norm = xs / 1e18
        floor0 = float(min(ys)) * 0.5
        amp0 = max(float(max(ys)) - floor0, 1e-3)
        try:
            params, _ = curve_fit(
                _floor_power_model,
                xs_norm,
                ys,
                p0=(floor0, amp0, 0.1),
                bounds=([0.0, 0.0, 0.0], [float(min(ys)) * 0.999, np.inf, 5.0]),
                maxfev=10000,
            )
        except (RuntimeError, ValueError):
            return None
        floor, amp, alpha = (float(value) for value in params)
        pred = _floor_power_model(xs_norm, floor, amp, alpha)
        ss_res = float(np.sum((ys - pred) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        rmse = math.sqrt(ss_res / xs.size)
        return {
            "floor": floor,
            "amplitude": amp,
            "alpha": alpha,
            "r2": r2,
            "rmse": rmse,
            "n": int(xs.size),
        }

    def _fit_log_linear(xs: np.ndarray, ys: np.ndarray) -> dict[str, float] | None:
        if xs.size < 2:
            return None
        lx = np.log(xs)
        ly = np.log(ys)
        slope, intercept = np.polyfit(lx, ly, 1)
        pred_log = intercept + slope * lx
        pred = np.exp(pred_log)
        ss_res = float(np.sum((ys - pred) ** 2))
        ss_tot = float(np.sum((ys - ys.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else None
        rmse_log = math.sqrt(float(np.sum((ly - pred_log) ** 2)) / xs.size)
        return {
            "slope": float(slope),
            "intercept": float(intercept),
            "r2": r2,
            "rmse_log": rmse_log,
            "n": int(xs.size),
        }

    endpoint_columns = ["run_id", "run_name", "scale", "scale_flops", "mix", "lr", "value"]
    if ENDPOINTS_PATH.exists():
        endpoints = pd.read_csv(ENDPOINTS_PATH, dtype={"scale": str, "lr": str})
        endpoints = endpoints[endpoints["metric_label"].eq(METRIC_LABEL) & endpoints["complete"].astype(bool)].copy()
        endpoints["scale_flops"] = endpoints["scale_flops"].astype(float)
        endpoints = endpoints[endpoint_columns].sort_values(["scale_flops", "mix", "lr"])
    else:
        endpoints = pd.DataFrame(columns=endpoint_columns)
    if EXTRAPOLATION_TARGETS_PATH.exists():
        targets = pd.read_csv(EXTRAPOLATION_TARGETS_PATH, dtype={"scale": str, "lr": str})
        targets = targets[targets["metric_label"].eq(METRIC_LABEL) & targets["complete"].astype(bool)].copy()
        targets["scale_flops"] = targets["scale_flops"].astype(float)
        targets = targets[endpoint_columns].sort_values(["scale_flops", "mix", "lr"])
    else:
        targets = pd.DataFrame(columns=endpoint_columns)

    scaling_fits: list[dict[str, Any]] = []
    if not endpoints.empty:
        scale_flops_lookup: dict[str, float] = {}
        for frame in (endpoints, targets):
            for scale_str, scale_value in zip(
                frame["scale"].astype(str),
                frame["scale_flops"].astype(float),
                strict=False,
            ):
                scale_flops_lookup.setdefault(scale_str, float(scale_value))
        cutoff_flops_map = {scale: scale_flops_lookup.get(scale, float("inf")) for scale in CUTOFF_SCALES}
        # Build a combined "training pool" of small-ladder endpoints + held-out endpoints
        # (1e21 is available as a candidate training scale when the slider goes that high).
        fit_pool = pd.concat([endpoints, targets[targets["scale"].isin(CUTOFF_SCALES)]], ignore_index=True)
        cell_groups = fit_pool.groupby(["mix", "lr"], sort=False)
        for (mix, lr), group in cell_groups:
            group = group.sort_values("scale_flops")
            for cutoff_index, cutoff_scale in enumerate(CUTOFF_SCALES):
                cutoff_flops = cutoff_flops_map.get(cutoff_scale, float("inf"))
                if math.isnan(cutoff_flops):
                    continue
                mask = group["scale_flops"] <= cutoff_flops + 1.0
                included = group[mask]
                if included.empty:
                    continue
                xs = included["scale_flops"].to_numpy(dtype=float)
                ys = included["value"].to_numpy(dtype=float)
                row: dict[str, Any] = {
                    "mix": mix,
                    "lr": str(lr),
                    "cutoff_index": cutoff_index,
                    "cutoff_scale": cutoff_scale,
                    "n": int(xs.size),
                    "min_scale": str(included["scale"].iloc[0]),
                    "max_scale": str(included["scale"].iloc[-1]),
                }
                fp = _fit_floor_power(xs, ys)
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
                ll = _fit_log_linear(xs, ys)
                if ll:
                    row.update(
                        {
                            "ll_slope": ll["slope"],
                            "ll_intercept": ll["intercept"],
                            "ll_r2": ll["r2"],
                            "ll_rmse_log": ll["rmse_log"],
                        }
                    )
                scaling_fits.append(row)

    return {
        "points": records_for_json(points),
        "predictions": records_for_json(predictions),
        "jointPredictions": records_for_json(joint_predictions),
        "jointModels": records_for_json(joint_models),
        "endpoints": records_for_json(endpoints),
        "heldoutTargets": records_for_json(targets),
        "scalingFits": [{key: finite_or_none(value) for key, value in row.items()} for row in scaling_fits],
        "scaleOrder": list(SCALE_ORDER),
        "smallLadderScales": list(SMALL_LADDER_SCALES),
        "heldOutScales": list(HELD_OUT_SCALES),
        "cutoffScales": list(CUTOFF_SCALES),
        "mixOrder": list(MIX_ORDER),
        "lrOrder": list(LR_ORDER),
        "methodOrder": list(METHOD_ORDER),
        "methodLabels": METHOD_LABELS,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Delphi Midtraining Interactive Scaling</title>
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
    h2 { margin: 30px 0 10px; font-size: 22px; }
    h3 { margin: 0 0 12px; font-size: 18px; }
    p { color: var(--muted); line-height: 1.45; }
    main { padding: 18px 32px 42px; max-width: 1540px; }
    .controls, .target-controls {
      display: flex;
      flex-wrap: wrap;
      gap: 16px 22px;
      align-items: center;
      padding: 14px 16px;
      background: var(--soft);
      border: 1px solid var(--line);
      border-radius: 8px;
    }
    .target-controls { margin: 14px 0 20px; }
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
    .formula-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
      margin: 14px 0 24px;
    }
    .formula-card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 14px 16px;
      background: #ffffff;
    }
    .formula-card h3 { margin-bottom: 8px; }
    .formula-card p { margin: 8px 0 0; font-size: 14px; }
    .math-line {
      overflow-x: auto;
      padding: 8px 0;
      color: var(--ink);
    }
    .chart { width: 100%; height: 650px; }
    #bestHeatmap { height: 900px; }
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
    .tables {
      display: grid;
      grid-template-columns: minmax(500px, 1fr) minmax(640px, 1.35fr);
      gap: 28px;
      align-items: start;
    }
    .tables-stacked {
      display: grid;
      grid-template-columns: 1fr;
      gap: 22px;
      align-items: start;
    }
    .tables-stacked > div { overflow-x: auto; }
    .tables > div { overflow-x: auto; }
    .note { font-size: 13px; color: var(--muted); }
    .summary {
      margin: 8px 0 12px;
      color: var(--muted);
      font-size: 14px;
    }
    details {
      margin-top: 18px;
      border-top: 1px solid var(--line);
      padding-top: 14px;
    }
    summary { cursor: pointer; font-weight: 650; }
  </style>
</head>
<body>
  <header>
    <h1>Delphi Midtraining Interactive Scaling</h1>
    <p>Interactive validation-trajectory prediction for math validation loss. Parametric prefix fits use SciPy optimization with MAE or Huber objectives; final model comparison uses endpoint MAE.</p>
  </header>
  <main>
    <section>
      <div class="controls">
        <label>mix <select id="mix"></select></label>
        <div>
          <div class="note">learning rates</div>
          <div id="lrChecks" class="checks"></div>
        </div>
        <label>prefix %
          <input id="prefix" type="range" min="10" max="90" step="5" value="90" />
          <span class="value" id="prefixValue">90%</span>
        </label>
        <label>functional form / method <select id="method"></select></label>
      </div>
    </section>

    <section>
      <h2>Curve Prediction Within A LR / Mix / Flop Cell</h2>
      <p>
        This first setting treats each completed run as its own cell: fixed flop
        scale, fixed data mix, and fixed learning rate. For a selected prefix
        \(p\), each parametric method fits only that cell's validation points with
        normalized progress \(\tau \le p\), then predicts the endpoint at
        \(\tau=1\). Each run's endpoint error is \(|\hat L(1)-L(1)|\); aggregate
        tables report MAE as the mean of those absolute endpoint errors, plus the
        worst-case absolute error.
      </p>
      <div class="formula-grid">
        <div class="formula-card">
          <h3>Shared endpoint model</h3>
          <div class="math-line">\(L(\tau) = F + A\,g(\tau; \theta),\quad g(1;\theta)=0\)</div>
          <p>\(F\) is the predicted final loss. \(A\) is the drop left from the prefix trajectory. Shape parameters \(\theta\) are initialized from a small grid and then optimized with SciPy.</p>
        </div>
        <div class="formula-card">
          <h3>Log</h3>
          <div class="math-line">\(g(\tau;s)=\frac{\log((1+s)/(\tau+s))}{\log((1+s)/s)}\)</div>
          <p>A sharp early drop with a long flattening tail.</p>
        </div>
        <div class="formula-card">
          <h3>Exponential</h3>
          <div class="math-line">\(g(\tau;r)=\frac{e^{-r\tau}-e^{-r}}{1-e^{-r}}\)</div>
          <p>A fast decay that asymptotes toward the endpoint.</p>
        </div>
        <div class="formula-card">
          <h3>Power</h3>
          <div class="math-line">\(g(\tau;s,a)=\frac{(\tau+s)^{-a}-(1+s)^{-a}}{s^{-a}-(1+s)^{-a}}\)</div>
          <p>A heavier-tailed curve; it can stay bendy deeper into training.</p>
        </div>
        <div class="formula-card">
          <h3>Rational</h3>
          <div class="math-line">\(g(\tau;t_0,\beta)=\frac{(1+(\tau/t_0)^\beta)^{-1}-(1+(1/t_0)^\beta)^{-1}}{1-(1+(1/t_0)^\beta)^{-1}}\)</div>
          <p>A smooth sideways-S / shoulder shape for trajectories that drop quickly then flatten.</p>
        </div>
        <div class="formula-card">
          <h3>MAE vs Huber Fit</h3>
          <div class="math-line">\(\min_{F,A}\sum_{\tau_i \le p}\rho(L_i - F - A g(\tau_i;\theta))\)</div>
          <p>
            MAE variants use bounded `scipy.optimize.minimize`; Huber variants use
            `scipy.optimize.least_squares(loss="huber")`. Shape parameters start
            from the small grid above, then SciPy optimizes \(F\), \(A\), and
            \(\theta\). All reported scores below are endpoint absolute error / MAE.
          </p>
        </div>
      </div>
    </section>

    <section>
      <h2>Per-Cell Within-Run Prediction</h2>
      <p class="note">Lines are observed trajectories. Diamonds are observed finals; x markers are predicted finals. Parametric curve methods draw fitted continuations from the selected prefix to the final step.</p>
      <div id="trajectory" class="chart"></div>
    </section>

    <section>
      <h2>Best Parametric Form By Scale And Prefix</h2>
      <p class="note">Each cell shows the parametric curve variant with lowest final-loss MAE at that scale and prefix.</p>
      <div id="bestHeatmap" class="chart"></div>
    </section>

    <section>
      <h2>Target MAE Config Search</h2>
      <p class="note">Pick a max absolute final-loss error target, then restrict the search to the mixture and learning-rate regime you care about. A config qualifies only if every completed run in that selected regime is at or below the target.</p>
      <div class="target-controls">
        <label>target max error
          <select id="target">
            <option>0.005</option>
            <option>0.01</option>
            <option selected>0.02</option>
            <option>0.05</option>
            <option>0.1</option>
          </select>
        </label>
        <label>target mix <select id="targetMix"></select></label>
        <label>target learning rate <select id="targetLr"></select></label>
      </div>
      <div class="summary" id="targetScopeSummary"></div>
      <div class="tables">
        <div>
          <h3>Selected Mix / LR</h3>
          <div id="regimeTable"></div>
        </div>
        <div>
          <h3>Per Scale In Selected Regime</h3>
          <div id="scaleTable"></div>
        </div>
      </div>
      <details>
        <summary>Global reference across all mixes and learning rates</summary>
        <div class="tables">
          <div>
            <h3>All Runs</h3>
            <div id="overallTable"></div>
          </div>
          <div>
            <h3>All Runs Per Scale</h3>
            <div id="globalScaleTable"></div>
          </div>
        </div>
      </details>
    </section>

    <section>
      <h2>Endpoint Scaling Law (Compute Vs Final Loss)</h2>
      <p>
        Chinchilla-style 3-parameter fit: \(L_\infty(C) = E + A\,(C/10^{18})^{-\alpha}\), where \(E\) is the
        irreducible-loss floor, \(C\) is base-model FLOPs, and \(L_\infty\) is final
        \(\texttt{math\_val\_loss}\). Fit per \((\textrm{mix}, \textrm{LR})\) on the small ladder
        (3e18 → 3e20) by \(\texttt{scipy.optimize.curve\_fit}\) with \(E < \min y\), \(A,\alpha \ge 0\).
        The 1e21 and 1e22 cells are never used by the fit; their actuals are plotted as triangles for the
        extrapolation check. The two-parameter log-log fit \(\log L = a + b\,\log C\) is available as a
        toggle for comparison — it lacks the asymptote so it under-predicts loss at very large compute.
      </p>
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

    <section>
      <h2>Joint Trajectory Fits Across LR, Mix, And Flop</h2>
      <p class="note">
        This second setting follows the scaling-law-discovery idea more directly:
        fit shared trajectory regressions using \(\tau\), flop scale, data mix,
        and learning rate as features. The global scope fits across all flops,
        mixes, and learning rates. The by-flop scope fits within each flop scale,
        sharing only across mix and learning rate. For each prefix, fitting uses
        only points with \(\tau \le p\), then predicts endpoints at \(\tau=1\).
      </p>
      <div class="formula-grid">
        <div class="formula-card">
          <h3>Source</h3>
          <p>
            The joint forms are inspired by
            <a href="https://arxiv.org/abs/2507.21184">Can Language Models Discover Scaling Laws?</a>
            and its
            <a href="https://linhaowei1.github.io/scaling_law_discovery/">SLDBench project page</a>.
            The key idea we are borrowing is shared structure over features, not
            an exact formula from the paper.
          </p>
        </div>
        <div class="formula-card">
          <h3>Joint Forms</h3>
          <div class="math-line">\(z(\mathrm{flop},m,\eta)^\top\beta_0 + z^\top\beta_1 e^{-k\tau} + z^\top\beta_2\tau\)</div>
          <div class="math-line">\(z^\top\beta_0 + z^\top\beta_1(\tau+s)^{-\alpha} + z^\top\beta_2\tau\)</div>
          <div class="math-line">\(z^\top\beta_0 + z^\top\beta_1\exp(-b e^{-k\tau}) + z^\top\beta_2\tau\)</div>
          <p>
            \(z\) contains mix, LR, and optionally log-flops with interactions.
            These are global regressions, not one curve per cell.
          </p>
        </div>
      </div>
      <div class="target-controls">
        <label>target max error
          <select id="jointTarget">
            <option>0.005</option>
            <option>0.01</option>
            <option selected>0.02</option>
            <option>0.05</option>
            <option>0.1</option>
          </select>
        </label>
        <label>joint mix <select id="jointMix"></select></label>
        <label>joint learning rate <select id="jointLr"></select></label>
        <label>joint method <select id="jointMethod"></select></label>
        <label>prefix %
          <input id="jointPrefix" type="range" min="10" max="90" step="5" value="90" />
          <span class="value" id="jointPrefixValue">90%</span>
        </label>
      </div>
      <div class="summary" id="jointScopeSummary"></div>
      <div id="jointTradeoff" class="chart"></div>
      <div id="jointCurves" class="chart"></div>
      <div id="jointScatter" class="chart"></div>
      <div class="tables">
        <div>
          <h3>Joint Configs Meeting Target</h3>
          <div id="jointConfigTable"></div>
        </div>
        <div>
          <h3>Joint Per-Scale Configs</h3>
          <div id="jointScaleTable"></div>
        </div>
      </div>
    </section>
  </main>
  <script id="payload" type="application/json">__PAYLOAD_JSON__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("payload").textContent);
    const points = DATA.points;
    const predictions = DATA.predictions;
    const jointPredictions = DATA.jointPredictions || [];
    const jointModels = (DATA.jointModels || []).map((model) => ({
      ...model,
      beta: JSON.parse(model.beta_json),
      theta: JSON.parse(model.theta_json),
    }));
    const scaleOrder = DATA.scaleOrder;
    const smallLadderScales = DATA.smallLadderScales;
    const heldOutScales = DATA.heldOutScales;
    const cutoffScales = DATA.cutoffScales || DATA.smallLadderScales;
    const mixOrder = DATA.mixOrder;
    const lrOrder = DATA.lrOrder;
    const methodOrder = DATA.methodOrder;
    const methodLabels = DATA.methodLabels;
    const endpoints = DATA.endpoints || [];
    const heldoutTargets = DATA.heldoutTargets || [];
    const scalingFits = DATA.scalingFits || [];
    const colors = {"33": "#4C78A8", "50": "#F58518", "67": "#54A24B", "83": "#E45756"};
    const scaleFlops = {"3e18": 3e18, "9e18": 9e18, "2e19": 2e19, "3e19": 3e19, "9e19": 9e19, "2e20": 2e20, "3e20": 3e20, "1e21": 1e21, "1e22": 1e22};

    function methodLabel(method) {
      return methodLabels[method] || method;
    }

    function baseMethod(method) {
      return method.replace(/_(mae|huber)$/, "");
    }

    function isCurve(method) {
      return method.startsWith("curve_");
    }

    function selectedLrs() {
      return [...document.querySelectorAll("#lrChecks input:checked")].map((input) => input.value);
    }

    function selectedPrefix() {
      return Number(document.getElementById("prefix").value) / 100;
    }

    function shapeValues(method, tau, shape1, shape2) {
      const base = baseMethod(method);
      if (base === "curve_log") {
        const denominator = Math.log((1 + shape1) / shape1);
        return Math.log((1 + shape1) / (tau + shape1)) / denominator;
      }
      if (base === "curve_exp") {
        const denominator = 1 - Math.exp(-shape1);
        return (Math.exp(-shape1 * tau) - Math.exp(-shape1)) / denominator;
      }
      if (base === "curve_power") {
        const raw = Math.pow(tau + shape1, -shape2);
        const rawStart = Math.pow(shape1, -shape2);
        const rawEnd = Math.pow(1 + shape1, -shape2);
        return (raw - rawEnd) / (rawStart - rawEnd);
      }
      if (base === "curve_rational") {
        const raw = 1 / (1 + Math.pow(tau / shape1, shape2));
        const rawEnd = 1 / (1 + Math.pow(1 / shape1, shape2));
        return (raw - rawEnd) / (1 - rawEnd);
      }
      return null;
    }

    function curveY(row, tau) {
      return row.param_floor + row.param_amplitude * shapeValues(
        row.method,
        tau,
        Number(row.param_shape_1),
        row.param_shape_2 == null ? null : Number(row.param_shape_2),
      );
    }

    function renderControls() {
      document.getElementById("mix").innerHTML = mixOrder.map((mix) => `<option value="${mix}">${mix}</option>`).join("");
      document.getElementById("method").innerHTML = methodOrder.map(
        (method) => `<option value="${method}">${methodLabel(method)}</option>`,
      ).join("");
      document.getElementById("method").value = methodOrder.includes("curve_rational_mae")
        ? "curve_rational_mae"
        : methodOrder[0];
      document.getElementById("lrChecks").innerHTML = lrOrder.map(
        (lr) => `<label><input type="checkbox" value="${lr}" checked />lr${lr}</label>`,
      ).join("");
      document.getElementById("targetMix").innerHTML = [
        '<option value="all">all mixes</option>',
        ...mixOrder.map((mix) => `<option value="${mix}">${mix}</option>`),
      ].join("");
      document.getElementById("targetMix").value = "p33m67";
      document.getElementById("targetLr").innerHTML = [
        '<option value="all">all learning rates</option>',
        ...lrOrder.map((lr) => `<option value="${lr}">lr${lr}</option>`),
      ].join("");
      document.getElementById("jointMix").innerHTML = [
        '<option value="all">all mixes</option>',
        ...mixOrder.map((mix) => `<option value="${mix}">${mix}</option>`),
      ].join("");
      document.getElementById("jointLr").innerHTML = [
        '<option value="all">all learning rates</option>',
        ...lrOrder.map((lr) => `<option value="${lr}">lr${lr}</option>`),
      ].join("");
      const jointMethods = [...new Map(jointPredictions.map((row) => [row.method, row.method_label || row.method])).entries()];
      document.getElementById("jointMethod").innerHTML = jointMethods.map(
        ([method, label]) => `<option value="${method}">${label}</option>`,
      ).join("");
      if (jointMethods.some(([method]) => method === "joint_by_flop_power_drift")) {
        document.getElementById("jointMethod").value = "joint_by_flop_power_drift";
      }
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
      document.querySelectorAll("select,input").forEach((element) => element.addEventListener("input", renderAll));
    }

    function renderTrajectory() {
      const mix = document.getElementById("mix").value;
      const lrs = selectedLrs();
      const prefix = selectedPrefix();
      document.getElementById("prefixValue").textContent = `${Math.round(prefix * 100)}%`;
      const method = document.getElementById("method").value;
      const traces = [];
      const groups = new Map();
      points.filter((point) => point.mix === mix && lrs.includes(String(point.lr))).forEach((point) => {
        const key = `${point.scale}|${point.lr}|${point.run_id}`;
        if (!groups.has(key)) {
          groups.set(key, []);
        }
        groups.get(key).push(point);
      });

      const seen = new Set();
      for (const group of groups.values()) {
        group.sort((left, right) => left.tau - right.tau);
        const first = group[0];
        const heldout = first.eval_split === "heldout_large";
        const legendGroup = `${first.lr}-${first.eval_split}`;
        traces.push({
          x: group.map((point) => point.tau),
          y: group.map((point) => point.value),
          mode: "lines",
          type: "scatter",
          name: `lr${first.lr} ${heldout ? "held-out" : "small"}`,
          legendgroup: legendGroup,
          showlegend: !seen.has(legendGroup),
          line: {color: colors[first.lr] || "#666", width: heldout ? 3 : 1, dash: heldout ? "solid" : "dot"},
          opacity: heldout ? 0.95 : 0.30,
          hovertext: group.map((point) => `${point.scale}<br>${point.run_name}<br>step=${point.step}/${point.final_step}`),
          hovertemplate: "%{hovertext}<br>tau=%{x:.3f}<br>loss=%{y:.5f}<extra></extra>",
        });
        seen.add(legendGroup);
      }

      const predRows = predictions.filter(
        (row) => row.mix === mix
          && lrs.includes(String(row.lr))
          && row.method === method
          && Math.abs(Number(row.prefix) - prefix) < 1e-9,
      );
      for (const lr of lrs) {
        const rows = predRows.filter((row) => String(row.lr) === lr);
        if (!rows.length) {
          continue;
        }
        traces.push({
          x: rows.map(() => 1),
          y: rows.map((row) => row.target),
          mode: "markers",
          type: "scatter",
          name: `lr${lr} observed final`,
          marker: {symbol: "diamond", size: 9, color: colors[lr]},
          hovertext: rows.map((row) => `${row.scale} ${row.target_kind}<br>${row.run_name}`),
          hovertemplate: "%{hovertext}<br>observed=%{y:.5f}<extra></extra>",
        });
        traces.push({
          x: rows.map(() => 1),
          y: rows.map((row) => row.predicted),
          mode: "markers",
          type: "scatter",
          name: `lr${lr} predicted final`,
          marker: {symbol: "x", size: 11, color: colors[lr], line: {width: 2}},
          hovertext: rows.map((row) => `${row.scale} ${row.target_kind}<br>${row.run_name}<br>error=${Number(row.error).toFixed(5)}`),
          hovertemplate: "%{hovertext}<br>predicted=%{y:.5f}<extra></extra>",
        });
        if (isCurve(method)) {
          rows.filter((row) => row.eval_split === "heldout_large" && row.param_floor != null).forEach((row) => {
            const xs = [];
            const ys = [];
            const start = Math.max(prefix, Number(row.prefix_actual_tau));
            for (let index = 0; index < 60; index += 1) {
              const tau = start + ((1 - start) * index) / 59;
              xs.push(tau);
              ys.push(curveY(row, tau));
            }
            traces.push({
              x: xs,
              y: ys,
              mode: "lines",
              type: "scatter",
              name: `lr${lr} fit`,
              showlegend: false,
              line: {color: colors[lr], dash: "dash", width: 2},
              hovertemplate: `${row.scale}<br>${row.run_name}<br>prefix-fit MAE=${Number(row.prefix_fit_mae).toFixed(5)}<br>tau=%{x:.3f}<br>fit=%{y:.5f}<extra></extra>`,
            });
          });
        }
      }
      Plotly.react("trajectory", traces, {
        height: 650,
        margin: {l: 60, r: 20, t: 40, b: 50},
        xaxis: {title: "normalized training progress", range: [0, 1.03]},
        yaxis: {title: "math_val_loss"},
        legend: {orientation: "v"},
      }, {responsive: true});
    }

    function completeCurveRows() {
      return predictions.filter((row) => row.complete && isCurve(row.method));
    }

    function statsBy(keys, rows = completeCurveRows()) {
      const groups = new Map();
      for (const row of rows) {
        const keyParts = keys.map((key) => key === "prefix_percent" ? Math.round(row.prefix * 100) : row[key]);
        const key = keyParts.join("|");
        if (!groups.has(key)) {
          groups.set(key, {rows: [], keyParts});
        }
        groups.get(key).rows.push(row);
      }
      return [...groups.values()].map((group) => {
        const errors = group.rows.map((row) => Number(row.abs_error));
        const runIds = new Set(group.rows.map((row) => row.run_id));
        const scales = new Set(group.rows.map((row) => row.scale));
        const stats = {
          mean_abs_error: errors.reduce((total, error) => total + error, 0) / errors.length,
          max_abs_error: Math.max(...errors),
          run_count: runIds.size,
          scale_count: scales.size,
        };
        keys.forEach((key, index) => {
          stats[key] = group.keyParts[index];
        });
        return stats;
      });
    }

    function renderHeatmap() {
      const stats = statsBy(["scale", "prefix_percent", "method"]);
      const prefixes = [...new Set(stats.map((row) => row.prefix_percent))].sort((left, right) => left - right);
      const z = [];
      const text = [];
      const custom = [];
      for (const prefix of prefixes) {
        const zRow = [];
        const textRow = [];
        const customRow = [];
        for (const scale of scaleOrder) {
          const options = stats
            .filter((row) => row.scale === scale && row.prefix_percent === prefix)
            .sort((left, right) => left.mean_abs_error - right.mean_abs_error);
          if (!options.length) {
            zRow.push(null);
            textRow.push("");
            customRow.push(["", null, null]);
            continue;
          }
          const best = options[0];
          zRow.push(best.mean_abs_error);
          textRow.push(`${methodLabel(best.method)}<br>${best.mean_abs_error.toFixed(4)}`);
          customRow.push([methodLabel(best.method), best.mean_abs_error, best.max_abs_error]);
        }
        z.push(zRow);
        text.push(textRow);
        custom.push(customRow);
      }
      Plotly.react("bestHeatmap", [{
        type: "heatmap",
        x: scaleOrder,
        y: prefixes.map((prefix) => `${prefix}%`),
        z,
        text,
        customdata: custom,
        texttemplate: "%{text}",
        colorscale: "Viridis",
        colorbar: {title: "mean MAE"},
        hovertemplate: "scale=%{x}<br>prefix=%{y}<br>method=%{customdata[0]}<br>mean MAE=%{customdata[1]:.5f}<br>max error=%{customdata[2]:.5f}<extra></extra>",
      }], {
        height: 900,
        margin: {l: 70, r: 30, t: 30, b: 50},
        xaxis: {title: "isoflop scale"},
        yaxis: {title: "prefix"},
      }, {responsive: true});
    }

    function tableHtml(rows, columns) {
      if (!rows.length) {
        return '<p class="note"><em>No config satisfies this target for every completed run in this scope.</em></p>';
      }
      return `<table><thead><tr>${columns.map((column) => `<th>${column.label}</th>`).join("")}</tr></thead><tbody>${
        rows.map((row) => `<tr>${columns.map((column) => `<td>${column.format ? column.format(row[column.key], row) : row[column.key]}</td>`).join("")}</tr>`).join("")
      }</tbody></table>`;
    }

    function targetRows() {
      const mix = document.getElementById("targetMix").value;
      const lr = document.getElementById("targetLr").value;
      return completeCurveRows().filter(
        (row) => (mix === "all" || row.mix === mix) && (lr === "all" || String(row.lr) === lr),
      );
    }

    function renderTargetTables() {
      const target = Number(document.getElementById("target").value);
      const rows = targetRows();
      const expectedRuns = new Set(rows.map((row) => row.run_id)).size;
      const expectedScales = new Set(rows.map((row) => row.scale)).size;
      const targetMix = document.getElementById("targetMix").value;
      const targetLr = document.getElementById("targetLr").value;
      const scopeLabel = `${targetMix === "all" ? "all mixes" : targetMix} / ${targetLr === "all" ? "all learning rates" : `lr${targetLr}`}`;
      document.getElementById("targetScopeSummary").textContent =
        `${scopeLabel}: ${expectedRuns} completed runs across ${expectedScales} scales are checked against max abs error <= ${target}.`;

      const selected = statsBy(["prefix_percent", "method"], rows)
        .filter((stat) => stat.run_count === expectedRuns && stat.max_abs_error <= target)
        .sort((left, right) => left.prefix_percent - right.prefix_percent
          || left.max_abs_error - right.max_abs_error
          || left.mean_abs_error - right.mean_abs_error)
        .slice(0, 12);

      const expectedByScale = Object.fromEntries(scaleOrder.map((scale) => [
        scale,
        new Set(rows.filter((row) => row.scale === scale).map((row) => row.run_id)).size,
      ]));
      const perScale = statsBy(["scale", "prefix_percent", "method"], rows)
        .filter((stat) => expectedByScale[stat.scale] > 0
          && stat.run_count === expectedByScale[stat.scale]
          && stat.max_abs_error <= target)
        .sort((left, right) => scaleOrder.indexOf(left.scale) - scaleOrder.indexOf(right.scale)
          || left.prefix_percent - right.prefix_percent
          || left.max_abs_error - right.max_abs_error
          || left.mean_abs_error - right.mean_abs_error)
        .filter((stat, index, all) => all.filter((other) => other.scale === stat.scale).indexOf(stat) < 5);

      const overallRows = completeCurveRows();
      const expectedOverall = new Set(overallRows.map((row) => row.run_id)).size;
      const globalOverall = statsBy(["prefix_percent", "method"], overallRows)
        .filter((stat) => stat.run_count === expectedOverall && stat.max_abs_error <= target)
        .sort((left, right) => left.prefix_percent - right.prefix_percent
          || left.max_abs_error - right.max_abs_error
          || left.mean_abs_error - right.mean_abs_error)
        .slice(0, 12);
      const expectedGlobalByScale = Object.fromEntries(scaleOrder.map((scale) => [
        scale,
        new Set(overallRows.filter((row) => row.scale === scale).map((row) => row.run_id)).size,
      ]));
      const globalPerScale = statsBy(["scale", "prefix_percent", "method"], overallRows)
        .filter((stat) => stat.run_count === expectedGlobalByScale[stat.scale] && stat.max_abs_error <= target)
        .sort((left, right) => scaleOrder.indexOf(left.scale) - scaleOrder.indexOf(right.scale)
          || left.prefix_percent - right.prefix_percent
          || left.max_abs_error - right.max_abs_error)
        .filter((stat, index, all) => all.filter((other) => other.scale === stat.scale).indexOf(stat) < 5);

      const selectedColumns = [
        {key: "prefix_percent", label: "prefix %"},
        {key: "method", label: "method", format: (value) => methodLabel(value)},
        {key: "max_abs_error", label: "max abs error", format: (value) => value.toFixed(5)},
        {key: "mean_abs_error", label: "mean abs error", format: (value) => value.toFixed(5)},
        {key: "run_count", label: "runs checked"},
        {key: "scale_count", label: "scales"},
      ];
      const scaleColumns = [
        {key: "scale", label: "scale"},
        {key: "prefix_percent", label: "prefix %"},
        {key: "method", label: "method", format: (value) => methodLabel(value)},
        {key: "max_abs_error", label: "max abs error", format: (value) => value.toFixed(5)},
        {key: "mean_abs_error", label: "mean abs error", format: (value) => value.toFixed(5)},
        {key: "run_count", label: "runs checked"},
      ];
      document.getElementById("regimeTable").innerHTML = tableHtml(selected, selectedColumns);
      document.getElementById("scaleTable").innerHTML = tableHtml(perScale, scaleColumns);
      document.getElementById("overallTable").innerHTML = tableHtml(globalOverall, selectedColumns);
      document.getElementById("globalScaleTable").innerHTML = tableHtml(globalPerScale, scaleColumns);
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

    function jointRows() {
      const mix = document.getElementById("jointMix").value;
      const lr = document.getElementById("jointLr").value;
      return jointPredictions.filter(
        (row) => row.complete && (mix === "all" || row.mix === mix) && (lr === "all" || String(row.lr) === lr),
      );
    }

    function jointStats(keys, rows = jointRows()) {
      const groups = new Map();
      for (const row of rows) {
        const keyParts = keys.map((key) => key === "prefix_percent" ? Math.round(row.prefix * 100) : row[key]);
        const key = keyParts.join("|");
        if (!groups.has(key)) {
          groups.set(key, {rows: [], keyParts});
        }
        groups.get(key).rows.push(row);
      }
      return [...groups.values()].map((group) => {
        const errors = group.rows.map((row) => Number(row.abs_error));
        const runIds = new Set(group.rows.map((row) => row.run_id));
        const scales = new Set(group.rows.map((row) => row.scale));
        const labels = new Set(group.rows.map((row) => row.method_label));
        const stats = {
          mean_abs_error: errors.reduce((total, error) => total + error, 0) / errors.length,
          max_abs_error: Math.max(...errors),
          run_count: runIds.size,
          scale_count: scales.size,
          method_label: [...labels][0],
        };
        keys.forEach((key, index) => {
          stats[key] = group.keyParts[index];
        });
        return stats;
      });
    }

    function jointMixMidFraction(mix) {
      if (mix === "p33m67") {
        return 0.67;
      }
      if (mix === "p50m50") {
        return 0.50;
      }
      return 0.33;
    }

    function jointStaticFeatures(row, includeFlops) {
      const mid = jointMixMidFraction(row.mix) - 0.50;
      const lr = Number(row.lr) / 100 - 0.58;
      const values = [1, mid, lr, mid * lr];
      if (includeFlops) {
        const logFlops = Math.log10(scaleFlops[row.scale]) - Math.log10(3e20);
        values.push(logFlops, logFlops * mid, logFlops * lr);
      }
      return values;
    }

    function jointBases(form, tau, theta) {
      if (form === "exp_drift") {
        const rate = Math.exp(theta[0]);
        return [Math.exp(-rate * tau), tau];
      }
      if (form === "power_drift") {
        const shift = Math.exp(theta[0]);
        const exponent = Math.exp(theta[1]);
        return [Math.pow(tau + shift, -exponent), tau];
      }
      if (form === "gompertz_shoulder") {
        const amplitude = Math.exp(theta[0]);
        const rate = Math.exp(theta[1]);
        return [Math.exp(-amplitude * Math.exp(-rate * tau)), tau];
      }
      return [0, tau];
    }

    function jointPredict(model, row, tau) {
      const includeFlops = model.include_flops === true || model.include_flops === "True" || model.include_flops === "true";
      const z = jointStaticFeatures(row, includeFlops);
      const [basis1, basis2] = jointBases(model.form, tau, model.theta);
      const design = [...z, ...z.map((value) => value * basis1), ...z.map((value) => value * basis2)];
      return design.reduce((total, value, index) => total + value * model.beta[index], 0);
    }

    function jointModelFor(row, method, prefix) {
      return jointModels.find((model) => model.method === method
        && Math.abs(Number(model.prefix) - prefix) < 1e-9
        && (model.scale === "all" || model.scale === row.scale));
    }

    function renderJointSection() {
      if (!jointPredictions.length) {
        document.getElementById("jointScopeSummary").textContent = "Run delphi_joint_trajectory_prediction.py to generate joint fits.";
        return;
      }
      const rows = jointRows();
      const target = Number(document.getElementById("jointTarget").value);
      const prefix = Number(document.getElementById("jointPrefix").value) / 100;
      document.getElementById("jointPrefixValue").textContent = `${Math.round(prefix * 100)}%`;
      const method = document.getElementById("jointMethod").value;
      const expectedRuns = new Set(rows.map((row) => row.run_id)).size;
      const expectedScales = new Set(rows.map((row) => row.scale)).size;
      const mix = document.getElementById("jointMix").value;
      const lr = document.getElementById("jointLr").value;
      document.getElementById("jointScopeSummary").textContent =
        `${mix === "all" ? "all mixes" : mix} / ${lr === "all" ? "all learning rates" : `lr${lr}`}: ${expectedRuns} completed runs across ${expectedScales} scales.`;

      const methodStats = jointStats(["prefix_percent", "method"], rows)
        .filter((stat) => stat.run_count === expectedRuns)
        .sort((left, right) => left.prefix_percent - right.prefix_percent);
      const methods = [...new Set(methodStats.map((stat) => stat.method))];
      const traces = methods.map((methodName) => {
        const methodRows = methodStats.filter((stat) => stat.method === methodName);
        return {
          type: "scatter",
          mode: "lines+markers",
          x: methodRows.map((stat) => stat.prefix_percent),
          y: methodRows.map((stat) => stat.mean_abs_error),
          customdata: methodRows.map((stat) => [stat.max_abs_error, stat.method_label]),
          name: methodRows[0]?.method_label || methodName,
          hovertemplate: "prefix=%{x}%<br>%{customdata[1]}<br>mean MAE=%{y:.5f}<br>max error=%{customdata[0]:.5f}<extra></extra>",
        };
      });
      Plotly.react("jointTradeoff", traces, {
        height: 520,
        margin: {l: 60, r: 20, t: 30, b: 50},
        xaxis: {title: "prefix %"},
        yaxis: {title: "endpoint MAE"},
        legend: {orientation: "v"},
      }, {responsive: true});

      const observedGroups = new Map();
      points.filter((point) => (mix === "all" || point.mix === mix) && (lr === "all" || String(point.lr) === lr)).forEach((point) => {
        if (!observedGroups.has(point.run_id)) {
          observedGroups.set(point.run_id, []);
        }
        observedGroups.get(point.run_id).push(point);
      });
      const curveTraces = [];
      const curveSeen = new Set();
      for (const group of observedGroups.values()) {
        group.sort((left, right) => left.tau - right.tau);
        const first = group[0];
        const legendGroup = `observed-${first.lr}`;
        curveTraces.push({
          x: group.map((point) => point.tau),
          y: group.map((point) => point.value),
          type: "scatter",
          mode: "lines",
          name: `observed lr${first.lr}`,
          legendgroup: legendGroup,
          showlegend: !curveSeen.has(legendGroup),
          line: {color: colors[first.lr] || "#666", width: 1, dash: first.eval_split === "heldout_large" ? "solid" : "dot"},
          opacity: first.eval_split === "heldout_large" ? 0.55 : 0.22,
          hovertext: group.map((point) => `${point.scale} ${point.mix} lr${point.lr}<br>${point.run_name}`),
          hovertemplate: "%{hovertext}<br>tau=%{x:.3f}<br>observed=%{y:.5f}<extra></extra>",
        });
        curveSeen.add(legendGroup);
      }
      for (const group of observedGroups.values()) {
        const first = group[0];
        const model = jointModelFor(first, method, prefix);
        if (!model) {
          continue;
        }
        const xs = [];
        const ys = [];
        for (let index = 0; index <= 40; index += 1) {
          const tau = index / 40;
          xs.push(tau);
          ys.push(jointPredict(model, first, tau));
        }
        const legendGroup = `fit-${first.lr}`;
        curveTraces.push({
          x: xs,
          y: ys,
          type: "scatter",
          mode: "lines",
          name: `joint fit lr${first.lr}`,
          legendgroup: legendGroup,
          showlegend: !curveSeen.has(legendGroup),
          line: {color: colors[first.lr] || "#666", width: 2},
          opacity: 0.85,
          hovertext: xs.map(() => `${first.scale} ${first.mix} lr${first.lr}<br>${first.run_name}<br>${model.method_label}`),
          hovertemplate: "%{hovertext}<br>tau=%{x:.3f}<br>fit=%{y:.5f}<extra></extra>",
        });
        curveSeen.add(legendGroup);
      }
      Plotly.react("jointCurves", curveTraces, {
        height: 620,
        margin: {l: 60, r: 20, t: 30, b: 50},
        xaxis: {title: "normalized training progress", range: [0, 1.03]},
        yaxis: {title: "math_val_loss"},
        legend: {orientation: "v"},
      }, {responsive: true});

      const scatterRows = rows.filter((row) => row.method === method && Math.abs(Number(row.prefix) - prefix) < 1e-9);
      const scatterTraces = lrOrder.map((lrValue) => {
        const lrRows = scatterRows.filter((row) => String(row.lr) === lrValue);
        return {
          type: "scatter",
          mode: "markers",
          x: lrRows.map((row) => row.target),
          y: lrRows.map((row) => row.predicted),
          name: `lr${lrValue}`,
          marker: {color: colors[lrValue], size: 9},
          text: lrRows.map((row) => `${row.scale} ${row.mix}<br>${row.run_name}<br>abs error=${Number(row.abs_error).toFixed(5)}`),
          hovertemplate: "%{text}<br>observed=%{x:.5f}<br>predicted=%{y:.5f}<extra></extra>",
        };
      }).filter((trace) => trace.x.length);
      const values = scatterRows.flatMap((row) => [Number(row.target), Number(row.predicted)]);
      const low = Math.min(...values);
      const high = Math.max(...values);
      if (Number.isFinite(low) && Number.isFinite(high)) {
        scatterTraces.push({
          type: "scatter",
          mode: "lines",
          x: [low, high],
          y: [low, high],
          name: "perfect",
          line: {color: "#111827", dash: "dot"},
          hoverinfo: "skip",
        });
      }
      Plotly.react("jointScatter", scatterTraces, {
        height: 520,
        margin: {l: 60, r: 20, t: 30, b: 50},
        xaxis: {title: "observed final math loss"},
        yaxis: {title: "predicted final math loss"},
      }, {responsive: true});

      const qualifying = methodStats
        .filter((stat) => stat.run_count === expectedRuns && stat.max_abs_error <= target)
        .sort((left, right) => left.prefix_percent - right.prefix_percent
          || left.max_abs_error - right.max_abs_error
          || left.mean_abs_error - right.mean_abs_error)
        .slice(0, 12);
      const expectedByScale = Object.fromEntries(scaleOrder.map((scale) => [
        scale,
        new Set(rows.filter((row) => row.scale === scale).map((row) => row.run_id)).size,
      ]));
      const byScale = jointStats(["scale", "prefix_percent", "method"], rows)
        .filter((stat) => expectedByScale[stat.scale] > 0
          && stat.run_count === expectedByScale[stat.scale]
          && stat.max_abs_error <= target)
        .sort((left, right) => scaleOrder.indexOf(left.scale) - scaleOrder.indexOf(right.scale)
          || left.prefix_percent - right.prefix_percent
          || left.max_abs_error - right.max_abs_error)
        .filter((stat, index, all) => all.filter((other) => other.scale === stat.scale).indexOf(stat) < 5);
      const configColumns = [
        {key: "prefix_percent", label: "prefix %"},
        {key: "method_label", label: "joint method"},
        {key: "max_abs_error", label: "max abs error", format: (value) => value.toFixed(5)},
        {key: "mean_abs_error", label: "mean abs error", format: (value) => value.toFixed(5)},
        {key: "run_count", label: "runs checked"},
        {key: "scale_count", label: "scales"},
      ];
      const scaleColumns = [
        {key: "scale", label: "scale"},
        {key: "prefix_percent", label: "prefix %"},
        {key: "method_label", label: "joint method"},
        {key: "max_abs_error", label: "max abs error", format: (value) => value.toFixed(5)},
        {key: "mean_abs_error", label: "mean abs error", format: (value) => value.toFixed(5)},
        {key: "run_count", label: "runs checked"},
      ];
      document.getElementById("jointConfigTable").innerHTML = tableHtml(qualifying, configColumns);
      document.getElementById("jointScaleTable").innerHTML = tableHtml(byScale, scaleColumns);
    }

    function renderAll() {
      renderTrajectory();
      renderHeatmap();
      renderTargetTables();
      renderScalingSection();
      renderJointSection();
      if (window.MathJax && window.MathJax.typesetPromise) {
        window.MathJax.typesetPromise();
      }
    }

    renderControls();
    renderAll();
  </script>
</body>
</html>
"""


def build_html(data: dict[str, Any]) -> str:
    payload_json = json.dumps(data, allow_nan=False, separators=(",", ":"))
    return HTML_TEMPLATE.replace("__PAYLOAD_JSON__", payload_json)


def main() -> None:
    args = parse_args()
    data = payload(args.points, args.predictions)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_html(data), encoding="utf-8")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
