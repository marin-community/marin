# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: E501

"""Endpoint scaling fit for the Delphi iso-token (token-matched) CPT ladder.

Mirrors the "Endpoint Scaling Law (Compute Vs Final Loss)" section of the published
midtraining report (``build_delphi_midtraining_interactive_report.py``) — the same
Chinchilla-style floor+power fit ``L(C) = E + A * (C/1e18)^-alpha`` and the 2-param
log-log fit — but on the iso-token sweep (W&B tag ``sweep:delphi-cpt-isotoken``),
where every base 3e18→1e22 is midtrained on the same fixed math-mix token budget,
so model scale N is not confounded with data budget D as it is on the iso-FLOP ladder.

Fits use the small ladder (3e18→3e20, with a fit-through control), and 1e21/1e22 are
held out as the extrapolation check, matching the published report exactly.

For direct comparison the report overlays the iso-FLOP K=0.20 ladder (same mix/LR
recipes as the iso-token runs): the small-ladder ``k0p20`` cells plus the 1e21/1e22
K=0.20 cells (``9p25b``/``32p07b`` run names), fit the same way. ALL of these are CPT
from the cooled end of pretraining — the small ladder initializes from the released HF
models, the 1e21/1e22 cells MODEL_ONLY (fresh optimizer + warmup) from the final,
>=98%-LR-decayed checkpoint of the finished base pretrain run (verified in the W&B
configs; the logbook claim that these were "cooldown resumed mid-WSD" is wrong). On the
iso-FLOP ladder the midtrain token budget D grows with scale (0.24B → 32B), which is
the confound the iso-token ladder removes.

W&B run data is cached incrementally under ``midtrain_wandb_data/`` (gitignored),
using the same per-run schema as ``download_midtrain_wandb.py`` (metadata.json,
config.json, summary.json, history.jsonl, history_meta.json; run files are skipped).
Finished runs are immutable and never refetched.

Run:
    uv run python scripts/analysis/delphi_isotoken_endpoint_scaling.py
    uv run python scripts/analysis/delphi_isotoken_endpoint_scaling.py --use-cache
    uv run python scripts/analysis/delphi_isotoken_endpoint_scaling.py --no-history
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from pathlib import Path
from typing import Any

import pandas as pd
import wandb
from build_delphi_midtraining_interactive_report import finite_or_none, fit_floor_power, fit_log_linear
from delphi_small_final_loss_scaling import MIDTRAIN_BUDGET_FRACTION, SCALE_PRETRAIN_TOKENS_B
from plotly.offline import get_plotlyjs

logger = logging.getLogger("delphi_isotoken_endpoint_scaling")

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "midtrain_wandb_data"
RUNS_DIR = DATA_DIR / "runs"
OUT_DIR = REPO_ROOT / "sk_midtrain_analysis_fable"
DEFAULT_OUTPUT_PATH = OUT_DIR / "delphi_isotoken_endpoint_scaling.html"

ENTITY = "marin-community"
PROJECT = "delphi-midtraining"
PROJECT_PATH = f"{ENTITY}/{PROJECT}"
SWEEP_TAG = "sweep:delphi-cpt-isotoken"

METRIC_KEY = "eval/nemotron_cc_math_v1/4plus/loss"
METRIC_LABEL = "math_val_loss"

RUN_PATTERN = re.compile(
    r"^delphi-(?P<scale>3e18|9e18|2e19|3e19|9e19|2e20|3e20|1e21|1e22)-"
    r"(?P<mix>p\d{2}m\d{2})-tok(?P<budget>\d+[bm])-lr(?P<lr>\d+)-a(?P<attempt>\d{3})$"
)

# Iso-FLOP K=0.20 overlay. The 3e18-3e20 cells are k0p20-named CPT runs from the released
# HF models; the 1e21/1e22 cells carry the absolute budget (9.25B/32.07B tokens) and a
# decimal LR in the name, and are likewise CPT — initialized MODEL_ONLY (fresh optimizer +
# warmup) from the final, ~fully-LR-decayed checkpoint of the finished pretrain run.
ISOFLOP_SERIES = "k0p20"
ISOFLOP_RUN_PATTERN = re.compile(
    r"^delphi-(?P<scale>3e18|9e18|2e19|3e19|9e19|2e20|3e20)-"
    r"(?P<mix>p\d{2}m\d{2})-k0p20-lr(?P<lr>\d+)-a(?P<attempt>\d{3})$"
)
ISOFLOP_LARGE_RUN_PATTERN = re.compile(
    r"^delphi-(?P<scale>1e21|1e22)-(?P<mix>p\d{2}m\d{2})-(?:9p25b|32p07b)-"
    r"lr(?P<lr_decimal>0\.\d+)-(?P<suffix>[a-z0-9]+)$"
)
LR_DECIMAL = {"33": "0.33", "50": "0.5", "67": "0.67", "83": "0.83"}
LR_FROM_DECIMAL = {decimal: lr for lr, decimal in LR_DECIMAL.items()}
ISOFLOP_LARGE_EXPECTED_FINAL_STEP = {"1e21": 4410, "1e22": 7646}

ALL_SCALE_FLOPS = {
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
SCALE_ORDER = tuple(ALL_SCALE_FLOPS)
SMALL_LADDER_SCALES = SCALE_ORDER[:7]
HELD_OUT_SCALES = SCALE_ORDER[7:]
# Like the published report: the fit-through control can extend past the small ladder to
# pull 1e21 into the training pool; 1e22 is always held out.
CUTOFF_SCALES = (*SMALL_LADDER_SCALES, "1e21")
DEFAULT_CUTOFF_SCALE = "3e20"

# Per-scale global batch size of the canonical Delphi bases (sequences of 4096 tokens);
# fixed-token budgets resolve to round(tokens / (batch * 4096)) steps (resolve_cpt_budget).
SCALE_BATCH_SIZE = {
    "3e18": 8,
    "9e18": 16,
    "2e19": 16,
    "3e19": 32,
    "9e19": 64,
    "2e20": 64,
    "3e20": 128,
    "1e21": 512,
    "1e22": 1024,
}
SEQ_LEN = 4096
COMPLETION_TOLERANCE = 5

BUDGET_LABEL_PATTERN = re.compile(r"^(?P<count>\d+)(?P<unit>[bm])$")
BUDGET_UNIT_TOKENS = {"b": 1_000_000_000, "m": 1_000_000}


def budget_tokens(budget_label: str) -> int:
    match = BUDGET_LABEL_PATTERN.match(budget_label)
    if match is None:
        raise ValueError(f"Unparseable budget label: {budget_label!r}")
    return int(match.group("count")) * BUDGET_UNIT_TOKENS[match.group("unit")]


def expected_final_step(scale: str, budget_label: str) -> int:
    steps = round(budget_tokens(budget_label) / (SCALE_BATCH_SIZE[scale] * SEQ_LEN))
    return steps - 1  # last logged step index


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--use-cache", action="store_true", help="Skip W&B entirely; read midtrain_wandb_data/ only")
    parser.add_argument("--no-history", action="store_true", help="Skip history.jsonl download (summaries only)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    return parser.parse_args()


def run_cache_dir(run_id: str) -> Path:
    return RUNS_DIR / run_id


def cached_state(run_id: str) -> str | None:
    metadata_path = run_cache_dir(run_id) / "metadata.json"
    if not metadata_path.exists():
        return None
    return json.loads(metadata_path.read_text()).get("state")


def export_run(run: Any, with_history: bool) -> None:
    """Write one run into the midtrain_wandb_data/runs/<id>/ schema."""
    out = run_cache_dir(run.id)
    out.mkdir(parents=True, exist_ok=True)
    metadata = {
        "id": run.id,
        "name": run.name,
        "state": run.state,
        "created_at": str(run.created_at),
        "url": run.url,
        "tags": list(run.tags),
        "entity": run.entity,
        "project": run.project,
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))
    (out / "config.json").write_text(json.dumps(dict(run.config), indent=2, default=str))
    (out / "summary.json").write_text(json.dumps(run.summary._json_dict, indent=2, default=str))
    if with_history:
        rows = 0
        keys: set[str] = set()
        with (out / "history.jsonl").open("w") as fh:
            for row in run.scan_history():
                fh.write(json.dumps(row, default=str) + "\n")
                keys.update(row)
                rows += 1
        (out / "history_meta.json").write_text(json.dumps({"row_count": rows, "keys": sorted(keys)}, indent=2))
    logger.info("exported %s (state=%s, history=%s)", run.name, run.state, with_history)


def refresh_cache(with_history: bool) -> None:
    api = wandb.Api(timeout=60)
    runs = api.runs(PROJECT_PATH, filters={"tags": SWEEP_TAG})
    for run in runs:
        if RUN_PATTERN.match(run.name) is None:
            logger.warning("skipping unrecognized iso-token run name: %s", run.name)
            continue
        if cached_state(run.id) == "finished":
            logger.info("cache hit (finished): %s", run.name)
            continue
        export_run(run, with_history=with_history)


def load_endpoints() -> pd.DataFrame:
    """Best finished, step-complete attempt per (scale, mix, budget, lr) from the cache."""
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(RUNS_DIR.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        match = RUN_PATTERN.match(metadata["name"])
        if match is None or SWEEP_TAG not in metadata.get("tags", []):
            continue
        if metadata["state"] != "finished":
            continue
        summary = json.loads((metadata_path.parent / "summary.json").read_text())
        cell = match.groupdict()
        step = summary.get("_step")
        expected = expected_final_step(cell["scale"], cell["budget"])
        if step is None or step < expected - COMPLETION_TOLERANCE:
            logger.warning("incomplete finished run %s: _step=%s expected=%s", metadata["name"], step, expected)
            continue
        value = summary.get(METRIC_KEY)
        if value is None:
            raise ValueError(f"Run {metadata['name']} is finished but missing {METRIC_KEY}")
        rows.append(
            {
                "run_id": metadata["id"],
                "run_name": metadata["name"],
                "scale": cell["scale"],
                "scale_flops": ALL_SCALE_FLOPS[cell["scale"]],
                "mix": cell["mix"],
                "budget": cell["budget"],
                "budget_tokens": budget_tokens(cell["budget"]),
                "series": f"tok{cell['budget']}",
                "tokens_b": budget_tokens(cell["budget"]) / 1e9,
                "lr": cell["lr"],
                "attempt": int(cell["attempt"]),
                "final_step": int(step),
                "value": float(value),
            }
        )
    if not rows:
        raise ValueError(f"No complete iso-token runs found under {RUNS_DIR}; run without --use-cache first")
    frame = pd.DataFrame(rows)
    frame = frame.sort_values("attempt").groupby(["scale", "mix", "budget", "lr"], as_index=False).last()
    return frame.sort_values(["budget_tokens", "scale_flops"]).reset_index(drop=True)


def refresh_isoflop_cache(recipes: pd.DataFrame, with_history: bool) -> None:
    """Fetch the iso-FLOP K=0.20 cells matching the iso-token (mix, lr) recipes."""
    api = wandb.Api(timeout=60)
    mixes = "|".join(sorted(recipes["mix"].unique()))
    lrs = sorted(recipes["lr"].unique())
    small_regex = rf"^delphi-({'|'.join(SMALL_LADDER_SCALES)})-({mixes})-k0p20-lr({'|'.join(lrs)})-a\d{{3}}$"
    decimals = "|".join(re.escape(LR_DECIMAL[lr]) for lr in lrs)
    large_regex = rf"^delphi-(1e21|1e22)-({mixes})-(9p25b|32p07b)-lr({decimals})-[a-z0-9]+$"
    for regex in (small_regex, large_regex):
        for run in api.runs(PROJECT_PATH, filters={"display_name": {"$regex": regex}}):
            if cached_state(run.id) == "finished":
                logger.info("cache hit (finished): %s", run.name)
                continue
            export_run(run, with_history=with_history)


def load_isoflop_endpoints() -> pd.DataFrame:
    """K=0.20 endpoints from the cache: best finished attempt per cell, 1e21/1e22 step-checked."""
    rows: list[dict[str, Any]] = []
    for metadata_path in sorted(RUNS_DIR.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text())
        name = metadata["name"]
        small = ISOFLOP_RUN_PATTERN.match(name)
        large = ISOFLOP_LARGE_RUN_PATTERN.match(name)
        if (small is None and large is None) or metadata["state"] != "finished":
            continue
        cell = (small or large).groupdict()
        if large is not None:
            cell["lr"] = LR_FROM_DECIMAL[cell.pop("lr_decimal")]
        summary = json.loads((metadata_path.parent / "summary.json").read_text())
        step = summary.get("_step")
        if large is not None:
            expected = ISOFLOP_LARGE_EXPECTED_FINAL_STEP[cell["scale"]]
            if step is None or step < expected - COMPLETION_TOLERANCE:
                logger.warning("incomplete 1e21/1e22 run %s: _step=%s expected=%s", name, step, expected)
                continue
        value = summary.get(METRIC_KEY)
        if value is None:
            logger.warning("skipping %s: missing %s", name, METRIC_KEY)
            continue
        rows.append(
            {
                "run_id": metadata["id"],
                "run_name": name,
                "scale": cell["scale"],
                "scale_flops": ALL_SCALE_FLOPS[cell["scale"]],
                "mix": cell["mix"],
                "budget": ISOFLOP_SERIES,
                "budget_tokens": -1,  # varies per scale; sort iso-FLOP first
                "series": ISOFLOP_SERIES,
                "tokens_b": round(MIDTRAIN_BUDGET_FRACTION * SCALE_PRETRAIN_TOKENS_B[cell["scale"]], 2),
                "lr": cell["lr"],
                "attempt": int(cell.get("attempt", 0)),
                "final_step": int(step),
                "value": float(value),
            }
        )
    if not rows:
        raise ValueError(f"No complete iso-FLOP K=0.20 runs found under {RUNS_DIR}; run without --use-cache first")
    frame = pd.DataFrame(rows)
    # Best attempt per small-ladder cell; latest (max step) per 1e21/1e22 cell.
    frame = frame.sort_values(["attempt", "final_step"]).groupby(["scale", "mix", "lr"], as_index=False).last()
    return frame.sort_values("scale_flops").reset_index(drop=True)


def compute_scaling_fits(endpoints: pd.DataFrame) -> list[dict[str, Any]]:
    """Per-(mix, series, lr) floor+power and log-log fits at each fit-through cutoff."""
    scaling_fits: list[dict[str, Any]] = []
    fit_pool = endpoints[endpoints["scale"].isin(CUTOFF_SCALES)]
    for (mix, series, lr), group in fit_pool.groupby(["mix", "series", "lr"], sort=False):
        group = group.sort_values("scale_flops")
        for cutoff_index, cutoff_scale in enumerate(CUTOFF_SCALES):
            included = group[group["scale_flops"] <= ALL_SCALE_FLOPS[cutoff_scale] + 1.0]
            if included.empty:
                continue
            xs = included["scale_flops"].to_numpy(dtype=float)
            ys = included["value"].to_numpy(dtype=float)
            row: dict[str, Any] = {
                "mix": mix,
                "series": series,
                "lr": str(lr),
                "cutoff_index": cutoff_index,
                "cutoff_scale": cutoff_scale,
                "n": int(xs.size),
                "min_scale": str(included["scale"].iloc[0]),
                "max_scale": str(included["scale"].iloc[-1]),
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
            scaling_fits.append(row)
    return scaling_fits


def build_payload(endpoints: pd.DataFrame, scaling_fits: list[dict[str, Any]]) -> dict[str, Any]:
    columns = ["run_id", "run_name", "scale", "scale_flops", "mix", "series", "tokens_b", "lr", "final_step", "value"]
    small = endpoints[endpoints["scale"].isin(SMALL_LADDER_SCALES)][columns]
    held = endpoints[endpoints["scale"].isin(HELD_OUT_SCALES)][columns]
    isotoken = endpoints[endpoints["series"].ne(ISOFLOP_SERIES)]
    budgets = sorted(isotoken["budget"].unique(), key=budget_tokens)
    return {
        "metricLabel": METRIC_LABEL,
        "endpoints": [{k: finite_or_none(v) for k, v in row.items()} for row in small.to_dict("records")],
        "heldoutTargets": [{k: finite_or_none(v) for k, v in row.items()} for row in held.to_dict("records")],
        "scalingFits": [{k: finite_or_none(v) for k, v in row.items()} for row in scaling_fits],
        "scaleOrder": list(SCALE_ORDER),
        "smallLadderScales": list(SMALL_LADDER_SCALES),
        "heldOutScales": list(HELD_OUT_SCALES),
        "cutoffScales": list(CUTOFF_SCALES),
        "defaultCutoffIndex": CUTOFF_SCALES.index(DEFAULT_CUTOFF_SCALE),
        "scaleFlops": ALL_SCALE_FLOPS,
        "budgets": budgets,
        "isoflopSeries": ISOFLOP_SERIES,
    }


HTML_TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Delphi Iso-Token Endpoint Scaling</title>
  <!-- Plotly is inlined so the page is fully self-contained (works offline / in sandboxed previews). -->
  <script>__PLOTLYJS__</script>
  <style>
    :root { color-scheme: light; --ink: #172033; --muted: #667085; --line: #e5e7eb; --bg: #ffffff; --soft: #f8fafc; }
    * { box-sizing: border-box; }
    body { margin: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; color: var(--ink); background: var(--bg); }
    header { padding: 24px 32px 14px; border-bottom: 1px solid var(--line); }
    h1 { margin: 0 0 8px; font-size: 30px; }
    h2 { margin: 30px 0 10px; font-size: 22px; }
    h3 { margin: 0 0 12px; font-size: 18px; }
    p { color: var(--muted); line-height: 1.45; }
    main { padding: 18px 32px 42px; max-width: 1540px; }
    .target-controls { display: flex; flex-wrap: wrap; gap: 16px 22px; align-items: center; padding: 14px 16px; background: var(--soft); border: 1px solid var(--line); border-radius: 8px; margin: 14px 0 20px; }
    label { font-size: 13px; color: var(--muted); display: flex; flex-direction: column; gap: 5px; }
    select, input[type=range] { min-width: 150px; }
    .value { color: var(--ink); font-weight: 650; font-variant-numeric: tabular-nums; }
    .note { font-size: 13px; }
    .chart { width: 100%; height: 650px; }
    table { border-collapse: collapse; margin: 10px 0 26px; min-width: 520px; }
    th, td { border: 1px solid var(--line); padding: 8px 10px; text-align: left; font-variant-numeric: tabular-nums; }
    th { background: #f3f4f6; }
    .tables-stacked { display: grid; gap: 8px; align-items: start; }
  </style>
</head>
<body>
  <header>
    <h1>Delphi Iso-Token Midtraining — Endpoint Scaling Law</h1>
    <p>
      Token-matched CPT ladder (debug-midtrain exp #7): every base 3e18 → 1e22 is midtrained on the
      <em>same</em> fixed math-mix token budget (mix p33m67, LR factor 0.5), so model scale N is not
      confounded with data budget D the way it is on the iso-FLOP ladder. Same fit and framing as the
      published iso-FLOP report: Chinchilla-style 3-parameter
      <code>L(C) = E + A*(C/1e18)^(-alpha)</code> on final <code>math_val_loss</code>, fit on the small
      ladder (3e18 → 3e20) by <code>scipy.optimize.curve_fit</code> with E &lt; min y and A, alpha ≥ 0;
      1e21 and 1e22 actuals are plotted as triangles for the extrapolation check. The 2-parameter log-log fit
      <code>log L = a + b·log C</code> is available as a toggle.
    </p>
    <p class="note">
      Reference point — on the iso-FLOP (K=0.20) ladder the same fit-through-3e20 procedure over-predicts the
      1e22 actual by +15.8% (Chinchilla) / +10.7% (log-log), mean abs over recipes: "the 1e22 miss". This page
      asks whether that miss survives when the token budget is held fixed.
    </p>
  </header>
  <main>
    <section>
      <h2>Endpoint Scaling (Compute Vs Final Loss)</h2>
      <div class="target-controls">
        <label>fit type <select id="fitType">
          <option value="floor_power" selected>floor + power (Chinchilla)</option>
          <option value="log_linear">log-log linear (2-param)</option>
        </select></label>
        <label>token budget <select id="budget"></select></label>
        <label>fit through
          <input id="cutoff" type="range" min="0" max="7" step="1" value="6" />
          <span class="value" id="cutoffValue">3e20</span>
        </label>
        <label class="checks"><span><input id="overlay" type="checkbox" checked /> overlay iso-FLOP K=0.20 ladder</span></label>
      </div>
      <p class="note">
        Slider sets the upper compute bound used to fit (both ladders). Open circles mark training cells dropped
        by the current cutoff; triangles are held-out actuals; X marks the fit's prediction at the held-out scale.
        Hover shows each cell's midtrain token budget D — fixed for the iso-token ladder, growing 0.24B → 32B
        along the iso-FLOP ladder. Both ladders are CPT from the cooled end of pretraining (the iso-FLOP
        1e21/1e22 cells initialize from the final base-run checkpoint, >=98% LR-decayed, fresh optimizer + warmup).
      </p>
      <div id="scalingPlot" class="chart"></div>
      <div class="tables-stacked">
        <div>
          <h3>Held-Out Predictions (1e21, 1e22)</h3>
          <div id="predTable"></div>
        </div>
        <div>
          <h3>Fit Quality</h3>
          <div id="fitTable"></div>
        </div>
      </div>
    </section>
  </main>
  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const DATA = JSON.parse(document.getElementById("payload").textContent);
    const scaleFlops = DATA.scaleFlops;
    const cutoffScales = DATA.cutoffScales;
    const heldOutScales = DATA.heldOutScales;
    const budgets = DATA.budgets;
    const palette = ["#4C78A8", "#F58518", "#54A24B", "#B279A2"];

    function lookupFit(series, cutoffIndex) {
      return DATA.scalingFits.find((row) => row.series === series && Number(row.cutoff_index) === Number(cutoffIndex));
    }

    function fitOk(fit, fitType) {
      if (!fit) return false;
      if (fitType === "floor_power") return fit.fp_floor != null && fit.fp_amplitude != null && fit.fp_alpha != null;
      return fit.ll_slope != null && fit.ll_intercept != null;
    }

    function predict(fit, fitType, flops) {
      if (!fitOk(fit, fitType)) return null;
      if (fitType === "floor_power") {
        return Number(fit.fp_floor) + Number(fit.fp_amplitude) * Math.pow(flops / 1e18, -Number(fit.fp_alpha));
      }
      return Math.exp(Number(fit.ll_intercept) + Number(fit.ll_slope) * Math.log(flops));
    }

    function tableHtml(rows, columns) {
      if (!rows.length) return '<p class="note"><em>No fit available at this cutoff.</em></p>';
      return `<table><thead><tr>${columns.map((c) => `<th>${c.label}</th>`).join("")}</tr></thead><tbody>${
        rows.map((row) => `<tr>${columns.map((c) => `<td>${c.format ? c.format(row[c.key], row) : row[c.key]}</td>`).join("")}</tr>`).join("")
      }</tbody></table>`;
    }

    function pointHover(row, note) {
      const tokens = row.tokens_b == null ? "" : `<br>D=${row.tokens_b}B midtrain tokens`;
      return `${row.scale}${note}<br>${row.run_name}${tokens}`;
    }

    function seriesArtifacts(series, label, color, fitType, cutoffIndex, cutoffScale, cutoffFlops) {
      const endpoints = DATA.endpoints.filter((row) => row.series === series);
      const targets = DATA.heldoutTargets.filter((row) => row.series === series);

      const trainRows = [];
      const dropRows = [];
      const heldRows = [];
      for (const row of endpoints) {
        (Number(row.scale_flops) <= cutoffFlops + 1e6 ? trainRows : dropRows).push(row);
      }
      for (const row of targets) {
        (Number(row.scale_flops) <= cutoffFlops + 1e6 ? trainRows : heldRows).push(row);
      }

      const traces = [];
      if (trainRows.length) {
        traces.push({
          x: trainRows.map((row) => row.scale_flops),
          y: trainRows.map((row) => row.value),
          mode: "markers", type: "scatter", name: `${label} train`, legendgroup: label,
          marker: {symbol: "circle", size: 10, color},
          hovertext: trainRows.map((row) => pointHover(row, "")),
          hovertemplate: "%{hovertext}<br>flops=%{x:.3e}<br>loss=%{y:.5f}<extra></extra>",
        });
      }
      if (dropRows.length) {
        traces.push({
          x: dropRows.map((row) => row.scale_flops),
          y: dropRows.map((row) => row.value),
          mode: "markers", type: "scatter", name: `${label} dropped`, legendgroup: label, showlegend: false,
          marker: {symbol: "circle-open", size: 11, color, line: {width: 2, color}},
          hovertext: dropRows.map((row) => pointHover(row, " (excluded by cutoff)")),
          hovertemplate: "%{hovertext}<br>flops=%{x:.3e}<br>loss=%{y:.5f}<extra></extra>",
        });
      }

      const fit = lookupFit(series, cutoffIndex);
      const fitTableRows = [];
      const predTableRows = [];
      if (fitOk(fit, fitType)) {
        const logLo = Math.log10(scaleFlops[DATA.scaleOrder[0]]);
        const logHi = Math.log10(scaleFlops[DATA.scaleOrder[DATA.scaleOrder.length - 1]]);
        const lineX = [];
        const lineY = [];
        for (let index = 0; index <= 60; index += 1) {
          const flops = Math.pow(10, logLo + ((logHi - logLo) * index) / 60);
          lineX.push(flops);
          lineY.push(predict(fit, fitType, flops));
        }
        traces.push({
          x: lineX, y: lineY, mode: "lines", type: "scatter", name: `${label} fit`, legendgroup: label,
          showlegend: false, line: {color, dash: "dot", width: 2}, hoverinfo: "skip",
        });
        if (fitType === "floor_power") {
          fitTableRows.push({recipe: label, n: fit.n, p1: Number(fit.fp_floor), p2: Number(fit.fp_amplitude), p3: Number(fit.fp_alpha), r2: fit.fp_r2 == null ? null : Number(fit.fp_r2), rmse: fit.fp_rmse == null ? null : Number(fit.fp_rmse)});
        } else {
          fitTableRows.push({recipe: label, n: fit.n, p1: Number(fit.ll_slope), p2: Number(fit.ll_intercept), p3: null, r2: fit.ll_r2 == null ? null : Number(fit.ll_r2), rmse: fit.ll_rmse_log == null ? null : Number(fit.ll_rmse_log)});
        }
      }

      if (heldRows.length) {
        traces.push({
          x: heldRows.map((row) => row.scale_flops),
          y: heldRows.map((row) => row.value),
          mode: "markers", type: "scatter", name: `${label} held-out`, legendgroup: label, showlegend: false,
          marker: {symbol: "triangle-up", size: 13, color, line: {color: "#111", width: 1}},
          hovertext: heldRows.map((row) => pointHover(row, " held-out actual")),
          hovertemplate: "%{hovertext}<br>flops=%{x:.3e}<br>actual=%{y:.5f}<extra></extra>",
        });
      }
      if (fitOk(fit, fitType)) {
        for (const row of heldRows) {
          const predicted = predict(fit, fitType, Number(row.scale_flops));
          traces.push({
            x: [row.scale_flops], y: [predicted], mode: "markers", type: "scatter", name: `${label} pred`,
            legendgroup: label, showlegend: false, marker: {symbol: "x", size: 12, color, line: {width: 2}},
            hovertext: [`${row.scale} predicted (cutoff ${cutoffScale})`],
            hovertemplate: "%{hovertext}<br>flops=%{x:.3e}<br>predicted=%{y:.5f}<extra></extra>",
          });
          const actual = Number(row.value);
          const error = predicted - actual;
          const pctError = actual !== 0 ? (error / actual) * 100 : null;
          predTableRows.push({recipe: label, scale: row.scale, actual, predicted, error, absError: Math.abs(error), pctError, absPctError: pctError == null ? null : Math.abs(pctError)});
        }
      }
      return {traces, fitTableRows, predTableRows};
    }

    function render() {
      const fitType = document.getElementById("fitType").value;
      const budget = document.getElementById("budget").value;
      const cutoffIndex = Number(document.getElementById("cutoff").value);
      const cutoffScale = cutoffScales[cutoffIndex];
      const cutoffFlops = scaleFlops[cutoffScale];
      document.getElementById("cutoffValue").textContent = cutoffScale;

      const seriesList = [
        {series: `tok${budget}`, label: `tok${budget} iso-token`, color: palette[budgets.indexOf(budget) % palette.length]},
      ];
      if (document.getElementById("overlay").checked) {
        seriesList.push({series: DATA.isoflopSeries, label: "K=0.20 iso-FLOP", color: "#E45756"});
      }

      const traces = [];
      const fitTableRows = [];
      const predTableRows = [];
      for (const entry of seriesList) {
        const art = seriesArtifacts(entry.series, entry.label, entry.color, fitType, cutoffIndex, cutoffScale, cutoffFlops);
        traces.push(...art.traces);
        fitTableRows.push(...art.fitTableRows);
        predTableRows.push(...art.predTableRows);
      }

      Plotly.react("scalingPlot", traces, {
        height: 620,
        margin: {l: 80, r: 20, t: 30, b: 60},
        xaxis: {title: "base-model pretraining compute (FLOPs, log scale)", type: "log", showgrid: true},
        yaxis: {title: `final ${DATA.metricLabel} (log scale)`, type: "log", showgrid: true},
        legend: {orientation: "v"},
        shapes: [{type: "line", xref: "x", yref: "paper", x0: cutoffFlops, x1: cutoffFlops, y0: 0, y1: 1, line: {color: "#94a3b8", dash: "dash", width: 1}}],
        annotations: [{x: Math.log10(cutoffFlops), xref: "x", y: 1.04, yref: "paper", showarrow: false, text: `fit cutoff (${cutoffScale})`, font: {size: 12, color: "#475569"}}],
      }, {responsive: true});

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
      document.getElementById("fitTable").innerHTML = tableHtml(fitTableRows, fitColumns);
      document.getElementById("predTable").innerHTML = tableHtml(predTableRows, predColumns);
    }

    const budgetSelect = document.getElementById("budget");
    for (const budget of budgets) {
      const option = document.createElement("option");
      option.value = budget;
      option.textContent = `tok${budget}`;
      budgetSelect.appendChild(option);
    }
    const cutoffInput = document.getElementById("cutoff");
    cutoffInput.max = String(cutoffScales.length - 1);
    cutoffInput.value = String(DATA.defaultCutoffIndex);
    for (const id of ["fitType", "budget", "cutoff", "overlay"]) {
      document.getElementById(id).addEventListener("input", render);
    }
    render();
  </script>
</body>
</html>
"""


def print_summary(endpoints: pd.DataFrame, scaling_fits: list[dict[str, Any]]) -> None:
    default_fits = [row for row in scaling_fits if row["cutoff_scale"] == DEFAULT_CUTOFF_SCALE]
    held = endpoints[endpoints["scale"].isin(HELD_OUT_SCALES)]
    for fit in default_fits:
        cell = f"{fit['mix']} {fit['series']} lr{fit['lr']}"
        logger.info(
            "%s fit through %s (n=%d): floor=%.4f amp=%.4f alpha=%.4f (R2=%.5f) | log-log slope=%.4f",
            cell,
            DEFAULT_CUTOFF_SCALE,
            fit["n"],
            fit.get("fp_floor", float("nan")),
            fit.get("fp_amplitude", float("nan")),
            fit.get("fp_alpha", float("nan")),
            fit.get("fp_r2", float("nan")),
            fit.get("ll_slope", float("nan")),
        )
        for _, row in held[held["series"].eq(fit["series"])].iterrows():
            fp_pred = fit["fp_floor"] + fit["fp_amplitude"] * (row["scale_flops"] / 1e18) ** -fit["fp_alpha"]
            ll_pred = math.exp(fit["ll_intercept"] + fit["ll_slope"] * math.log(row["scale_flops"]))
            logger.info(
                "  %s actual=%.5f | chinchilla pred=%.5f (%+.2f%%) | log-log pred=%.5f (%+.2f%%)",
                row["scale"],
                row["value"],
                fp_pred,
                (fp_pred / row["value"] - 1) * 100,
                ll_pred,
                (ll_pred / row["value"] - 1) * 100,
            )


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args()
    if not args.use_cache:
        refresh_cache(with_history=not args.no_history)
    isotoken = load_endpoints()
    if not args.use_cache:
        refresh_isoflop_cache(isotoken, with_history=not args.no_history)
    isoflop = load_isoflop_endpoints()
    endpoints = pd.concat([isotoken, isoflop], ignore_index=True)
    scaling_fits = compute_scaling_fits(endpoints)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    isotoken.to_csv(args.output.parent / "isotoken_endpoints.csv", index=False)
    isoflop.to_csv(args.output.parent / "isoflop_k020_endpoints.csv", index=False)
    pd.DataFrame(scaling_fits).to_csv(args.output.parent / "isotoken_scaling_fits.csv", index=False)
    payload = build_payload(endpoints, scaling_fits)
    html = HTML_TEMPLATE.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    html = html.replace("__PLOTLYJS__", get_plotlyjs())
    args.output.write_text(html)
    logger.info("wrote %s (%.1f KB)", args.output, args.output.stat().st_size / 1024)
    print_summary(endpoints, scaling_fits)


if __name__ == "__main__":
    main()
