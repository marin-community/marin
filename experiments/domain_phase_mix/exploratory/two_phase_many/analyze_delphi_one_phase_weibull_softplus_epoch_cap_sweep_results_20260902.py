# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs>=2025.7",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scipy>=1.14",
#   "wandb>=0.21",
# ]
# ///

"""Collect and report the measured Delphi WSPU epoch-cap sweep."""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
import wandb
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO_ROOT / "lib/marin/src"))

from marin.evaluation.olmo_base_eval.aggregate import table9_macro  # noqa: E402
from marin.evaluation.olmo_base_eval.components import table9_components  # noqa: E402

OUTPUT_DIR = REPO_ROOT / (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "delphi_one_phase_weibull_softplus_epoch_cap_sweep_20260902"
)
CANDIDATE_SUMMARY = OUTPUT_DIR / "candidate_summary.csv"
PRIOR_MEASURED = OUTPUT_DIR / "prior_measured_results.csv"
ATLAS = REPO_ROOT / (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "delphi_3e18_optimum_validation_atlas_20260901/observed_candidates.csv"
)
NOISE_RESULTS = REPO_ROOT / (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "delphi_3e18_fixed_aggregate_phase_snr_20260724/same_seed_delta_noise.csv"
)
TRAINING_ROOT = (
    "marin-us-east5/pinlin_calvin_xu/data_mixture/delphi_one_phase_weibull_softplus_epoch_cap_sweep_3e18_20260902"
)
IRIS_ROOT = "/calvinxu/dm-delphi-3e18-onephase-wspu-epochcaps-v6e8-20260902"
TABLE9_GROUP = "olmo_base_eval_table9_delphi_3e18_one_phase_weibull_softplus_epoch_cap_sweep"
FINAL_STEP = 3006
EXPECTED_ROWS = 12
TARGET_METRICS = {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}
TARGET_STYLES = {
    "uncheatable": ("Optimized for Uncheatable", "#178A72"),
    "table9": ("Optimized for Table-9", "#D95F32"),
}
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


@dataclass(frozen=True)
class SeedMatchSpec:
    """One WSPU or OLMix result in the matched-data-seed confirmation."""

    target: str
    policy: str
    epoch_cap: int
    data_seed: int
    metric_project: str
    metric_run_id: str
    metric_key: str
    training_run_id: str


SEED_MATCH_SPECS = (
    SeedMatchSpec(
        "uncheatable",
        "wspu",
        6,
        666200,
        "marin",
        "wspu_uncheatable_seed666200_wspu_uncheatable_cap06-be060e",
        "eval/uncheatable_eval/bpb",
        "wspu_uncheatable_seed666200_wspu_uncheatable_cap06-be060e",
    ),
    SeedMatchSpec(
        "uncheatable",
        "olmix",
        4,
        666200,
        "marin",
        "olmix_onephase_uncheatable_d001_kl005_cap4_3e18-800ea1",
        "eval/uncheatable_eval/bpb",
        "olmix_onephase_uncheatable_d001_kl005_cap4_3e18-800ea1",
    ),
    SeedMatchSpec(
        "table9",
        "wspu",
        6,
        662009,
        "marin-eval",
        "077oz9yd",
        "olmo_base_easy/table9_51_component_macro_bpb",
        "wspu_table9_seed662009_wspu_table9_cap06-aa4659",
    ),
    SeedMatchSpec(
        "table9",
        "wspu",
        7,
        662009,
        "marin-eval",
        "esr6cjuw",
        "olmo_base_easy/table9_51_component_macro_bpb",
        "wspu_table9_seed662009_wspu_table9_cap07-ce8961",
    ),
    SeedMatchSpec(
        "table9",
        "wspu",
        8,
        662009,
        "marin-eval",
        "22eqjh7q",
        "olmo_base_easy/table9_51_component_macro_bpb",
        "wspu_table9_seed662009_wspu_table9_cap08-d6c27c",
    ),
    SeedMatchSpec(
        "table9",
        "olmix",
        4,
        662009,
        "marin-eval",
        "o518aq9w",
        "olmo_base_easy/table9_51_component_macro_bpb",
        "olmix_onephase_table9_d001_kl0p005_cap4_3e18-eff7f7",
    ),
)


def _read_text(filesystem: gcsfs.GCSFileSystem, path: str) -> str:
    with filesystem.open(path, "rt") as handle:
        return handle.read()


def _collect_training_row(filesystem: gcsfs.GCSFileSystem, candidate: dict[str, Any]) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    matches = filesystem.glob(f"{TRAINING_ROOT}/onephase_wspu_{candidate_id}-*/checkpoints/eval_metrics.jsonl")
    if len(matches) != 1:
        raise ValueError(f"{candidate_id}: expected one endpoint file, found {len(matches)}")
    eval_path = matches[0]
    run_dir = eval_path.removesuffix("/checkpoints/eval_metrics.jsonl")
    status = _read_text(filesystem, f"{run_dir}/.executor_status").strip()
    if status != "SUCCESS":
        raise ValueError(f"{candidate_id}: executor status is {status!r}")
    rows = [json.loads(line) for line in _read_text(filesystem, eval_path).splitlines() if line.strip()]
    endpoints = [row for row in rows if int(row.get("step", -1)) == FINAL_STEP]
    if len(endpoints) != 1:
        raise ValueError(f"{candidate_id}: expected one exact step-{FINAL_STEP} endpoint, found {len(endpoints)}")
    endpoint = endpoints[0]
    return {
        **candidate,
        "epoch_cap": int(candidate["epoch_cap"]),
        "predicted_target_bpb": float(candidate["runtime_predicted_bpb"]),
        "uncheatable_bpb": float(endpoint["eval/uncheatable_eval/bpb"]),
        "uncheatable_macro_bpb": float(endpoint["eval/uncheatable_eval/macro_bpb"]),
        "github_cpp_bpb": float(endpoint["eval/uncheatable_eval/github_cpp/bpb"]),
        "github_python_bpb": float(endpoint["eval/uncheatable_eval/github_python/bpb"]),
        "table9_macro_bpb": math.nan,
        "table9_wandb_url": "",
        "eval_metrics_uri": f"gs://{eval_path}",
        "executor_status": status,
        "final_step": FINAL_STEP,
    }


def _candidate_id(run: Any, candidate_ids: tuple[str, ...]) -> str | None:
    tag_map = {tag.split("=", 1)[0]: tag.split("=", 1)[1] for tag in run.tags if "=" in tag}
    tagged = tag_map.get("source_run")
    if tagged in candidate_ids:
        return tagged
    return next((candidate_id for candidate_id in candidate_ids if candidate_id in run.name), None)


def _collect_table9(candidate_ids: tuple[str, ...]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    api = wandb.Api(timeout=120)
    runs = list(api.runs("marin-community/marin-eval", filters={"group": TABLE9_GROUP}, per_page=200))
    finished: dict[str, dict[str, Any]] = {}
    components: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda item: item.created_at):
        candidate_id = _candidate_id(run, candidate_ids)
        summary = dict(run.summary)
        macro = summary.get("olmo_base_easy/table9_macro_bpb")
        if candidate_id is None or run.state != "finished" or macro is None:
            continue
        values = {
            component: float(summary[f"olmo_base_easy/table9/{component}/bpb"]) for component in table9_components()
        }
        reconstructed = table9_macro(values)
        if not math.isclose(float(macro), reconstructed, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{candidate_id}: Table-9 component reconstruction mismatch")
        finished[candidate_id] = {"macro": float(macro), "url": run.url}
        components.extend(
            {
                "candidate_id": candidate_id,
                "component_position": position,
                "component": component,
                "bpb": values[component],
            }
            for position, component in enumerate(table9_components())
        )
    return finished, pd.DataFrame(components)


def _collect_seed_match_results() -> pd.DataFrame:
    api = wandb.Api(timeout=120)
    rows = []
    for spec in SEED_MATCH_SPECS:
        metric_run = api.run(f"marin-community/{spec.metric_project}/{spec.metric_run_id}")
        training_run = api.run(f"marin-community/marin/{spec.training_run_id}")
        if metric_run.state != "finished" or training_run.state != "finished":
            raise ValueError(f"{spec.metric_run_id}: matched-seed metric or training run is incomplete")
        observed_seed = training_run.config.get("data_seed")
        if observed_seed is None or int(observed_seed) != spec.data_seed:
            raise ValueError(f"{spec.training_run_id}: expected data seed {spec.data_seed}, found {observed_seed}")
        value = metric_run.summary.get(spec.metric_key)
        if value is None or not math.isfinite(float(value)):
            raise ValueError(f"{spec.metric_run_id}: missing finite metric {spec.metric_key}")
        rows.append(
            {
                "target": spec.target,
                "policy": spec.policy,
                "epoch_cap": spec.epoch_cap,
                "data_seed": spec.data_seed,
                "bpb": float(value),
                "metric_run_id": spec.metric_run_id,
                "metric_run_url": metric_run.url,
                "training_run_id": spec.training_run_id,
            }
        )
    return pd.DataFrame(rows).sort_values(["target", "policy", "epoch_cap"]).reset_index(drop=True)


def collect_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(CANDIDATE_SUMMARY)
    candidates = candidates[
        candidates.target.eq("table9") | (candidates.target.eq("uncheatable") & candidates.epoch_cap.le(6))
    ].copy()
    if len(candidates) != EXPECTED_ROWS or candidates.candidate_id.duplicated().any():
        raise ValueError("Frozen runtime candidate inventory is incomplete or non-unique")

    filesystem = gcsfs.GCSFileSystem(token="google_default")
    records = candidates.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=EXPECTED_ROWS) as pool:
        results = pd.DataFrame(pool.map(lambda candidate: _collect_training_row(filesystem, candidate), records))

    candidate_ids = tuple(results.candidate_id.astype(str))
    table9, components = _collect_table9(candidate_ids)
    if set(table9) != set(candidate_ids):
        raise ValueError(f"Native Table-9 inventory mismatch: missing={sorted(set(candidate_ids) - set(table9))}")
    for index, row in results.iterrows():
        native = table9[str(row.candidate_id)]
        results.loc[index, "table9_macro_bpb"] = native["macro"]
        results.loc[index, "table9_wandb_url"] = native["url"]
    results = results.sort_values(["target", "epoch_cap"]).reset_index(drop=True)
    if not results.executor_status.eq("SUCCESS").all() or results.table9_macro_bpb.isna().any():
        raise ValueError("Measured result inventory is incomplete")
    return results, components.sort_values(["candidate_id", "component_position"]).reset_index(drop=True)


def _noise_scales() -> dict[str, float]:
    rows = pd.read_csv(NOISE_RESULTS)
    anchors = {"uncheatable": "uncheatable_frontier", "table9": "table9_frontier"}
    return {
        target: float(
            rows.loc[rows.anchor_id.eq(anchor) & rows.target.eq(target), "same_seed_delta_noise_sd_bpb"].iloc[0]
        )
        for target, anchor in anchors.items()
    }


def _historical_frontiers() -> dict[str, dict[str, Any]]:
    atlas = pd.read_csv(ATLAS)
    main = atlas[atlas.policy_class.eq("One phase") & atlas.is_main_range.astype(bool)]
    frontiers: dict[str, dict[str, Any]] = {}
    for target, metric in TARGET_METRICS.items():
        rows = main[main[metric].notna()]
        best = rows.loc[rows[metric].astype(float).idxmin()]
        frontiers[target] = {
            "candidate_id": str(best.candidate_id),
            "bpb": float(best[metric]),
            "sweep_id": str(best.sweep_id),
        }
    return frontiers


def _diagnostics(results: pd.DataFrame) -> dict[str, dict[str, Any]]:
    noise = _noise_scales()
    frontiers = _historical_frontiers()
    predecessor = pd.read_csv(PRIOR_MEASURED)
    diagnostics: dict[str, dict[str, Any]] = {}
    for target, metric in TARGET_METRICS.items():
        rows = results[results.target.eq(target)].sort_values("epoch_cap")
        predicted = rows.predicted_target_bpb.to_numpy(float)
        observed = rows[metric].to_numpy(float)
        predicted_best = int(np.argmin(predicted))
        observed_best = int(np.argmin(observed))
        prior_rows = predecessor[predecessor.candidate_family.eq("shared_shape_dsp") & predecessor.target.eq(target)]
        prior_best = float(prior_rows.measured_bpb.min())
        best_value = float(observed[observed_best])
        frontier = frontiers[target]
        diagnostics[target] = {
            "rows": len(rows),
            "prediction_rmse_bpb": float(np.sqrt(np.mean((predicted - observed) ** 2))),
            "mean_observed_minus_predicted_bpb": float(np.mean(observed - predicted)),
            "spearman": float(spearmanr(predicted, observed).statistic),
            "predicted_best_cap": int(rows.iloc[predicted_best].epoch_cap),
            "observed_best_cap": int(rows.iloc[observed_best].epoch_cap),
            "observed_best_candidate_id": str(rows.iloc[observed_best].candidate_id),
            "observed_best_bpb": best_value,
            "predicted_selection_regret_bpb": float(observed[predicted_best] - best_value),
            "predicted_selection_regret_noise_sd": float((observed[predicted_best] - best_value) / noise[target]),
            "predecessor_best_bpb": prior_best,
            "delta_vs_predecessor_bpb": best_value - prior_best,
            "delta_vs_predecessor_noise_sd": (best_value - prior_best) / noise[target],
            "historical_one_phase_frontier": frontier,
            "delta_vs_historical_frontier_bpb": best_value - float(frontier["bpb"]),
            "delta_vs_historical_frontier_noise_sd": (best_value - float(frontier["bpb"])) / noise[target],
            "new_historical_frontier": best_value < float(frontier["bpb"]),
        }
    return diagnostics


def _write_plot(
    results: pd.DataFrame,
    diagnostics: dict[str, dict[str, Any]],
    seed_matches: pd.DataFrame,
) -> Path:
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable", "Table-9 macro"),
        horizontal_spacing=0.1,
    )
    predecessor = pd.read_csv(PRIOR_MEASURED)
    for column, (metric_target, metric) in enumerate(TARGET_METRICS.items(), start=1):
        for target, (label, color) in TARGET_STYLES.items():
            rows = results[results.target.eq(target)].sort_values("epoch_cap")
            figure.add_trace(
                go.Scatter(
                    x=rows.epoch_cap,
                    y=rows[metric],
                    mode="lines+markers",
                    name=label,
                    legendgroup=target,
                    showlegend=column == 1,
                    line={"color": color, "width": 3},
                    marker={"size": 9},
                    customdata=rows.candidate_id,
                    hovertemplate="<b>%{customdata}</b><br>Cap %{x}<br>Measured BPB %{y:.6f}<extra></extra>",
                ),
                row=1,
                col=column,
            )

        target_rows = results[results.target.eq(metric_target)].sort_values("epoch_cap")
        label, color = TARGET_STYLES[metric_target]
        figure.add_trace(
            go.Scatter(
                x=target_rows.epoch_cap,
                y=target_rows.predicted_target_bpb,
                mode="lines+markers",
                name=f"WSPU prediction: {label.removeprefix('Optimized for ')}",
                legendgroup=f"prediction-{metric_target}",
                visible="legendonly",
                line={"color": color, "width": 2, "dash": "dash"},
                marker={"symbol": "circle-open", "size": 7},
                hovertemplate="Cap %{x}<br>Predicted BPB %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )

        matched = seed_matches[seed_matches.target.eq(metric_target)]
        for policy, name, symbol in (
            ("wspu", "Matched-seed WSPU", "square-open"),
            ("olmix", "Matched-seed OLMix", "diamond-open"),
        ):
            policy_rows = matched[matched.policy.eq(policy)].sort_values("epoch_cap")
            figure.add_trace(
                go.Scatter(
                    x=policy_rows.epoch_cap,
                    y=policy_rows.bpb,
                    mode="markers",
                    name=name,
                    legendgroup=f"seed-match-{policy}",
                    showlegend=column == 1,
                    marker={
                        "color": "#111111",
                        "size": 13,
                        "symbol": symbol,
                        "line": {"color": "#111111", "width": 2},
                    },
                    customdata=np.column_stack(
                        [policy_rows.data_seed, policy_rows.metric_run_id, policy_rows.training_run_id]
                    ),
                    hovertemplate=(
                        f"<b>{name}</b><br>Cap %{{x}}<br>Measured BPB %{{y:.6f}}"
                        "<br>Data seed %{customdata[0]}<br>Metric run %{customdata[1]}"
                        "<br>Training run %{customdata[2]}<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )

        prior = predecessor[
            predecessor.candidate_family.eq("shared_shape_dsp") & predecessor.target.eq(metric_target)
        ].sort_values("epoch_cap")
        figure.add_trace(
            go.Scatter(
                x=prior.epoch_cap,
                y=prior.measured_bpb,
                mode="lines+markers",
                name="Shared-shape DSP predecessor",
                legendgroup="predecessor",
                showlegend=column == 1,
                line={"color": "#8594A3", "width": 2, "dash": "dot"},
                marker={"size": 7, "symbol": "circle-open"},
                hovertemplate="Predecessor cap %{x}<br>Measured BPB %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )

        frontier = diagnostics[metric_target]["historical_one_phase_frontier"]
        figure.add_hline(
            y=float(frontier["bpb"]),
            line={"color": "#17324D", "width": 2, "dash": "dash"},
            annotation_text=f"Historical 1p frontier {float(frontier['bpb']):.6f}",
            annotation_position="bottom right",
            row=1,
            col=column,
        )
        best = diagnostics[metric_target]
        figure.add_trace(
            go.Scatter(
                x=[best["observed_best_cap"]],
                y=[best["observed_best_bpb"]],
                mode="markers",
                name="Best WSPU row",
                legendgroup="best-wspu",
                showlegend=column == 1,
                marker={"color": "#17324D", "size": 12, "symbol": "star"},
                hovertemplate="Best WSPU cap %{x}<br>Measured BPB %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )

    figure.update_layout(
        title={"text": "Delphi 3e18 WSPU one-phase epoch-cap validation", "x": 0.5, "xanchor": "center"},
        height=680,
        margin={"l": 70, "r": 35, "t": 165, "b": 120},
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.03,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13},
        },
        paper_bgcolor="#F8F3E8",
        plot_bgcolor="#F8F3E8",
        font={"family": "Avenir Next, sans-serif", "size": 15, "color": "#17324D"},
        hoverlabel={"bgcolor": "#FFF9EE", "font": {"color": "#17324D"}},
    )
    figure.update_xaxes(title_text="Whole-run materialized epoch cap", dtick=1, gridcolor="#DCE5EA")
    figure.update_yaxes(title_text="BPB (lower is better)", gridcolor="#DCE5EA")
    figure.add_annotation(
        x=0.5,
        y=-0.2,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=(
            "Original tracks use the sweep seed. Open square: WSPU rerun at the selected OLMix seed; "
            "open diamond: matched OLMix.<br>Seeds: 666200 (Uncheatable), 662009 (Table-9). "
            "Dashed model predictions are hidden by default."
        ),
        font={"size": 12, "color": "#52657A"},
    )
    path = OUTPUT_DIR / "measured_cap_performance.html"
    pio.write_html(figure, path, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG)
    return path


def write_outputs() -> None:
    results, components = collect_results()
    seed_matches = _collect_seed_match_results()
    diagnostics = _diagnostics(results)
    noise = _noise_scales()

    measured_path = OUTPUT_DIR / "measured_results.csv"
    component_path = OUTPUT_DIR / "measured_table9_components.csv"
    summary_path = OUTPUT_DIR / "measured_summary.json"
    report_path = OUTPUT_DIR / "results.md"
    results.to_csv(measured_path, index=False)
    components.to_csv(component_path, index=False)
    plot_path = _write_plot(results, diagnostics, seed_matches)

    uncheatable = diagnostics["uncheatable"]
    table9 = diagnostics["table9"]
    uncheatable_rows = results[results.target.eq("uncheatable")].sort_values("epoch_cap")
    table9_rows = results[results.target.eq("table9")].sort_values("epoch_cap")
    uncheatable_cap4 = float(uncheatable_rows.loc[uncheatable_rows.epoch_cap.eq(4), "uncheatable_bpb"].iloc[0])
    table9_cap5 = float(table9_rows.loc[table9_rows.epoch_cap.eq(5), "table9_macro_bpb"].iloc[0])
    summary = {
        "iris_root": IRIS_ROOT,
        "training_root": f"gs://{TRAINING_ROOT}",
        "final_step": FINAL_STEP,
        "runtime_rows": len(results),
        "native_table9_components_per_row": len(table9_components()),
        "all_training_executor_status_success": bool(results.executor_status.eq("SUCCESS").all()),
        "diagnostics": diagnostics,
        "noise_calibration": {
            "uncheatable_same_seed_delta_sd_bpb": noise["uncheatable"],
            "table9_same_seed_delta_sd_bpb": noise["table9"],
            "noise_df": 6,
        },
        "coarse_plateau_diagnostics": {
            "uncheatable_cap4_minus_cap6_bpb": uncheatable_cap4 - float(uncheatable["observed_best_bpb"]),
            "uncheatable_cap4_minus_cap6_noise_sd": (
                (uncheatable_cap4 - float(uncheatable["observed_best_bpb"])) / noise["uncheatable"]
            ),
            "table9_cap5_minus_cap6_bpb": table9_cap5 - float(table9["observed_best_bpb"]),
            "table9_cap5_minus_cap6_noise_sd": (table9_cap5 - float(table9["observed_best_bpb"])) / noise["table9"],
        },
        "limitations": [
            "Each mixture has one common data/trainer seed; cap differences do not have direct replicate intervals.",
            "Noise-SD ratios use an older six-degree-of-freedom same-seed anchor and are descriptive, not tests.",
            "Historical frontier comparisons are cross-experiment and not paired confirmations.",
        ],
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    report = f"""# Delphi 3e18 WSPU one-phase epoch-cap validation

## Verdict

All 12 runtime-distinct trainings and all 12 native Table-9 evaluations completed successfully. The sweep is a
useful validation of `weibull_softplus_unscaled` (WSPU), but it does **not** establish a new frontier on either
primary metric.

[Open the measured cap-performance plot](measured_cap_performance.html).

- **Uncheatable:** the best WSPU row is `{uncheatable["observed_best_candidate_id"]}` at cap
  `{uncheatable["observed_best_cap"]}`, with `{uncheatable["observed_best_bpb"]:.6f}` BPB. This is
  `{uncheatable["delta_vs_historical_frontier_bpb"]:+.6f}` BPB relative to the historical one-phase frontier
  `{uncheatable["historical_one_phase_frontier"]["candidate_id"]}` at
  `{uncheatable["historical_one_phase_frontier"]["bpb"]:.6f}`. WSPU selected the observed best cap exactly, but
  remains `{uncheatable["delta_vs_predecessor_bpb"]:+.6f}` BPB behind the shared-shape DSP sweep's best row.
- **Table-9:** the best WSPU row is `{table9["observed_best_candidate_id"]}` at cap
  `{table9["observed_best_cap"]}`, with `{table9["observed_best_bpb"]:.6f}` BPB. This is
  `{table9["delta_vs_historical_frontier_bpb"]:+.6f}` BPB relative to the historical one-phase frontier
  `{table9["historical_one_phase_frontier"]["candidate_id"]}` at
  `{table9["historical_one_phase_frontier"]["bpb"]:.6f}`. It improves on the shared-shape DSP target path by
  `{-table9["delta_vs_predecessor_bpb"]:.6f}` BPB. WSPU predicted cap 8 rather than cap 6, but the resulting
  selection regret is only `{table9["predicted_selection_regret_bpb"]:.6f}` BPB, or
  `{table9["predicted_selection_regret_noise_sd"]:.2f}` conservative noise SD.

## Interpretation

The response paths are smooth and scientifically sensible. Uncheatable improves rapidly through cap 4 and then
forms a shallow cap-4-to-6 plateau; cap 4 is only
`{summary["coarse_plateau_diagnostics"]["uncheatable_cap4_minus_cap6_bpb"]:.6f}` BPB above cap 6. Table-9
improves through cap 6 and is flat to slightly worse at caps 7-8. The coarse optimum region is therefore cap 4-6
for Uncheatable and cap 5-8 for Table-9, not an exact cap selected from a single seed.

WSPU's ranking transfers substantially better than its absolute calibration: target-path Spearman is
`{uncheatable["spearman"]:.3f}` for Uncheatable and `{table9["spearman"]:.3f}` for Table-9, while predictions are
optimistic by `{uncheatable["mean_observed_minus_predicted_bpb"]:.6f}` and
`{table9["mean_observed_minus_predicted_bpb"]:.6f}` BPB on average. The strong Table-9 gain over the predecessor
shows that the successor's materialized policy is better behaved, but the remaining
`{table9["delta_vs_historical_frontier_bpb"]:.6f}` BPB gap to the global one-phase atlas frontier is material.

## Limitations

Every candidate uses one common trainer/data seed. The conservative noise anchors have only six degrees of
freedom, so their SD ratios calibrate effect size rather than provide confidence intervals. A frontier claim would
require first beating the atlas reference and then confirming that candidate with paired repeats; neither metric
passes the first gate here.

## Provenance

- Iris root: `{IRIS_ROOT}`
- Exact endpoint: step `{FINAL_STEP}`
- Training root: `gs://{TRAINING_ROOT}`
- Native Table-9 group: `{TABLE9_GROUP}`
- `measured_results.csv`: 12 exact endpoints
- `measured_table9_components.csv`: 51 reconstructed Table-9 components per endpoint
"""
    report_path.write_text(report)
    output_paths = (measured_path, component_path, summary_path, report_path, plot_path)
    print(
        json.dumps(
            {"outputs": [str(path) for path in output_paths], "summary": summary},
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    write_outputs()
