# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "gcsfs>=2025.1",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scipy>=1.14",
#   "wandb>=0.21",
# ]
# ///

"""Analyze measured Delphi aggregate-linear-V one-phase challengers."""

from __future__ import annotations

import hashlib
import json
import math
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import gcsfs
import numpy as np
import pandas as pd
import plotly.express as px
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
    "delphi_one_phase_surrogate_challenger_validations_20260831"
)
CANDIDATE_SUMMARY = OUTPUT_DIR / "candidate_summary.csv"
CANDIDATE_MANIFEST = OUTPUT_DIR / "manifest.json"
DSP_RESULTS = REPO_ROOT / (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "delphi_one_phase_dsp_epoch_cap_sweep_20260828/measured_results.csv"
)
DSP_COMPONENTS = REPO_ROOT / (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "delphi_one_phase_dsp_epoch_cap_sweep_20260828/measured_table9_components.csv"
)
NOISE_RESULTS = REPO_ROOT / (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "delphi_3e18_fixed_aggregate_phase_snr_20260724/same_seed_delta_noise.csv"
)
TRAIN_PROJECT = "marin-community/marin"
TRAIN_GROUP = "delphi_3e18_one_phase_aggregate_v_validation"
EVAL_PROJECT = "marin-community/marin-eval"
TABLE9_GROUP = "olmo_base_eval_table9_delphi_3e18_one_phase_aggregate_v_validation"
TRAINING_ROOT = (
    "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/"
    "delphi_one_phase_surrogate_challenger_validations_3e18_20260831"
)
IRIS_ROOT = "/calvinxu/dm-delphi-3e18-onephase-aggregate-v-caps4to10-v6e8-20260831"
EXPECTED_CANDIDATE_WEIGHTS_SHA256 = "0e98d5d98354308516050dec9bc09766df06f42367fb5014ed31541739311546"
FINAL_STEP = 3006
TARGETS = ("uncheatable_bpb", "table9_macro_bpb")
CAPS = (4, 6, 8, 10)
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _gcs_text(filesystem: gcsfs.GCSFileSystem, uri: str) -> str:
    with filesystem.open(uri.removeprefix("gs://"), "rb") as handle:
        return handle.read().decode()


def _conservative_noise_scales() -> dict[str, float]:
    """Load conservative same-seed delta noise anchors for the two targets."""
    frame = pd.read_csv(NOISE_RESULTS)
    anchors = {
        "uncheatable_bpb": ("uncheatable_frontier", "uncheatable"),
        "table9_macro_bpb": ("table9_frontier", "table9"),
    }
    scales: dict[str, float] = {}
    for metric, (anchor_id, target) in anchors.items():
        row = frame.loc[frame["anchor_id"].eq(anchor_id) & frame["target"].eq(target)]
        if len(row) != 1:
            raise ValueError(f"Missing unique noise anchor for {metric}")
        scales[metric] = float(row.iloc[0]["same_seed_delta_noise_sd_bpb"])
    return scales


def _candidate_rows() -> pd.DataFrame:
    manifest = json.loads(CANDIDATE_MANIFEST.read_text())
    observed_hash = manifest["outputs"]["candidate_weights.csv"]
    if observed_hash != EXPECTED_CANDIDATE_WEIGHTS_SHA256:
        raise ValueError(f"Candidate weights changed: {observed_hash}")
    candidates = pd.read_csv(CANDIDATE_SUMMARY)
    selected = candidates.loc[
        candidates["model"].eq("aggregate_linear_v")
        & candidates["target"].isin(TARGETS)
        & candidates["epoch_cap"].isin(CAPS)
    ].copy()
    if len(selected) != 8 or selected["candidate_id"].duplicated().any():
        raise ValueError("Expected eight unique aggregate-linear-V candidates")
    return selected.sort_values(["target", "epoch_cap"]).reset_index(drop=True)


def _finished_table9_runs(api: wandb.Api) -> dict[str, Any]:
    runs = list(api.runs(EVAL_PROJECT, filters={"group": TABLE9_GROUP}, per_page=100))
    finished: dict[str, Any] = {}
    for run in sorted(runs, key=lambda item: item.created_at):
        if run.state != "finished" or run.summary.get("olmo_base_easy/table9_macro_bpb") is None:
            continue
        slug = run.name.removeprefix("t9_onephase_av_")
        finished[slug] = run
    if len(finished) != 8:
        raise ValueError(f"Expected eight finished native Table-9 rows, found {len(finished)}")
    return finished


def collect_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = _candidate_rows()
    by_id = candidates.set_index("candidate_id")
    api = wandb.Api(timeout=240)
    training_runs = list(api.runs(TRAIN_PROJECT, filters={"group": TRAIN_GROUP}, per_page=100))
    if len(training_runs) != 8:
        raise ValueError(f"Expected eight training runs, found {len(training_runs)}")
    native_runs = _finished_table9_runs(api)
    filesystem = gcsfs.GCSFileSystem(token="google_default")

    rows: list[dict[str, Any]] = []
    component_rows: list[dict[str, Any]] = []
    for run in training_runs:
        tags = {tag.split("=", 1)[0]: tag.split("=", 1)[1] for tag in run.tags if "=" in tag}
        candidate_id = tags.get("source_run")
        if candidate_id not in by_id.index:
            raise ValueError(f"{run.name}: unknown source candidate {candidate_id!r}")
        candidate = by_id.loc[str(candidate_id)]
        expected_root = f"{TRAINING_ROOT}/{run.name}"
        checkpoint_root = str(run.config["trainer"]["checkpointer"]["base_path"])
        if checkpoint_root != f"{expected_root}/checkpoints":
            raise ValueError(f"{run.name}: checkpoint root is misplaced: {checkpoint_root}")
        status = _gcs_text(filesystem, f"{expected_root}/.executor_status").strip()
        if status != "SUCCESS":
            raise ValueError(f"{candidate_id}: executor status is {status!r}")

        metric_rows = [
            json.loads(line)
            for line in _gcs_text(filesystem, f"{checkpoint_root}/eval_metrics.jsonl").splitlines()
            if line.strip()
        ]
        final_rows = [row for row in metric_rows if int(row.get("step", -1)) == FINAL_STEP]
        if len(final_rows) != 1:
            raise ValueError(f"{candidate_id}: expected one step-{FINAL_STEP} row, found {len(final_rows)}")
        final = final_rows[0]

        slug = str(candidate_id).removeprefix("aggregate_linear_v_")
        native = native_runs[slug]
        native_summary = dict(native.summary)
        components = {
            component: float(native_summary[f"olmo_base_easy/table9/{component}/bpb"])
            for component in table9_components()
        }
        table9_value = float(native_summary["olmo_base_easy/table9_macro_bpb"])
        reconstructed = table9_macro(components)
        if not math.isclose(reconstructed, table9_value, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"{candidate_id}: native Table-9 macro does not reconstruct")
        for position, component in enumerate(table9_components()):
            component_rows.append(
                {
                    "candidate_id": candidate_id,
                    "target": candidate["target"],
                    "epoch_cap": int(candidate["epoch_cap"]),
                    "component_position": position,
                    "component": component,
                    "bpb": components[component],
                }
            )

        rows.append(
            {
                "candidate_id": candidate_id,
                "target": candidate["target"],
                "epoch_cap": int(candidate["epoch_cap"]),
                "predicted_target_bpb": float(candidate["runtime_predicted_bpb"]),
                "uncheatable_bpb": float(final["eval/uncheatable_eval/bpb"]),
                "uncheatable_macro_bpb": float(final["eval/uncheatable_eval/macro_bpb"]),
                "github_cpp_bpb": float(final["eval/uncheatable_eval/github_cpp/bpb"]),
                "github_python_bpb": float(final["eval/uncheatable_eval/github_python/bpb"]),
                "table9_macro_bpb": table9_value,
                "max_materialized_epoch": float(candidate["max_materialized_epoch"]),
                "effective_buckets": float(candidate["effective_buckets"]),
                "tv_to_proportional": float(candidate["tv_to_proportional"]),
                "nearest_panel_tv": float(candidate["nearest_panel_tv"]),
                "largest_bucket": candidate["largest_bucket"],
                "largest_weight": float(candidate["largest_weight"]),
                "final_step": FINAL_STEP,
                "executor_status": status,
                "training_wandb_state": run.state,
                "training_wandb_url": run.url,
                "native_table9_wandb_url": native.url,
                "eval_metrics_uri": f"{checkpoint_root}/eval_metrics.jsonl",
            }
        )

    result = pd.DataFrame(rows).sort_values(["target", "epoch_cap"]).reset_index(drop=True)
    components = (
        pd.DataFrame(component_rows).sort_values(["target", "epoch_cap", "component_position"]).reset_index(drop=True)
    )
    if len(result) != 8 or len(components) != 8 * len(table9_components()):
        raise ValueError("Measured challenger inventory is incomplete")
    return result, components


def _diagnostics(rows: pd.DataFrame) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        group = rows.loc[rows["target"].eq(target)].sort_values("epoch_cap")
        predicted = group["predicted_target_bpb"].to_numpy(dtype=float)
        observed = group[target].to_numpy(dtype=float)
        predicted_best = int(np.argmin(predicted))
        observed_best = int(np.argmin(observed))
        diagnostics[target] = {
            "rows": len(group),
            "rmse_bpb": float(np.sqrt(np.mean(np.square(predicted - observed)))),
            "mae_bpb": float(np.mean(np.abs(predicted - observed))),
            "mean_observed_minus_predicted_bpb": float(np.mean(observed - predicted)),
            "spearman": float(spearmanr(predicted, observed).statistic),
            "predicted_best_cap": int(group.iloc[predicted_best]["epoch_cap"]),
            "observed_best_cap": int(group.iloc[observed_best]["epoch_cap"]),
            "selection_regret_bpb": float(observed[predicted_best] - observed[observed_best]),
            "observed_best_bpb": float(observed[observed_best]),
            "observed_best_candidate_id": str(group.iloc[observed_best]["candidate_id"]),
        }
    return diagnostics


def _dsp_comparison(rows: pd.DataFrame) -> pd.DataFrame:
    dsp = pd.read_csv(DSP_RESULTS)
    dsp = dsp.loc[dsp["epoch_cap"].isin(CAPS)].copy()
    joined = rows.merge(dsp, on=["target", "epoch_cap"], suffixes=("_aggregate_v", "_dsp"), validate="one_to_one")
    for metric in ("uncheatable_bpb", "table9_macro_bpb", "github_cpp_bpb", "github_python_bpb"):
        joined[f"aggregate_v_minus_dsp_{metric}"] = joined[f"{metric}_aggregate_v"] - joined[f"{metric}_dsp"]
    return joined


def _plot(rows: pd.DataFrame, dsp: pd.DataFrame) -> Path:
    colors = px.colors.sample_colorscale("RdYlGn_r", [0.15, 0.85])
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Uncheatable BPB", "Table-9 macro BPB"),
        horizontal_spacing=0.1,
    )
    panels = (("uncheatable_bpb", "uncheatable_bpb"), ("table9_macro_bpb", "table9_macro_bpb"))
    for column, (metric, target) in enumerate(panels, start=1):
        for frame, label, dash, color in (
            (rows, "Aggregate-linear-V", "solid", colors[0]),
            (dsp, "Shared-shape DSP", "dash", colors[1]),
        ):
            group = frame.loc[frame["target"].eq(target)].sort_values("epoch_cap")
            figure.add_trace(
                go.Scatter(
                    x=group["epoch_cap"],
                    y=group[metric],
                    mode="lines+markers",
                    name=label,
                    legendgroup=label,
                    showlegend=column == 1,
                    line={"color": color, "width": 3, "dash": dash},
                    marker={"size": 10},
                    customdata=group["candidate_id"],
                    hovertemplate="<b>%{customdata}</b><br>Cap %{x}<br>BPB %{y:.6f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
    figure.update_layout(
        title="Measured one-phase optima: aggregate-linear-V versus shared-shape DSP",
        height=640,
        margin={"l": 70, "r": 35, "t": 120, "b": 80},
        paper_bgcolor="#F8F3E8",
        plot_bgcolor="#F8F3E8",
        font={"family": "Avenir Next, sans-serif", "size": 15, "color": "#17324D"},
        legend={"orientation": "h", "y": 1.08, "x": 0.5, "xanchor": "center"},
    )
    figure.update_xaxes(title="Whole-run epoch cap", tickvals=list(CAPS), gridcolor="#DCE5EA")
    figure.update_yaxes(title="BPB (lower is better)", gridcolor="#DCE5EA")
    path = OUTPUT_DIR / "measured_comparison.html"
    pio.write_html(figure, path, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG)
    return path


def _markdown_table(rows: pd.DataFrame) -> str:
    lines = [
        "| candidate | cap | predicted target | Uncheatable | Table-9 | GitHub C++ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows.itertuples(index=False):
        lines.append(
            f"| {row.candidate_id} | {row.epoch_cap} | {row.predicted_target_bpb:.6f} | "
            f"{row.uncheatable_bpb:.6f} | {row.table9_macro_bpb:.6f} | {row.github_cpp_bpb:.6f} |"
        )
    return "\n".join(lines)


def write_outputs() -> None:
    rows, components = collect_results()
    diagnostics = _diagnostics(rows)
    comparison = _dsp_comparison(rows)
    dsp = pd.read_csv(DSP_RESULTS)
    noise = _conservative_noise_scales()
    plot_path = _plot(rows, dsp)

    measured_path = OUTPUT_DIR / "measured_results.csv"
    components_path = OUTPUT_DIR / "measured_table9_components.csv"
    comparison_path = OUTPUT_DIR / "measured_vs_dsp.csv"
    rows.to_csv(measured_path, index=False)
    components.to_csv(components_path, index=False)
    comparison.to_csv(comparison_path, index=False)

    best_uncheatable = rows.loc[rows["uncheatable_bpb"].idxmin()]
    best_table9 = rows.loc[rows["table9_macro_bpb"].idxmin()]
    best_dsp_uncheatable = dsp.loc[dsp["uncheatable_bpb"].idxmin()]
    best_dsp_table9 = dsp.loc[dsp["table9_macro_bpb"].idxmin()]
    dsp_table9_target = dsp.loc[dsp["target"].eq("table9_macro_bpb")]
    best_dsp_table9_same_target = dsp_table9_target.loc[dsp_table9_target["table9_macro_bpb"].idxmin()]
    dsp_components = pd.read_csv(DSP_COMPONENTS)
    best_component_delta = (
        components.loc[components["candidate_id"].eq(best_table9["candidate_id"]), ["component", "bpb"]]
        .rename(columns={"bpb": "aggregate_v_bpb"})
        .merge(
            dsp_components.loc[
                dsp_components["candidate_id"].eq(best_dsp_table9["candidate_id"]), ["component", "bpb"]
            ].rename(columns={"bpb": "dsp_bpb"}),
            on="component",
            validate="one_to_one",
        )
    )
    best_component_delta["aggregate_v_minus_dsp_bpb"] = (
        best_component_delta["aggregate_v_bpb"] - best_component_delta["dsp_bpb"]
    )
    best_component_delta = best_component_delta.sort_values("aggregate_v_minus_dsp_bpb").reset_index(drop=True)
    component_delta_path = OUTPUT_DIR / "best_table9_component_delta_vs_dsp.csv"
    best_component_delta.to_csv(component_delta_path, index=False)
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "iris_root": IRIS_ROOT,
        "training_root": TRAINING_ROOT,
        "final_step": FINAL_STEP,
        "rows": len(rows),
        "native_table9_components_per_row": len(table9_components()),
        "diagnostics": diagnostics,
        "best_aggregate_v": {
            "uncheatable": {
                "candidate_id": best_uncheatable["candidate_id"],
                "bpb": float(best_uncheatable["uncheatable_bpb"]),
            },
            "table9": {
                "candidate_id": best_table9["candidate_id"],
                "bpb": float(best_table9["table9_macro_bpb"]),
            },
        },
        "best_dsp": {
            "uncheatable": {
                "candidate_id": best_dsp_uncheatable["candidate_id"],
                "bpb": float(best_dsp_uncheatable["uncheatable_bpb"]),
            },
            "table9": {
                "candidate_id": best_dsp_table9["candidate_id"],
                "bpb": float(best_dsp_table9["table9_macro_bpb"]),
            },
        },
        "best_aggregate_v_minus_best_dsp": {
            "uncheatable_bpb": float(best_uncheatable["uncheatable_bpb"] - best_dsp_uncheatable["uncheatable_bpb"]),
            "table9_macro_bpb": float(best_table9["table9_macro_bpb"] - best_dsp_table9["table9_macro_bpb"]),
        },
        "best_aggregate_v_minus_best_same_target_dsp": {
            "uncheatable_bpb": float(best_uncheatable["uncheatable_bpb"] - best_dsp_uncheatable["uncheatable_bpb"]),
            "table9_macro_bpb": float(best_table9["table9_macro_bpb"] - best_dsp_table9_same_target["table9_macro_bpb"]),
        },
        "best_table9_component_comparison": {
            "improved_components": int(best_component_delta["aggregate_v_minus_dsp_bpb"].lt(0).sum()),
            "total_components": len(best_component_delta),
            "median_delta_bpb": float(best_component_delta["aggregate_v_minus_dsp_bpb"].median()),
            "largest_improvement_component": str(best_component_delta.iloc[0]["component"]),
            "largest_improvement_bpb": float(best_component_delta.iloc[0]["aggregate_v_minus_dsp_bpb"]),
            "largest_regression_component": str(best_component_delta.iloc[-1]["component"]),
            "largest_regression_bpb": float(best_component_delta.iloc[-1]["aggregate_v_minus_dsp_bpb"]),
        },
        "noise_calibration": {
            "uncheatable_same_seed_delta_sd_bpb": noise["uncheatable_bpb"],
            "table9_same_seed_delta_sd_bpb": noise["table9_macro_bpb"],
            "noise_df": 6,
            "uncheatable_gap_vs_best_dsp_sd": (
                float(best_uncheatable["uncheatable_bpb"] - best_dsp_uncheatable["uncheatable_bpb"])
                / noise["uncheatable_bpb"]
            ),
            "table9_gain_vs_best_dsp_sd": (
                float(best_dsp_table9["table9_macro_bpb"] - best_table9["table9_macro_bpb"]) / noise["table9_macro_bpb"]
            ),
            "table9_gain_vs_same_target_dsp_sd": (
                float(best_dsp_table9_same_target["table9_macro_bpb"] - best_table9["table9_macro_bpb"])
                / noise["table9_macro_bpb"]
            ),
        },
        "support_distance": {
            "aggregate_v_nearest_panel_tv_range": [
                float(rows["nearest_panel_tv"].min()),
                float(rows["nearest_panel_tv"].max()),
            ],
            "dsp_nearest_panel_tv_range": [
                float(dsp["nearest_panel_tv"].min()),
                float(dsp["nearest_panel_tv"].max()),
            ],
        },
        "limitations": [
            "One common trainer/data seed per mixture; close cap differences are not replicated.",
            "The aggregate-linear-V candidates are 0.325-0.440 TV from their nearest fit-panel support.",
            "Discovery-model predictions and measured endpoints use the same candidate selection event.",
        ],
    }
    summary_path = OUTPUT_DIR / "measured_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    uncheatable = diagnostics["uncheatable_bpb"]
    table9 = diagnostics["table9_macro_bpb"]
    uncheatable_group = rows.loc[rows["target"].eq("uncheatable_bpb")].set_index("epoch_cap")
    predicted_cap8_to_cap10 = abs(
        float(uncheatable_group.loc[8, "predicted_target_bpb"])
        - float(uncheatable_group.loc[10, "predicted_target_bpb"])
    )
    table9_gain_vs_dsp = -summary["best_aggregate_v_minus_best_dsp"]["table9_macro_bpb"]
    table9_gain_vs_same_target_dsp = -summary["best_aggregate_v_minus_best_same_target_dsp"]["table9_macro_bpb"]
    report = f"""# Delphi 3e18 aggregate-linear-V challenger validation

## Result

The result is target-dependent. Shared-shape DSP remains better on Uncheatable, while the aggregate-linear-V
cap-8 Table-9 candidate is the best measured Table-9 policy across these two fresh one-phase sweeps.

[Open the measured comparison](measured_comparison.html).

- **Uncheatable:** cap 8 and cap 10 are a numerical prediction tie, differing by only
  `{predicted_cap8_to_cap10:.7f}` BPB. Their measured difference is also unresolved: selecting cap 10 instead of
  the observed cap-8 minimum costs `{uncheatable['selection_regret_bpb']:.6f}` BPB, only
  `{uncheatable['selection_regret_bpb'] / noise['uncheatable_bpb']:.2f}` conservative same-seed noise SD.
- **Table-9:** the model selects cap {table9['predicted_best_cap']} while the observed best cap is
  {table9['observed_best_cap']}. Selection regret is {table9['selection_regret_bpb']:.6f} BPB,
  about `{table9['selection_regret_bpb'] / noise['table9_macro_bpb']:.1f}` conservative noise SD; Spearman is
  {table9['spearman']:.3f}. This is the model's resolved cap-argmin failure.
- The best aggregate-linear-V row is {summary['best_aggregate_v']['uncheatable']['bpb']:.6f} on Uncheatable and
  {summary['best_aggregate_v']['table9']['bpb']:.6f} on Table-9. These are respectively
  {summary['best_aggregate_v_minus_best_dsp']['uncheatable_bpb']:+.6f} and
  {summary['best_aggregate_v_minus_best_dsp']['table9_macro_bpb']:+.6f} BPB relative to the best measured DSP rows,
  about `{summary['noise_calibration']['uncheatable_gap_vs_best_dsp_sd']:.1f}` and
  `{summary['noise_calibration']['table9_gain_vs_best_dsp_sd']:.1f}` conservative noise SDs respectively.

## Measured candidates

{_markdown_table(rows)}

## Interpretation

The aggregate-V structure identifies a Table-9-relevant candidate family that DSP misses: its cap-8 policy improves
on the best DSP Table-9 row across both target families by {table9_gain_vs_dsp:.6f} BPB, about
{table9_gain_vs_dsp / noise['table9_macro_bpb']:.1f} conservative noise SDs. Relative to the same-target DSP row,
the gain is {table9_gain_vs_same_target_dsp:.6f} BPB, about
{table9_gain_vs_same_target_dsp / noise['table9_macro_bpb']:.1f} SDs. This is a strong discovery candidate, but not
yet a validated frontier. The model predicts cap 10, whereas the measured curve turns upward after cap 8; its
Table-9 rank correlation is only {table9['spearman']:.3f}, and its absolute level is too optimistic.

The macro improvement is not driven by one component: the aggregate-V policy has lower BPB on
{summary['best_table9_component_comparison']['improved_components']} of
{summary['best_table9_component_comparison']['total_components']} components, with a median change of
{summary['best_table9_component_comparison']['median_delta_bpb']:+.6f} BPB. The largest improvement is
`{summary['best_table9_component_comparison']['largest_improvement_component']}`
({summary['best_table9_component_comparison']['largest_improvement_bpb']:+.6f}), while the largest regression is
`{summary['best_table9_component_comparison']['largest_regression_component']}`
({summary['best_table9_component_comparison']['largest_regression_bpb']:+.6f}).

Those 51 component deltas are correlated outputs from one paired comparison, not independent replications. Their
useful qualitative pattern is directional: the largest gains concentrate in math and code tasks, while the largest
regressions concentrate in world-knowledge and QA tasks.

For Uncheatable, DSP remains the stronger policy generator. The best aggregate-V row is
{summary['best_aggregate_v_minus_best_dsp']['uncheatable_bpb']:+.6f} BPB worse than the best DSP row, about
{summary['noise_calibration']['uncheatable_gap_vs_best_dsp_sd']:.1f} conservative noise SDs. The evidence therefore
supports using different structural heads for the two objectives, not replacing DSP wholesale. It also shows why
heldout rank fit alone is insufficient: Table-9 cap-argmin placement can fail even when the candidate family contains
a substantially better measured policy.

## Limitations

Every candidate uses one common trainer/data seed, making DSP-versus-aggregate-V differences common-random-number
paired comparisons. The conservative noise anchors have only six degrees of freedom and wide uncertainty intervals;
their SD ratios are calibration aids, not test statistics. The aggregate-linear-V candidates are 0.325-0.440 TV
from their nearest fit-panel support, but DSP candidates are also extrapolative at 0.291-0.543 TV. The Table-9 result
is the best observed across the two named sweeps, not yet a globally validated frontier claim; it needs paired
confirmation against the relevant incumbent.

## Provenance

- Iris root: `{IRIS_ROOT}`
- Exact endpoint: step `{FINAL_STEP}`
- Training root: `{TRAINING_ROOT}`
- Native Table-9 group: `{TABLE9_GROUP}`
- Candidate weights SHA-256: `{EXPECTED_CANDIDATE_WEIGHTS_SHA256}`
"""
    report_path = OUTPUT_DIR / "results.md"
    report_path.write_text(report)

    outputs = (
        measured_path,
        components_path,
        comparison_path,
        component_delta_path,
        summary_path,
        report_path,
        plot_path,
    )
    print(json.dumps({"outputs": {path.name: _sha256(path) for path in outputs}, "summary": summary}, indent=2))


if __name__ == "__main__":
    write_outputs()
