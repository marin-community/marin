# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy>=2.0",
#   "plotly>=6.0",
#   "scipy>=1.14",
#   "wandb>=0.21",
# ]
# ///

from __future__ import annotations

import csv
import hashlib
import json
import math
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
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
    "delphi_one_phase_dsp_epoch_cap_sweep_20260828"
)
CANDIDATE_SUMMARY = OUTPUT_DIR / "candidate_summary.csv"
NOISE_RESULTS = REPO_ROOT / (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "delphi_3e18_fixed_aggregate_phase_snr_20260724/same_seed_delta_noise.csv"
)
TRAINING_ROOT = "gs://marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_one_phase_dsp_epoch_cap_sweep_3e18_20260828"
IRIS_ROOT = "/calvinxu/dm-delphi-3e18-onephase-dsp-epochcaps-v6e8-retry6-20260829"
TABLE9_GROUP = "olmo_base_eval_table9_delphi_3e18_one_phase_dsp_epoch_cap_sweep"
FINAL_STEP = 3006
TARGET_METRICS = {
    "uncheatable_bpb": "uncheatable_bpb",
    "table9_macro_bpb": "table9_macro_bpb",
}
TARGET_STYLES = {
    "uncheatable_bpb": ("Optimized for Uncheatable", "#178A72"),
    "table9_macro_bpb": ("Optimized for Table-9", "#D95F32"),
}
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


def gcs_cat(uri: str) -> str:
    """Read a small regional GCS artifact through the authenticated CLI."""
    return subprocess.run(
        ["gcloud", "storage", "cat", uri],
        check=True,
        capture_output=True,
        text=True,
    ).stdout


def candidate_predictions() -> tuple[list[dict[str, str]], dict[str, dict[str, str]]]:
    with CANDIDATE_SUMMARY.open() as handle:
        rows = list(csv.DictReader(handle))
    return rows, {row["candidate_id"]: row for row in rows}


def conservative_noise_scales() -> dict[str, float]:
    """Load conservative same-seed delta noise anchors for the two targets."""
    with NOISE_RESULTS.open() as handle:
        rows = list(csv.DictReader(handle))
    anchors = {
        "uncheatable_bpb": ("uncheatable_frontier", "uncheatable"),
        "table9_macro_bpb": ("table9_frontier", "table9"),
    }
    return {
        target: float(
            next(row["same_seed_delta_noise_sd_bpb"] for row in rows if (row["anchor_id"], row["target"]) == anchor)
        )
        for target, anchor in anchors.items()
    }


def finished_table9_runs(api: wandb.Api) -> dict[str, object]:
    runs = list(
        api.runs(
            "marin-community/marin-eval",
            filters={"group": TABLE9_GROUP},
            per_page=200,
        )
    )
    finished: dict[str, object] = {}
    for run in sorted(runs, key=lambda item: item.created_at):
        if run.state != "finished":
            continue
        candidate_id = run.name.removeprefix("t9_onephase_dsp_")
        if run.summary.get("olmo_base_easy/table9_macro_bpb") is not None:
            finished[candidate_id] = run
    return finished


def collect_results() -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float]]:
    candidate_rows, predictions = candidate_predictions()
    fit_panel_best = {
        target: float(next(row for row in candidate_rows if row["target"] == target)["fit_panel_best_observed_bpb"])
        for target in TARGET_METRICS
    }

    api = wandb.Api(timeout=120)
    training_runs = list(
        api.runs(
            "marin-community/marin",
            filters={"tags": {"$in": ["whole-run-epoch-cap-sweep"]}},
            per_page=200,
        )
    )
    if len(training_runs) != 11:
        raise RuntimeError(f"Expected 11 training W&B runs, found {len(training_runs)}")
    native_runs = finished_table9_runs(api)

    rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for run in training_runs:
        tag_map = {tag.split("=", 1)[0]: tag.split("=", 1)[1] for tag in run.tags if "=" in tag}
        candidate_id = tag_map["source_run"]
        prediction = predictions[candidate_id]
        eval_uri = f"{TRAINING_ROOT}/{run.name}/checkpoints/eval_metrics.jsonl"
        status_uri = f"{TRAINING_ROOT}/{run.name}/.executor_status"

        executor_status = gcs_cat(status_uri).strip()
        if executor_status != "SUCCESS":
            raise RuntimeError(f"{candidate_id}: executor status is {executor_status!r}")

        eval_rows = [json.loads(line) for line in gcs_cat(eval_uri).splitlines() if line.strip()]
        final_rows = [row for row in eval_rows if row.get("step") == FINAL_STEP]
        if len(final_rows) != 1:
            raise RuntimeError(f"{candidate_id}: expected one step-{FINAL_STEP} row, found {len(final_rows)}")
        final = final_rows[0]

        native = native_runs.get(candidate_id)
        if native is None:
            raise RuntimeError(f"{candidate_id}: no finished native Table-9 result")
        native_summary = dict(native.summary)
        table9_value = float(native_summary["olmo_base_easy/table9_macro_bpb"])
        components = {
            component: float(native_summary[f"olmo_base_easy/table9/{component}/bpb"])
            for component in table9_components()
        }
        reconstructed = table9_macro(components)
        if not math.isclose(reconstructed, table9_value, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"{candidate_id}: 51-component macro {reconstructed:.15f} != native {table9_value:.15f}")
        for position, component in enumerate(table9_components()):
            component_rows.append(
                {
                    "candidate_id": candidate_id,
                    "target": prediction["target"],
                    "epoch_cap": int(prediction["epoch_cap"]),
                    "component_position": position,
                    "component": component,
                    "bpb": components[component],
                }
            )

        rows.append(
            {
                "candidate_id": candidate_id,
                "target": prediction["target"],
                "epoch_cap": int(prediction["epoch_cap"]),
                "predicted_target_bpb": float(prediction["runtime_predicted_bpb"]),
                "uncheatable_bpb": float(final["eval/uncheatable_eval/bpb"]),
                "uncheatable_macro_bpb": float(final["eval/uncheatable_eval/macro_bpb"]),
                "github_cpp_bpb": float(final["eval/uncheatable_eval/github_cpp/bpb"]),
                "github_python_bpb": float(final["eval/uncheatable_eval/github_python/bpb"]),
                "table9_macro_bpb": table9_value,
                "max_materialized_epoch": float(prediction["max_materialized_epoch"]),
                "effective_buckets": float(prediction["effective_buckets"]),
                "tv_to_proportional": float(prediction["tv_to_proportional"]),
                "nearest_panel_tv": float(prediction["nearest_panel_tv"]),
                "largest_bucket": prediction["largest_bucket"],
                "largest_weight": float(prediction["largest_weight"]),
                "final_step": FINAL_STEP,
                "executor_status": executor_status,
                "training_wandb_state": run.state,
                "training_wandb_url": run.url,
                "native_table9_wandb_url": native.url,
                "eval_metrics_uri": eval_uri,
            }
        )

    rows.sort(key=lambda row: (str(row["target"]), int(row["epoch_cap"])))
    component_rows.sort(key=lambda row: (str(row["target"]), int(row["epoch_cap"]), int(row["component_position"])))
    return rows, component_rows, fit_panel_best


def model_diagnostics(rows: list[dict[str, object]], fit_panel_best: dict[str, float]) -> dict[str, dict[str, object]]:
    diagnostics: dict[str, dict[str, object]] = {}
    for target, metric in TARGET_METRICS.items():
        selected = [row for row in rows if row["target"] == target]
        predicted = np.asarray([row["predicted_target_bpb"] for row in selected], dtype=float)
        observed = np.asarray([row[metric] for row in selected], dtype=float)
        caps = np.asarray([row["epoch_cap"] for row in selected], dtype=int)
        predicted_best = int(np.argmin(predicted))
        observed_best = int(np.argmin(observed))
        diagnostics[target] = {
            "rows": len(selected),
            "prediction_rmse_bpb": float(np.sqrt(np.mean((predicted - observed) ** 2))),
            "prediction_mae_bpb": float(np.mean(np.abs(predicted - observed))),
            "mean_observed_minus_predicted_bpb": float(np.mean(observed - predicted)),
            "spearman": float(spearmanr(predicted, observed).statistic),
            "predicted_best_cap": int(caps[predicted_best]),
            "observed_best_cap": int(caps[observed_best]),
            "predicted_selection_regret_bpb": float(observed[predicted_best] - observed[observed_best]),
            "observed_best_bpb": float(observed[observed_best]),
            "fit_panel_best_observed_bpb": fit_panel_best[target],
            "improvement_vs_fit_panel_best_bpb": float(fit_panel_best[target] - observed[observed_best]),
        }
    return diagnostics


def markdown_table(rows: list[dict[str, object]]) -> str:
    lines = [
        "| candidate | cap | predicted target | Uncheatable | Table-9 | GitHub C++ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f'| {row["candidate_id"]} | {row["epoch_cap"]} | {float(row["predicted_target_bpb"]):.6f} | '
            f'{float(row["uncheatable_bpb"]):.6f} | {float(row["table9_macro_bpb"]):.6f} | '
            f'{float(row["github_cpp_bpb"]):.6f} |'
        )
    return "\n".join(lines)


def write_cap_performance_plot(rows: list[dict[str, object]]) -> Path:
    """Plot measured performance against the whole-run epoch cap."""
    panels = (
        ("uncheatable_bpb", "Uncheatable BPB"),
        ("table9_macro_bpb", "Table-9 macro BPB"),
    )
    figure = make_subplots(rows=1, cols=2, subplot_titles=tuple(title for _, title in panels), horizontal_spacing=0.1)

    for column, (metric, _) in enumerate(panels, start=1):
        for target, (label, color) in TARGET_STYLES.items():
            selected = sorted(
                (row for row in rows if row["target"] == target),
                key=lambda row: int(row["epoch_cap"]),
            )
            figure.add_trace(
                go.Scatter(
                    x=[row["epoch_cap"] for row in selected],
                    y=[row[metric] for row in selected],
                    mode="lines+markers",
                    name=label,
                    legendgroup=target,
                    showlegend=column == 1,
                    line={"color": color, "width": 3},
                    marker={"color": color, "size": 10, "line": {"color": "#FFF9EE", "width": 1.5}},
                    customdata=[row["candidate_id"] for row in selected],
                    hovertemplate=("<b>%{customdata}</b><br>Epoch cap: %{x}<br>Measured BPB: %{y:.6f}<extra></extra>"),
                ),
                row=1,
                col=column,
            )

        matching_target = metric
        prediction_rows = sorted(
            (row for row in rows if row["target"] == matching_target),
            key=lambda row: int(row["epoch_cap"]),
        )
        prediction_label, prediction_color = TARGET_STYLES[matching_target]
        figure.add_trace(
            go.Scatter(
                x=[row["epoch_cap"] for row in prediction_rows],
                y=[row["predicted_target_bpb"] for row in prediction_rows],
                mode="lines+markers",
                name=f"DSP prediction: {prediction_label.removeprefix('Optimized for ')}",
                legendgroup=f"prediction-{matching_target}",
                visible="legendonly",
                line={"color": prediction_color, "width": 2, "dash": "dash"},
                marker={"color": prediction_color, "size": 7, "symbol": "circle-open"},
                hovertemplate="Epoch cap: %{x}<br>DSP prediction: %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )

        best = min(rows, key=lambda row: float(row[metric]))
        figure.add_trace(
            go.Scatter(
                x=[best["epoch_cap"]],
                y=[best[metric]],
                mode="markers",
                name="Best observed",
                legendgroup="best-observed",
                showlegend=column == 1,
                marker={"color": "#17324D", "size": 15, "symbol": "star", "line": {"color": "#FFF9EE", "width": 1}},
                customdata=[best["candidate_id"]],
                hovertemplate="<b>%{customdata}</b><br>Best observed: %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )

    figure.update_layout(
        title={
            "text": "Delphi 3e18 one-phase epoch-cap sweep",
            "x": 0.5,
            "xanchor": "center",
            "font": {"size": 28, "family": "Avenir Next, sans-serif", "color": "#17324D"},
        },
        legend={
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "xanchor": "center",
            "x": 0.5,
            "font": {"size": 13},
        },
        margin={"l": 75, "r": 35, "t": 135, "b": 105},
        height=650,
        paper_bgcolor="#F8F3E8",
        plot_bgcolor="#F8F3E8",
        font={"family": "Avenir Next, sans-serif", "size": 15, "color": "#17324D"},
        hoverlabel={"bgcolor": "#FFF9EE", "font": {"color": "#17324D"}},
    )
    figure.update_xaxes(
        title_text="Whole-run materialized epoch cap",
        tickmode="array",
        tickvals=[2, 4, 6, 8, 10, 12],
        gridcolor="#DCE5EA",
        linecolor="#17324D",
        row=1,
    )
    figure.update_yaxes(title_text="BPB (lower is better)", gridcolor="#DCE5EA", linecolor="#17324D", row=1, col=1)
    figure.update_yaxes(title_text="BPB (lower is better)", gridcolor="#DCE5EA", linecolor="#17324D", row=1, col=2)
    figure.add_annotation(
        x=0.5,
        y=-0.2,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=(
            "Solid lines are measured endpoints. Dashed DSP predictions are hidden by default; click their legend "
            "entries to show them. One common trainer/data seed per mixture."
        ),
        font={"size": 13, "color": "#52657A"},
    )

    plot_path = OUTPUT_DIR / "cap_performance.html"
    pio.write_html(figure, plot_path, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG)
    return plot_path


def write_outputs() -> None:
    rows, component_rows, fit_panel_best = collect_results()
    diagnostics = model_diagnostics(rows, fit_panel_best)
    noise = conservative_noise_scales()

    measured_path = OUTPUT_DIR / "measured_results.csv"
    with measured_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    components_path = OUTPUT_DIR / "measured_table9_components.csv"
    with components_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(component_rows[0]))
        writer.writeheader()
        writer.writerows(component_rows)

    plot_path = write_cap_performance_plot(rows)

    best_uncheatable = min(rows, key=lambda row: float(row["uncheatable_bpb"]))
    best_table9 = min(rows, key=lambda row: float(row["table9_macro_bpb"]))
    best_table9_target = min(
        (row for row in rows if row["target"] == "table9_macro_bpb"),
        key=lambda row: float(row["table9_macro_bpb"]),
    )
    uncheatable_rows = [row for row in rows if row["target"] == "uncheatable_bpb"]
    cap2_uncheatable = next(row for row in uncheatable_rows if row["epoch_cap"] == 2)
    cap4_uncheatable = next(row for row in uncheatable_rows if row["epoch_cap"] == 4)
    cap6_uncheatable = next(row for row in uncheatable_rows if row["epoch_cap"] == 6)
    cap8_uncheatable = next(row for row in uncheatable_rows if row["epoch_cap"] == 8)
    table9_rows = [row for row in rows if row["target"] == "table9_macro_bpb"]
    cap4_table9 = next(row for row in table9_rows if row["epoch_cap"] == 4)
    cap12_table9 = next(row for row in table9_rows if row["epoch_cap"] == 12)

    uncheatable_cap2_to_cap4 = float(cap2_uncheatable["uncheatable_bpb"]) - float(cap4_uncheatable["uncheatable_bpb"])
    uncheatable_cap4_to_best = float(cap4_uncheatable["uncheatable_bpb"]) - float(best_uncheatable["uncheatable_bpb"])
    uncheatable_cap6_to_cap8 = abs(
        float(cap6_uncheatable["uncheatable_bpb"]) - float(cap8_uncheatable["uncheatable_bpb"])
    )
    table9_cap4_to_cap6 = float(cap4_table9["table9_macro_bpb"]) - float(best_table9_target["table9_macro_bpb"])
    table9_cap12_selection_regret = float(cap12_table9["table9_macro_bpb"]) - float(
        best_table9_target["table9_macro_bpb"]
    )
    cross_target_table9_gap = float(best_table9_target["table9_macro_bpb"]) - float(best_table9["table9_macro_bpb"])

    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "iris_root": IRIS_ROOT,
        "training_root": TRAINING_ROOT,
        "final_step": FINAL_STEP,
        "runtime_rows": len(rows),
        "native_table9_components_per_row": len(table9_components()),
        "all_training_executor_status_success": all(row["executor_status"] == "SUCCESS" for row in rows),
        "diagnostics": diagnostics,
        "best_across_all_candidates": {
            "uncheatable": {
                "candidate_id": best_uncheatable["candidate_id"],
                "bpb": best_uncheatable["uncheatable_bpb"],
            },
            "table9": {
                "candidate_id": best_table9["candidate_id"],
                "bpb": best_table9["table9_macro_bpb"],
            },
        },
        "cross_target_table9_result": {
            "best_table9_target_candidate_id": best_table9_target["candidate_id"],
            "best_table9_target_bpb": best_table9_target["table9_macro_bpb"],
            "best_overall_candidate_id": best_table9["candidate_id"],
            "best_overall_bpb": best_table9["table9_macro_bpb"],
            "table9_target_minus_overall_bpb": cross_target_table9_gap,
            "gap_in_conservative_noise_sd": cross_target_table9_gap / noise["table9_macro_bpb"],
        },
        "overall_improvement_vs_fit_panel_best_bpb": {
            "uncheatable_bpb": float(fit_panel_best["uncheatable_bpb"] - float(best_uncheatable["uncheatable_bpb"])),
            "table9_macro_bpb": float(fit_panel_best["table9_macro_bpb"] - float(best_table9["table9_macro_bpb"])),
        },
        "noise_calibration": {
            "uncheatable_same_seed_delta_sd_bpb": noise["uncheatable_bpb"],
            "table9_same_seed_delta_sd_bpb": noise["table9_macro_bpb"],
            "noise_df": 6,
            "uncheatable_cap2_to_cap4_gain_sd": uncheatable_cap2_to_cap4 / noise["uncheatable_bpb"],
            "uncheatable_cap4_to_best_gain_sd": uncheatable_cap4_to_best / noise["uncheatable_bpb"],
            "uncheatable_cap6_to_cap8_gap_sd": uncheatable_cap6_to_cap8 / noise["uncheatable_bpb"],
            "table9_cap4_to_cap6_gain_sd": table9_cap4_to_cap6 / noise["table9_macro_bpb"],
            "table9_cap12_selection_regret_sd": table9_cap12_selection_regret / noise["table9_macro_bpb"],
        },
        "uncheatable_cap4_to_best_gain_bpb": uncheatable_cap4_to_best,
        "limitations": [
            "One common trainer/data seed per mixture; cap-to-cap differences have no replicate confidence intervals.",
            "The 20 nominal cells alias to 11 runtime-distinct mixtures.",
            (
                "Training W&B finalization is incomplete for four rows, but exact step-3006 "
                "eval_metrics.jsonl and successful executor status are present for all 11."
            ),
        ],
    }
    summary_path = OUTPUT_DIR / "measured_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")

    uncheatable_diag = diagnostics["uncheatable_bpb"]
    table9_diag = diagnostics["table9_macro_bpb"]
    table9_cap6_minus_cap4 = float(best_table9_target["table9_macro_bpb"]) - float(cap4_table9["table9_macro_bpb"])
    report = f"""# Delphi 3e18 one-phase DSP epoch-cap sweep: measured results

## Result

The whole-run epoch cap is useful, but the two DSP heads behave very differently out of sample.

[Open the interactive cap-performance plot](cap_performance.html).

- **Uncheatable:** cap 2 to cap 4 improves BPB by `{uncheatable_cap2_to_cap4:.6f}`, about
  `{uncheatable_cap2_to_cap4 / noise['uncheatable_bpb']:.1f}` conservative same-seed noise SDs. Caps 4 through
  10 span only `{uncheatable_cap4_to_best:.6f}` BPB, about
  `{uncheatable_cap4_to_best / noise['uncheatable_bpb']:.1f}` SDs, so the apparent exact DSP ordering
  (Spearman `{uncheatable_diag["spearman"]:.3f}`) does not resolve a preferred cap within that plateau.
- **Table-9:** the response is U-shaped, with the Table-9-targeted cap-6 row best at
  `{float(best_table9_target["table9_macro_bpb"]):.6f}`. The surrogate instead predicts monotonic improvement
  through cap 12, yielding Spearman `{table9_diag["spearman"]:.3f}` and
  `{table9_diag["predicted_selection_regret_bpb"]:.6f}` BPB selection regret, about
  `{table9_cap12_selection_regret / noise['table9_macro_bpb']:.1f}` conservative noise SDs. This is a real
  argmin-placement failure, not merely endpoint noise.
- **Cross-target:** the Uncheatable-targeted cap-6 row is nominally `{cross_target_table9_gap:.6f}` BPB below
  the best Table-9-targeted row, only `{cross_target_table9_gap / noise['table9_macro_bpb']:.2f}` conservative
  noise SD. The experiment does not resolve which target head produces the better Table-9 policy.
- Relative to the canonical 280-row fit panel, the best same-target rows improve Uncheatable by
  `{uncheatable_diag["improvement_vs_fit_panel_best_bpb"]:.6f}` BPB and Table-9 by
  `{table9_diag["improvement_vs_fit_panel_best_bpb"]:.6f}` BPB. Across both target families, the best Table-9
  row improves on the panel best by
  `{summary["overall_improvement_vs_fit_panel_best_bpb"]["table9_macro_bpb"]:.6f}` BPB. These are descriptive
  cross-experiment comparisons to the canonical panel, not fresh replicated frontier claims.

## Uncheatable-targeted mixtures

{markdown_table(uncheatable_rows)}

## Table-9-targeted mixtures

{markdown_table(table9_rows)}

## Interpretation

The cap is a strong and simple regularizer, but loosening it is not uniformly beneficial. The Uncheatable
path reaches a broad, noise-unresolved plateau after cap 4. The Table-9 head extrapolates toward increasingly
concentrated, far-from-support mixtures (the cap-12 candidate places 44.3% on `literature_high`) and misses the observed
interior optimum. High held-out rank correlation on the historical panel therefore did not transfer to
optimization along this extrapolative path.

The 20 nominal target/cap cells collapse to 11 runtime-distinct mixtures: Uncheatable caps 12-20 alias cap 10,
and Table-9 caps 14-20 alias cap 12. Every training executor succeeded and every native Table-9 evaluation
completed. Four training W&B runs are marked crashed because final syncing was interrupted; their exact
step-3006 metrics are recovered from `checkpoints/eval_metrics.jsonl` and independently paired with successful
native evaluations.

## Limitation and next gate

Each mixture has one common data/trainer seed. The `{summary["uncheatable_cap4_to_best_gain_bpb"]:.6f}` BPB
spread from Uncheatable cap 4 to cap 10, and the `{table9_cap6_minus_cap4:+.6f}` BPB cap-6 versus cap-4
Table-9 difference, require paired repeats before choosing among nearby caps. The noise anchors above have only
six degrees of freedom and wide uncertainty intervals, so their SD ratios are calibration aids rather than test
statistics. The robust conclusion is the coarse one: cap 2 is too strict, very loose Table-9 caps are harmful,
and the useful region is approximately cap 4-10 for Uncheatable and cap 4-6 for Table-9.

## Provenance

- Iris root: `{IRIS_ROOT}`
- Exact endpoint: step `{FINAL_STEP}`
- Training metrics: `{TRAINING_ROOT}/<run>/checkpoints/eval_metrics.jsonl`
- Native Table-9 group: `{TABLE9_GROUP}`
- `measured_results.csv`: 11 endpoint rows
- `measured_table9_components.csv`: 51 canonical Table-9 components per row
"""
    report_path = OUTPUT_DIR / "results.md"
    report_path.write_text(report)

    hashes = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest()
        for path in (measured_path, components_path, summary_path, report_path, plot_path)
    }
    print(json.dumps({"outputs": hashes, "summary": summary}, indent=2, sort_keys=True))


if __name__ == "__main__":
    write_outputs()
