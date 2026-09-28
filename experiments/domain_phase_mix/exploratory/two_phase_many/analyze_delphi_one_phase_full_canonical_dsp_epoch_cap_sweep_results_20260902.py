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
#   "tabulate>=0.9",
#   "wandb>=0.21",
# ]
# ///

"""Collect and report the Delphi full-canonical DSP epoch-cap sweep."""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
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
    "delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_20260901"
)
CANDIDATE_SUMMARY = OUTPUT_DIR / "candidate_summary.csv"
CANDIDATE_WEIGHTS = OUTPUT_DIR / "candidate_weights.csv"
SHARED_SHAPE_DIR = REPO_ROOT / (
    "experiments/domain_phase_mix/exploratory/two_phase_many/reference_outputs/"
    "delphi_one_phase_dsp_epoch_cap_sweep_20260828"
)
TRAINING_ROOT = (
    "marin-us-east5/pinlin_calvin_xu/data_mixture/" "delphi_one_phase_full_canonical_dsp_epoch_cap_sweep_3e18_20260901"
)
TABLE9_GROUP = "olmo_base_eval_table9_delphi_3e18_one_phase_full_canonical_dsp_epoch_cap_sweep"
FINAL_STEP = 3006
EXPECTED_ROWS = 16
TARGET_STYLES = {
    "uncheatable_bpb": ("Optimized for Uncheatable", "#178A72"),
    "table9_macro_bpb": ("Optimized for Table-9", "#D95F32"),
}
PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    "toImageButtonOptions": {"format": "png", "scale": 4},
}


def _read_text(filesystem: gcsfs.GCSFileSystem, path: str) -> str:
    with filesystem.open(path, "rt") as handle:
        return handle.read()


def _collect_training_row(
    filesystem: gcsfs.GCSFileSystem,
    candidate: dict[str, Any],
) -> dict[str, Any]:
    candidate_id = str(candidate["candidate_id"])
    matches = filesystem.glob(
        f"{TRAINING_ROOT}/onephase_fullcanonical_dsp_{candidate_id}-*/checkpoints/eval_metrics.jsonl"
    )
    if len(matches) != 1:
        raise ValueError(f"{candidate_id}: expected one eval file, found {len(matches)}")
    eval_path = matches[0]
    run_dir = eval_path.removesuffix("/checkpoints/eval_metrics.jsonl")
    status = _read_text(filesystem, f"{run_dir}/.executor_status").strip()
    if status != "SUCCESS":
        raise ValueError(f"{candidate_id}: executor status is {status!r}")
    rows = [json.loads(line) for line in _read_text(filesystem, eval_path).splitlines() if line.strip()]
    endpoints = [row for row in rows if int(row.get("step", -1)) == FINAL_STEP]
    if len(endpoints) != 1:
        raise ValueError(f"{candidate_id}: expected one exact endpoint, found {len(endpoints)}")
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
    return next((candidate_id for candidate_id in candidate_ids if run.name.endswith(candidate_id)), None)


def _collect_table9(candidate_ids: tuple[str, ...]) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    api = wandb.Api(timeout=120)
    runs = list(api.runs("marin-community/marin-eval", filters={"group": TABLE9_GROUP}, per_page=200))
    finished: dict[str, dict[str, Any]] = {}
    components: list[dict[str, Any]] = []
    for run in sorted(runs, key=lambda item: item.created_at):
        candidate_id = _candidate_id(run, candidate_ids)
        summary = dict(run.summary)
        if candidate_id is None or run.state != "finished":
            continue
        macro = summary.get("olmo_base_easy/table9_macro_bpb")
        if macro is None:
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


def _candidate_tv_from_proportional(candidate_ids: tuple[str, ...]) -> pd.DataFrame:
    weights = pd.read_csv(CANDIDATE_WEIGHTS)
    required_columns = {"candidate_id", "domain", "weight", "proportional_weight"}
    missing_columns = required_columns - set(weights.columns)
    if missing_columns:
        raise ValueError(f"Candidate weights are missing columns: {sorted(missing_columns)}")
    if weights.duplicated(["candidate_id", "domain"]).any():
        raise ValueError("Candidate weights contain duplicate candidate/domain rows")

    observed_ids = set(weights.candidate_id.astype(str))
    expected_ids = set(candidate_ids)
    if observed_ids != expected_ids:
        raise ValueError(
            "Candidate-weight inventory does not match the frozen candidate summary: "
            f"missing={sorted(expected_ids - observed_ids)}, extra={sorted(observed_ids - expected_ids)}"
        )

    inventory = weights.groupby("candidate_id", as_index=False).agg(
        domain_count=("domain", "nunique"),
        weight_sum=("weight", "sum"),
        proportional_weight_sum=("proportional_weight", "sum"),
    )
    expected_domain_count = int(inventory.domain_count.iloc[0])
    if expected_domain_count != 39 or not inventory.domain_count.eq(expected_domain_count).all():
        raise ValueError("Every candidate must contain the same complete 39-bucket inventory")
    if not np.allclose(inventory.weight_sum, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("Candidate mixture weights do not sum to one")
    if not np.allclose(inventory.proportional_weight_sum, 1.0, rtol=0.0, atol=1e-9):
        raise ValueError("Proportional mixture weights do not sum to one")

    weights["absolute_weight_delta"] = (weights.weight - weights.proportional_weight).abs()
    distance = (
        weights.groupby("candidate_id", as_index=False)
        .absolute_weight_delta.sum()
        .rename(columns={"absolute_weight_delta": "l1_distance_from_proportional"})
    )
    distance["tv_distance_from_proportional"] = 0.5 * distance.l1_distance_from_proportional
    if not distance.tv_distance_from_proportional.between(0.0, 1.0).all():
        raise ValueError("TV distance must lie in [0, 1]")
    return distance


def collect_results() -> tuple[pd.DataFrame, pd.DataFrame]:
    candidates = pd.read_csv(CANDIDATE_SUMMARY)
    if len(candidates) != EXPECTED_ROWS or candidates.candidate_id.duplicated().any():
        raise ValueError("Frozen full-canonical candidate inventory is incomplete or non-unique")
    filesystem = gcsfs.GCSFileSystem(token="google_default")
    records = candidates.to_dict(orient="records")
    with ThreadPoolExecutor(max_workers=16) as pool:
        rows = list(pool.map(lambda candidate: _collect_training_row(filesystem, candidate), records))
    results = pd.DataFrame(rows)
    candidate_ids = tuple(str(candidate_id) for candidate_id in results.candidate_id)
    results = results.merge(_candidate_tv_from_proportional(candidate_ids), on="candidate_id", validate="one_to_one")
    table9, components = _collect_table9(candidate_ids)
    for index, row in results.iterrows():
        native = table9.get(str(row.candidate_id))
        if native is not None:
            results.loc[index, "table9_macro_bpb"] = native["macro"]
            results.loc[index, "table9_wandb_url"] = native["url"]
    results = results.sort_values(["target", "epoch_cap"]).reset_index(drop=True)
    if len(results) != EXPECTED_ROWS or not results.executor_status.eq("SUCCESS").all():
        raise ValueError("Training result inventory is incomplete")
    return results, components


def _diagnostics(results: pd.DataFrame, target: str, metric: str) -> dict[str, Any]:
    rows = results[results.target.eq(target) & results[metric].notna()].sort_values("epoch_cap")
    if len(rows) < 3:
        return {"rows": len(rows), "status": "insufficient native evaluations"}
    predicted = rows.predicted_target_bpb.to_numpy(float)
    observed = rows[metric].to_numpy(float)
    predicted_best = int(np.argmin(predicted))
    observed_best = int(np.argmin(observed))
    return {
        "rows": len(rows),
        "status": "complete" if len(rows) == 8 else "partial",
        "rmse_bpb": float(np.sqrt(np.mean((predicted - observed) ** 2))),
        "spearman": float(spearmanr(predicted, observed).statistic),
        "predicted_best_cap": int(rows.iloc[predicted_best].epoch_cap),
        "observed_best_cap": int(rows.iloc[observed_best].epoch_cap),
        "predicted_selection_regret_bpb": float(observed[predicted_best] - observed[observed_best]),
    }


def _tv_diagnostics(results: pd.DataFrame, target: str) -> dict[str, Any]:
    rows = results[results.target.eq(target)].sort_values("epoch_cap")
    distances = rows.tv_distance_from_proportional.to_numpy(float)
    return {
        "rows": len(rows),
        "monotone_nondecreasing": bool(np.all(np.diff(distances) >= -1e-12)),
        "minimum": float(distances.min()),
        "maximum": float(distances.max()),
        "by_epoch_cap": {
            str(int(epoch_cap)): float(distance) for epoch_cap, distance in zip(rows.epoch_cap, distances, strict=True)
        },
    }


def _write_plot(results: pd.DataFrame, shared_summary: dict[str, Any]) -> Path:
    panels = (("uncheatable_bpb", "Uncheatable BPB"), ("table9_macro_bpb", "Table-9 macro BPB"))
    figure = make_subplots(
        rows=1,
        cols=3,
        subplot_titles=[*[title for _, title in panels], "Mixture divergence"],
        column_widths=[0.36, 0.36, 0.28],
        horizontal_spacing=0.08,
    )
    for column, (metric, _) in enumerate(panels, start=1):
        for target, (label, color) in TARGET_STYLES.items():
            selected = results[results.target.eq(target)].sort_values("epoch_cap")
            measured = selected[selected[metric].notna()]
            figure.add_trace(
                go.Scatter(
                    x=measured.epoch_cap,
                    y=measured[metric],
                    mode="lines+markers",
                    name=label,
                    legendgroup=target,
                    showlegend=column == 1,
                    line={"color": color, "width": 3},
                    marker={"size": 10, "color": color},
                    customdata=measured.candidate_id,
                    hovertemplate="<b>%{customdata}</b><br>Cap %{x}<br>Measured %{y:.6f}<extra></extra>",
                ),
                row=1,
                col=column,
            )
        matching = results[results.target.eq(metric)].sort_values("epoch_cap")
        label, color = TARGET_STYLES[metric]
        figure.add_trace(
            go.Scatter(
                x=matching.epoch_cap,
                y=matching.predicted_target_bpb,
                mode="lines+markers",
                name=f"Full DSP prediction: {label.removeprefix('Optimized for ')}",
                legendgroup=f"prediction-{metric}",
                visible="legendonly",
                line={"color": color, "width": 2, "dash": "dash"},
                marker={"symbol": "circle-open", "size": 8},
                hovertemplate="Cap %{x}<br>Predicted %{y:.6f}<extra></extra>",
            ),
            row=1,
            col=column,
        )
        shared_key = "uncheatable" if metric == "uncheatable_bpb" else "table9"
        shared_best = float(shared_summary["best_across_all_candidates"][shared_key]["bpb"])
        figure.add_hline(
            y=shared_best,
            line={"color": "#17324D", "width": 1.5, "dash": "dot"},
            annotation_text="Shared-shape DSP best",
            annotation_position="bottom right",
            row=1,
            col=column,
        )
    for target, (label, color) in TARGET_STYLES.items():
        selected = results[results.target.eq(target)].sort_values("epoch_cap")
        figure.add_trace(
            go.Scatter(
                x=selected.epoch_cap,
                y=selected.tv_distance_from_proportional,
                mode="lines+markers",
                name=f"{label}: TV",
                legendgroup=target,
                showlegend=False,
                line={"color": color, "width": 3},
                marker={"size": 10, "color": color},
                customdata=selected.candidate_id,
                hovertemplate=("<b>%{customdata}</b><br>Cap %{x}<br>TV from proportional %{y:.4f}<extra></extra>"),
            ),
            row=1,
            col=3,
        )
    table9_count = int(results.table9_macro_bpb.notna().sum())
    figure.update_layout(
        title={"text": "Delphi 3e18 full-canonical DSP epoch-cap validation", "x": 0.5, "xanchor": "center"},
        paper_bgcolor="#F8F3E8",
        plot_bgcolor="#F8F3E8",
        font={"family": "Avenir Next, sans-serif", "size": 15, "color": "#17324D"},
        height=700,
        margin={"l": 70, "r": 30, "t": 135, "b": 150},
        legend={"orientation": "h", "x": 0.5, "xanchor": "center", "y": 1.03, "yanchor": "bottom"},
    )
    figure.update_xaxes(title_text="Whole-run materialized epoch cap", dtick=2, gridcolor="#DCE5EA")
    figure.update_yaxes(title_text="BPB (lower is better)", gridcolor="#DCE5EA", row=1, col=1)
    figure.update_yaxes(title_text="BPB (lower is better)", gridcolor="#DCE5EA", row=1, col=2)
    figure.update_yaxes(
        title_text="TV distance from proportional",
        range=[0.0, 1.0],
        gridcolor="#DCE5EA",
        row=1,
        col=3,
    )
    figure.add_annotation(
        x=0.5,
        y=-0.27,
        xref="paper",
        yref="paper",
        showarrow=False,
        text=(
            f"All 16 Uncheatable endpoints are measured with one common seed. Native Table-9 is "
            f"{table9_count}/16 complete; missing cells are omitted. Dashed model predictions are hidden by default.<br>"
            "TV is half the L1 distance between each optimized mixture and the proportional baseline."
        ),
        font={"size": 13, "color": "#52657A"},
    )
    path = OUTPUT_DIR / "cap_performance.html"
    pio.write_html(figure, path, include_plotlyjs=True, full_html=True, config=PLOT_CONFIG)
    return path


def _markdown_table(results: pd.DataFrame) -> str:
    columns = [
        "candidate_id",
        "epoch_cap",
        "predicted_target_bpb",
        "uncheatable_bpb",
        "table9_macro_bpb",
        "github_cpp_bpb",
        "tv_distance_from_proportional",
    ]
    return results[columns].to_markdown(index=False, floatfmt=".6f")


def write_outputs() -> dict[str, Any]:
    results, components = collect_results()
    shared_summary = json.loads((SHARED_SHAPE_DIR / "measured_summary.json").read_text())
    best_uncheatable = results.loc[results.uncheatable_bpb.idxmin()]
    shared_uncheatable = float(shared_summary["best_across_all_candidates"]["uncheatable"]["bpb"])
    table9_complete = int(results.table9_macro_bpb.notna().sum())
    summary = {
        "generated_at": datetime.now(UTC).isoformat(),
        "training_rows": len(results),
        "table9_rows": table9_complete,
        "table9_expected_rows": EXPECTED_ROWS,
        "best_uncheatable": {
            "candidate_id": best_uncheatable.candidate_id,
            "bpb": float(best_uncheatable.uncheatable_bpb),
            "epoch_cap": int(best_uncheatable.epoch_cap),
        },
        "shared_shape_best_uncheatable_bpb": shared_uncheatable,
        "full_minus_shared_shape_best_uncheatable_bpb": float(best_uncheatable.uncheatable_bpb - shared_uncheatable),
        "diagnostics": {
            "uncheatable_bpb": _diagnostics(results, "uncheatable_bpb", "uncheatable_bpb"),
            "table9_macro_bpb": _diagnostics(results, "table9_macro_bpb", "table9_macro_bpb"),
        },
        "tv_distance_from_proportional": {target: _tv_diagnostics(results, target) for target in TARGET_STYLES},
        "limitations": [
            "One common data/trainer seed per mixture; cap-to-cap differences do not "
            "have replicate confidence intervals.",
            "Native Table-9 is incomplete until the collision-proof eval-only repair finishes.",
        ],
    }
    results.to_csv(OUTPUT_DIR / "measured_results.csv", index=False)
    components.to_csv(OUTPUT_DIR / "measured_table9_components.csv", index=False)
    (OUTPUT_DIR / "measured_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    plot_path = _write_plot(results, shared_summary)

    diagnostic = summary["diagnostics"]["uncheatable_bpb"]
    uncheatable_tv = summary["tv_distance_from_proportional"]["uncheatable_bpb"]
    table9_tv = summary["tv_distance_from_proportional"]["table9_macro_bpb"]
    report = [
        "# Delphi 3e18 full-canonical DSP epoch-cap validation",
        "",
        "## Result",
        "",
        (
            f"All 16 trainings and inline Uncheatable endpoints are complete. The best full-canonical candidate is "
            f"`{best_uncheatable.candidate_id}` at cap {int(best_uncheatable.epoch_cap)}, with Uncheatable BPB "
            f"`{float(best_uncheatable.uncheatable_bpb):.6f}`. This is "
            f"`{float(best_uncheatable.uncheatable_bpb - shared_uncheatable):+.6f}` BPB relative to the previous "
            f"shared-shape DSP best (`{shared_uncheatable:.6f}`), so the added per-bucket nonlinear flexibility did not "
            "improve the validated optimum."
        ),
        "",
        (
            "The materialized optima move monotonically away from proportional as the cap loosens. "
            f"Uncheatable-target TV rises from `{uncheatable_tv['minimum']:.4f}` to "
            f"`{uncheatable_tv['maximum']:.4f}`; Table-9-target TV rises from `{table9_tv['minimum']:.4f}` to "
            f"`{table9_tv['maximum']:.4f}`. Thus the full-canonical optimizer uses each relaxation to make a more "
            "extreme mixture even though measured Uncheatable performance stops improving after cap 4-6."
        ),
        "",
        (
            "The full-canonical model predicts monotonically better outcomes as the cap loosens, but measured "
            f"Uncheatable performance turns upward after cap 4-6. On its own target path, RMSE is "
            f"`{diagnostic['rmse_bpb']:.6f}`, Spearman is `{diagnostic['spearman']:.3f}`, and selecting the predicted "
            f"cap-16 optimum incurs `{diagnostic['predicted_selection_regret_bpb']:.6f}` BPB regret."
        ),
        "",
        (
            f"Native Table-9 is only {table9_complete}/{EXPECTED_ROWS} complete because long evaluator names collided "
            "after Iris child-name truncation. The trainings are intact; a short-name eval-only repair is required "
            "before drawing a Table-9 conclusion."
        ),
        "",
        f"[Open the interactive cap plot]({plot_path.name}).",
        "",
        "## Measurements",
        "",
        _markdown_table(results),
        "",
        "## Limitations",
        "",
        "- One common data/trainer seed per candidate; differences have no replicate confidence intervals.",
        "- Missing Table-9 cells are blank rather than imputed.",
        "",
    ]
    (OUTPUT_DIR / "results.md").write_text("\n".join(report))
    return summary


def main() -> None:
    summary = write_outputs()
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
