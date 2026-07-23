# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy>=1.7",
#   "fsspec>=2025.7",
#   "gcsfs>=2025.7",
#   "kaleido==0.2.1",
#   "numpy>=2.0",
#   "pandas>=2.2",
#   "plotly>=6.0",
#   "scikit-learn>=1.6",
#   "scipy>=1.15",
#   "tabulate>=0.9",
# ]
# ///
"""Fit and visualize expanded-3e18 raw surrogate optima.

The fits use the original 280-row Delphi 3e18 two-phase swarm, 238
coordinate-disjoint tied policies trained at the same setting, and 480
phase-varying 3e18 extension policies. Hyperparameter choices remain frozen
from the original Delphi 3e18 Observatory fits.
"""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

REPO_ROOT = Path(__file__).resolve().parents[4]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    analyze_delphi_phase_policy_sample_efficiency_20260721 as analysis,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/delphi_expanded_fit_raw_optima_20260721"
PLOT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}, "responsive": True}
PHASE_0_COLOR = "#d95f2d"
PROPORTIONAL_COLOR = "#78909c"
EXPECTED_BASE_ROWS = 280
EXPECTED_TIED_ROWS = 238
EXPECTED_EXTENSION_ROWS = 480
EXPECTED_TOTAL_ROWS = EXPECTED_BASE_ROWS + EXPECTED_TIED_ROWS + EXPECTED_EXTENSION_ROWS


@dataclass(frozen=True)
class CandidateSpec:
    """One endpoint fit and raw optimum to audit."""

    target: str
    model_id: str
    label: str


@dataclass(frozen=True)
class CandidateResult:
    """Persisted raw optimum and its per-domain diagnostics."""

    spec: CandidateSpec
    summary: dict[str, Any]
    weights: pd.DataFrame


CANDIDATES = (
    CandidateSpec("uncheatable", "separate_heads", "Uncheatable / Separate heads"),
    CandidateSpec("table9", "compact_retained_state", "Table-9 / Compact retained state"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--optimizer-starts", type=int, default=16)
    parser.add_argument("--optimizer-seed", type=int, default=20260721)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def policy_hashes(weights: np.ndarray) -> list[str]:
    """Hash normalized policy coordinates independently of source metadata."""
    rounded = np.round(np.asarray(weights, dtype=np.float64), decimals=14)
    return [hashlib.sha256(policy.tobytes()).hexdigest() for policy in rounded]


def source_provenance(
    reference: pooled.Dataset,
    tied_frame: pd.DataFrame,
    extension: analysis.CoordinatePool,
    target: str,
) -> pd.DataFrame:
    rows = [
        {
            "target": target,
            "role": "base_two_phase_fit",
            "source": "Delphi 3e18 two-phase fit swarm",
            "training_series": "delphi_3e18_fit_swarm",
            "rows": len(reference.frame),
            "response_scale": "Delphi 3e18",
        },
        {
            "target": target,
            "role": "tied_spine",
            "source": "Independently trained Delphi 3e18 one-phase swarm",
            "training_series": "delphi_3e18_one_phase_scheduled_new_training",
            "rows": len(tied_frame),
            "response_scale": "Delphi 3e18",
        },
    ]
    for series, count in extension.frame["training_series"].value_counts().sort_index().items():
        rows.append(
            {
                "target": target,
                "role": "phase_extension",
                "source": "Completed Delphi 3e18 phase-varying development panel",
                "training_series": str(series),
                "rows": int(count),
                "response_scale": "Delphi 3e18",
            }
        )
    return pd.DataFrame(rows)


def endpoint_training_set(
    target: str,
) -> tuple[pooled.Dataset, analysis.CoordinatePool, pd.DataFrame]:
    reference = observatory.load_delphi_3e18_fit_dataset(target)
    heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(reference)
    single, _single_indices = observatory.load_delphi_3e18_single_phase_dataset(
        target,
        reference,
        heldout_frame,
        heldout_weights,
    )
    target_column = analysis.TARGET_COLUMNS[target]
    base_frame = reference.frame.copy()
    base_frame[target_column] = reference.y
    single.frame[target_column] = single.y
    tied_frame, tied_weights = analysis.tied_independent_pool(single)
    extension, evaluation = analysis.heldout_pools(reference)

    if len(reference.frame) != EXPECTED_BASE_ROWS:
        raise ValueError(f"Expected {EXPECTED_BASE_ROWS} base rows, found {len(reference.frame)}")
    if len(tied_frame) != EXPECTED_TIED_ROWS:
        raise ValueError(f"Expected {EXPECTED_TIED_ROWS} tied rows, found {len(tied_frame)}")
    if len(extension.frame) != EXPECTED_EXTENSION_ROWS:
        raise ValueError(f"Expected {EXPECTED_EXTENSION_ROWS} extension rows, found {len(extension.frame)}")
    if set(extension.frame["training_series"]) != set(analysis.EXTENSION_SERIES):
        raise ValueError("The phase-extension source set differs from the frozen learning-curve protocol")
    if not extension.frame["training_state"].eq("finished").all():
        raise ValueError("The phase extension contains unfinished checkpoints")
    if not extension.frame["checkpoint_declared_complete"].eq(1).all():
        raise ValueError("The phase extension contains incomplete checkpoints")

    train = analysis.combined_dataset(
        reference,
        (
            (base_frame, reference.weights),
            (tied_frame, tied_weights),
            (extension.frame, extension.weights),
        ),
        target,
        f"delphi_3e18_expanded_endpoint_{target}",
    )
    if train.n != EXPECTED_TOTAL_ROWS:
        raise ValueError(f"Expected {EXPECTED_TOTAL_ROWS} total rows, found {train.n}")
    coordinate_hashes = policy_hashes(train.weights)
    if len(set(coordinate_hashes)) != EXPECTED_TOTAL_ROWS:
        raise ValueError("The expanded endpoint fit contains duplicate policy coordinates")
    if not np.isfinite(train.y).all():
        raise ValueError("The expanded endpoint fit contains a non-finite target")

    provenance = source_provenance(reference, tied_frame, extension, target)
    if int(provenance["rows"].sum()) != EXPECTED_TOTAL_ROWS:
        raise AssertionError("Provenance counts do not sum to the fitted row count")
    if set(provenance["response_scale"]) != {"Delphi 3e18"}:
        raise AssertionError("A non-3e18 response source entered the fit")
    return train, evaluation, provenance


def clean_domain(domain: str) -> str:
    for prefix in ("dolma3_", "dolmino_"):
        if domain.startswith(prefix):
            domain = domain.removeprefix(prefix)
            break
    return domain.replace("cc/", "CC: ").replace("_", " ")


def candidate_paths(output_dir: Path, spec: CandidateSpec) -> tuple[Path, Path]:
    stem = f"{spec.target}_{spec.model_id}_raw_optimum"
    return output_dir / f"{stem}.json", output_dir / f"{stem}_weights.csv"


def weights_frame(train: pooled.Dataset, optimum: analysis.RawOptimum) -> pd.DataFrame:
    alpha0, alpha1 = observatory.phase_fractions(train)
    proportional = observatory.natural_weights(train, alpha0)
    aggregate = alpha0 * optimum.weights[0] + alpha1 * optimum.weights[1]
    phase0_exposure = optimum.weights[0] * train.c0
    phase1_exposure = optimum.weights[1] * train.c1
    return pd.DataFrame(
        {
            "domain": train.domain_names,
            "domain_label": [clean_domain(domain) for domain in train.domain_names],
            "domain_group": [observatory.domain_group(domain) for domain in train.domain_names],
            "proportional_weight": proportional,
            "phase_0_weight": optimum.weights[0],
            "phase_1_weight": optimum.weights[1],
            "aggregate_weight": aggregate,
            "phase_0_exposure": phase0_exposure,
            "phase_1_exposure": phase1_exposure,
            "aggregate_exposure": phase0_exposure + phase1_exposure,
            "proportional_exposure": proportional * (train.c0 + train.c1),
        }
    )


def fit_candidate(
    spec: CandidateSpec,
    output_dir: Path,
    optimizer_starts: int,
    optimizer_seed: int,
    overwrite: bool,
) -> tuple[CandidateResult, pd.DataFrame]:
    summary_path, weights_path = candidate_paths(output_dir, spec)
    train, evaluation, provenance = endpoint_training_set(spec.target)
    if summary_path.exists() and weights_path.exists() and not overwrite:
        return CandidateResult(spec, json.loads(summary_path.read_text()), pd.read_csv(weights_path)), provenance

    frozen = analysis.frozen_spec(spec.target, observatory.TWO_PHASE, spec.model_id)
    model = analysis.fit_frozen_model(train, frozen)
    optimum = analysis.optimize_raw_model(
        train,
        model,
        frozen,
        seed=optimizer_seed,
        count=optimizer_starts,
        previous=None,
    )
    record = analysis.raw_optimum_record(
        optimum,
        train,
        evaluation,
        spec.target,
        spec.model_id,
        "tied_spine_plus_two_phase",
        EXPECTED_EXTENSION_ROWS,
        EXPECTED_TIED_ROWS,
        optimizer_seed,
        None,
    )
    target_column = analysis.TARGET_COLUMNS[spec.target]
    fit_payload = json.loads(
        (analysis.CACHE_ROOT / spec.target / observatory.TWO_PHASE / f"{spec.model_id}.json").read_text()
    )
    summary = {
        **record,
        "candidate_label": spec.label,
        "fit_row_count": train.n,
        "base_two_phase_rows": EXPECTED_BASE_ROWS,
        "independent_tied_rows": EXPECTED_TIED_ROWS,
        "phase_extension_rows": EXPECTED_EXTENSION_ROWS,
        "all_response_labels_from_delphi_3e18": True,
        "uses_300m_response_labels": False,
        "hyperparameter_source": "Frozen Delphi 3e18 Observatory fit",
        "frozen_tuning": frozen.tuning,
        "parameter_count": int(fit_payload["fitDetail"]["parameterCount"]),
        "current_development_frontier_bpb": float(evaluation.frame[target_column].min()),
        "predicted_gain_vs_development_frontier": float(evaluation.frame[target_column].min() - optimum.predicted_bpb),
    }
    frame = weights_frame(train, optimum)
    frame.to_csv(weights_path, index=False)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return CandidateResult(spec, summary, frame), provenance


def add_bar_panel(
    figure: go.Figure,
    frame: pd.DataFrame,
    row: int,
    col: int,
    candidate_column: str,
    baseline_column: str,
    axis_title: str,
) -> None:
    customdata = np.column_stack(
        [
            frame["domain"],
            frame["domain_group"],
            frame[candidate_column],
            frame[baseline_column],
            frame[candidate_column] - frame[baseline_column],
        ]
    )
    values = (
        (frame[baseline_column], "Proportional", PROPORTIONAL_COLOR, 0.78, "proportional"),
        (frame[candidate_column], "Raw predicted optimum", PHASE_0_COLOR, 0.94, "optimum"),
    )
    for series, label, color, opacity, legendgroup in values:
        if candidate_column.endswith("weight"):
            text = [f"{value:.3f}" if value >= 0.008 else "" for value in series]
        else:
            text = [f"{value:.2f}" if value >= 0.15 else "" for value in series]
        figure.add_trace(
            go.Bar(
                x=series,
                y=frame["domain_label"],
                orientation="h",
                name=label,
                legendgroup=legendgroup,
                showlegend=row == 1 and col == 1,
                marker_color=color,
                opacity=opacity,
                text=text if legendgroup == "optimum" else None,
                textposition="outside",
                cliponaxis=False,
                customdata=customdata,
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "group=%{customdata[1]}<br>"
                    f"{axis_title}: %{{x:.6f}}<br>"
                    "optimum=%{customdata[2]:.6f}<br>"
                    "proportional=%{customdata[3]:.6f}<br>"
                    "optimum - proportional=%{customdata[4]:+.6f}<extra></extra>"
                ),
            ),
            row=row,
            col=col,
        )
    figure.update_xaxes(title_text=axis_title, rangemode="tozero", row=row, col=col)
    figure.update_yaxes(
        categoryorder="array",
        categoryarray=frame["domain_label"].tolist(),
        tickfont={"size": 10},
        row=row,
        col=col,
    )


def plot_results(results: list[CandidateResult]) -> go.Figure:
    column_titles = ["Phase 0 weights", "Phase 1 weights", "Aggregate weights", "Aggregate exposure"]
    row_titles = []
    for result in results:
        summary = result.summary
        row_titles.append(
            f"{result.spec.label}<br>pred={summary['predicted_bpb']:.6f}<br>"
            f"max epoch={summary['max_simulated_epochs']:.2f}<br>phase TV={summary['phase_total_variation']:.3f}"
        )
    figure = make_subplots(
        rows=len(results),
        cols=4,
        subplot_titles=column_titles + [""] * (4 * (len(results) - 1)),
        row_titles=row_titles,
        shared_yaxes="rows",
        horizontal_spacing=0.035,
        vertical_spacing=0.06,
    )
    panels = (
        ("phase_0_weight", "proportional_weight", "mixture weight"),
        ("phase_1_weight", "proportional_weight", "mixture weight"),
        ("aggregate_weight", "proportional_weight", "mixture weight"),
        ("aggregate_exposure", "proportional_exposure", "realized simulated epochs"),
    )
    for row, result in enumerate(results, start=1):
        for col, panel in enumerate(panels, start=1):
            add_bar_panel(figure, result.weights, row, col, *panel)

    figure.update_layout(
        title={
            "text": (
                "Expanded Delphi 3e18 fits: unregularized predicted optima versus proportional"
                "<br><span style='font-size:15px;color:#44546a'>No deployment KL, phase-information budget, "
                "epoch cap, or trust region; aggregate weights use the observed 0.8 / 0.2 phase split.</span>"
            ),
            "x": 0.5,
            "xanchor": "center",
            "y": 0.995,
            "yanchor": "top",
        },
        barmode="group",
        template="plotly_white",
        width=2700,
        height=2650,
        margin={"l": 250, "r": 330, "t": 230, "b": 130},
        legend={
            "orientation": "h",
            "x": 0.5,
            "xanchor": "center",
            "y": -0.035,
            "yanchor": "top",
            "bgcolor": "rgba(255,255,255,0.94)",
            "bordercolor": "#d9e0ea",
            "borderwidth": 1,
        },
        paper_bgcolor="#fbfaf6",
        plot_bgcolor="white",
    )
    for annotation in figure.layout.annotations:
        if annotation.text in row_titles:
            annotation.update(textangle=0, x=1.01, xanchor="left", align="left", font={"size": 13})
    return figure


def render_html(
    results: list[CandidateResult],
    provenance: pd.DataFrame,
    output_dir: Path,
) -> Path:
    figure = plot_results(results)
    figure.write_image(output_dir / "expanded_3e18_raw_optima_mixtures.png", scale=1)
    plot = pio.to_html(figure, include_plotlyjs=True, full_html=False, config=PLOT_CONFIG)
    health_rows = []
    for result in results:
        summary = result.summary
        health_rows.append(
            {
                "Target / model": result.spec.label,
                "Predicted BPB": summary["predicted_bpb"],
                "Current dev frontier": summary["current_development_frontier_bpb"],
                "Predicted gain": summary["predicted_gain_vs_development_frontier"],
                "Max weight": summary["max_bucket_weight"],
                "Max epochs": summary["max_simulated_epochs"],
                "Phase TV": summary["phase_total_variation"],
                "Aggregate TV to proportional": summary["aggregate_tv_to_proportional"],
                "Nearest fit TV": summary["nearest_fit_policy_tv"],
                "Support distance": summary["standardized_fit_support_distance"],
                "Optimizer successful starts": f"{summary['successful_starts']}/{summary['finite_starts']}",
                "Parameters": summary["parameter_count"],
            }
        )
    health = pd.DataFrame(health_rows)
    numeric = health.select_dtypes(include=["number"]).columns
    health[numeric] = health[numeric].round(6)
    unique_provenance = provenance.drop_duplicates(["role", "training_series", "rows"]).copy()
    facts = [
        ("Training setting", "Delphi 3e18, 39 buckets, 80% / 20% WSD phases"),
        ("Unique fitted policies", f"{EXPECTED_TOTAL_ROWS} = 280 base + 238 tied + 480 phase varying"),
        ("Response-label scale", "Every fitted BPB is observed from a Delphi 3e18 checkpoint"),
        ("300M outcomes", "None used in either fit"),
        (
            "Coordinate inheritance",
            "Some policy coordinates mirror earlier designs; their fitted outcomes are still 3e18",
        ),
        ("Hyperparameters", "Frozen from the original 280-row Delphi 3e18 Observatory fits"),
        ("Optimization", "Continuous raw two-phase surface; no deployment regularization"),
        ("Numerical warning", "All starts were finite, but SciPy did not certify convergence"),
    ]
    fact_html = "".join(
        f"<div class='fact'><strong>{html.escape(name)}</strong><span>{html.escape(value)}</span></div>"
        for name, value in facts
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Expanded Delphi 3e18 raw optima</title>
<style>
  :root {{ --ink:#173042; --muted:#64748b; --paper:#fbfaf6; --line:#d9d4c8; --accent:#d95f2d; }}
  body {{ margin:0; background:var(--paper); color:var(--ink); font-family:Georgia, 'Times New Roman', serif; }}
  main {{ max-width:2700px; margin:0 auto; padding:18px 30px 60px; }}
  .plot {{ border:1px solid var(--line); background:white; box-shadow:0 12px 30px rgba(23,48,66,.08); }}
  h2 {{ font-size:30px; margin:42px 0 16px; }}
  p {{ font-size:18px; line-height:1.55; max-width:1200px; color:var(--muted); }}
  .facts {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(280px,1fr)); gap:12px; }}
  .fact {{ border:1px solid var(--line); border-top:4px solid var(--accent); background:white; padding:16px; }}
  .fact strong, .fact span {{ display:block; }}
  .fact strong {{ font:700 13px/1.2 ui-sans-serif, sans-serif; letter-spacing:.08em; text-transform:uppercase; }}
  .fact span {{ margin-top:8px; font-size:17px; line-height:1.35; }}
  table {{ width:100%; border-collapse:collapse; background:white; font:14px/1.35 ui-sans-serif, sans-serif; }}
  th, td {{ border:1px solid var(--line); padding:9px 10px; text-align:right; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align:left; }}
</style>
</head>
<body><main>
<section class="plot">{plot}</section>
<h2>Fit and optimization fact sheet</h2>
<div class="facts">{fact_html}</div>
<h2>Raw-optimum health</h2>
{health.to_html(index=False, border=0)}
<h2>Training provenance</h2>
<p>The provenance table is deduplicated across targets because the same policy coordinates are fitted to separate
target labels.</p>
{unique_provenance.to_html(index=False, border=0)}
</main></body></html>
"""
    path = output_dir / "expanded_3e18_raw_optima_mixtures.html"
    path.write_text(document)
    return path


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    results: list[CandidateResult] = []
    provenance_frames: list[pd.DataFrame] = []
    for spec in CANDIDATES:
        result, provenance = fit_candidate(
            spec,
            args.output_dir,
            args.optimizer_starts,
            args.optimizer_seed,
            args.overwrite,
        )
        results.append(result)
        provenance_frames.append(provenance)

    provenance = pd.concat(provenance_frames, ignore_index=True)
    provenance.to_csv(args.output_dir / "training_provenance.csv", index=False)
    manifest = pd.DataFrame([result.summary for result in results])
    manifest.to_csv(args.output_dir / "raw_optimum_manifest.csv", index=False)
    all_weights = pd.concat(
        [result.weights.assign(target=result.spec.target, model=result.spec.model_id) for result in results],
        ignore_index=True,
    )
    all_weights.to_csv(args.output_dir / "raw_optimum_weights.csv", index=False)
    path = render_html(results, provenance, args.output_dir)
    print(
        manifest[
            [
                "target",
                "model",
                "predicted_bpb",
                "current_development_frontier_bpb",
                "max_bucket_weight",
                "max_simulated_epochs",
                "phase_total_variation",
                "nearest_fit_policy_tv",
                "optimizer_converged",
                "successful_starts",
            ]
        ].to_string(index=False)
    )
    print(path)


if __name__ == "__main__":
    main()
