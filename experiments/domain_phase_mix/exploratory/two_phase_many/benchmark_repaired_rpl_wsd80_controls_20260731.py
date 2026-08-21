# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "matplotlib",
#   "numpy",
#   "pandas",
#   "plotly",
#   "pyarrow",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Evaluate the frozen repaired RPL estimator on WSD80 control targets.

The estimator and nonlinear shape grid are exactly those preregistered as
WSD80-SUR-046. Three code targets are positive controls for phase gain; two
broad-text targets are negative controls against an invented phase gain.
Every target and fold protocol is independently resumable.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    audit_wsd80_cross_metric_rpl_20260730 as prior_audit,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_expanded_300m_pareto_baseline_20260731 as baseline,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_repaired_rpl_300m_20260731 as repaired_300m,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_wsd80_incumbents_20260728 as incumbent,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_estimator_repair_20260731 as repaired,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    retained_power_law_model_20260728 as parent,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    starcoder_wsd80_panel_20260728 as wsd80,
)

DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "repaired_rpl_wsd80_controls_20260731"
PRIOR_OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "wsd80_cross_metric_rpl_20260730"
PROTOCOL_VERSION = "repaired-rpl-wsd80-controls-v1"
PROTOCOLS = ("random", "blocked")
TARGETS = (
    prior_audit.PRIMARY_TARGET,
    "eval/uncheatable_eval/github_python-llama3/bpb",
    "eval/uncheatable_eval/github_cpp-llama3/bpb",
    "eval/paloma/c4_en-llama3/bpb",
    "eval/paloma/falcon-refinedweb-llama3/bpb",
)
POSITIVE_TARGETS = frozenset(TARGETS[:3])
NEGATIVE_TARGETS = frozenset(TARGETS[3:])
OUTER_SPLITS = 3
INNER_SPLITS = 3
OUTER_SEED = 0
INNER_SEED_BASE = 31_000
FULL_FIT_SEED = 32_000
GRID_RESOLUTION = 201
POSITIVE_REGRET_LIMIT = 0.006
POSITIVE_GAIN_FLOOR = 0.004
NEGATIVE_REGRET_LIMIT = 0.005
NEGATIVE_GAIN_LIMIT = 0.005
PRIMARY_OPTIMUM_DISTANCE_LIMIT = 0.10
INTERIOR_RMSE_RATIO_LIMIT = 1.05
EXPORT_CONFIG = {"toImageButtonOptions": {"format": "png", "scale": 4}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--targets", default=",".join(TARGETS))
    parser.add_argument("--protocols", default=",".join(PROTOCOLS))
    parser.add_argument("--workers", type=int, default=5)
    parser.add_argument("--grid", type=int, default=GRID_RESOLUTION)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--no-collect", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_csv(value: str) -> tuple[str, ...]:
    return tuple(item.strip() for item in value.split(",") if item.strip())


def protocol_payload(grid: int) -> dict[str, Any]:
    sources = (
        Path(__file__),
        Path(repaired.__file__),
        Path(parent.__file__),
        Path(prior_audit.__file__),
        Path(incumbent.__file__),
    )
    payload = {
        "version": PROTOCOL_VERSION,
        "candidate": "WSD80-SUR-046",
        "targets": TARGETS,
        "positive_targets": sorted(POSITIVE_TARGETS),
        "negative_targets": sorted(NEGATIVE_TARGETS),
        "protocols": PROTOCOLS,
        "outer_splits": OUTER_SPLITS,
        "inner_splits": INNER_SPLITS,
        "outer_seed": OUTER_SEED,
        "inner_seed_base": INNER_SEED_BASE,
        "full_fit_seed": FULL_FIT_SEED,
        "grid": grid,
        "selection": {
            "core_rmse_ratio_limit": repaired.CORE_RMSE_RATIO_LIMIT,
            "regret_at_1_slack": repaired.REGRET_AT_1_SLACK,
            "lower_tail_fraction": repaired.LOWER_TAIL_FRACTION,
            "lower_tail_min_count": repaired.LOWER_TAIL_MIN_COUNT,
            "phase_penalty_multiplier": repaired.PHASE_PENALTY_MULTIPLIER,
        },
        "acceptance": acceptance_gate(),
        "source_hashes": {str(path.relative_to(REPO_ROOT)): baseline.file_hash(path) for path in sources},
    }
    encoded = json.dumps(baseline.json_ready(payload), sort_keys=True, separators=(",", ":")).encode()
    return {**payload, "protocol_hash": hashlib.sha256(encoded).hexdigest()}


def acceptance_gate() -> dict[str, float | str]:
    return {
        "primary_protocol": "random",
        "positive_regret_at_1_max": POSITIVE_REGRET_LIMIT,
        "positive_predicted_phase_gain_min": POSITIVE_GAIN_FLOOR,
        "negative_regret_at_1_max": NEGATIVE_REGRET_LIMIT,
        "negative_predicted_phase_gain_max": NEGATIVE_GAIN_LIMIT,
        "primary_optimum_distance_max": PRIMARY_OPTIMUM_DISTANCE_LIMIT,
        "interior_rmse_ratio_vs_original_rpl_max": INTERIOR_RMSE_RATIO_LIMIT,
        "note": (
            "All thresholds were frozen before evaluating the repaired estimator. "
            "Blocked-region results are mandatory diagnostics but not thresholded because "
            "the original target-specific RPL audit reported only random-fold refits."
        ),
    }


def fold_builder(
    protocol: str,
    weights: np.ndarray,
    indices: np.ndarray,
    splits: int,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    if protocol == "random":
        folds = incumbent.random_folds(weights, indices, splits, seed)
    elif protocol == "blocked":
        folds = incumbent.mixture_blocked_folds(weights, indices, splits, seed)
    else:
        raise ValueError(f"unknown fold protocol: {protocol}")
    return tuple((np.asarray(train, dtype=int), np.asarray(test, dtype=int)) for train, test in folds)


def local_folds(
    protocol: str,
    global_weights: np.ndarray,
    global_indices: np.ndarray,
    seed: int,
) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
    positions = np.arange(len(global_indices))
    folds = fold_builder(protocol, global_weights, global_indices, INNER_SPLITS, seed)
    return tuple(
        (
            positions[np.isin(global_indices, train)],
            positions[np.isin(global_indices, test)],
        )
        for train, test in folds
    )


def selection_context(weights: np.ndarray) -> repaired.SelectionContext:
    phase_0 = weights[:, 0, 1]
    phase_1 = weights[:, 1, 1]
    tied = np.isclose(phase_0, phase_1, atol=1e-9)
    # Fibers were designed in nominal 80/20 coordinates. The latent-state model
    # separately uses realized step fractions for physical epoch accounting.
    aggregate = wsd80.PHASE_0_FRACTION * phase_0 + wsd80.PHASE_1_FRACTION * phase_1
    tied_by_aggregate = {round(float(aggregate[index]), 9): int(index) for index in np.flatnonzero(tied)}
    pair_tied = []
    pair_asymmetric = []
    for index in np.flatnonzero(~tied):
        counterpart = tied_by_aggregate.get(round(float(aggregate[index]), 9))
        if counterpart is None:
            continue
        pair_tied.append(counterpart)
        pair_asymmetric.append(int(index))
    return repaired.SelectionContext(
        tied=tied,
        pair_tied=np.asarray(pair_tied, dtype=int),
        pair_asymmetric=np.asarray(pair_asymmetric, dtype=int),
    )


def fit_model(
    weights: np.ndarray,
    target: np.ndarray,
    geometry: parent.Geometry,
    folds: tuple[tuple[np.ndarray, np.ndarray], ...],
    workers: int,
) -> repaired.Fitted:
    return repaired.fit(
        weights,
        target,
        geometry,
        folds,
        selection_context(weights),
        workers=workers,
    )


def cell_dir(output_dir: Path, protocol: str, target: str) -> Path:
    digest = hashlib.sha256(target.encode()).hexdigest()[:12]
    return output_dir / "cells" / protocol / digest


def cell_complete(path: Path, protocol_hash: str) -> bool:
    required = (
        path / "complete.json",
        path / "scores.csv",
        path / "selection.json",
        path / "optimum.json",
        path / "fold_selections.json",
        path / "fold_coefficients.csv",
        path / "coefficient_stability.csv",
    )
    if any(not item.exists() for item in required):
        return False
    marker = json.loads((path / "complete.json").read_text())
    return marker.get("protocol_hash") == protocol_hash


def coefficient_rows(model: repaired.Fitted, fold: int | str) -> list[dict[str, int | float | str]]:
    return repaired_300m.coefficient_rows(model, fold)


def run_cell(
    output_dir: Path,
    frozen: dict[str, Any],
    panel: wsd80.Panel,
    frame: pd.DataFrame,
    replicates: pd.DataFrame,
    metric: str,
    protocol: str,
    workers: int,
    grid: int,
    force: bool,
) -> None:
    path = cell_dir(output_dir, protocol, metric)
    protocol_hash = str(frozen["protocol_hash"])
    if not force and cell_complete(path, protocol_hash):
        print(f"skip complete {protocol}/{prior_audit.metric_label(metric)}", flush=True)
        return
    path.mkdir(parents=True, exist_ok=True)
    target = frame[metric].to_numpy(dtype=float)
    geometry = parent.Geometry(
        c0=panel.c0,
        c1=panel.c1,
        phase_0_fraction=wsd80.REALIZED_PHASE_0_FRACTION,
    )
    indices = np.arange(len(target))
    outer = fold_builder(protocol, panel.weights, indices, OUTER_SPLITS, OUTER_SEED)
    predicted = np.full(len(target), np.nan, dtype=float)
    fold_rows = []
    coefficients = []
    for fold_id, (train, test) in enumerate(outer):
        print(
            f"{protocol}/{prior_audit.metric_label(metric)}: outer fold "
            f"{fold_id + 1}/{len(outer)} ({len(train)} train, {len(test)} test)",
            flush=True,
        )
        inner = local_folds(
            protocol,
            panel.weights,
            train,
            INNER_SEED_BASE + fold_id,
        )
        model = fit_model(
            panel.weights[train],
            target[train],
            geometry,
            inner,
            workers,
        )
        predicted[test] = model.predict(panel.weights[test])
        fold_rows.append(
            {
                "outer_fold": fold_id,
                "train_rows": len(train),
                "test_rows": len(test),
                "shape": asdict(model.shape),
                "ridge": model.ridge,
                "selection": asdict(model.selection),
            }
        )
        coefficients.extend(coefficient_rows(model, fold_id))
    if not np.isfinite(predicted).all():
        raise RuntimeError(f"incomplete predictions for {protocol}/{metric}")

    sigma = prior_audit.pooled_seed_sigma(replicates, metric)
    masks, _best = prior_audit.subset_masks(panel, target)
    scores = pd.DataFrame(
        prior_audit.score_predictions(
            metric,
            protocol,
            target,
            predicted,
            masks,
            sigma,
        )
    )
    selection = prior_audit.discrete_selection(
        metric,
        protocol,
        target,
        predicted,
        masks,
        panel,
    )
    full_inner = local_folds(
        protocol,
        panel.weights,
        indices,
        FULL_FIT_SEED,
    )
    full_fit = fit_model(panel.weights, target, geometry, full_inner, workers)
    coefficients.extend(coefficient_rows(full_fit, "full"))
    optimum, _grid = prior_audit.continuous_optimum(
        metric,
        protocol,
        target,
        full_fit,
        panel,
        grid,
    )

    pd.DataFrame(
        {
            "row_index": indices,
            "wandb_run_id": panel.frame["wandb_run_id"].astype(str),
            "phase_0_starcoder": panel.phase_0[:, 1],
            "phase_1_starcoder": panel.phase_1[:, 1],
            "observed": target,
            "predicted": predicted,
            "residual": predicted - target,
        }
    ).to_csv(path / "predictions.csv", index=False)
    scores.to_csv(path / "scores.csv", index=False)
    baseline.write_json(path / "selection.json", selection)
    baseline.write_json(path / "optimum.json", optimum)
    baseline.write_json(path / "fold_selections.json", fold_rows)
    baseline.write_json(
        path / "full_fit.json",
        {
            "shape": asdict(full_fit.shape),
            "ridge": full_fit.ridge,
            "selection": asdict(full_fit.selection),
            "parameter_diagnostics": repaired_300m.parameter_diagnostics(full_fit),
            "aggregate_matched_pair_count": len(selection_context(panel.weights).pair_tied),
        },
    )
    coefficient_frame = pd.DataFrame(coefficients)
    coefficient_frame.to_csv(path / "fold_coefficients.csv", index=False)
    repaired_300m.coefficient_stability(coefficient_frame).to_csv(
        path / "coefficient_stability.csv",
        index=False,
    )
    baseline.write_json(
        path / "complete.json",
        {
            "protocol_hash": protocol_hash,
            "metric": metric,
            "protocol": protocol,
        },
    )
    print(f"completed {protocol}/{prior_audit.metric_label(metric)}", flush=True)


def prior_random_results() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    scores = pd.read_csv(PRIOR_OUTPUT_DIR / "retuned_cross_metric_scores.csv")
    selection = pd.read_csv(PRIOR_OUTPUT_DIR / "retuned_cross_metric_selection.csv")
    optima = pd.read_csv(PRIOR_OUTPUT_DIR / "retuned_cross_metric_optima.csv")
    return (
        scores[scores["metric"].isin(TARGETS)].copy(),
        selection[selection["metric"].isin(TARGETS)].copy(),
        optima[optima["metric"].isin(TARGETS)].copy(),
    )


def control_plot(summary: pd.DataFrame, output_path: Path) -> None:
    random = summary[summary["protocol"].eq("random")].copy()
    colors = ["#1a9850" if target in POSITIVE_TARGETS else "#d73027" for target in random["metric"]]
    figure = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=("Selected-policy regret", "Predicted phase gain"),
    )
    figure.add_trace(
        go.Bar(
            x=random["label"],
            y=random["regret_at_1"],
            marker={"color": colors},
            name="Repaired RPL",
            customdata=np.column_stack([random["prior_regret_at_1"]]),
            hovertemplate=("%{x}<br>repaired %{y:.6f}<br>original RPL %{customdata[0]:.6f}" "<extra></extra>"),
        ),
        row=1,
        col=1,
    )
    figure.add_trace(
        go.Bar(
            x=random["label"],
            y=random["predicted_two_phase_gain"],
            marker={"color": colors},
            name="Repaired RPL",
            showlegend=False,
            customdata=np.column_stack(
                [
                    random["observed_sampled_two_phase_gain"],
                    random["prior_predicted_two_phase_gain"],
                ]
            ),
            hovertemplate=(
                "%{x}<br>repaired predicted %{y:.6f}"
                "<br>observed sampled %{customdata[0]:.6f}"
                "<br>original RPL predicted %{customdata[1]:.6f}<extra></extra>"
            ),
        ),
        row=1,
        col=2,
    )
    figure.update_xaxes(tickangle=-30)
    figure.update_yaxes(title="BPB", row=1, col=1)
    figure.update_yaxes(title="BPB", row=1, col=2)
    figure.update_layout(
        title="Repaired RPL on WSD80 positive and negative controls",
        width=1500,
        height=650,
        margin={"l": 90, "r": 50, "t": 100, "b": 210},
        paper_bgcolor="#fffdf7",
        plot_bgcolor="#fffdf7",
        showlegend=False,
    )
    figure.write_html(output_path, include_plotlyjs=True, config=EXPORT_CONFIG)


def collect_results(output_dir: Path, frozen: dict[str, Any]) -> None:
    score_frames = []
    selections = []
    optima = []
    complete = []
    for protocol in PROTOCOLS:
        for metric in TARGETS:
            path = cell_dir(output_dir, protocol, metric)
            if not cell_complete(path, str(frozen["protocol_hash"])):
                continue
            complete.append((protocol, metric))
            score_frames.append(pd.read_csv(path / "scores.csv"))
            selections.append(json.loads((path / "selection.json").read_text()))
            optima.append(json.loads((path / "optimum.json").read_text()))
    scores = pd.concat(score_frames, ignore_index=True) if score_frames else pd.DataFrame()
    selection = pd.DataFrame(selections)
    optimum = pd.DataFrame(optima)
    scores.to_csv(output_dir / "scores.csv", index=False)
    selection.to_csv(output_dir / "selection.csv", index=False)
    optimum.to_csv(output_dir / "optima.csv", index=False)

    summary = pd.DataFrame()
    gate = pd.DataFrame()
    if not scores.empty:
        interior = scores[scores["subset"].eq("interior")][
            ["metric", "label", "protocol", "rmse", "rmse_sigma", "median_absolute", "spearman"]
        ]
        summary = interior.merge(
            selection[
                [
                    "metric",
                    "protocol",
                    "regret_at_1",
                    "regret_at_5",
                    "selected_distance",
                ]
            ],
            on=["metric", "protocol"],
            validate="one_to_one",
        ).merge(
            optimum[
                [
                    "metric",
                    "protocol",
                    "observed_sampled_two_phase_gain",
                    "predicted_two_phase_gain",
                    "optimum_distance_interior",
                    "predicted_best_interior_phase_0",
                    "predicted_best_interior_phase_1",
                ]
            ],
            on=["metric", "protocol"],
            validate="one_to_one",
        )
        prior_scores, prior_selection, prior_optima = prior_random_results()
        prior_interior = prior_scores[prior_scores["subset"].eq("interior")][["metric", "rmse"]].rename(
            columns={"rmse": "prior_interior_rmse"}
        )
        prior_selection = prior_selection[["metric", "regret_at_1"]].rename(columns={"regret_at_1": "prior_regret_at_1"})
        prior_optima = prior_optima[["metric", "predicted_two_phase_gain"]].rename(
            columns={"predicted_two_phase_gain": "prior_predicted_two_phase_gain"}
        )
        summary = (
            summary.merge(prior_interior, on="metric", how="left", validate="many_to_one")
            .merge(prior_selection, on="metric", how="left", validate="many_to_one")
            .merge(prior_optima, on="metric", how="left", validate="many_to_one")
        )
        summary["interior_rmse_ratio_vs_original_rpl"] = summary["rmse"] / summary["prior_interior_rmse"]
        summary.to_csv(output_dir / "control_summary.csv", index=False)
        control_plot(summary, output_dir / "control_summary.html")

        random = summary[summary["protocol"].eq("random")].copy()
        gate_rows = []
        for row in random.to_dict(orient="records"):
            metric = str(row["metric"])
            label = str(row["label"])
            interior_rmse_ratio = float(row["interior_rmse_ratio_vs_original_rpl"])
            regret_at_1 = float(row["regret_at_1"])
            predicted_phase_gain = float(row["predicted_two_phase_gain"])
            optimum_distance = float(row["optimum_distance_interior"])
            positive = metric in POSITIVE_TARGETS
            checks = {
                "interior_rmse_ratio": interior_rmse_ratio <= INTERIOR_RMSE_RATIO_LIMIT,
                "regret_at_1": regret_at_1 <= (POSITIVE_REGRET_LIMIT if positive else NEGATIVE_REGRET_LIMIT),
                "predicted_phase_gain": (
                    predicted_phase_gain >= POSITIVE_GAIN_FLOOR
                    if positive
                    else predicted_phase_gain <= NEGATIVE_GAIN_LIMIT
                ),
            }
            if metric == prior_audit.PRIMARY_TARGET:
                checks["primary_optimum_distance"] = optimum_distance <= PRIMARY_OPTIMUM_DISTANCE_LIMIT
            for check, passed in checks.items():
                gate_rows.append(
                    {
                        "metric": metric,
                        "label": label,
                        "control": "positive" if positive else "negative",
                        "check": check,
                        "passed": bool(passed),
                    }
                )
        gate = pd.DataFrame(gate_rows)
        gate.to_csv(output_dir / "acceptance_results.csv", index=False)

    frozen_complete = len(complete) == len(PROTOCOLS) * len(TARGETS)
    baseline.write_json(
        output_dir / "status.json",
        {
            "protocol_hash": frozen["protocol_hash"],
            "complete_cells": [{"protocol": protocol, "metric": metric} for protocol, metric in complete],
            "frozen_complete": frozen_complete,
            "random_gate_passed": bool(not gate.empty and gate["passed"].all()),
        },
    )
    report = [
        "# Repaired RPL WSD80 Controls",
        "",
        f"- Protocol: `{frozen['protocol_hash']}`",
        f"- Complete cells: {len(complete)}/{len(PROTOCOLS) * len(TARGETS)}",
        "- Candidate: `WSD80-SUR-046`; model equation and nonlinear grid are unchanged.",
        "- Green rows are code-positive controls; red rows are broad-text-negative controls.",
        "",
    ]
    if not summary.empty:
        report.extend(
            [
                "## Control Summary",
                "",
                summary[
                    [
                        "label",
                        "protocol",
                        "rmse",
                        "interior_rmse_ratio_vs_original_rpl",
                        "regret_at_1",
                        "observed_sampled_two_phase_gain",
                        "predicted_two_phase_gain",
                        "optimum_distance_interior",
                    ]
                ].to_markdown(index=False, floatfmt=".6f"),
                "",
            ]
        )
    if not gate.empty:
        report.extend(
            [
                "## Frozen Random-Fold Gate",
                "",
                gate.to_markdown(index=False),
                "",
                f"Overall gate: `{'PASS' if gate['passed'].all() else 'FAIL'}`.",
                "",
            ]
        )
    report.extend(
        [
            "## Interpretation Rule",
            "",
            (
                "Estimator repair is promoted only if it preserves the code-target phase signal "
                "and removes the broad-text false gain without exceeding the frozen interior-error "
                "ratio. Blocked-region results diagnose transfer but cannot rescue a failed random "
                "gate."
            ),
            "",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(report))


def main() -> None:
    args = parse_args()
    targets = parse_csv(args.targets)
    protocols = parse_csv(args.protocols)
    unknown_targets = sorted(set(targets) - set(TARGETS))
    unknown_protocols = sorted(set(protocols) - set(PROTOCOLS))
    if unknown_targets:
        raise ValueError(f"unknown targets: {unknown_targets}")
    if unknown_protocols:
        raise ValueError(f"unknown protocols: {unknown_protocols}")
    if args.workers < 1:
        raise ValueError("--workers must be positive")
    if args.grid < 21:
        raise ValueError("--grid must be at least 21")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    frozen = protocol_payload(args.grid)
    baseline.write_json(args.output_dir / "protocol.json", frozen)
    baseline.write_json(args.output_dir / "acceptance_gate.json", acceptance_gate())
    panel, frame, available = prior_audit.load_metric_panel()
    missing = sorted(set(TARGETS) - set(available))
    if missing:
        raise ValueError(f"control targets are unavailable or incomplete: {missing}")
    replicates = prior_audit.load_metric_replicates(TARGETS)
    context = selection_context(panel.weights)
    baseline.write_json(
        args.output_dir / "design_audit.json",
        {
            "rows": len(panel.y),
            "tied_rows": int(context.tied.sum()),
            "asymmetric_rows": int((~context.tied).sum()),
            "aggregate_matched_asymmetric_to_tied_pairs": len(context.pair_tied),
        },
    )
    if args.prepare_only:
        collect_results(args.output_dir, frozen)
        print(f"prepared protocol {frozen['protocol_hash']} in {args.output_dir}", flush=True)
        return

    for protocol in protocols:
        for metric in targets:
            run_cell(
                args.output_dir,
                frozen,
                panel,
                frame,
                replicates,
                metric,
                protocol,
                args.workers,
                args.grid,
                args.force,
            )
            if not args.no_collect:
                collect_results(args.output_dir, frozen)
    if not args.no_collect:
        collect_results(args.output_dir, frozen)


if __name__ == "__main__":
    main()
