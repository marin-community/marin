# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = ["numpy", "pandas", "plotly", "scikit-learn", "scipy"]
# ///
"""Materialize simple one-phase surrogate challengers for Delphi 3e18.

The frozen epoch-cap sweep showed that shared-shape DSP extrapolates through the
Table-9 optimum. This successor keeps the same 280-row fit panel, exact 1/2048
runtime grid, and whole-run epoch caps, but compares two response bases:

* ``aggregate_linear_v`` is genuinely single-phase. Each bucket has nonnegative
  under- and over-exposure slopes around an empirical log-exposure centre.
* ``corrected_separate_heads`` is the flexible benchmark. Its centres are built
  from the one-phase fit panel rather than the historical two-phase panel.

Three mixture-blocked fits are ensembled for each model and target. The script
does not launch training; it emits exact candidate mixtures, cross-model
predictions, support diagnostics, and a self-contained comparison report.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import json
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
for entry in (str(SCRIPT_DIR), str(REPO_ROOT)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import benchmark_single_phase_surrogates_20260824 as base  # noqa: E402
import materialize_delphi_one_phase_dsp_epoch_cap_sweep_20260828 as frozen  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import plotly.io as pio  # noqa: E402
import select_delphi_phase0_prefix_candidates_20260824 as prefix_materializer  # noqa: E402
import swarm39_harness_20260725 as swarm39  # noqa: E402
import swarm39_models_20260725 as zoo  # noqa: E402
from plotly.subplots import make_subplots  # noqa: E402
from scipy.optimize import minimize  # noqa: E402

OUTPUT_DIR = SCRIPT_DIR / "reference_outputs" / "delphi_one_phase_surrogate_challengers_20260831"
CAPS = (4, 6, 10)
TARGETS = (swarm39.UNCHEATABLE, swarm39.TABLE9)
TARGET_LABELS = {
    swarm39.UNCHEATABLE: "Uncheatable",
    swarm39.TABLE9: "Table-9 macro",
}
MODEL_KEYS = ("aggregate_linear_v", "corrected_separate_heads")
PARTITION_SEEDS = (0, 1, 2)
CENTRE_SHIFTS = (-2.0, -1.0, 0.0, 1.0, 2.0)
N_FOLDS = 5
MIXTURE_BLOCK_SIZE = frozen.MIXTURE_BLOCK_SIZE
MAX_EXCHANGE_STEPS = frozen.MAX_EXCHANGE_STEPS
REFINE_TOLERANCE = frozen.REFINE_TOLERANCE
PLOT_CONFIG = frozen.PLOT_CONFIG


@dataclass(frozen=True)
class SurrogateEnsemble:
    """Three blocked-partition fits sharing one response basis."""

    key: str
    target: str
    model: swarm39.Model
    fits: tuple[swarm39.Fit, ...]
    template: swarm39.Panel

    def member_predictions(self, weights: np.ndarray) -> np.ndarray:
        query = policy_panel(self.template, weights)
        return np.asarray([fit.predict(query, self.model) for fit in self.fits])

    def predict(self, weights: np.ndarray) -> np.ndarray:
        return np.median(self.member_predictions(weights), axis=0)


@dataclass(frozen=True)
class Candidate:
    """Continuous and exact-runtime forms of one constrained optimum."""

    model_key: str
    target: str
    cap: int
    continuous_weights: np.ndarray
    runtime_counts: np.ndarray
    continuous_prediction: float
    runtime_prediction: float
    exchange_steps: int
    optimizer_successes: int

    @property
    def runtime_weights(self) -> np.ndarray:
        return self.runtime_counts / MIXTURE_BLOCK_SIZE


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--caps", type=int, nargs="+", default=list(CAPS))
    return parser.parse_args()


def aggregate_linear_model(panel: swarm39.Panel) -> swarm39.Model:
    """Return the aggregate-only V-shaped log-exposure response."""
    log_exposure = np.log1p(panel.epochs)
    positive = np.where(panel.epochs > 1e-8, log_exposure, np.nan)
    with np.errstate(invalid="ignore"):
        centre = np.nanmedian(positive, axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)

    def shapes() -> Iterable[dict]:
        for shift in CENTRE_SHIFTS:
            yield {"centre": np.clip(centre + shift, -2.0, 8.0).tolist(), "centre_shift": shift}

    def build(query: swarm39.Panel, shape: dict) -> swarm39.Design:
        value = np.log1p(query.epochs)
        location = np.asarray(shape["centre"], dtype=float)
        under = np.maximum(location - value, 0.0)
        over = np.maximum(value - location, 0.0)
        return swarm39.Design(
            matrix=np.hstack([under, over]),
            names=tuple(
                [
                    *(f"underexposure:{bucket}" for bucket in query.buckets),
                    *(f"overexposure:{bucket}" for bucket in query.buckets),
                ]
            ),
        )

    return swarm39.Model("aggregate_linear_v", build, shapes)


def corrected_separate_heads_model(panel: swarm39.Panel) -> swarm39.Model:
    """Build separate-head centres from the one-phase panel being fitted."""
    return swarm39.Model(
        "corrected_separate_heads",
        zoo.build_separate_heads,
        lambda: zoo.separate_head_shapes(panel),
    )


def policy_panel(template: swarm39.Panel, weights: np.ndarray) -> swarm39.Panel:
    """Construct tied one-phase query points from a matrix of mixture weights.

    Constrained optimizers evaluate intermediate points slightly off the simplex.
    Feasibility is therefore checked on accepted optima and materialized counts,
    not at this response-model boundary.
    """
    values = np.atleast_2d(np.asarray(weights, dtype=float))
    if values.shape[1] != len(template.buckets):
        raise ValueError(f"Expected {len(template.buckets)} buckets, got {values.shape[1]}")
    if not np.isfinite(values).all():
        raise ValueError("Query mixtures must be finite")
    if np.any(values < -1e-12):
        raise ValueError("Query mixtures must be nonnegative")
    rows = len(values)
    return dataclasses.replace(
        template,
        split="candidate",
        phase0=values,
        phase1=values,
        targets={},
        series=np.repeat("candidate", rows),
        policy_class=np.repeat("single_phase", rows),
        group=np.arange(rows),
        row_id=np.asarray([f"candidate_{index}" for index in range(rows)]),
    )


def fit_ensemble(panel: swarm39.Panel, target: str, model_key: str) -> SurrogateEnsemble:
    if model_key == "aggregate_linear_v":
        model = aggregate_linear_model(panel)
    elif model_key == "corrected_separate_heads":
        model = corrected_separate_heads_model(panel)
    else:
        raise ValueError(f"Unknown model {model_key!r}")
    fits = tuple(
        swarm39.fit_model(
            panel,
            model,
            target,
            n_splits=N_FOLDS,
            seed=seed,
            split_fn=swarm39.mixture_blocked_splits,
        )
        for seed in PARTITION_SEEDS
    )
    return SurrogateEnsemble(model_key, target, model, fits, panel)


def optimization_starts(
    panel: swarm39.Panel,
    ensemble: SurrogateEnsemble,
    upper: np.ndarray,
    previous: np.ndarray | None,
) -> list[np.ndarray]:
    observed = panel.targets[ensemble.target]
    predicted = ensemble.predict(panel.phase0)
    candidates = [panel.proportional]
    candidates.extend(panel.phase0[index] for index in np.argsort(observed)[:24])
    candidates.extend(panel.phase0[index] for index in np.argsort(predicted)[:24])
    if previous is not None:
        candidates.append(previous)
    seed = 20260831 + 101 * MODEL_KEYS.index(ensemble.key) + 17 * TARGETS.index(ensemble.target)
    rng = np.random.default_rng(seed)
    for concentration in (8.0, 32.0, 128.0):
        alpha = 1.0 + concentration * panel.proportional
        candidates.extend(rng.dirichlet(alpha) for _ in range(12))
    starts = []
    seen: set[tuple[float, ...]] = set()
    for candidate in candidates:
        projected = frozen.project_capped_simplex(candidate, upper)
        key = tuple(np.round(projected, 12))
        if key in seen:
            continue
        seen.add(key)
        starts.append(projected)
    return starts


def continuous_optimum(
    predict: Callable[[np.ndarray], np.ndarray],
    upper: np.ndarray,
    starts: list[np.ndarray],
) -> tuple[np.ndarray, float, int]:
    def objective(weights: np.ndarray) -> float:
        return float(predict(weights[None, :])[0])

    constraint = {"type": "eq", "fun": lambda weights: float(weights.sum() - 1.0)}
    best_weights = starts[0]
    best_value = objective(best_weights)
    successes = 0
    for start in starts:
        result = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, float(limit)) for limit in upper],
            constraints=[constraint],
            options={"ftol": 1e-13, "maxiter": 1_500},
        )
        if not result.success:
            continue
        successes += 1
        value = float(result.fun)
        if value < best_value:
            best_weights = np.asarray(result.x, dtype=float)
            best_value = value
    if successes == 0:
        raise RuntimeError("No constrained optimization start converged")
    if not np.isclose(best_weights.sum(), 1.0, atol=1e-9):
        raise ValueError("Continuous optimum does not sum to one")
    if np.any(best_weights > upper + 1e-9):
        raise ValueError("Continuous optimum violates the epoch cap")
    return best_weights, best_value, successes


def refine_runtime_counts(
    predict: Callable[[np.ndarray], np.ndarray],
    initial_counts: np.ndarray,
    maximum_counts: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Apply improving one-count exchanges on the exact runtime grid."""
    current = np.asarray(initial_counts, dtype=np.int64).copy()
    current_value = float(predict((current / MIXTURE_BLOCK_SIZE)[None, :])[0])
    for step in range(MAX_EXCHANGE_STEPS):
        donors = np.flatnonzero(current > 0)
        receivers = np.flatnonzero(current < maximum_counts)
        proposals = []
        moves = []
        for donor in donors:
            for receiver in receivers:
                if donor == receiver:
                    continue
                proposal = current.copy()
                proposal[donor] -= 1
                proposal[receiver] += 1
                proposals.append(proposal)
                moves.append((donor, receiver))
        if not proposals:
            return current, step
        values = predict(np.asarray(proposals, dtype=float) / MIXTURE_BLOCK_SIZE)
        choice = int(np.argmin(values))
        if float(values[choice]) >= current_value - REFINE_TOLERANCE:
            return current, step
        donor, receiver = moves[choice]
        current[donor] -= 1
        current[receiver] += 1
        current_value = float(values[choice])
    raise RuntimeError(f"Runtime exchange refinement exceeded {MAX_EXCHANGE_STEPS} steps")


def materialize_ensemble(
    panel: swarm39.Panel,
    ensemble: SurrogateEnsemble,
    scales: np.ndarray,
    caps: tuple[int, ...],
) -> list[Candidate]:
    candidates = []
    previous: np.ndarray | None = None
    for cap in caps:
        upper = np.minimum(1.0, cap / scales)
        starts = optimization_starts(panel, ensemble, upper, previous)
        continuous, continuous_value, successes = continuous_optimum(ensemble.predict, upper, starts)
        maximum_counts = np.floor(upper * MIXTURE_BLOCK_SIZE + 1e-12).astype(np.int64)
        if int(maximum_counts.sum()) < MIXTURE_BLOCK_SIZE:
            raise ValueError(f"Cap {cap} is infeasible after runtime quantization")
        initial = prefix_materializer.constrained_counts(continuous, maximum_counts)
        counts, exchange_steps = refine_runtime_counts(ensemble.predict, initial, maximum_counts)
        weights = counts / MIXTURE_BLOCK_SIZE
        epochs = scales * weights
        if int(counts.sum()) != MIXTURE_BLOCK_SIZE:
            raise ValueError("Runtime counts do not fill the mixture block")
        if float(epochs.max()) > cap + 1e-10:
            raise ValueError(f"Runtime candidate exceeds cap {cap}: {epochs.max()}")
        if not np.array_equal(prefix_materializer.runtime_counts(weights), counts):
            raise ValueError("Runtime candidate is unstable under realization")
        candidates.append(
            Candidate(
                model_key=ensemble.key,
                target=ensemble.target,
                cap=cap,
                continuous_weights=continuous,
                runtime_counts=counts,
                continuous_prediction=continuous_value,
                runtime_prediction=float(ensemble.predict(weights[None, :])[0]),
                exchange_steps=exchange_steps,
                optimizer_successes=successes,
            )
        )
        previous = continuous
    return candidates


def heldout_scores(ensembles: dict[tuple[str, str], SurrogateEnsemble]) -> pd.DataFrame:
    heldout = base.single_phase_heldout("delphi_3e18")
    rows = []
    for (model_key, target), ensemble in ensembles.items():
        query = heldout.subset(np.isfinite(heldout.targets[target]))
        score = base.score(ensemble.predict(query.phase0), query.targets[target])
        rows.append({"model": model_key, "target": target} | score)
    return pd.DataFrame(rows)


def candidate_tables(
    panel: swarm39.Panel,
    candidates: list[Candidate],
    ensembles: dict[tuple[str, str], SurrogateEnsemble],
    scales: np.ndarray,
    caps: tuple[int, ...],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    weight_rows = []
    candidate_weights: dict[tuple[str, str, int], np.ndarray] = {}
    for candidate in candidates:
        weights = candidate.runtime_weights
        epochs = scales * weights
        distances = 0.5 * np.abs(panel.phase0 - weights).sum(axis=1)
        nearest = int(np.argmin(distances))
        ensemble = ensembles[(candidate.model_key, candidate.target)]
        member_predictions = ensemble.member_predictions(weights[None, :])[:, 0]
        candidate_id = f"{candidate.model_key}_{candidate.target.removesuffix('_bpb')}_cap{candidate.cap:02d}"
        candidate_weights[(candidate.model_key, candidate.target, candidate.cap)] = weights
        row = {
            "candidate_id": candidate_id,
            "model": candidate.model_key,
            "target": candidate.target,
            "target_label": TARGET_LABELS[candidate.target],
            "epoch_cap": candidate.cap,
            "continuous_predicted_bpb": candidate.continuous_prediction,
            "runtime_predicted_bpb": candidate.runtime_prediction,
            "partition_prediction_sd": float(np.std(member_predictions, ddof=1)),
            "max_materialized_epoch": float(epochs.max()),
            "support_buckets": int(np.count_nonzero(weights)),
            "effective_buckets": float(np.exp(-np.sum(weights[weights > 0] * np.log(weights[weights > 0])))),
            "tv_to_proportional": float(0.5 * np.abs(weights - panel.proportional).sum()),
            "nearest_panel_row_id": str(panel.row_id[nearest]),
            "nearest_panel_tv": float(distances[nearest]),
            "largest_bucket": str(panel.buckets[int(np.argmax(weights))]),
            "largest_weight": float(weights.max()),
            "exchange_steps": candidate.exchange_steps,
            "optimizer_successes": candidate.optimizer_successes,
        }
        for prediction_target in TARGETS:
            for model_key in MODEL_KEYS:
                row[f"predicted_{prediction_target}_by_{model_key}"] = float(
                    ensembles[(model_key, prediction_target)].predict(weights[None, :])[0]
                )
        summary_rows.append(row)
        for index, bucket in enumerate(panel.buckets):
            weight_rows.append(
                {
                    "candidate_id": candidate_id,
                    "model": candidate.model_key,
                    "target": candidate.target,
                    "target_label": TARGET_LABELS[candidate.target],
                    "epoch_cap": candidate.cap,
                    "domain": bucket,
                    "runtime_count": int(candidate.runtime_counts[index]),
                    "weight": float(weights[index]),
                    "proportional_weight": float(panel.proportional[index]),
                    "materialized_epochs": float(epochs[index]),
                }
            )
    comparison_rows = []
    for target in TARGETS:
        for cap in caps:
            first = candidate_weights[(MODEL_KEYS[0], target, cap)]
            second = candidate_weights[(MODEL_KEYS[1], target, cap)]
            comparison_rows.append(
                {
                    "target": target,
                    "target_label": TARGET_LABELS[target],
                    "epoch_cap": cap,
                    "cross_model_tv": float(0.5 * np.abs(first - second).sum()),
                    "cross_model_hellinger": frozen.hellinger(first, second),
                }
            )
    return pd.DataFrame(summary_rows), pd.DataFrame(weight_rows), pd.DataFrame(comparison_rows)


def model_fits_payload(ensembles: dict[tuple[str, str], SurrogateEnsemble]) -> dict:
    payload = {}
    for (model_key, target), ensemble in ensembles.items():
        payload[f"{model_key}/{target}"] = [
            {
                "partition_seed": seed,
                "shape": fit.shape,
                "l2": fit.l2,
                "intercept": fit.intercept,
                "coefficients": fit.coefficients.tolist(),
                "names": list(fit.names),
                "oof_rmse": fit.oof_rmse,
            }
            for seed, fit in zip(PARTITION_SEEDS, ensemble.fits, strict=True)
        ]
    return payload


def build_figures(summary: pd.DataFrame, comparison: pd.DataFrame, caps: tuple[int, ...]) -> list[go.Figure]:
    colors = {MODEL_KEYS[0]: "#147d6f", MODEL_KEYS[1]: "#d9542d"}
    objective = make_subplots(rows=1, cols=2, subplot_titles=[TARGET_LABELS[target] for target in TARGETS])
    support = make_subplots(rows=1, cols=2, subplot_titles=[TARGET_LABELS[target] for target in TARGETS])
    for column, target in enumerate(TARGETS, start=1):
        for model_key in MODEL_KEYS:
            frame = summary[(summary.target == target) & (summary.model == model_key)].sort_values("epoch_cap")
            objective.add_trace(
                go.Scatter(
                    x=frame.epoch_cap,
                    y=frame.runtime_predicted_bpb,
                    mode="lines+markers",
                    name=model_key.replace("_", " "),
                    legendgroup=model_key,
                    showlegend=column == 1,
                    line={"color": colors[model_key], "width": 3},
                    error_y={"type": "data", "array": frame.partition_prediction_sd, "visible": True},
                    customdata=np.column_stack([frame.nearest_panel_tv, frame.tv_to_proportional, frame.largest_weight]),
                    hovertemplate=(
                        "Cap %{x}<br>Predicted BPB %{y:.6f}<br>Nearest panel TV %{customdata[0]:.3f}"
                        "<br>TV to proportional %{customdata[1]:.3f}<br>Largest weight %{customdata[2]:.1%}"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=column,
            )
            support.add_trace(
                go.Scatter(
                    x=frame.epoch_cap,
                    y=frame.nearest_panel_tv,
                    mode="lines+markers",
                    name=model_key.replace("_", " "),
                    legendgroup=model_key,
                    showlegend=column == 1,
                    line={"color": colors[model_key], "width": 3},
                ),
                row=1,
                col=column,
            )
    for figure, title, y_title in (
        (objective, "Each model's own constrained optimum", "Predicted BPB"),
        (support, "How far the candidate lies from measured mixtures", "Nearest-panel TV"),
    ):
        figure.update_xaxes(title_text="Whole-run epoch cap", tickmode="array", tickvals=list(caps))
        figure.update_yaxes(title_text=y_title)
        figure.update_layout(title=title)
        frozen.base_layout(figure, height=520)
    disagreement = go.Figure()
    for target in TARGETS:
        frame = comparison[comparison.target == target]
        disagreement.add_trace(
            go.Scatter(
                x=frame.epoch_cap,
                y=frame.cross_model_tv,
                mode="lines+markers",
                name=TARGET_LABELS[target],
            )
        )
    disagreement.update_layout(title="Candidate disagreement between the two models")
    disagreement.update_xaxes(title="Whole-run epoch cap", tickmode="array", tickvals=list(caps))
    disagreement.update_yaxes(title="Total variation between candidate mixtures")
    frozen.base_layout(disagreement, height=480)
    return [objective, support, disagreement]


def render_report(
    output_path: Path,
    summary: pd.DataFrame,
    comparison: pd.DataFrame,
    scores: pd.DataFrame,
    panel_path: Path,
    figures: list[go.Figure],
    caps: tuple[int, ...],
) -> None:
    fragments = [
        pio.to_html(
            figure,
            include_plotlyjs=index == 0,
            full_html=False,
            config=PLOT_CONFIG,
            div_id=f"figure-{index}",
        )
        for index, figure in enumerate(figures)
    ]
    score_table = scores[["model", "target_label", "rmse", "spearman", "regret@1", "regret@3", "calibration"]].to_html(
        index=False, classes="audit-table", float_format=lambda value: f"{value:.6f}"
    )
    audit_columns = [
        "candidate_id",
        "runtime_predicted_bpb",
        "predicted_uncheatable_bpb_by_aggregate_linear_v",
        "predicted_uncheatable_bpb_by_corrected_separate_heads",
        "predicted_table9_macro_bpb_by_aggregate_linear_v",
        "predicted_table9_macro_bpb_by_corrected_separate_heads",
        "partition_prediction_sd",
        "nearest_panel_tv",
        "tv_to_proportional",
        "largest_bucket",
        "largest_weight",
    ]
    candidate_table = summary[audit_columns].to_html(
        index=False, classes="audit-table", float_format=lambda value: f"{value:.6f}"
    )
    comparison_table = comparison.to_html(index=False, classes="audit-table", float_format=lambda value: f"{value:.6f}")
    document = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Delphi one-phase surrogate challengers</title>
<style>
:root{{--ink:#17324a;--muted:#5d7080;--paper:#fbf7ef;--card:#fffdf8;--line:#d8cdbd;--accent:#d9542d;}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font-family:"Avenir Next",Avenir,sans-serif}}
main{{max-width:1500px;margin:auto;padding:54px 34px 90px}} h1,h2{{font-family:Georgia,serif;letter-spacing:-.025em}}
h1{{font-size:clamp(42px,6vw,76px);line-height:.98;max-width:1100px}} h2{{font-size:34px;margin-top:58px}}
.dek{{font-size:20px;line-height:1.55;max-width:1050px;color:var(--muted)}}
.warning{{border-left:6px solid var(--accent);padding:18px 24px;background:#fff4e9}}
.warning{{font-size:17px;line-height:1.5}}
.figure{{margin:30px -12px 54px}}
.table-wrap{{overflow:auto;background:var(--card);border:1px solid var(--line);padding:8px}}
.audit-table{{border-collapse:collapse;width:100%;font-size:14px}}
.audit-table th,.audit-table td{{padding:10px 12px;border-bottom:1px solid #e7ded1;text-align:right}}
.audit-table th:first-child,.audit-table td:first-child{{text-align:left}} code{{background:#efe8dc;padding:2px 5px}}
.provenance{{color:var(--muted);font-size:14px;overflow-wrap:anywhere}}
</style></head><body><main>
<p class="provenance">Local candidate materialization · no training submitted</p>
<h1>Two simpler alternatives to shared-shape DSP</h1>
<p class="dek">Both models fit the same 280 Delphi 3e18 one-phase endpoints and optimize tied mixtures on
the exact 1/2048 runtime grid. The aggregate V-head is the simple mechanistic candidate; corrected separate
heads is a flexible benchmark whose centres are derived from the one-phase panel.</p>
<div class="warning"><strong>Selection gate:</strong> these are model-generated policies, not measured
frontiers. Cross-model agreement and support distance determine which exact rows are worth validating.</div>
<h2>Heldout behavior</h2><div class="table-wrap">{score_table}</div>
<section class="figure">{fragments[0]}</section><section class="figure">{fragments[1]}</section>
<section class="figure">{fragments[2]}</section>
<h2>Exact candidates and cross-predictions</h2><div class="table-wrap">{candidate_table}</div>
<h2>Cross-model geometry</h2><div class="table-wrap">{comparison_table}</div>
<h2>Provenance</h2><p class="provenance">Panel: {html.escape(str(panel_path.relative_to(REPO_ROOT)))}<br>
Panel SHA-256: {frozen.file_sha256(panel_path)}<br>Partition seeds: {PARTITION_SEEDS}<br>
Caps: {caps}<br>Runtime mixture block: {MIXTURE_BLOCK_SIZE}</p>
</main></body></html>"""
    output_path.write_text(document)


def main() -> None:
    args = parse_args()
    caps = tuple(args.caps)
    if not caps or caps != tuple(sorted(set(caps))) or any(cap <= 0 for cap in caps):
        raise ValueError(f"Caps must be unique positive integers in increasing order, got {caps}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    panel_path = swarm39.CANONICAL / f"{base.ONE_PHASE_DATASET['delphi_3e18']}.csv"
    panel = base.one_phase_panel("delphi_3e18")
    if len(panel) != 280 or any(np.isfinite(panel.targets[target]).sum() != 280 for target in TARGETS):
        raise ValueError("The complete 280-row one-phase panel is required for both targets")
    scales = panel.c0 + panel.c1
    ensembles = {
        (model_key, target): fit_ensemble(panel, target, model_key) for model_key in MODEL_KEYS for target in TARGETS
    }
    candidates = [
        candidate for ensemble in ensembles.values() for candidate in materialize_ensemble(panel, ensemble, scales, caps)
    ]
    summary, weights, comparison = candidate_tables(panel, candidates, ensembles, scales, caps)
    scores = heldout_scores(ensembles)
    scores["target_label"] = scores.target.map(TARGET_LABELS)

    paths = {
        "candidate_summary.csv": summary,
        "candidate_weights.csv": weights,
        "cross_model_comparison.csv": comparison,
        "heldout_scores.csv": scores,
    }
    for name, frame in paths.items():
        frame.to_csv(args.output_dir / name, index=False)
    fits_path = args.output_dir / "model_fits.json"
    fits_path.write_text(json.dumps(model_fits_payload(ensembles), indent=2, sort_keys=True) + "\n")
    report_path = args.output_dir / "index.html"
    render_report(report_path, summary, comparison, scores, panel_path, build_figures(summary, comparison, caps), caps)

    weight_hashes = weights.groupby("candidate_id").apply(
        lambda frame: hashlib.sha256(frame.runtime_count.to_numpy(dtype=np.int64).tobytes()).hexdigest(),
        include_groups=False,
    )
    manifest = {
        "experiment": "Delphi 3e18 one-phase surrogate challengers",
        "training_status": "not_submitted",
        "fit_panel": str(panel_path.relative_to(REPO_ROOT)),
        "fit_panel_sha256": frozen.file_sha256(panel_path),
        "fit_rows": len(panel),
        "models": list(MODEL_KEYS),
        "targets": list(TARGETS),
        "caps": list(caps),
        "partition_seeds": list(PARTITION_SEEDS),
        "mixture_block_size": MIXTURE_BLOCK_SIZE,
        "candidate_count": len(summary),
        "unique_runtime_mixtures": int(weight_hashes.nunique()),
        "outputs": {
            path.name: frozen.file_sha256(path)
            for path in [*(args.output_dir / name for name in paths), fits_path, report_path]
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(summary.to_string(index=False))
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
