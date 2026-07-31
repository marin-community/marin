# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "cvxpy",
#   "fsspec",
#   "gcsfs",
#   "numpy",
#   "pandas",
#   "plotly",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
# ]
# ///
"""Audit algebraic phase tying against independently refitted one-phase models.

The 238-row Delphi one-phase augmented swarm is the only fitting data. The
remaining one-phase policies, including the exposed adversarial panel, are
read only by the ``evaluate`` stage after the fitting protocol is frozen.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_partially_pooled_phase_bowls as pooled,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round54_single_phase_refit"
FIT_SERIES = "delphi_one_phase_augmented_swarm_3e18_20260715"
ADVERSARIAL_SERIES = "delphi_3e18_adversarial_stress_panel_20260716"
TARGET_COLUMNS = {
    "uncheatable": "uncheatable_bpb",
    "table9": "table9_macro_bpb",
}
MODEL_IDS = tuple(observatory.DELPHI_3E18_MODEL_IDS)
SEEDS = (0, 1, 2)
EXPECTED_FIT_ROWS = 238
PREREGISTRATION = OUTPUT_DIR / "preregistration.json"


def json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_value(payload), indent=2, sort_keys=True, allow_nan=False) + "\n")


def preregistration_payload() -> dict[str, Any]:
    return {
        "round": 54,
        "frozenAt": datetime.now(UTC).isoformat(),
        "purpose": "Compare algebraic phase tying with a direct one-phase refit of each frozen baseline form.",
        "fitData": {
            "series": FIT_SERIES,
            "expectedRows": EXPECTED_FIT_ROWS,
            "targets": TARGET_COLUMNS,
            "policy": "single_phase_tied",
            "coordinateOverlap": "coordinate_disjoint",
        },
        "models": list(MODEL_IDS),
        "fitProtocol": {
            "policyClass": observatory.SINGLE_PHASE,
            "oofSeeds": list(SEEDS),
            "foldsPerSeed": 5,
            "hyperparameters": "Existing Observatory selection procedure, unchanged.",
            "singlePhaseInput": "phase_0_weight == phase_1_weight",
        },
        "evaluationSegments": [
            "one_phase_fit_oof",
            "historical_one_phase_heldout",
            "adversarial_one_phase_target_matched",
            "adversarial_one_phase_cross_target",
        ],
        "metrics": [
            "rmse",
            "mae",
            "bias_predicted_minus_observed",
            "spearman",
            "observed_on_predicted_slope",
            "regret_at_1",
            "regret_at_3",
            "regret_at_5",
            "selected_optimism",
            "optimism_gt_0p05_count",
            "worst_optimism",
            "low_tail_rmse",
            "lower_tail_optimism",
        ],
        "adversarialStrata": ["candidate_target", "selection"],
        "decisionRule": (
            "This is a required restriction audit, not candidate promotion. Direct refitting is preferable only if "
            "it improves OOF and heldout calibration/regret without changing the functional form."
        ),
        "dataUseBoundary": (
            "The fit stage may read outcomes only from the designated 238-row series. The evaluate stage may read "
            "all exposed development outcomes only after this file exists."
        ),
    }


def freeze() -> None:
    if PREREGISTRATION.exists():
        existing = json.loads(PREREGISTRATION.read_text())
        expected = preregistration_payload()
        existing.pop("frozenAt", None)
        expected.pop("frozenAt", None)
        if existing != expected:
            raise ValueError("Existing round-54 preregistration differs from the current protocol")
        print(f"preregistration already frozen: {PREREGISTRATION}")
        return
    write_json(PREREGISTRATION, preregistration_payload())
    digest = hashlib.sha256(PREREGISTRATION.read_bytes()).hexdigest()
    print(f"froze round-54 protocol: {PREREGISTRATION} sha256={digest}")


def parse_weights(frame: pd.DataFrame, domains: Sequence[str]) -> np.ndarray:
    def one(value: str) -> list[float]:
        parsed = json.loads(value)
        return [float(parsed[domain]) for domain in domains]

    phase0 = np.asarray([one(value) for value in frame["phase_0_weights_json"]], dtype=float)
    phase1 = np.asarray([one(value) for value in frame["phase_1_weights_json"]], dtype=float)
    weights = np.stack([phase0, phase1], axis=1)
    if not np.allclose(weights.sum(axis=2), 1.0, atol=1e-9):
        raise ValueError("One-phase weights do not sum to one")
    return weights


def fit_frame() -> pd.DataFrame:
    frame = pd.read_csv(observatory.DELPHI_3E18_HELDOUTS)
    selected = frame[
        (frame["training_series"] == FIT_SERIES)
        & (frame["policy_class"] == "single_phase_tied")
        & (frame["fit_panel_overlap"] == "coordinate_disjoint")
        & (frame["training_state"] == "finished")
        & (frame["checkpoint_declared_complete"] == 1)
    ].reset_index(drop=True)
    if len(selected) != EXPECTED_FIT_ROWS:
        raise ValueError(f"Expected {EXPECTED_FIT_ROWS} one-phase fit rows, found {len(selected)}")
    return selected


def one_phase_dataset(target_id: str, frame: pd.DataFrame) -> pooled.Dataset:
    reference = observatory.load_delphi_3e18_fit_dataset(target_id)
    weights = parse_weights(frame, reference.domain_names)
    if not np.allclose(weights[:, 0], weights[:, 1], atol=1e-12):
        raise ValueError("The one-phase fit series contains a phase-varying policy")
    target = frame[TARGET_COLUMNS[target_id]].to_numpy(dtype=float)
    if not np.isfinite(target).all():
        raise ValueError(f"The one-phase fit series has incomplete {target_id} outcomes")
    return pooled.Dataset(
        name=f"delphi_3e18_single_{target_id}",
        frame=frame,
        y=target,
        weights=weights,
        c0=np.asarray(reference.c0, dtype=float),
        c1=np.asarray(reference.c1, dtype=float),
        domain_names=list(reference.domain_names),
    )


def fit_result_path(target_id: str, model_id: str) -> Path:
    return OUTPUT_DIR / "fits" / target_id / f"{model_id}.json"


def fit_models() -> None:
    if not PREREGISTRATION.exists():
        raise ValueError("Freeze the round-54 protocol before fitting")
    frame = fit_frame()
    all_frame = pd.read_csv(observatory.DELPHI_3E18_HELDOUTS)
    all_single = all_frame[
        (all_frame["policy_class"] == "single_phase_tied")
        & (all_frame["fit_panel_overlap"] == "coordinate_disjoint")
        & (all_frame["training_state"] == "finished")
        & (all_frame["checkpoint_declared_complete"] == 1)
    ].reset_index(drop=True)
    for target_id in TARGET_COLUMNS:
        dataset = one_phase_dataset(target_id, frame)
        all_weights = parse_weights(all_single, dataset.domain_names)
        for model_id in MODEL_IDS:
            path = fit_result_path(target_id, model_id)
            if path.exists():
                print(f"fit cache hit: {target_id}/{model_id}", flush=True)
                continue
            print(f"fitting independent one-phase model: {target_id}/{model_id}", flush=True)
            model, oof_prediction, full_prediction, tuning = observatory.fit_one_model(
                dataset,
                model_id,
                observatory.SINGLE_PHASE,
                SEEDS,
            )
            all_prediction = observatory.predict_model(
                model,
                dataset,
                model_id,
                observatory.SINGLE_PHASE,
                all_weights,
            )
            detail = observatory.fit_detail(
                dataset,
                model_id,
                model,
                oof_prediction,
                full_prediction,
                tuning,
                observatory.SINGLE_PHASE,
                protocol=(
                    "Three-seed, five-fold OOF on 238 coordinate-disjoint one-phase Delphi 3e18 swarm rows; "
                    "unchanged Observatory hyperparameter selection; full refit projected onto all disjoint "
                    "one-phase development policies."
                ),
                oof_seeds=SEEDS,
            )
            write_json(
                path,
                {
                    "target": target_id,
                    "model": model_id,
                    "fitHeldoutIds": frame["heldout_id"].astype(str).tolist(),
                    "evaluationHeldoutIds": all_single["heldout_id"].astype(str).tolist(),
                    "oofPrediction": oof_prediction,
                    "fullFitPrediction": full_prediction,
                    "evaluationPrediction": all_prediction,
                    "fitDetail": detail,
                },
            )


def selection_from_tags(value: str) -> str:
    tags = json.loads(value)
    matches = [tag.removeprefix("selection=") for tag in tags if tag.startswith("selection=")]
    return matches[0] if matches else "not_applicable"


def metric_row(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    observed = np.asarray(observed, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    if len(observed) < 2 or not np.isfinite(observed).all() or not np.isfinite(predicted).all():
        raise ValueError("Metric segment must contain at least two finite observations")
    residual = predicted - observed
    slope = float(np.polyfit(predicted, observed, 1)[0]) if np.ptp(predicted) > 1e-12 else float("nan")
    rank = float(spearmanr(observed, predicted).statistic) if len(observed) >= 3 else float("nan")
    order = np.argsort(predicted)
    best = float(np.min(observed))
    tail_count = min(len(observed), max(5, math.ceil(0.15 * len(observed))))
    tail = order[:tail_count]
    optimism = observed - predicted
    selected = int(order[0])
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "mae": float(np.mean(np.abs(residual))),
        "bias_predicted_minus_observed": float(np.mean(residual)),
        "spearman": rank,
        "observed_on_predicted_slope": slope,
        "regret_at_1": float(observed[selected] - best),
        "regret_at_3": float(np.min(observed[order[: min(3, len(order))]]) - best),
        "regret_at_5": float(np.min(observed[order[: min(5, len(order))]]) - best),
        "selected_optimism": float(optimism[selected]),
        "optimism_gt_0p05_count": int(np.sum(optimism > 0.05)),
        "worst_optimism": float(np.max(optimism)),
        "low_tail_rmse": float(np.sqrt(np.mean(residual[tail] ** 2))),
        "lower_tail_optimism": float(np.mean(np.maximum(optimism[tail], 0.0))),
    }


def two_phase_predictions(target_id: str, model_id: str, heldout_indices: np.ndarray) -> np.ndarray:
    path = observatory.cache_path("delphi_3e18", target_id, observatory.TWO_PHASE, model_id)
    payload = json.loads(path.read_text())
    prediction = np.asarray(payload["prediction"], dtype=float)
    expected = 280 + len(pd.read_csv(observatory.DELPHI_3E18_HELDOUTS))
    if len(prediction) != expected:
        raise ValueError(f"Unexpected cached prediction length for {target_id}/{model_id}: {len(prediction)}")
    return prediction[280 + heldout_indices]


def add_segment_metrics(
    rows: list[dict[str, Any]],
    prediction_frame: pd.DataFrame,
    *,
    target_id: str,
    model_id: str,
    fit_mode: str,
    segment: str,
    mask: np.ndarray,
    prediction_column: str,
) -> None:
    subset = prediction_frame.loc[mask]
    if len(subset) < 2:
        return
    rows.append(
        {
            "target": target_id,
            "model": model_id,
            "fit_mode": fit_mode,
            "segment": segment,
            "candidate_target": "all",
            "selection_stratum": "all",
            **metric_row(
                subset["observed"].to_numpy(dtype=float),
                subset[prediction_column].to_numpy(dtype=float),
            ),
        }
    )


def evaluate() -> None:
    if not PREREGISTRATION.exists():
        raise ValueError("Freeze and fit the round-54 protocol before evaluation")
    missing = [fit_result_path(target, model) for target in TARGET_COLUMNS for model in MODEL_IDS]
    missing = [path for path in missing if not path.exists()]
    if missing:
        raise ValueError(f"Missing {len(missing)} frozen one-phase fits; run --stage fit first")

    all_frame = pd.read_csv(observatory.DELPHI_3E18_HELDOUTS)
    all_mask = (
        (all_frame["policy_class"] == "single_phase_tied")
        & (all_frame["fit_panel_overlap"] == "coordinate_disjoint")
        & (all_frame["training_state"] == "finished")
        & (all_frame["checkpoint_declared_complete"] == 1)
    )
    all_indices = np.flatnonzero(all_mask.to_numpy())
    single = all_frame.loc[all_mask].reset_index(drop=True)
    single["selection_stratum"] = single["tags_json"].map(selection_from_tags)
    metric_rows: list[dict[str, Any]] = []
    prediction_parts: list[pd.DataFrame] = []

    for target_id, target_column in TARGET_COLUMNS.items():
        for model_id in MODEL_IDS:
            payload = json.loads(fit_result_path(target_id, model_id).read_text())
            direct = np.asarray(payload["evaluationPrediction"], dtype=float)
            fit_ids = set(payload["fitHeldoutIds"])
            fit_positions = np.flatnonzero(single["heldout_id"].astype(str).isin(fit_ids).to_numpy())
            if len(fit_positions) != EXPECTED_FIT_ROWS:
                raise ValueError(f"Could not map all fit rows for {target_id}/{model_id}")
            direct[fit_positions] = np.asarray(payload["oofPrediction"], dtype=float)
            tied = two_phase_predictions(target_id, model_id, all_indices)
            prediction_frame = pd.DataFrame(
                {
                    "target": target_id,
                    "model": model_id,
                    "heldout_id": single["heldout_id"].astype(str),
                    "training_series": single["training_series"].astype(str),
                    "candidate_target": single["objective"].astype(str),
                    "selection_stratum": single["selection_stratum"].astype(str),
                    "observed": single[target_column].to_numpy(dtype=float),
                    "independent_one_phase_prediction": direct,
                    "algebraic_tied_two_phase_prediction": tied,
                    "is_one_phase_fit_row": single["heldout_id"].astype(str).isin(fit_ids).to_numpy(),
                    "is_adversarial": single["training_series"].eq(ADVERSARIAL_SERIES).to_numpy(),
                }
            )
            prediction_parts.append(prediction_frame)
            fit_mask = prediction_frame["is_one_phase_fit_row"].to_numpy(dtype=bool)
            adversarial_mask = prediction_frame["is_adversarial"].to_numpy(dtype=bool)
            historical_mask = ~(fit_mask | adversarial_mask)
            for fit_mode, prediction_column in (
                ("independent_one_phase_refit", "independent_one_phase_prediction"),
                ("algebraic_restriction_of_two_phase_fit", "algebraic_tied_two_phase_prediction"),
            ):
                add_segment_metrics(
                    metric_rows,
                    prediction_frame,
                    target_id=target_id,
                    model_id=model_id,
                    fit_mode=fit_mode,
                    segment="one_phase_fit_panel",
                    mask=fit_mask,
                    prediction_column=prediction_column,
                )
                add_segment_metrics(
                    metric_rows,
                    prediction_frame,
                    target_id=target_id,
                    model_id=model_id,
                    fit_mode=fit_mode,
                    segment="historical_one_phase_heldout",
                    mask=historical_mask,
                    prediction_column=prediction_column,
                )
                add_segment_metrics(
                    metric_rows,
                    prediction_frame,
                    target_id=target_id,
                    model_id=model_id,
                    fit_mode=fit_mode,
                    segment="adversarial_one_phase_all",
                    mask=adversarial_mask,
                    prediction_column=prediction_column,
                )
                for candidate_target in ("uncheatable", "table9"):
                    target_mask = adversarial_mask & prediction_frame["candidate_target"].eq(candidate_target).to_numpy()
                    add_segment_metrics(
                        metric_rows,
                        prediction_frame,
                        target_id=target_id,
                        model_id=model_id,
                        fit_mode=fit_mode,
                        segment=(
                            "adversarial_one_phase_target_matched"
                            if candidate_target == target_id
                            else "adversarial_one_phase_cross_target"
                        ),
                        mask=target_mask,
                        prediction_column=prediction_column,
                    )
                    if candidate_target == target_id:
                        for selection in ("baseline_ranked", "challenger_ranked", "high_disagreement"):
                            stratum_mask = target_mask & prediction_frame["selection_stratum"].eq(selection).to_numpy()
                            add_segment_metrics(
                                metric_rows,
                                prediction_frame,
                                target_id=target_id,
                                model_id=model_id,
                                fit_mode=fit_mode,
                                segment="adversarial_one_phase_target_matched_stratum",
                                mask=stratum_mask,
                                prediction_column=prediction_column,
                            )
                            if metric_rows:
                                metric_rows[-1]["candidate_target"] = candidate_target
                                metric_rows[-1]["selection_stratum"] = selection

    predictions = pd.concat(prediction_parts, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    predictions.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    make_plots(metrics, predictions)
    write_report(metrics, predictions)
    print(
        metrics[metrics["segment"].isin(["one_phase_fit_panel", "historical_one_phase_heldout"])].to_string(index=False)
    )


def make_plots(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    models = list(MODEL_IDS)
    colors = {
        "independent_one_phase_refit": "#d95f02",
        "algebraic_restriction_of_two_phase_fit": "#1b4f72",
    }
    figure = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Uncheatable: one-phase fit panel",
            "Uncheatable: historical heldout",
            "Table-9: one-phase fit panel",
            "Table-9: historical heldout",
        ),
    )
    for row_index, target in enumerate(("uncheatable", "table9"), start=1):
        for col_index, segment in enumerate(("one_phase_fit_panel", "historical_one_phase_heldout"), start=1):
            selected = metrics[(metrics["target"] == target) & (metrics["segment"] == segment)]
            for fit_mode in colors:
                subset = selected[selected["fit_mode"] == fit_mode].set_index("model").reindex(models)
                figure.add_trace(
                    go.Bar(
                        x=models,
                        y=subset["rmse"],
                        name=fit_mode.replace("_", " "),
                        marker_color=colors[fit_mode],
                        legendgroup=fit_mode,
                        showlegend=row_index == 1 and col_index == 1,
                    ),
                    row=row_index,
                    col=col_index,
                )
    figure.update_layout(
        barmode="group",
        template="plotly_white",
        title="Direct one-phase refit versus algebraic restriction",
        height=850,
        width=1500,
        margin=dict(l=70, r=40, t=100, b=170),
    )
    figure.update_xaxes(tickangle=-40)
    figure.update_yaxes(title_text="RMSE")
    figure.write_html(OUTPUT_DIR / "restriction_refit_rmse.html", include_plotlyjs="cdn")

    scatter = make_subplots(rows=1, cols=2, subplot_titles=("Uncheatable", "Table-9"))
    for column, target in enumerate(("uncheatable", "table9"), start=1):
        subset = predictions[
            (predictions["target"] == target) & ~predictions["is_one_phase_fit_row"] & ~predictions["is_adversarial"]
        ]
        for model_id in models:
            model = subset[subset["model"] == model_id]
            scatter.add_trace(
                go.Scatter(
                    x=model["observed"],
                    y=model["independent_one_phase_prediction"],
                    mode="markers",
                    name=model_id,
                    legendgroup=model_id,
                    showlegend=column == 1,
                    customdata=np.stack([model["heldout_id"], model["training_series"]], axis=1),
                    hovertemplate="observed=%{x:.4f}<br>predicted=%{y:.4f}<br>%{customdata[0]}<br>%{customdata[1]}<extra></extra>",
                ),
                row=1,
                col=column,
            )
        low = float(min(subset["observed"].min(), subset["independent_one_phase_prediction"].min()))
        high = float(max(subset["observed"].max(), subset["independent_one_phase_prediction"].max()))
        scatter.add_trace(
            go.Scatter(
                x=[low, high], y=[low, high], mode="lines", line=dict(color="#666", dash="dash"), showlegend=False
            ),
            row=1,
            col=column,
        )
    scatter.update_layout(
        template="plotly_white",
        title="Independent one-phase refits on historical heldouts",
        height=650,
        width=1450,
    )
    scatter.update_xaxes(title_text="Observed BPB")
    scatter.update_yaxes(title_text="Predicted BPB")
    scatter.write_html(OUTPUT_DIR / "historical_one_phase_calibration.html", include_plotlyjs="cdn")


def write_report(metrics: pd.DataFrame, predictions: pd.DataFrame) -> None:
    lines = [
        "# Round 54: independently fitted Delphi one-phase restrictions",
        "",
        "## Boundary",
        "",
        f"- Fit data: exactly {EXPECTED_FIT_ROWS} coordinate-disjoint policies from `{FIT_SERIES}`.",
        "- The existing two-phase fits are evaluated algebraically on tied inputs without refitting.",
        "- Each independent one-phase model uses the same restricted functional form and unchanged Observatory selection procedure.",
        "- Exposed adversarial outcomes were read only after `preregistration.json` and every model fit existed.",
        "",
        "## Headline comparison",
        "",
    ]
    headline = metrics[metrics["segment"].isin(["one_phase_fit_panel", "historical_one_phase_heldout"])].copy()
    columns = [
        "target",
        "model",
        "fit_mode",
        "segment",
        "n",
        "rmse",
        "spearman",
        "observed_on_predicted_slope",
        "regret_at_1",
        "optimism_gt_0p05_count",
        "worst_optimism",
    ]
    lines.extend(headline[columns].to_markdown(index=False).splitlines())
    lines.extend(["", "## Interpretation", ""])
    historical = headline[headline["segment"] == "historical_one_phase_heldout"]
    for target in TARGET_COLUMNS:
        target_rows = historical[historical["target"] == target]
        independent = target_rows[target_rows["fit_mode"] == "independent_one_phase_refit"]
        tied = target_rows[target_rows["fit_mode"] == "algebraic_restriction_of_two_phase_fit"]
        best_independent = independent.loc[independent["rmse"].idxmin()]
        best_tied = tied.loc[tied["rmse"].idxmin()]
        lines.append(
            f"- **{target}:** best independent refit RMSE is {best_independent['rmse']:.5f} "
            f"(`{best_independent['model']}`); best algebraically tied two-phase-fit RMSE is "
            f"{best_tied['rmse']:.5f} (`{best_tied['model']}`)."
        )
    lines.extend(
        [
            "",
            "These protocols answer different questions. OOF on the 238-row one-phase swarm measures the fitted one-phase "
            "restriction. Algebraic tying measures whether parameters learned from phase-varying policies transfer to the "
            "tied subspace. A gap is evidence about identification and transfer, not permission to call one protocol the other.",
            "",
            "## Artifacts",
            "",
            "- `metrics.csv`: segment- and stratum-specific diagnostics.",
            "- `predictions.csv`: all one-phase predictions and provenance.",
            "- `restriction_refit_rmse.html`: fit-panel and historical-heldout RMSE comparison.",
            "- `historical_one_phase_calibration.html`: observed-versus-predicted historical heldouts.",
            "",
            f"Prediction rows: {len(predictions)}.",
        ]
    )
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "fit", "evaluate", "all"), default="all")
    args = parser.parse_args()
    if args.stage in {"freeze", "all"}:
        freeze()
    if args.stage in {"fit", "all"}:
        fit_models()
    if args.stage in {"evaluate", "all"}:
        evaluate()


if __name__ == "__main__":
    main()
