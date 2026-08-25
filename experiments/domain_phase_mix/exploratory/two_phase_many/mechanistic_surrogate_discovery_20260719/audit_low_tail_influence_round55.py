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
"""Measure sensitivity of fitted surfaces to the lowest observed fit rows.

This is a diagnostic, not a trimming proposal. Structural hyperparameters are
frozen at the full-panel Observatory selections. Only model coefficients are
refit after removing the lowest-loss rows.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.optimize import minimize
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[5]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    benchmark_hierarchical_coverage_grp_20260715 as hierarchical,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    export_mixture_fit_observatory as observatory,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (  # noqa: E402
    fit_production_grp_quality_variants as family_grp,
)

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = SCRIPT_DIR.parent / "reference_outputs/mechanistic_surrogate_discovery_20260719/round55_low_tail_influence"
PREREGISTRATION = OUTPUT_DIR / "preregistration.json"
MODEL_IDS = ("separate_heads", "hierarchical_phase_bucket_replay")
TARGETS = ("uncheatable", "table9")
EXCLUSION_COUNTS = (0, 1, 3, 7, 14)
RANDOM_SEED = 20260719


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


def freeze() -> None:
    payload = {
        "round": 55,
        "frozenAt": datetime.now(UTC).isoformat(),
        "purpose": "Test whether a few low-observed-loss fit rows determine surface extrapolation and decisions.",
        "models": list(MODEL_IDS),
        "targets": list(TARGETS),
        "exclusionCounts": list(EXCLUSION_COUNTS),
        "fitPolicy": "two_phase",
        "hyperparameterPolicy": "Freeze every structural hyperparameter at the full 280-row Observatory selection.",
        "refitPolicy": "Refit only the constrained linear/nonnegative response head after deleting the k lowest outcomes.",
        "optimization": {
            "parameterization": "two independent simplex logits with one reference logit removed per phase",
            "starts": "proportional, empirical fit frontier, and four seeded Dirichlet policies",
            "bounds": [-10.0, 10.0],
            "maxiter": 250,
        },
        "interpretation": (
            "Any instability is evidence against identified raw optimization. Stability does not validate the surface. "
            "No removed-row result may be used as a training or trimming rule."
        ),
    }
    if PREREGISTRATION.exists():
        existing = json.loads(PREREGISTRATION.read_text())
        existing.pop("frozenAt", None)
        payload.pop("frozenAt", None)
        if existing != payload:
            raise ValueError("Existing round-55 preregistration differs from the current protocol")
        return
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PREREGISTRATION.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def cached_tuning(target: str, model_id: str) -> dict[str, Any]:
    path = observatory.cache_path("delphi_3e18", target, observatory.TWO_PHASE, model_id)
    return json.loads(path.read_text())["fitDetail"]["tuning"]


def fit_predictor(
    dataset: Any,
    indices: np.ndarray,
    model_id: str,
    tuning: Mapping[str, Any],
) -> Callable[[np.ndarray], np.ndarray]:
    if model_id == "separate_heads":
        model = observatory.separate_fit(dataset, indices, float(tuning["l2"]), observatory.TWO_PHASE)
        return lambda weights: observatory.separate_predict(model, dataset, weights, observatory.TWO_PHASE)
    if model_id == "hierarchical_phase_bucket_replay":
        values = tuning["shapeParameters"]
        shape = family_grp.Shape(
            exponent=float(values["exponent"]),
            late_multiplier=float(values["lateMultiplier"]),
            forgetting_rate=float(values["forgettingRate"]),
            penalty_threshold=float(values["penaltyThreshold"]),
        )
        config = hierarchical.Config(
            variant=hierarchical.Variant.HIERARCHICAL_PHASE_BUCKET_REPLAY,
            shape_index=0,
            shape=shape,
            l2=float(tuning["l2"]),
            residual_shrink=float(tuning["residualShrink"]),
            undercoverage_fraction=0.0,
            coverage_gate_ratio=0.0,
        )
        model = observatory.hierarchical_phase_replay_fit(dataset, indices, config)
        return model.predict
    raise ValueError(model_id)


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    values = np.exp(shifted)
    return values / values.sum()


def unpack(theta: np.ndarray, domains: int) -> np.ndarray:
    phase0 = softmax(np.concatenate([theta[: domains - 1], [0.0]]))
    phase1 = softmax(np.concatenate([theta[domains - 1 :], [0.0]]))
    return np.stack([phase0, phase1], axis=0)


def logits(weights: np.ndarray) -> np.ndarray:
    clipped = np.maximum(weights, 1e-9)
    phase0 = np.log(clipped[0, :-1]) - np.log(clipped[0, -1])
    phase1 = np.log(clipped[1, :-1]) - np.log(clipped[1, -1])
    return np.concatenate([phase0, phase1])


def raw_optimum(
    dataset: Any,
    predictor: Callable[[np.ndarray], np.ndarray],
    empirical_best: np.ndarray,
) -> tuple[np.ndarray, float, bool]:
    alpha0, _alpha1 = observatory.phase_fractions(dataset)
    natural = observatory.natural_weights(dataset, alpha0)
    rng = np.random.default_rng(RANDOM_SEED)
    starts = [np.stack([natural, natural]), empirical_best]
    starts.extend(
        np.stack([rng.dirichlet(0.4 * np.ones(dataset.m)), rng.dirichlet(0.4 * np.ones(dataset.m))]) for _ in range(4)
    )
    best: tuple[float, np.ndarray, bool] | None = None

    def objective(theta: np.ndarray) -> float:
        weights = unpack(theta, dataset.m)
        return float(predictor(weights[None, :, :])[0])

    for start in starts:
        result = minimize(
            objective,
            logits(start),
            method="L-BFGS-B",
            bounds=[(-10.0, 10.0)] * (2 * (dataset.m - 1)),
            options={"maxiter": 250, "ftol": 1e-11},
        )
        candidate = (float(result.fun), unpack(result.x, dataset.m), bool(result.success))
        if best is None or candidate[0] < best[0]:
            best = candidate
    if best is None:
        raise RuntimeError("No raw optimum candidates")
    return best[1], best[0], best[2]


def metric_row(observed: np.ndarray, predicted: np.ndarray) -> dict[str, float | int]:
    residual = predicted - observed
    optimism = observed - predicted
    order = np.argsort(predicted)
    slope = float(np.polyfit(predicted, observed, 1)[0]) if np.ptp(predicted) > 1e-12 else float("nan")
    return {
        "n": len(observed),
        "rmse": float(np.sqrt(np.mean(residual**2))),
        "bias": float(np.mean(residual)),
        "spearman": float(spearmanr(observed, predicted).statistic),
        "observed_on_predicted_slope": slope,
        "regret_at_1": float(observed[order[0]] - np.min(observed)),
        "regret_at_3": float(np.min(observed[order[:3]]) - np.min(observed)),
        "regret_at_5": float(np.min(observed[order[:5]]) - np.min(observed)),
        "optimism_gt_0p05_count": int(np.sum(optimism > 0.05)),
        "worst_optimism": float(np.max(optimism)),
    }


def run() -> None:
    if not PREREGISTRATION.exists():
        raise ValueError("Freeze the round-55 protocol before running")
    metric_rows: list[dict[str, Any]] = []
    optimum_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    for target in TARGETS:
        dataset = observatory.load_delphi_3e18_fit_dataset(target)
        heldout_frame, heldout_weights = observatory.load_delphi_3e18_heldouts(dataset)
        disjoint = heldout_frame["fit_panel_overlap"].eq("coordinate_disjoint").to_numpy()
        heldout_observed = heldout_frame.loc[
            disjoint, {"uncheatable": "uncheatable_bpb", "table9": "table9_macro_bpb"}[target]
        ].to_numpy(float)
        heldout_weights = heldout_weights[disjoint]
        fit_order = np.argsort(dataset.y)
        empirical_best = dataset.weights[int(fit_order[0])]
        for model_id in MODEL_IDS:
            tuning = cached_tuning(target, model_id)
            baseline_optimum: np.ndarray | None = None
            for excluded_count in EXCLUSION_COUNTS:
                excluded = fit_order[:excluded_count]
                retained = np.setdiff1d(np.arange(dataset.n), excluded, assume_unique=True)
                predictor = fit_predictor(dataset, retained, model_id, tuning)
                fit_prediction = predictor(dataset.weights)
                heldout_prediction = predictor(heldout_weights)
                optimum, optimum_value, success = raw_optimum(dataset, predictor, empirical_best)
                if baseline_optimum is None:
                    baseline_optimum = optimum
                distance = np.abs(dataset.weights - optimum[None, :, :]).sum(axis=(1, 2))
                phase_tv = 0.5 * float(np.abs(optimum[0] - optimum[1]).sum())
                max_epoch = float(max(np.max(optimum[0] * dataset.c0), np.max(optimum[1] * dataset.c1)))
                for segment, observed, predicted in (
                    ("full_fit_panel", dataset.y, fit_prediction),
                    ("retained_fit_panel", dataset.y[retained], fit_prediction[retained]),
                    ("excluded_low_tail", dataset.y[excluded], fit_prediction[excluded]),
                    ("coordinate_disjoint_archive", heldout_observed, heldout_prediction),
                ):
                    if len(observed) < 3:
                        continue
                    metric_rows.append(
                        {
                            "target": target,
                            "model": model_id,
                            "excluded_count": excluded_count,
                            "segment": segment,
                            **metric_row(np.asarray(observed), np.asarray(predicted)),
                        }
                    )
                optimum_rows.append(
                    {
                        "target": target,
                        "model": model_id,
                        "excluded_count": excluded_count,
                        "predicted_raw_optimum": optimum_value,
                        "optimizer_success": success,
                        "phase_tv": phase_tv,
                        "max_weight": float(np.max(optimum)),
                        "max_simulated_epoch": max_epoch,
                        "nearest_fit_l1": float(np.min(distance)),
                        "l1_shift_from_full_fit_optimum": float(np.abs(optimum - baseline_optimum).sum()),
                    }
                )
                for phase in range(2):
                    for domain_index, domain in enumerate(dataset.domain_names):
                        weight_rows.append(
                            {
                                "target": target,
                                "model": model_id,
                                "excluded_count": excluded_count,
                                "phase": phase,
                                "domain": domain,
                                "weight": float(optimum[phase, domain_index]),
                            }
                        )
    metrics = pd.DataFrame(metric_rows)
    optima = pd.DataFrame(optimum_rows)
    weights = pd.DataFrame(weight_rows)
    metrics.to_csv(OUTPUT_DIR / "metrics.csv", index=False)
    optima.to_csv(OUTPUT_DIR / "raw_optima.csv", index=False)
    weights.to_csv(OUTPUT_DIR / "raw_optimum_weights.csv", index=False)
    render(metrics, optima)
    report(metrics, optima)


def render(metrics: pd.DataFrame, optima: pd.DataFrame) -> None:
    archive = metrics[metrics["segment"] == "coordinate_disjoint_archive"]
    figure = px.line(
        archive,
        x="excluded_count",
        y="rmse",
        color="model",
        facet_col="target",
        markers=True,
        title="Archive RMSE after excluding the best-observed fit rows",
        template="plotly_white",
    )
    figure.write_html(OUTPUT_DIR / "archive_rmse_sensitivity.html", include_plotlyjs="cdn")
    figure = px.line(
        optima,
        x="excluded_count",
        y="l1_shift_from_full_fit_optimum",
        color="model",
        facet_col="target",
        markers=True,
        title="Raw-optimum movement under low-tail deletion",
        template="plotly_white",
    )
    figure.write_html(OUTPUT_DIR / "raw_optimum_sensitivity.html", include_plotlyjs="cdn")


def report(metrics: pd.DataFrame, optima: pd.DataFrame) -> None:
    archive = metrics[metrics["segment"] == "coordinate_disjoint_archive"]
    lines = [
        "# Round 55: low-observed-tail influence audit",
        "",
        "This is a sensitivity diagnostic, not a trimming procedure. Structural hyperparameters remain frozen at the full-panel selection; only response coefficients are refit.",
        "",
        "## Coordinate-disjoint archive",
        "",
        *archive[
            [
                "target",
                "model",
                "excluded_count",
                "rmse",
                "spearman",
                "observed_on_predicted_slope",
                "regret_at_1",
                "optimism_gt_0p05_count",
                "worst_optimism",
            ]
        ]
        .to_markdown(index=False)
        .splitlines(),
        "",
        "## Raw-optimum stability",
        "",
        *optima.to_markdown(index=False).splitlines(),
        "",
        "A large shift after deleting one or a few best observations is evidence that the raw optimum is not identified. Stability is necessary but not sufficient because the same structural extrapolation bias can persist across all deletions.",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("freeze", "run", "all"), default="all")
    args = parser.parse_args()
    if args.stage in {"freeze", "all"}:
        freeze()
    if args.stage in {"run", "all"}:
        run()


if __name__ == "__main__":
    main()
