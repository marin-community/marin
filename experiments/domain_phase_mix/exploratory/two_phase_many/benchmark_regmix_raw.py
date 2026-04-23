# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["lightgbm", "numpy", "pandas", "scikit-learn"]
# ///
"""Fit RegMix-style regressors on the many-domain two-phase packet across subset sizes."""

from __future__ import annotations

import argparse
import ast
from functools import cache
import json
from pathlib import Path

import fsspec
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.regmix_regression_kfold import sample_configs
from experiments.domain_phase_mix.exploratory.two_phase_many.dataset_metadata import (
    load_two_phase_many_candidate_summary_spec,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    CALIBRATION_CV_WEIGHT,
    CALIBRATION_DEPOPT_WEIGHT,
    CALIBRATION_FOLDMEAN_WEIGHT,
    CALIBRATION_SUPPORT_WEIGHT,
    CALIBRATION_TAIL_WEIGHT,
)
from experiments.domain_phase_mix.static_batch_selection import (
    average_phase_tv_distance,
    retrospective_generic_selection,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    OBJECTIVE_METRIC,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
)
from experiments.domain_phase_mix.two_phase_many_regmix_raw_subset_optima import (
    REGMIX_RAW_SUBSET_OPTIMA_SOURCE_EXPERIMENT,
    REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES,
    regmix_raw_subset_optimum_run_name,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    CSV_PATH,
    _mean_phase_tv_distance,
    _phase_weights_from_array,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CURVE_POINTS_CSV = SCRIPT_DIR / "two_phase_many_regmix_raw_curve_points.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_regmix_raw_summary.json"
COMPARISON_CSV = SCRIPT_DIR / "two_phase_many_regmix_vs_grp_raw_comparison.csv"
SUMMARY_MD = SCRIPT_DIR / "two_phase_many_regmix_raw.md"
GRP_SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_grp_power_family_penalty_raw_summary.json"
CHECKPOINT_ROOT = "marin-us-east5/checkpoints/" + REGMIX_RAW_SUBSET_OPTIMA_SOURCE_EXPERIMENT

METHOD = "feature_bayes_linear"
KFOLD_SPLITS = 5
KFOLD_SEED = 0
REGMIX_SAMPLE_SEED = 123
REGMIX_SAMPLE_COUNT = 250_000
REGMIX_TOP_K = 128
LGBM_MAX_ITER = 1000
EARLY_STOPPING_ROUNDS = 10
LGBM_PARAMS = {
    "boosting_type": "gbdt",
    "objective": "regression",
    "metric": "l2",
    "learning_rate": 1e-2,
    "verbosity": -1,
    "random_state": KFOLD_SEED,
    "n_jobs": 1,
    "n_estimators": LGBM_MAX_ITER,
}


def _flatten_weights(weights: np.ndarray) -> np.ndarray:
    return np.asarray(weights, dtype=float).reshape(weights.shape[0], -1)


def _normalize_phase_weights(weights: np.ndarray) -> np.ndarray:
    totals = weights.sum(axis=-1, keepdims=True)
    return np.divide(weights, np.maximum(totals, 1e-12), out=np.zeros_like(weights), where=totals > 0.0)


@cache
def _sample_bank(n_domains: int, n_phases: int) -> np.ndarray:
    sampled = sample_configs(REGMIX_SAMPLE_COUNT, n_domains=n_domains, n_phases=n_phases, seed=REGMIX_SAMPLE_SEED)
    bank = sampled.reshape(REGMIX_SAMPLE_COUNT, n_phases, n_domains)
    return _normalize_phase_weights(bank)


def _fit_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
) -> tuple[lgb.LGBMRegressor, int]:
    reg = lgb.LGBMRegressor(**LGBM_PARAMS)
    reg.fit(
        x_train,
        y_train,
        eval_set=[(x_valid, y_valid)],
        eval_metric="l2",
        callbacks=[lgb.early_stopping(stopping_rounds=EARLY_STOPPING_ROUNDS, verbose=False)],
    )
    best_iteration = int(reg.best_iteration_ or LGBM_MAX_ITER)
    return reg, best_iteration


def _fit_full_model(x: np.ndarray, y: np.ndarray, *, n_estimators: int) -> lgb.LGBMRegressor:
    reg = lgb.LGBMRegressor(**{**LGBM_PARAMS, "n_estimators": int(max(1, n_estimators))})
    reg.fit(x, y)
    return reg


def _best_observed_in_subset(packet, subset_indices: np.ndarray) -> tuple[str, float]:
    subset_values = packet.base.y[subset_indices]
    best_local_idx = int(np.argmin(subset_values))
    best_idx = int(subset_indices[best_local_idx])
    return (
        str(packet.base.frame.iloc[best_idx][packet.base.name_col]),
        float(packet.base.y[best_idx]),
    )


def _optimize_regmix_model(
    model: lgb.LGBMRegressor,
    sample_bank: np.ndarray,
    *,
    top_k: int = REGMIX_TOP_K,
) -> tuple[float, np.ndarray]:
    sample_features = _flatten_weights(sample_bank)
    predictions = model.predict(sample_features)
    top_count = min(int(top_k), len(sample_bank))
    top_idx = np.argsort(predictions)[:top_count]
    optimum = sample_bank[top_idx].mean(axis=0)
    optimum = _normalize_phase_weights(optimum)
    optimum_pred = float(model.predict(_flatten_weights(optimum[None, :, :]))[0])
    return optimum_pred, optimum


def _lower_tail_optimism(y: np.ndarray, oof_pred: np.ndarray) -> float:
    tail_count = max(5, int(np.ceil(0.15 * float(len(y)))))
    tail_idx = np.argsort(oof_pred)[:tail_count]
    return float(np.mean(np.maximum(y[tail_idx] - oof_pred[tail_idx], 0.0)))


def _fit_subset_point(subset_size: int) -> dict[str, object]:
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_regmix_raw",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    if subset_size == len(packet.base.y):
        subset_indices = np.arange(len(packet.base.y), dtype=int)
    else:
        selection = retrospective_generic_selection(spec, method=METHOD, k=subset_size, seed=KFOLD_SEED)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)

    weights = packet.base.w[subset_indices]
    y = packet.base.y[subset_indices]
    x = _flatten_weights(weights)
    sample_bank = _sample_bank(packet.base.m, 2)
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=KFOLD_SEED)

    oof = np.zeros_like(y)
    fold_regrets: list[float] = []
    fold_depopt_scores: list[float] = []
    fold_support_scores: list[float] = []
    best_iterations: list[int] = []

    for tr, te in kf.split(x):
        model, best_iteration = _fit_model(x[tr], y[tr], x[te], y[te])
        best_iterations.append(best_iteration)
        pred = model.predict(x[te])
        oof[te] = pred
        chosen = int(np.argmin(pred))
        fold_regrets.append(float(y[te][chosen] - np.min(y[te])))

        raw_pred, raw_weights = _optimize_regmix_model(model, sample_bank)
        distances = average_phase_tv_distance(weights[te], raw_weights[None, :, :])
        nearest_count = min(8, len(te))
        nearest_idx = np.argsort(distances)[:nearest_count]
        fold_depopt_scores.append(max(float(np.min(y[te][nearest_idx])) - raw_pred, 0.0))
        fold_support_scores.append(float(distances[nearest_idx[0]]))

    cv_rmse = float(np.sqrt(np.mean((oof - y) ** 2)))
    cv_regret_at_1 = float(y[int(np.argmin(oof))] - np.min(y))
    cv_foldmean_regret_at_1 = float(np.mean(fold_regrets))
    lower_tail_optimism = _lower_tail_optimism(y, oof)
    cv_depopt_best8 = float(np.mean(fold_depopt_scores))
    cv_rawopt_nearest_tv = float(np.mean(fold_support_scores))
    objective = (
        CALIBRATION_CV_WEIGHT * cv_rmse
        + CALIBRATION_FOLDMEAN_WEIGHT * cv_foldmean_regret_at_1
        + CALIBRATION_TAIL_WEIGHT * lower_tail_optimism
        + CALIBRATION_DEPOPT_WEIGHT * cv_depopt_best8
        + CALIBRATION_SUPPORT_WEIGHT * cv_rawopt_nearest_tv
    )

    full_model = _fit_full_model(x, y, n_estimators=round(float(np.mean(best_iterations))))
    predicted_optimum_value, deployment = _optimize_regmix_model(full_model, sample_bank)
    fullswarm_predictions = full_model.predict(_flatten_weights(packet.base.w))
    chosen_idx = int(np.argmin(fullswarm_predictions))
    distances_full = average_phase_tv_distance(packet.base.w, deployment[None, :, :])
    nearest_idx = int(np.argmin(distances_full))
    subset_best_run_name, subset_best_bpb = _best_observed_in_subset(packet, subset_indices)
    best_full_observed_bpb = float(np.min(packet.base.y))

    return {
        "subset_size": subset_size,
        "predicted_optimum_value": predicted_optimum_value,
        "subset_best_observed_run_name": subset_best_run_name,
        "subset_best_observed_bpb": subset_best_bpb,
        "fullswarm_chosen_run_name": str(packet.base.frame.iloc[chosen_idx][packet.base.name_col]),
        "fullswarm_chosen_value": float(packet.base.y[chosen_idx]),
        "fullswarm_regret_at_1": float(packet.base.y[chosen_idx] - best_full_observed_bpb),
        "nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "nearest_observed_value": float(packet.base.y[nearest_idx]),
        "nearest_observed_tv_distance": float(distances_full[nearest_idx]),
        "tuning_objective": objective,
        "tuning_cv_rmse": cv_rmse,
        "tuning_cv_regret_at_1": cv_regret_at_1,
        "tuning_cv_foldmean_regret_at_1": cv_foldmean_regret_at_1,
        "tuning_lower_tail_optimism": lower_tail_optimism,
        "tuning_cv_depopt_best8": cv_depopt_best8,
        "tuning_cv_rawopt_nearest_tv": cv_rawopt_nearest_tv,
        "phase0_support_below_1e4": int(np.sum(deployment[0] < 1e-4)),
        "phase1_support_below_1e4": int(np.sum(deployment[1] < 1e-4)),
        "phase0_max_weight": float(np.max(deployment[0])),
        "phase1_max_weight": float(np.max(deployment[1])),
        "phase_weights": _phase_weights_from_array(packet.base.domain_names, deployment),
        "actual_validated_bpb": None,
    }


def _rows_with_movements(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    prev_weights: np.ndarray | None = None
    enriched: list[dict[str, object]] = []
    for row in sorted(rows, key=lambda item: int(item["subset_size"])):
        current_weights = np.asarray(
            [
                [row["phase_weights"]["phase_0"][domain_name] for domain_name in packet.base.domain_names],
                [row["phase_weights"]["phase_1"][domain_name] for domain_name in packet.base.domain_names],
            ],
            dtype=float,
        )
        move_tv = None if prev_weights is None else float(_mean_phase_tv_distance(prev_weights, current_weights))
        enriched_row = dict(row)
        enriched_row["optimum_move_mean_phase_tv_vs_prev"] = move_tv
        enriched.append(enriched_row)
        prev_weights = current_weights
    return enriched


def _validated_subset_bpbs() -> dict[int, float]:
    fs = fsspec.filesystem("gs")
    realized: dict[int, float] = {}
    for subset_size in REGMIX_RAW_SUBSET_OPTIMA_SUBSET_SIZES:
        run_name = regmix_raw_subset_optimum_run_name(subset_size)
        matches = sorted(fs.glob(f"{CHECKPOINT_ROOT}/{run_name}-*/checkpoints/eval_metrics.jsonl"))
        if not matches:
            continue
        payload: dict[str, float] | None = None
        with fs.open(matches[-1], "r") as handle:
            for line in handle:
                if line.strip():
                    payload = json.loads(line)
        if payload is None:
            continue
        value = payload.get(OBJECTIVE_METRIC)
        if value is not None:
            realized[subset_size] = float(value)
    return realized


def _apply_realized_validated_bpbs(
    rows: list[dict[str, object]], *, best_observed_bpb: float
) -> list[dict[str, object]]:
    realized = _validated_subset_bpbs()
    enriched: list[dict[str, object]] = []
    for row in rows:
        enriched_row = dict(row)
        actual_validated_bpb = realized.get(int(row["subset_size"]))
        enriched_row["actual_validated_bpb"] = actual_validated_bpb
        if actual_validated_bpb is None:
            enriched_row["validated_prediction_error"] = None
            enriched_row["validated_regret_at_1"] = None
        else:
            enriched_row["validated_prediction_error"] = float(actual_validated_bpb) - float(
                row["predicted_optimum_value"]
            )
            enriched_row["validated_regret_at_1"] = float(actual_validated_bpb) - best_observed_bpb
        enriched.append(enriched_row)
    return enriched


def _grp_rows_by_subset_size() -> dict[int, dict[str, object]]:
    payload = json.loads(GRP_SUMMARY_JSON.read_text())
    return {int(row["subset_size"]): row for row in payload["rows"]}


def _load_existing_rows() -> list[dict[str, object]]:
    if SUMMARY_JSON.exists():
        payload = json.loads(SUMMARY_JSON.read_text())
        rows = payload.get("rows")
        if isinstance(rows, list):
            return rows
    if CURVE_POINTS_CSV.exists():
        frame = pd.read_csv(CURVE_POINTS_CSV)
        if "phase_weights" in frame.columns:
            frame["phase_weights"] = frame["phase_weights"].map(
                lambda payload: ast.literal_eval(payload) if isinstance(payload, str) and payload else payload
            )
        return frame.to_dict(orient="records")
    raise FileNotFoundError(f"Missing existing RegMix outputs at {SUMMARY_JSON} or {CURVE_POINTS_CSV}")


def _write_outputs(rows: list[dict[str, object]]) -> None:
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    rows = _rows_with_movements(rows)
    frame = pd.DataFrame(rows).sort_values("subset_size").reset_index(drop=True)
    curve_for_csv = frame.copy()
    curve_for_csv["phase_weights"] = curve_for_csv["phase_weights"].map(json.dumps)
    curve_for_csv.to_csv(CURVE_POINTS_CSV, index=False)

    grp_rows = _grp_rows_by_subset_size()
    comparison_rows: list[dict[str, object]] = []
    for row in rows:
        grp_row = grp_rows.get(int(row["subset_size"]))
        comparison_rows.append(
            {
                "subset_size": int(row["subset_size"]),
                "regmix_predicted_optimum_value": float(row["predicted_optimum_value"]),
                "grp_predicted_optimum_value": None if grp_row is None else float(grp_row["predicted_optimum_value"]),
                "regmix_nearest_observed_tv_distance": float(row["nearest_observed_tv_distance"]),
                "grp_nearest_observed_tv_distance": (
                    None if grp_row is None else float(grp_row["nearest_observed_tv_distance"])
                ),
                "regmix_tuning_cv_rmse": float(row["tuning_cv_rmse"]),
                "grp_tuning_cv_rmse": None if grp_row is None else float(grp_row["tuning_cv_rmse"]),
                "regmix_tuning_cv_foldmean_regret_at_1": float(row["tuning_cv_foldmean_regret_at_1"]),
                "grp_tuning_cv_foldmean_regret_at_1": (
                    None if grp_row is None else float(grp_row["tuning_cv_foldmean_regret_at_1"])
                ),
                "regmix_tuning_lower_tail_optimism": float(row["tuning_lower_tail_optimism"]),
                "grp_tuning_lower_tail_optimism": (
                    None if grp_row is None else float(grp_row["tuning_lower_tail_optimism"])
                ),
                "regmix_optimum_move_mean_phase_tv_vs_prev": row["optimum_move_mean_phase_tv_vs_prev"],
                "grp_optimum_move_mean_phase_tv_vs_prev": (
                    None if grp_row is None else grp_row["optimum_move_mean_phase_tv_vs_prev"]
                ),
            }
        )
    pd.DataFrame(comparison_rows).to_csv(COMPARISON_CSV, index=False)

    summary_payload = {
        "objective_metric": OBJECTIVE_METRIC,
        "variant": "regmix",
        "curve_points_csv": str(CURVE_POINTS_CSV),
        "comparison_csv": str(COMPARISON_CSV),
        "sample_count": REGMIX_SAMPLE_COUNT,
        "top_k": REGMIX_TOP_K,
        "best_observed_bpb": float(np.min(packet.base.y)),
        "rows": rows,
    }
    SUMMARY_JSON.write_text(json.dumps(summary_payload, indent=2))

    best_row = min(rows, key=lambda row: float(row["predicted_optimum_value"]))
    lines = [
        "# Two-phase Many-Domain RegMix Local Optima",
        "",
        f"- Objective metric: `{OBJECTIVE_METRIC}`",
        f"- Sample bank: `{REGMIX_SAMPLE_COUNT}` mixed samples, top-`{REGMIX_TOP_K}` averaging",
        f"- Full packet size: `{len(packet.base.y)}`",
        "",
        "## Best Local Optimum",
        "",
        f"- Subset size: `{int(best_row['subset_size'])}`",
        f"- Predicted optimum BPB: `{float(best_row['predicted_optimum_value']):.6f}`",
        f"- Nearest observed TV: `{float(best_row['nearest_observed_tv_distance']):.6f}`",
        f"- CV RMSE: `{float(best_row['tuning_cv_rmse']):.6f}`",
        f"- CV Mean Regret@1: `{float(best_row['tuning_cv_foldmean_regret_at_1']):.6f}`",
        f"- Tail optimism: `{float(best_row['tuning_lower_tail_optimism']):.6f}`",
        "",
    ]
    SUMMARY_MD.write_text("\n".join(lines))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-validated-only",
        action="store_true",
        help="Reuse existing local RegMix subset fits and only refresh realized validated BPBs from checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    best_observed_bpb = float(np.min(packet.base.y))
    if args.refresh_validated_only:
        rows = _load_existing_rows()
    else:
        subset_sizes = (
            *GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
            len(packet.base.y),
        )
        rows = []
        for subset_size in subset_sizes:
            row = _fit_subset_point(int(subset_size))
            rows.append(row)
            print(
                f"Finished subset_size={int(row['subset_size'])} "
                f"predicted={float(row['predicted_optimum_value']):.6f} "
                f"nearest_tv={float(row['nearest_observed_tv_distance']):.6f}"
            )
    rows = _apply_realized_validated_bpbs(rows, best_observed_bpb=best_observed_bpb)
    _write_outputs(rows)
    print(f"Wrote {CURVE_POINTS_CSV}")
    print(f"Wrote {SUMMARY_JSON}")
    print(f"Wrote {COMPARISON_CSV}")


if __name__ == "__main__":
    main()
