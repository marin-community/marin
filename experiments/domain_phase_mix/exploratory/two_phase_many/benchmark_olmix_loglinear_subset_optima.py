# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scikit-learn", "scipy"]
# ///
"""Fit subset-size Olmix loglinear optima on the many-domain two-phase packet."""

from __future__ import annotations

import argparse
from functools import cache
import json
from pathlib import Path

import fsspec
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

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
    build_dataset_spec_from_frame,
    retrospective_generic_selection,
)
from experiments.domain_phase_mix.two_phase_dolma3_dolmino_top_level import (
    create_two_phase_dolma3_dolmino_top_level_experiment,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
    OBJECTIVE_METRIC,
)
from experiments.domain_phase_mix.two_phase_many_genericfamily_retuned_subset_optima import (
    CSV_PATH,
    _mean_phase_tv_distance,
    _phase_weights_from_array,
)
from experiments.domain_phase_mix.two_phase_many_olmix_loglinear_sl_verb import (
    fit_olmix_loglinear_model,
    solve_olmix_loglinear_schedule,
)

SCRIPT_DIR = Path(__file__).resolve().parent
CURVE_POINTS_CSV = SCRIPT_DIR / "two_phase_many_olmix_loglinear_subset_curve_points.csv"
SUMMARY_JSON = SCRIPT_DIR / "two_phase_many_olmix_loglinear_subset_summary.json"
SUMMARY_MD = SCRIPT_DIR / "two_phase_many_olmix_loglinear_subset.md"

KFOLD_SPLITS = 5
KFOLD_SEED = 0
SOURCE_EXPERIMENT = "pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_loglinear_subset_optima_uncheatable_bpb"
RUN_NAME_PREFIX = "baseline_olmix_loglinear_optimum"
CHECKPOINT_ROOT = "marin-us-east5/checkpoints/" + SOURCE_EXPERIMENT


@cache
def _olmix_priors() -> tuple[np.ndarray, np.ndarray]:
    experiment = create_two_phase_dolma3_dolmino_top_level_experiment(name="two_phase_many_olmix_loglinear_subset")
    natural_proportions = np.asarray([float(domain.total_weight) for domain in experiment.domains], dtype=float)
    natural_proportions = natural_proportions / natural_proportions.sum()
    phase_fractions = np.asarray(
        [phase.end_fraction - phase.start_fraction for phase in experiment.phase_schedule.phases],
        dtype=float,
    )
    return natural_proportions, phase_fractions


def olmix_loglinear_subset_optimum_run_name(subset_size: int) -> str:
    """Return the canonical run name for one Olmix subset-fit optimum."""
    return f"{RUN_NAME_PREFIX}_k{subset_size:03d}_uncheatable_bpb"


def _weights_from_phase_weights(
    phase_weights: dict[str, dict[str, float]],
    domain_names: list[str],
) -> np.ndarray:
    return np.asarray(
        [
            [float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names],
            [float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names],
        ],
        dtype=float,
    )


def _best_observed_in_subset(packet, subset_indices: np.ndarray) -> tuple[str, float]:
    subset_values = packet.base.y[subset_indices]
    best_local_idx = int(np.argmin(subset_values))
    best_idx = int(subset_indices[best_local_idx])
    return (
        str(packet.base.frame.iloc[best_idx][packet.base.name_col]),
        float(packet.base.y[best_idx]),
    )


def _lower_tail_optimism(y: np.ndarray, oof_pred: np.ndarray) -> float:
    tail_count = max(5, int(np.ceil(0.15 * float(len(y)))))
    tail_idx = np.argsort(oof_pred)[:tail_count]
    return float(np.mean(np.maximum(y[tail_idx] - oof_pred[tail_idx], 0.0)))


def _solve_fit_optimum(fit, *, phase_names: list[str], domain_names: list[str]) -> tuple[np.ndarray, float, float]:
    natural_proportions, phase_fractions = _olmix_priors()
    phase_weights, predicted_objective, regularized_objective = solve_olmix_loglinear_schedule(
        fit,
        natural_proportions=natural_proportions,
        phase_fractions=phase_fractions,
        phase_names=phase_names,
        domain_names=domain_names,
    )
    weights = _weights_from_phase_weights(phase_weights, domain_names)
    return weights, float(predicted_objective), float(regularized_objective)


def _validated_subset_bpbs() -> dict[int, float]:
    fs = fsspec.filesystem("gs")
    realized: dict[int, float] = {}
    for subset_size in (*GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES, 242):
        run_name = olmix_loglinear_subset_optimum_run_name(int(subset_size))
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
            realized[int(subset_size)] = float(value)
    return realized


def _apply_realized_validated_bpbs(
    rows: list[dict[str, object]],
    *,
    best_observed_bpb: float,
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


def _fit_subset_point(subset_size: int) -> dict[str, object]:
    _, spec, _ = load_two_phase_many_candidate_summary_spec(
        CSV_PATH,
        objective_metric=OBJECTIVE_METRIC,
        name="two_phase_many_olmix_loglinear_subset",
    )
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    if subset_size == len(packet.base.y):
        subset_indices = np.arange(len(packet.base.y), dtype=int)
    else:
        selection = retrospective_generic_selection(spec, method="feature_bayes_linear", k=subset_size, seed=KFOLD_SEED)
        subset_indices = np.asarray(selection.selected_indices, dtype=int)

    train_frame = packet.base.frame.iloc[subset_indices].reset_index(drop=True)
    train_spec = build_dataset_spec_from_frame(
        train_frame,
        objective_metric=OBJECTIVE_METRIC,
        name=f"two_phase_many_olmix_loglinear_subset_k{subset_size}",
    )
    weights = np.asarray(train_spec.weights, dtype=float)
    y = np.asarray(train_spec.y, dtype=float)
    kf = KFold(n_splits=KFOLD_SPLITS, shuffle=True, random_state=KFOLD_SEED)

    oof = np.zeros_like(y)
    fold_regrets: list[float] = []
    fold_depopt_scores: list[float] = []
    fold_support_scores: list[float] = []

    for fold_idx, (tr, te) in enumerate(kf.split(weights)):
        fit = fit_olmix_loglinear_model(weights[tr], y[tr], seed=fold_idx)
        pred = np.asarray(fit.predict(weights[te]), dtype=float)
        oof[te] = pred
        chosen = int(np.argmin(pred))
        fold_regrets.append(float(y[te][chosen] - np.min(y[te])))

        fold_weights, raw_pred, _ = _solve_fit_optimum(
            fit,
            phase_names=train_spec.phase_names,
            domain_names=train_spec.domain_names,
        )
        distances = average_phase_tv_distance(weights[te], fold_weights[None, :, :])
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

    full_fit = fit_olmix_loglinear_model(weights, y, seed=KFOLD_SEED)
    deployment, predicted_optimum_value, regularized_objective = _solve_fit_optimum(
        full_fit,
        phase_names=train_spec.phase_names,
        domain_names=train_spec.domain_names,
    )
    fullswarm_predictions = np.asarray(full_fit.predict(packet.base.w), dtype=float)
    chosen_idx = int(np.argmin(fullswarm_predictions))
    distances_full = average_phase_tv_distance(packet.base.w, deployment[None, :, :])
    nearest_idx = int(np.argmin(distances_full))
    subset_best_run_name, subset_best_bpb = _best_observed_in_subset(packet, subset_indices)
    best_full_observed_bpb = float(np.min(packet.base.y))

    return {
        "subset_size": subset_size,
        "predicted_optimum_value": predicted_optimum_value,
        "regularized_objective": regularized_objective,
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


def _load_existing_rows() -> list[dict[str, object]]:
    if SUMMARY_JSON.exists():
        payload = json.loads(SUMMARY_JSON.read_text())
        rows = payload.get("rows")
        if isinstance(rows, list):
            return rows
    frame = pd.read_csv(CURVE_POINTS_CSV)
    if "phase_weights" in frame.columns:
        frame["phase_weights"] = frame["phase_weights"].map(
            lambda payload: json.loads(payload) if isinstance(payload, str) and payload else payload
        )
    return frame.to_dict(orient="records")


def _write_outputs(rows: list[dict[str, object]]) -> None:
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    rows = _rows_with_movements(rows)
    best_observed_bpb = float(np.min(packet.base.y))
    rows = _apply_realized_validated_bpbs(rows, best_observed_bpb=best_observed_bpb)
    frame = pd.DataFrame(rows).sort_values("subset_size").reset_index(drop=True)
    curve_for_csv = frame.copy()
    curve_for_csv["phase_weights"] = curve_for_csv["phase_weights"].map(json.dumps)
    curve_for_csv.to_csv(CURVE_POINTS_CSV, index=False)
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "objective_metric": OBJECTIVE_METRIC,
                "variant": "olmix_loglinear_subset",
                "curve_points_csv": str(CURVE_POINTS_CSV),
                "best_observed_bpb": best_observed_bpb,
                "rows": frame.replace({np.nan: None}).to_dict(orient="records"),
            },
            indent=2,
        )
        + "\n"
    )

    best_row = min(rows, key=lambda row: float(row["predicted_optimum_value"]))
    lines = [
        "# Two-phase Many-Domain Olmix Loglinear Subset Optima",
        "",
        f"- Objective metric: `{OBJECTIVE_METRIC}`",
        f"- Full packet size: `{len(packet.base.y)}`",
        "",
        "## Best Local Optimum",
        "",
        f"- Subset size: `{int(best_row['subset_size'])}`",
        f"- Predicted optimum BPB: `{float(best_row['predicted_optimum_value']):.6f}`",
        f"- Regularized objective: `{float(best_row['regularized_objective']):.6f}`",
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
        help="Reuse existing subset-fit curve points and only refresh realized validated BPBs from checkpoints.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    if args.refresh_validated_only:
        rows = _load_existing_rows()
    else:
        subset_sizes = (
            *GENERICFAMILY_OBSERVED_ONLY_TRUSTBLEND_SUBSET_OPTIMA_REPRESENTATIVE_SUBSET_SIZES,
            len(packet.base.y),
        )
        rows: list[dict[str, object]] = []
        for subset_size in subset_sizes:
            row = _fit_subset_point(int(subset_size))
            rows.append(row)
            print(
                f"Finished subset_size={int(row['subset_size'])} "
                f"predicted={float(row['predicted_optimum_value']):.6f} "
                f"nearest_tv={float(row['nearest_observed_tv_distance']):.6f}",
                flush=True,
            )
    _write_outputs(rows)
    print(f"Wrote {CURVE_POINTS_CSV}")
    print(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
