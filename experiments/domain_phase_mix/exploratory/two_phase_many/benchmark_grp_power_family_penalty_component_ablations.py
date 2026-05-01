# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas"]
# ///
"""Evaluate local ablations of GRP (Power-Family Penalty)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    CALIBRATION_CV_WEIGHT,
    CALIBRATION_DEPOPT_WEIGHT,
    CALIBRATION_FOLDMEAN_WEIGHT,
    CALIBRATION_SUPPORT_WEIGHT,
    CALIBRATION_TAIL_WEIGHT,
    LOWER_TAIL_FRAC,
    TRUSTBLEND_TOPK_ACTUAL,
    build_penalty_calibration_surrogate,
    optimize_penalty_calibration_model,
    penalty_calibration_params_from_metrics,
    penalty_calibration_variant_parameter_counts,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.build_grp_evaluation_table import (
    PENALTY_CALIBRATION_BEST_CSV,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
    regression_metrics,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    load_generic_family_packet,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance

SCRIPT_DIR = Path(__file__).resolve().parent
OUTPUT_CSV = SCRIPT_DIR / "grp_power_family_penalty_component_ablations.csv"
OUTPUT_MD = SCRIPT_DIR / "grp_power_family_penalty_component_ablations.md"
OUTPUT_JSON = SCRIPT_DIR / "grp_power_family_penalty_component_ablations_summary.json"
VARIANT_NAME = "power_family_penalty"
CV_SEED = 0


@dataclass(frozen=True)
class AblationSpec:
    """One structural ablation of GRP (Power-Family Penalty)."""

    name: str
    label: str
    include_singletons: bool = True
    include_pairs: bool = True
    include_family_totals: bool = True
    include_global_group_penalty: bool = True
    include_family_group_penalty: bool = True
    include_family_total_penalty: bool = True
    reg_override: float | None = None


ABLATIONS = (
    AblationSpec(name="power_family_penalty", label="GRP (Power-Family Penalty)"),
    AblationSpec(
        name="power_family_penalty_groups_only",
        label="GRP w/ source-type totals only",
        include_singletons=False,
        include_pairs=False,
        include_family_totals=True,
        include_family_group_penalty=False,
    ),
    AblationSpec(
        name="power_family_penalty_no_l2",
        label="GRP w/o L2 regularization",
        reg_override=0.0,
    ),
)


def _best_params() -> dict[str, float]:
    frame = pd.read_csv(PENALTY_CALIBRATION_BEST_CSV)
    matches = frame.loc[(frame["variant"] == VARIANT_NAME) & (frame["stage"] == "refine")]
    if matches.empty:
        raise ValueError(f"Missing refined row for {VARIANT_NAME!r} in {PENALTY_CALIBRATION_BEST_CSV}")
    return penalty_calibration_params_from_metrics(dict(matches.iloc[0]), VARIANT_NAME)


def _ablation_params(ablation: AblationSpec, base_params: dict[str, float]) -> dict[str, float]:
    params = dict(base_params)
    if ablation.reg_override is not None:
        params["reg"] = float(ablation.reg_override)
    return params


def _ablation_param_count(packet, ablation: AblationSpec) -> int:
    counts = penalty_calibration_variant_parameter_counts(
        packet,
        VARIANT_NAME,
        include_singletons=ablation.include_singletons,
        include_pairs=ablation.include_pairs,
        include_family_totals=ablation.include_family_totals,
        include_global_group_penalty=ablation.include_global_group_penalty,
        include_family_group_penalty=ablation.include_family_group_penalty,
        include_family_total_penalty=ablation.include_family_total_penalty,
    )
    nonlinear = counts["nonlinear_param_count"] - int(ablation.reg_override is not None)
    return int(counts["linear_head_param_count"] + nonlinear)


def _build_model(packet, ablation: AblationSpec, params: dict[str, float]):
    return build_penalty_calibration_surrogate(
        packet,
        params=params,
        variant_name=VARIANT_NAME,
        include_singletons=ablation.include_singletons,
        include_pairs=ablation.include_pairs,
        include_family_totals=ablation.include_family_totals,
        include_global_group_penalty=ablation.include_global_group_penalty,
        include_family_group_penalty=ablation.include_family_group_penalty,
        include_family_total_penalty=ablation.include_family_total_penalty,
    )


def _oof_metrics(packet, ablation: AblationSpec, params: dict[str, float], *, seed: int) -> dict[str, float]:
    y = packet.base.y
    weights = packet.base.w
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros_like(y)
    fold_regrets: list[float] = []
    depopt_scores: list[float] = []
    rawopt_support_scores: list[float] = []

    for tr, te in kf.split(weights):
        model = _build_model(packet, ablation, params).fit(weights[tr], y[tr])
        pred = model.predict(weights[te])
        oof[te] = pred
        fold_regrets.append(float(y[te][int(np.argmin(pred))] - np.min(y[te])))

        raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, n_random=1, seed=seed)
        raw_weights = np.stack([phase0, phase1], axis=0)
        distances = average_phase_tv_distance(weights[te], raw_weights[None, :, :])
        nearest_count = min(TRUSTBLEND_TOPK_ACTUAL, len(te))
        nearest_idx = np.argsort(distances)[:nearest_count]
        depopt_scores.append(max(float(np.min(y[te][nearest_idx])) - float(raw_result.fun), 0.0))
        rawopt_support_scores.append(float(distances[nearest_idx[0]]))

    residuals = oof - y
    cv_rmse = float(np.sqrt(np.mean(residuals**2)))
    tail_count = max(5, int(np.ceil(float(LOWER_TAIL_FRAC) * float(len(y)))))
    tail_idx = np.argsort(oof)[:tail_count]
    lower_tail_optimism = float(np.mean(np.maximum(y[tail_idx] - oof[tail_idx], 0.0)))
    mean_regret = float(np.mean(fold_regrets))
    mean_depopt = float(np.mean(depopt_scores))
    mean_support = float(np.mean(rawopt_support_scores))
    objective = (
        CALIBRATION_CV_WEIGHT * cv_rmse
        + CALIBRATION_FOLDMEAN_WEIGHT * mean_regret
        + CALIBRATION_TAIL_WEIGHT * lower_tail_optimism
        + CALIBRATION_DEPOPT_WEIGHT * mean_depopt
        + CALIBRATION_SUPPORT_WEIGHT * mean_support
    )
    cv = regression_metrics(packet.base.frame, packet.base.name_col, y, oof)
    return {
        "cv_r2": float(cv["r2"]),
        "cv_rmse": cv_rmse,
        "cv_spearman": float(cv["spearman"]),
        "cv_regret_at_1": float(cv["regret_at_1"]),
        "cv_foldmean_regret_at_1": mean_regret,
        "lower_tail_optimism": lower_tail_optimism,
        "cv_depopt_best8": mean_depopt,
        "cv_rawopt_nearest_tv": mean_support,
        "objective": objective,
    }


def _full_fit_metrics(packet, ablation: AblationSpec, params: dict[str, float]) -> dict[str, float]:
    model = _build_model(packet, ablation, params).fit(packet.base.w, packet.base.y)
    train_pred = model.predict(packet.base.w)
    train = regression_metrics(packet.base.frame, packet.base.name_col, packet.base.y, train_pred)
    raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, seed=CV_SEED)
    raw_weights = np.stack([phase0, phase1], axis=0)
    raw_distances = average_phase_tv_distance(packet.base.w, raw_weights[None, :, :])
    nearest_idx = int(np.argmin(raw_distances))
    return {
        "train_r2": float(train["r2"]),
        "train_rmse": float(train["rmse"]),
        "train_spearman": float(train["spearman"]),
        "train_regret_at_1": float(train["regret_at_1"]),
        "raw_predicted_optimum_value": float(raw_result.fun),
        "raw_nearest_observed_tv": float(raw_distances[nearest_idx]),
        "raw_nearest_observed_run_name": str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        "raw_nearest_observed_value": float(packet.base.y[nearest_idx]),
        "raw_phase0_lt_1e4": int(np.sum(phase0 < 1e-4)),
        "raw_phase1_lt_1e4": int(np.sum(phase1 < 1e-4)),
    }


def _rows() -> list[dict[str, object]]:
    packet = load_generic_family_packet(target=MANY_DOMAIN_TARGET)
    base_params = _best_params()
    rows: list[dict[str, object]] = []
    for ablation in ABLATIONS:
        params = _ablation_params(ablation, base_params)
        row = {
            "variant": ablation.name,
            "label": ablation.label,
            "total_param_count": _ablation_param_count(packet, ablation),
            "include_singletons": ablation.include_singletons,
            "include_pairs": ablation.include_pairs,
            "include_family_totals": ablation.include_family_totals,
            "include_global_group_penalty": ablation.include_global_group_penalty,
            "include_family_group_penalty": ablation.include_family_group_penalty,
            "include_family_total_penalty": ablation.include_family_total_penalty,
            "surrogate_variant": VARIANT_NAME,
            "reg": float(params["reg"]),
            "retuned": False,
            "objective_metric": MANY_DOMAIN_TARGET,
            "notes": (
                "Component ablation of the fixed full-retune power_family_penalty nonlinear parameters; "
                "no additional nonlinear retuning. The source-type-totals-only row necessarily drops the "
                "group penalty because singleton/pair groups are removed."
            ),
        }
        row.update(_full_fit_metrics(packet, ablation, params))
        row.update(_oof_metrics(packet, ablation, params, seed=CV_SEED))
        rows.append(row)
    return rows


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = [
        "label",
        "total_param_count",
        "cv_r2",
        "cv_rmse",
        "cv_spearman",
        "cv_foldmean_regret_at_1",
        "lower_tail_optimism",
        "cv_depopt_best8",
        "cv_rawopt_nearest_tv",
        "raw_predicted_optimum_value",
    ]
    display = frame[columns].copy()
    display.columns = [
        "variant",
        "params",
        "cv_r2",
        "cv_rmse",
        "cv_spearman",
        "fold_regret1",
        "tail_opt",
        "cv_depopt8",
        "cv_raw_tv",
        "raw_opt_bpb",
    ]
    return display.to_markdown(index=False, floatfmt=".4f") + "\n"


def main() -> None:
    frame = pd.DataFrame(_rows()).sort_values("objective").reset_index(drop=True)
    frame.to_csv(OUTPUT_CSV, index=False)
    OUTPUT_MD.write_text(_markdown_table(frame), encoding="utf-8")
    OUTPUT_JSON.write_text(
        json.dumps(
            {
                "variant_name": VARIANT_NAME,
                "objective_metric": MANY_DOMAIN_TARGET,
                "cv_seed": CV_SEED,
                "rows": frame.to_dict(orient="records"),
                "csv": str(OUTPUT_CSV),
                "markdown": str(OUTPUT_MD),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_CSV}")
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
