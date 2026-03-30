# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Anchor-aware tuning study for CCGlobalPremium-RetainedTotal."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    PENALTY_KIND_GROUP_LOG_THRESHOLD,
    PREMIUM_MODE_GLOBAL,
    SIGNAL_KIND_RETAINED_TOTAL,
    CCPairStructuredSurrogate,
    load_two_phase_many_packet,
    regression_metrics,
)
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    ccglobalpremium_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    ccpairtotal_retainedtotal_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_CSV = SCRIPT_DIR / "ccglobalpremium_anchor_tuning_results.csv"
SUMMARY_JSON = SCRIPT_DIR / "ccglobalpremium_anchor_tuning_summary.json"

VALIDATED_GLOBAL_BPB = 1.0438120365142822
VALIDATED_PAIR_BPB = 1.047936201095581

CURRENT_PARAMS = {
    "signal_kind": SIGNAL_KIND_RETAINED_TOTAL,
    "alpha": 8.0,
    "eta": 3.0,
    "lam": 1.0,
    "premium_mode": PREMIUM_MODE_GLOBAL,
    "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
    "tau": 1.0,
    "reg": 0.01,
}
GENERIC_TUNED_REFERENCE = {
    "cv_rmse": 0.010701309236816867,
    "anchor_mae": 0.0007870597827117631,
    "pred_validated_global": 1.0439171480723388,
    "pred_validated_pair": 1.0464730285069155,
}


def _summary_weights(summary: dict[str, object], domain_names: list[str]) -> np.ndarray:
    phase_weights = summary["phase_weights"]
    phase0 = np.asarray([float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names], dtype=float)
    phase1 = np.asarray([float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names], dtype=float)
    return np.stack([phase0, phase1], axis=0)


def _evaluate_params(
    data,
    params: dict[str, float | str],
    valid_weights: np.ndarray,
    valid_y: np.ndarray,
) -> dict[str, float | str | bool]:
    model = CCPairStructuredSurrogate(data, params).fit(data.w, data.y)
    train_pred = model.predict(data.w)
    cv_pred = model.cv_predict(data.w, data.y, seed=0, n_splits=5)
    valid_pred = model.predict(valid_weights)
    err = valid_pred - valid_y

    train = regression_metrics(data.frame, data.name_col, data.y, train_pred)
    cv = regression_metrics(data.frame, data.name_col, data.y, cv_pred)
    return {
        "alpha": float(params["alpha"]),
        "eta": float(params["eta"]),
        "lam": float(params["lam"]),
        "tau": float(params["tau"]),
        "reg": float(params["reg"]),
        "train_rmse": float(train["rmse"]),
        "train_r2": float(train["r2"]),
        "train_spearman": float(train["spearman"]),
        "cv_rmse": float(cv["rmse"]),
        "cv_r2": float(cv["r2"]),
        "cv_spearman": float(cv["spearman"]),
        "cv_regret_at_1": float(cv["regret_at_1"]),
        "anchor_mae": float(np.mean(np.abs(err))),
        "anchor_rmse": float(np.sqrt(np.mean(err**2))),
        "anchor_rank_correct": bool(int(np.argmin(valid_pred)) == int(np.argmin(valid_y))),
        "pred_validated_global": float(valid_pred[0]),
        "pred_validated_pair": float(valid_pred[1]),
        "err_validated_global": float(err[0]),
        "err_validated_pair": float(err[1]),
        "joint_score": float(cv["rmse"] + np.mean(np.abs(err))),
    }


def _sample_params(rng: np.random.Generator) -> dict[str, float | str]:
    return {
        "signal_kind": SIGNAL_KIND_RETAINED_TOTAL,
        "premium_mode": PREMIUM_MODE_GLOBAL,
        "pen_kind": PENALTY_KIND_GROUP_LOG_THRESHOLD,
        "alpha": float(np.exp(rng.uniform(np.log(1.0), np.log(24.0)))),
        "eta": float(np.exp(rng.uniform(np.log(1.0), np.log(24.0)))),
        "lam": float(np.exp(rng.uniform(np.log(1e-3), np.log(2.0)))),
        "tau": float(rng.uniform(0.4, 4.0)),
        "reg": float(np.exp(rng.uniform(np.log(1e-8), np.log(5e-2)))),
    }


def main() -> None:
    data = load_two_phase_many_packet()
    valid_weights = np.stack(
        [
            _summary_weights(ccglobalpremium_retainedtotal_summary(), data.domain_names),
            _summary_weights(ccpairtotal_retainedtotal_summary(), data.domain_names),
        ],
        axis=0,
    )
    valid_y = np.asarray([VALIDATED_GLOBAL_BPB, VALIDATED_PAIR_BPB], dtype=float)

    rng = np.random.default_rng(0)
    rows = [
        {"label": "current", **_evaluate_params(data, CURRENT_PARAMS, valid_weights, valid_y)},
    ]
    for idx in range(192):
        rows.append(
            {
                "label": f"sample_{idx:03d}",
                **_evaluate_params(data, _sample_params(rng), valid_weights, valid_y),
            }
        )

    results = (
        pd.DataFrame(rows).sort_values(["anchor_mae", "cv_rmse", "joint_score"], ascending=True).reset_index(drop=True)
    )
    RESULTS_CSV.write_text(results.to_csv(index=False))

    current = results[results["label"] == "current"].iloc[0].to_dict()
    best_anchor = results.iloc[0].to_dict()
    feasible = results[results["cv_rmse"] <= GENERIC_TUNED_REFERENCE["cv_rmse"] + 5e-4]
    best_balanced = feasible.sort_values(["anchor_mae", "joint_score"], ascending=True).iloc[0].to_dict()
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "current": current,
                "best_anchor": best_anchor,
                "best_balanced_with_generic_like_cv": best_balanced,
                "generic_tuned_reference": GENERIC_TUNED_REFERENCE,
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Results: {RESULTS_CSV}")
    print(f"Summary: {SUMMARY_JSON}")
    print(json.dumps(json.loads(SUMMARY_JSON.read_text()), indent=2))


if __name__ == "__main__":
    main()
