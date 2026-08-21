# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Tune the many-domain GRP/GenericFamily follow-up with a reproducible SciPy search."""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import spearmanr
from sklearn.model_selection import KFold

from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    GENERIC_FAMILY_NAMES,
    TUNED_GENERIC_FAMILY_PARAMS,
    GenericFamilyRetainedTotalSurrogate,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.two_phase_many_ccglobalpremium_baselines import (
    ccglobalpremium_retainedtotal_summary,
)
from experiments.domain_phase_mix.two_phase_many_ccpairtotal_baseline import (
    ccpairtotal_retainedtotal_summary,
)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_JSON = SCRIPT_DIR / "genericfamily_tuning_results.json"
RESULTS_CSV = SCRIPT_DIR / "genericfamily_tuning_trials.csv"

VALIDATED_GLOBAL_BPB = 1.04381
VALIDATED_PAIR_BPB = 1.04794

CV_WEIGHT = 1.0
ANCHOR_WEIGHT = 1.0
REGRET_WEIGHT = 0.02


def _summary_weights(summary: dict[str, object], domain_names: list[str]) -> np.ndarray:
    phase_weights = summary["phase_weights"]
    phase0 = np.asarray([float(phase_weights["phase_0"][domain_name]) for domain_name in domain_names], dtype=float)
    phase1 = np.asarray([float(phase_weights["phase_1"][domain_name]) for domain_name in domain_names], dtype=float)
    return np.stack([phase0, phase1], axis=0)


def _pack(params: dict[str, float]) -> np.ndarray:
    beta = float(np.clip(params["beta"], 1e-8, 1.0))
    return np.asarray(
        [
            np.log(float(params["alpha"])),
            np.log(float(params["eta"])),
            np.log(float(params["lam"])),
            float(params["tau"]),
            np.log(float(params["reg"])),
            np.log(beta / (1.0 - beta)),
        ],
        dtype=float,
    )


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0))))


def _unpack(z: np.ndarray) -> dict[str, float]:
    return {
        "alpha": float(np.exp(np.clip(z[0], -8.0, 8.0))),
        "eta": float(np.exp(np.clip(z[1], -8.0, 8.0))),
        "lam": float(np.exp(np.clip(z[2], -12.0, 4.0))),
        "tau": float(np.clip(z[3], -2.0, 8.0)),
        "reg": float(np.exp(np.clip(z[4], -18.0, -2.0))),
        "beta": float(np.clip(_sigmoid(float(z[5])), 1e-6, 1.0 - 1e-6)),
    }


def _evaluate_params(
    z: np.ndarray,
    packet,
    valid_weights: np.ndarray,
    valid_y: np.ndarray,
    *,
    seed: int = 0,
) -> dict[str, float | bool]:
    params = _unpack(z)
    kf = KFold(n_splits=5, shuffle=True, random_state=seed)
    oof = np.zeros_like(packet.base.y)
    fold_regrets: list[float] = []

    for _fold, (tr, te) in enumerate(kf.split(packet.base.w)):
        model = GenericFamilyRetainedTotalSurrogate(
            packet,
            params=params,
            family_totals=GENERIC_FAMILY_NAMES,
            quality_discount=True,
        ).fit(packet.base.w[tr], packet.base.y[tr])
        pred = model.predict(packet.base.w[te])
        oof[te] = pred
        fold_regrets.append(float(packet.base.y[te][int(np.argmin(pred))] - np.min(packet.base.y[te])))

    full_model = GenericFamilyRetainedTotalSurrogate(
        packet,
        params=params,
        family_totals=GENERIC_FAMILY_NAMES,
        quality_discount=True,
    ).fit(packet.base.w, packet.base.y)
    train_pred = full_model.predict(packet.base.w)
    anchor_pred = full_model.predict(valid_weights)
    anchor_err = anchor_pred - valid_y

    train_res = train_pred - packet.base.y
    cv_res = oof - packet.base.y
    sst = float(np.sum((packet.base.y - np.mean(packet.base.y)) ** 2))

    train_rmse = float(np.sqrt(np.mean(train_res**2)))
    cv_rmse = float(np.sqrt(np.mean(cv_res**2)))
    anchor_mae = float(np.mean(np.abs(anchor_err)))
    foldmean_regret = float(np.mean(fold_regrets))
    objective = CV_WEIGHT * cv_rmse + ANCHOR_WEIGHT * anchor_mae + REGRET_WEIGHT * foldmean_regret

    return {
        **params,
        "objective": objective,
        "train_rmse": train_rmse,
        "train_r2": float(1.0 - float(np.sum(train_res**2)) / sst),
        "train_spearman": float(spearmanr(packet.base.y, train_pred).statistic),
        "cv_rmse": cv_rmse,
        "cv_r2": float(1.0 - float(np.sum(cv_res**2)) / sst),
        "cv_spearman": float(spearmanr(packet.base.y, oof).statistic),
        "cv_regret_at_1": float(packet.base.y[int(np.argmin(oof))] - np.min(packet.base.y)),
        "cv_foldmean_regret_at_1": foldmean_regret,
        "anchor_mae": anchor_mae,
        "anchor_rmse": float(np.sqrt(np.mean(anchor_err**2))),
        "anchor_rank_correct": bool(int(np.argmin(anchor_pred)) == int(np.argmin(valid_y))),
        "pred_validated_global": float(anchor_pred[0]),
        "pred_validated_pair": float(anchor_pred[1]),
        "err_validated_global": float(anchor_err[0]),
        "err_validated_pair": float(anchor_err[1]),
    }


def main() -> None:
    packet = load_generic_family_packet()
    valid_weights = np.stack(
        [
            _summary_weights(ccglobalpremium_retainedtotal_summary(), packet.base.domain_names),
            _summary_weights(ccpairtotal_retainedtotal_summary(), packet.base.domain_names),
        ],
        axis=0,
    )
    valid_y = np.asarray([VALIDATED_GLOBAL_BPB, VALIDATED_PAIR_BPB], dtype=float)

    starts = [
        ("chatgpt_tuned", _pack(TUNED_GENERIC_FAMILY_PARAMS)),
        (
            "broad_beta_prior",
            _pack(
                {
                    "alpha": 11.533461482593735,
                    "eta": 10.859113730214359,
                    "lam": 0.3422735488822989,
                    "tau": 2.843180828656475,
                    "reg": 0.0001896587113845684,
                    "beta": 0.9324427249160729,
                }
            ),
        ),
        ("neutral_1", np.array([0.0, 0.0, -2.0, 2.0, -8.0, 0.0], dtype=float)),
        ("neutral_2", np.array([1.0, 1.0, -4.0, 4.0, -10.0, -1.0], dtype=float)),
        ("neutral_3", np.array([2.0, 2.0, -3.0, 3.0, -7.0, 1.0], dtype=float)),
    ]

    rows: list[dict[str, float | bool | str]] = []
    best_row: dict[str, float | bool | str] | None = None
    best_z: np.ndarray | None = None

    for label, start in starts:
        t0 = perf_counter()

        def objective(z: np.ndarray) -> float:
            return float(_evaluate_params(z, packet, valid_weights, valid_y)["objective"])

        result = minimize(
            objective,
            start,
            method="Nelder-Mead",
            options={"maxiter": 900, "xatol": 1e-4, "fatol": 1e-6},
        )
        row = _evaluate_params(np.asarray(result.x, dtype=float), packet, valid_weights, valid_y)
        row = {
            "start_label": label,
            "success": bool(result.success),
            "nit": int(result.nit),
            "nfev": int(result.nfev),
            "duration": float(perf_counter() - t0),
            **row,
        }
        rows.append(row)
        if best_row is None or float(row["objective"]) < float(best_row["objective"]):
            best_row = row
            best_z = np.asarray(result.x, dtype=float)

    if best_row is None or best_z is None:
        raise RuntimeError("No tuning run succeeded")

    trials = pd.DataFrame(rows).sort_values("objective", ascending=True).reset_index(drop=True)
    RESULTS_CSV.write_text(trials.to_csv(index=False))
    RESULTS_JSON.write_text(
        json.dumps(
            {
                "reference_params": TUNED_GENERIC_FAMILY_PARAMS,
                "best_run": best_row,
                "best_run_params_only": _unpack(best_z),
                "delta_vs_reference": {
                    key: float(_unpack(best_z)[key] - TUNED_GENERIC_FAMILY_PARAMS[key])
                    for key in TUNED_GENERIC_FAMILY_PARAMS
                },
            },
            indent=2,
            sort_keys=True,
        )
    )
    print(f"Trials: {RESULTS_CSV}")
    print(f"Summary: {RESULTS_JSON}")
    print(RESULTS_JSON.read_text())


if __name__ == "__main__":
    main()
