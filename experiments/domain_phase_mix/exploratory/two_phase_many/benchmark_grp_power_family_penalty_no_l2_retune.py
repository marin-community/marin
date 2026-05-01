# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.11"
# dependencies = ["numpy", "pandas", "scipy", "scikit-learn"]
# ///
"""Retune GRP (Power-Family Penalty) with L2 regularization removed."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    compute_penalty_calibration_metrics,
    pack_penalty_calibration_params,
    penalty_calibration_oof_metrics,
    penalty_calibration_param_keys,
    penalty_calibration_params_from_metrics,
    penalty_calibration_variant_parameter_counts,
    unpack_penalty_calibration_params,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.structured_epoch_family import (
    MANY_DOMAIN_TARGET,
)

SCRIPT_DIR = Path(__file__).resolve().parent
VARIANT_NAME = "power_family_penalty"
OUTPUT_STEM = "grp_power_family_penalty_no_l2_retune"
PENALTY_CALIBRATION_BEST_CSV = SCRIPT_DIR / "grp_penalty_calibration_variants_best.csv"
COARSE_CSV = SCRIPT_DIR / f"{OUTPUT_STEM}_coarse.csv"
REFINE_CSV = SCRIPT_DIR / f"{OUTPUT_STEM}_refine.csv"
BEST_CSV = SCRIPT_DIR / f"{OUTPUT_STEM}_best.csv"
SUMMARY_JSON = SCRIPT_DIR / f"{OUTPUT_STEM}_summary.json"
MARKDOWN_MD = SCRIPT_DIR / f"{OUTPUT_STEM}.md"
CV_SEED = 0
REG_FIXED = 0.0
REG_PACK_PLACEHOLDER_LOG = -18.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="Powell")
    parser.add_argument("--coarse-top-k", type=int, default=3)
    return parser.parse_args()


def _base_best_params() -> dict[str, float]:
    frame = pd.read_csv(PENALTY_CALIBRATION_BEST_CSV)
    matches = frame.loc[(frame["variant"] == VARIANT_NAME) & (frame["stage"] == "refine")]
    if matches.empty:
        raise ValueError(f"Missing refined row for {VARIANT_NAME!r} in {PENALTY_CALIBRATION_BEST_CSV}")
    params = penalty_calibration_params_from_metrics(dict(matches.iloc[0]), VARIANT_NAME)
    params["reg"] = REG_FIXED
    return params


def _no_l2_param_keys() -> tuple[str, ...]:
    return tuple(key for key in penalty_calibration_param_keys(VARIANT_NAME) if key != "reg")


def _pack_no_l2_params(params: dict[str, float]) -> np.ndarray:
    full = dict(params)
    full["reg"] = float(np.exp(REG_PACK_PLACEHOLDER_LOG))
    packed = pack_penalty_calibration_params(full, VARIANT_NAME)
    return np.concatenate([packed[:2], packed[3:]])


def _unpack_no_l2_params(z: np.ndarray) -> dict[str, float]:
    full_z = np.insert(np.asarray(z, dtype=float), 2, REG_PACK_PLACEHOLDER_LOG)
    params = unpack_penalty_calibration_params(full_z, VARIANT_NAME)
    params["reg"] = REG_FIXED
    return params


def _with_updates(base: dict[str, float], **updates: float) -> dict[str, float]:
    row = dict(base)
    row.update({key: float(value) for key, value in updates.items()})
    row["reg"] = REG_FIXED
    return row


def _dedupe_start_bank(rows: list[dict[str, float]]) -> tuple[dict[str, float], ...]:
    seen: set[tuple[tuple[str, float], ...]] = set()
    deduped: list[dict[str, float]] = []
    for row in rows:
        key = tuple(sorted((key, round(float(value), 8)) for key, value in row.items()))
        if key in seen:
            continue
        seen.add(key)
        deduped.append({key: float(value) for key, value in row.items()})
    return tuple(deduped)


def _start_bank() -> tuple[dict[str, float], ...]:
    base = _base_best_params()
    starts = [
        dict(base),
        _with_updates(base, eta=base["eta"] * 0.8, lam=max(base["lam"] * 0.5, 1e-8)),
        _with_updates(base, eta=base["eta"] * 1.2, lam=min(base["lam"] * 2.0 + 1e-8, 1.0)),
        _with_updates(base, beta=max(base["beta"] - 0.08, 0.05), tau_broad_text=base["tau_broad_text"] - 0.4),
        _with_updates(base, beta=min(base["beta"] + 0.08, 0.95), tau_tech_code=base["tau_tech_code"] + 0.5),
        _with_updates(base, a_broad_text=np.clip(base["a_broad_text"] * 0.75, 0.02, 2.0)),
        _with_updates(base, a_tech_code=np.clip(base["a_tech_code"] * 1.5, 0.02, 2.0)),
        _with_updates(base, a_reasoning=np.clip(base["a_reasoning"] * 1.35, 0.02, 2.0)),
        _with_updates(
            base,
            tau_broad_text=base["tau_broad_text"] + 0.5,
            tau_tech_code=base["tau_tech_code"] - 0.6,
            tau_reasoning=base["tau_reasoning"] + 0.4,
        ),
    ]
    return _dedupe_start_bank(starts)


def _parameter_counts(packet) -> dict[str, int]:
    counts = penalty_calibration_variant_parameter_counts(packet, VARIANT_NAME)
    counts["nonlinear_param_count"] -= 1
    counts["total_param_count"] -= 1
    return counts


def _coarse_rows(packet, start_bank: tuple[dict[str, float], ...]) -> pd.DataFrame:
    rows: list[dict[str, float | bool]] = []
    for start_id, params in enumerate(start_bank):
        rows.append(
            {
                "variant": "power_family_penalty_no_l2",
                "surrogate_variant": VARIANT_NAME,
                "stage": "coarse",
                "start_id": int(start_id),
                **params,
                **penalty_calibration_oof_metrics(packet, params, variant_name=VARIANT_NAME, seed=CV_SEED),
            }
        )
    return pd.DataFrame.from_records(rows).sort_values(
        ["objective", "cv_rmse", "cv_depopt_best8"],
        ascending=[True, True, True],
    )


def _refine_rows(
    packet,
    start_bank: tuple[dict[str, float], ...],
    *,
    coarse_top_k: int,
    method: str,
) -> tuple[pd.DataFrame, dict[str, float | bool], Any]:
    coarse_frame = _coarse_rows(packet, start_bank)
    chosen_ids = coarse_frame["start_id"].head(int(coarse_top_k)).tolist()

    best_metrics: dict[str, float | bool] | None = None
    best_result: Any | None = None
    best_objective = float("inf")
    refine_rows: list[dict[str, float | bool]] = []

    for chosen_rank, start_id in enumerate(chosen_ids):
        start = _pack_no_l2_params(start_bank[start_id])
        cache: dict[tuple[float, ...], float] = {}

        def objective(z: np.ndarray, _cache: dict[tuple[float, ...], float] = cache) -> float:
            key = tuple(np.round(np.asarray(z, dtype=float), 8))
            if key not in _cache:
                params = _unpack_no_l2_params(z)
                metrics = penalty_calibration_oof_metrics(packet, params, variant_name=VARIANT_NAME, seed=CV_SEED)
                _cache[key] = float(metrics["objective"])
            return _cache[key]

        options = {
            "L-BFGS-B": {"maxiter": 80, "ftol": 1e-6},
            "Nelder-Mead": {"maxiter": 400, "xatol": 1e-4, "fatol": 1e-6},
            "Powell": {"maxiter": 30, "xtol": 1e-4, "ftol": 1e-6},
        }.get(method, {"maxiter": 120})
        result = minimize(objective, start, method=method, options=options)
        params = _unpack_no_l2_params(np.asarray(result.x, dtype=float))
        metrics = penalty_calibration_oof_metrics(packet, params, variant_name=VARIANT_NAME, seed=CV_SEED)
        row = {
            "variant": "power_family_penalty_no_l2",
            "surrogate_variant": VARIANT_NAME,
            "stage": "refine",
            "chosen_rank": int(chosen_rank),
            "start_id": int(start_id),
            "success": bool(result.success),
            "message": str(result.message),
            **params,
            **metrics,
        }
        refine_rows.append(row)
        if float(row["objective"]) < best_objective:
            best_objective = float(row["objective"])
            best_metrics = row
            best_result = result

    if best_metrics is None or best_result is None:
        raise RuntimeError("No-L2 retune failed to produce a best result")
    return coarse_frame, best_metrics, pd.DataFrame.from_records(refine_rows)


def _best_row(packet, best_metrics: dict[str, float | bool]) -> dict[str, Any]:
    params = {key: float(best_metrics[key]) for key in _no_l2_param_keys()}
    params["reg"] = REG_FIXED
    model = build_penalty_calibration_surrogate(packet, params=params, variant_name=VARIANT_NAME).fit(
        packet.base.w,
        packet.base.y,
    )
    full_metrics = compute_penalty_calibration_metrics(packet, model, seed=CV_SEED)
    counts = _parameter_counts(packet)
    return {
        "variant": "power_family_penalty_no_l2",
        "surrogate_variant": VARIANT_NAME,
        "stage": "refine",
        "success": bool(best_metrics["success"]),
        "message": str(best_metrics["message"]),
        **params,
        **full_metrics,
        **counts,
        "retuned": True,
        "objective_metric": MANY_DOMAIN_TARGET,
        "notes": "Full nonlinear retune of power_family_penalty with reg fixed exactly to 0.0.",
    }


def _markdown(best_row: dict[str, Any]) -> str:
    columns = [
        "variant",
        "surrogate_variant",
        "nonlinear_param_count",
        "total_param_count",
        "train_rmse",
        "cv_rmse",
        "cv_foldmean_regret_at_1",
        "lower_tail_optimism",
        "cv_depopt_best8",
        "cv_rawopt_nearest_tv",
        "raw_predicted_optimum_value",
    ]
    frame = pd.DataFrame([best_row])[columns]
    return frame.to_markdown(index=False, floatfmt=".6f") + "\n"


def main() -> None:
    args = _parse_args()
    packet = load_generic_family_packet(target=MANY_DOMAIN_TARGET)
    start_bank = _start_bank()
    coarse_frame, best_metrics, refine_frame = _refine_rows(
        packet,
        start_bank,
        coarse_top_k=args.coarse_top_k,
        method=args.method,
    )
    best_row = _best_row(packet, best_metrics)

    coarse_frame.to_csv(COARSE_CSV, index=False)
    refine_frame.sort_values("objective").to_csv(REFINE_CSV, index=False)
    pd.DataFrame([best_row]).to_csv(BEST_CSV, index=False)
    MARKDOWN_MD.write_text(_markdown(best_row), encoding="utf-8")
    SUMMARY_JSON.write_text(
        json.dumps(
            {
                "variant": "power_family_penalty_no_l2",
                "surrogate_variant": VARIANT_NAME,
                "objective_metric": MANY_DOMAIN_TARGET,
                "method": args.method,
                "coarse_top_k": int(args.coarse_top_k),
                "start_bank_size": len(start_bank),
                "best_row": best_row,
                "coarse_csv": str(COARSE_CSV),
                "refine_csv": str(REFINE_CSV),
                "best_csv": str(BEST_CSV),
                "markdown": str(MARKDOWN_MD),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {COARSE_CSV}")
    print(f"Wrote {REFINE_CSV}")
    print(f"Wrote {BEST_CSV}")
    print(f"Wrote {MARKDOWN_MD}")
    print(f"Wrote {SUMMARY_JSON}")


if __name__ == "__main__":
    main()
