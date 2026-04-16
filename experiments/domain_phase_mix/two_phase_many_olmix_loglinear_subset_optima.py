# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Exact subset-fit predicted-optimum deployments for the Olmix loglinear baseline."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from functools import cache
import gzip
import json
from pathlib import Path
from typing import Any

import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    OBJECTIVE_METRIC,
)

OLMIX_LOGLINEAR_SUBSET_OPTIMA_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_olmix_loglinear_subset_optima_uncheatable_bpb"
)
OLMIX_LOGLINEAR_SUBSET_OPTIMA_BASE_RUN_ID = 460
OLMIX_LOGLINEAR_SUBSET_OPTIMA_SUBSET_SIZES = (20, 40, 60, 80, 100, 140, 180, 220, 242)
OLMIX_LOGLINEAR_SUBSET_OPTIMA_POLICY = "feature_bayes_linear_olmix_loglinear_optimum"
OLMIX_LOGLINEAR_SUBSET_OPTIMA_VARIANT = "olmix_loglinear"
OLMIX_LOGLINEAR_SUBSET_OPTIMA_TUNING_METHOD = "OlmixLoglinear+KL"
_SUMMARY_DIR = Path(__file__).resolve().parent / "exploratory" / "two_phase_many"
SUMMARY_JSON_PATH = _SUMMARY_DIR / "two_phase_many_olmix_loglinear_subset_summary.json"
_EMBEDDED_ROWS_GZIP_BASE64 = (
    "H4sIAMmM3WkC/+2d6XIdSXKsX0XWv6/q5hK56VVkMhgaxJDQkAAHANkayfTu8i8PlrPUkgVNt+lHSyY21V1VpyqX"
    "CA9Pj4h//a9frm+ef1x/vfp5/fXu0/Xz7aerX7//+su//NP9j69f/98//fIX/ePpt+vHb1c3Xx6ebu+vHn/cX91f"
    "f7vVJb/wd+e8y7/MXakn/uAyP/kQc/bJV9dyC7GeXP14+/nx9vnq+vnK62I3uZwth5JScDFYcUVX399eP94+PV89"
    "/Pp0+/hT7zjzFt7/Mnfl88+rT3dPz9f3N7f98SUks+ZS9k7v49vcPcdvbqHGXHwszdVmuvrh+/Pdtx/frr49/Ly9"
    "+nZ7fX/1/cv10y0/9PPp6vvj7c/30ev/xV19u/6Pq99u7z5/ee6vkHItpbTUii855PJ+4dOP798fHp+vfr39+vDb"
    "lb81XuL1P/uL53i9mK+txexatfB+4dxzgr3+95dnPOlf/tfLv3H9758evn67jlfXj/9x9/MwF07vlyw5azGkUmLW"
    "M16uurn5/9f6iev7T1efbp/uPt9ffdFTX26L0UVzKVbfil4zrd6ndzzc5k2/VVJrXr+YtFRO7rp5vPt22+/7ev3b"
    "8Y+5lkrWfSlFi1V/rNz2+ltatKlW1kGNpeR4csvtpx831893D/f9tn/Xujj5Nm/OLGkdNa26WsPWvW/fp8tjKyHr"
    "EV6/q5tPb/16e/P8+HB/d/PUb/5y/fhJW+T2+Mf1APNNK7H50PTyaewJb6+QSww5xphb1Hfkdnr7/fPt4/P13f03"
    "/e3oV32KfWijdqNlc3XlrqPJZHy1l12Ndtj07/f85e6eHdlf8tcfT3f3t0/HY+xrrLXKAuSoNe7y9r3vYxxyMO9a"
    "MBd1e99dR/c+PHw6LL27+7v7zycj6zUswbOCko8yP6s3vv1ecM4wJKFljU0+nY/PMlEna8dVjYpezjf2lC9l5vL3"
    "EXSh1JCC19LWzjgbhi+311+fv5wMGutBr9GqtkRpZe7yo3FKoYWmyfTaoxbPtugX2cyHx7/3T/58+/D58fr7l78f"
    "f4kMY4oyCM5nq66EOHD70Uaojg0QWtUy0fedvuvd/acfT8+Pd3JLx79Y5Uf0cSE1vXoti/e8/QybLfoiK68lrLE8"
    "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"
    "fVDtYb/NTv5b/8Dzqz88Pq4AAA="
)


@dataclass(frozen=True)
class OlmixLoglinearSubsetOptimumSummary:
    """Summary for one exact subset-fit Olmix predicted optimum."""

    subset_size: int
    run_id: int
    run_name: str
    policy: str
    objective_metric: str
    variant_name: str
    tuning_method: str
    predicted_optimum_value: float
    regularized_objective: float
    subset_best_observed_run_name: str
    subset_best_observed_bpb: float
    fullswarm_chosen_run_name: str
    fullswarm_chosen_value: float
    fullswarm_regret_at_1: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_tv_distance: float
    optimum_move_mean_phase_tv_vs_prev: float | None
    tuning_objective: float
    tuning_cv_rmse: float
    tuning_cv_regret_at_1: float
    tuning_cv_foldmean_regret_at_1: float
    tuning_lower_tail_optimism: float
    tuning_cv_depopt_best8: float
    tuning_cv_rawopt_nearest_tv: float
    phase0_max_weight: float
    phase1_max_weight: float
    phase0_support_below_1e4: int
    phase1_support_below_1e4: int
    phase_weights: dict[str, dict[str, float]]


def olmix_loglinear_subset_optimum_run_id(subset_size: int) -> int:
    """Return the canonical run id for one Olmix subset-fit optimum."""
    if subset_size not in OLMIX_LOGLINEAR_SUBSET_OPTIMA_SUBSET_SIZES:
        raise ValueError(f"Unsupported subset size: {subset_size}")
    return OLMIX_LOGLINEAR_SUBSET_OPTIMA_BASE_RUN_ID + OLMIX_LOGLINEAR_SUBSET_OPTIMA_SUBSET_SIZES.index(subset_size)


def olmix_loglinear_subset_optimum_run_name(subset_size: int) -> str:
    """Return the canonical run name for one Olmix subset-fit optimum."""
    return f"baseline_olmix_loglinear_optimum_k{subset_size:03d}_uncheatable_bpb"


def _embedded_rows() -> list[dict[str, Any]]:
    payload = gzip.decompress(base64.b64decode(_EMBEDDED_ROWS_GZIP_BASE64))
    return json.loads(payload.decode("utf-8"))


def _load_rows() -> list[dict[str, Any]]:
    if SUMMARY_JSON_PATH.exists():
        payload = json.loads(SUMMARY_JSON_PATH.read_text())
        rows = payload.get("rows")
        if isinstance(rows, list):
            return rows
    return _embedded_rows()


@cache
def _raw_subset_rows_by_size() -> dict[int, dict[str, Any]]:
    rows = _load_rows()
    rows_by_size = {int(row["subset_size"]): row for row in rows}
    missing = set(OLMIX_LOGLINEAR_SUBSET_OPTIMA_SUBSET_SIZES).difference(rows_by_size)
    if missing:
        raise ValueError(f"Missing embedded/disk Olmix subset summaries for sizes: {sorted(missing)}")
    return rows_by_size


@cache
def olmix_loglinear_subset_optima_summaries(
    subset_sizes: tuple[int, ...] = OLMIX_LOGLINEAR_SUBSET_OPTIMA_SUBSET_SIZES,
) -> tuple[OlmixLoglinearSubsetOptimumSummary, ...]:
    """Return exact subset-fit Olmix predicted-optimum summaries."""
    rows_by_size = _raw_subset_rows_by_size()
    summaries: list[OlmixLoglinearSubsetOptimumSummary] = []
    for subset_size in subset_sizes:
        if subset_size not in OLMIX_LOGLINEAR_SUBSET_OPTIMA_SUBSET_SIZES:
            raise ValueError(f"Unsupported subset size: {subset_size}")
        row = rows_by_size[subset_size]
        summaries.append(
            OlmixLoglinearSubsetOptimumSummary(
                subset_size=subset_size,
                run_id=olmix_loglinear_subset_optimum_run_id(subset_size),
                run_name=olmix_loglinear_subset_optimum_run_name(subset_size),
                policy=OLMIX_LOGLINEAR_SUBSET_OPTIMA_POLICY,
                objective_metric=OBJECTIVE_METRIC,
                variant_name=OLMIX_LOGLINEAR_SUBSET_OPTIMA_VARIANT,
                tuning_method=OLMIX_LOGLINEAR_SUBSET_OPTIMA_TUNING_METHOD,
                predicted_optimum_value=float(row["predicted_optimum_value"]),
                regularized_objective=float(row["regularized_objective"]),
                subset_best_observed_run_name=str(row["subset_best_observed_run_name"]),
                subset_best_observed_bpb=float(row["subset_best_observed_bpb"]),
                fullswarm_chosen_run_name=str(row["fullswarm_chosen_run_name"]),
                fullswarm_chosen_value=float(row["fullswarm_chosen_value"]),
                fullswarm_regret_at_1=float(row["fullswarm_regret_at_1"]),
                nearest_observed_run_name=str(row["nearest_observed_run_name"]),
                nearest_observed_value=float(row["nearest_observed_value"]),
                nearest_observed_tv_distance=float(row["nearest_observed_tv_distance"]),
                optimum_move_mean_phase_tv_vs_prev=(
                    None
                    if row["optimum_move_mean_phase_tv_vs_prev"] is None
                    else float(row["optimum_move_mean_phase_tv_vs_prev"])
                ),
                tuning_objective=float(row["tuning_objective"]),
                tuning_cv_rmse=float(row["tuning_cv_rmse"]),
                tuning_cv_regret_at_1=float(row["tuning_cv_regret_at_1"]),
                tuning_cv_foldmean_regret_at_1=float(row["tuning_cv_foldmean_regret_at_1"]),
                tuning_lower_tail_optimism=float(row["tuning_lower_tail_optimism"]),
                tuning_cv_depopt_best8=float(row["tuning_cv_depopt_best8"]),
                tuning_cv_rawopt_nearest_tv=float(row["tuning_cv_rawopt_nearest_tv"]),
                phase0_max_weight=float(row["phase0_max_weight"]),
                phase1_max_weight=float(row["phase1_max_weight"]),
                phase0_support_below_1e4=int(row["phase0_support_below_1e4"]),
                phase1_support_below_1e4=int(row["phase1_support_below_1e4"]),
                phase_weights=row["phase_weights"],
            )
        )
    return tuple(summaries)


def olmix_loglinear_subset_optima_summaries_json(
    subset_sizes: tuple[int, ...] = OLMIX_LOGLINEAR_SUBSET_OPTIMA_SUBSET_SIZES,
) -> str:
    """Return the exact subset-fit Olmix summaries as JSON."""
    return json.dumps(
        [summary.__dict__ for summary in olmix_loglinear_subset_optima_summaries(subset_sizes)],
        indent=2,
        sort_keys=True,
    )


def olmix_loglinear_subset_optima_summaries_frame(
    subset_sizes: tuple[int, ...] = OLMIX_LOGLINEAR_SUBSET_OPTIMA_SUBSET_SIZES,
) -> pd.DataFrame:
    """Return a flat summary frame for the exact subset-fit Olmix sweep."""
    return pd.DataFrame([summary.__dict__ for summary in olmix_loglinear_subset_optima_summaries(subset_sizes)])


def create_olmix_loglinear_subset_optima_weight_configs(
    subset_sizes: tuple[int, ...] = OLMIX_LOGLINEAR_SUBSET_OPTIMA_SUBSET_SIZES,
) -> tuple[WeightConfig, ...]:
    """Return weight configs for the exact subset-fit Olmix predicted-optimum sweep."""
    return tuple(
        WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
        for summary in olmix_loglinear_subset_optima_summaries(subset_sizes)
    )
