# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Raw-optimum GRP baselines for penalty / pair-CES follow-up variants."""

from __future__ import annotations

import csv
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from functools import cache
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_followup import (
    family_shares,
    load_generic_family_packet,
)
from experiments.domain_phase_mix.exploratory.two_phase_many.surrogate_search.generic_family_penalty_calibration import (
    build_penalty_calibration_surrogate,
    optimize_penalty_calibration_model,
)
from experiments.domain_phase_mix.static_batch_selection import average_phase_tv_distance
from experiments.domain_phase_mix.two_phase_many_genericfamily_observed_only_trustblend_subset_optima import (
    OBJECTIVE_METRIC,
    _phase_weights_from_array,
)

GENERICFAMILY_PENALTY_RAW_OPTIMA_SOURCE_EXPERIMENT = (
    "pinlin_calvin_xu/data_mixture/ngd3dm2_genericfamily_penalty_raw_optima_uncheatable_bpb"
)
BENCHMARK_BEST_CSV = (
    Path(__file__).resolve().parent / "exploratory" / "two_phase_many" / "grp_penalty_calibration_variants_best.csv"
)
NO_L2_RETUNE_BEST_CSV = (
    Path(__file__).resolve().parent / "exploratory" / "two_phase_many" / "grp_power_family_penalty_no_l2_retune_best.csv"
)
_EMBEDDED_BEST_ROWS: dict[str, dict[str, str]] = {
    "power_family_penalty": {
        "variant": "power_family_penalty",
        "stage": "refine",
        "success": "True",
        "message": "Optimization terminated successfully.",
        "eta": "6.627794351309641",
        "lam": "6.14421235332821e-06",
        "reg": "0.21636759823649684",
        "beta": "0.2629059619755788",
        "a_broad_text": "0.6462737477673589",
        "a_tech_code": "0.1657586322714625",
        "a_reasoning": "0.2076641777781618",
        "tau_broad_text": "3.193090495213877",
        "tau_tech_code": "6.2042610686315145",
        "tau_reasoning": "5.136810831800622",
        "cv_rmse": "0.009522115044355527",
        "cv_foldmean_regret_at_1": "0.0028093338012695757",
        "lower_tail_optimism": "0.0025588961641979974",
        "cv_depopt_best8": "0.0457087169538863",
        "cv_rawopt_nearest_tv": "0.5055017245968771",
        "objective": "0.020567918757875406",
    },
    "power_boxcox_family_penalty": {
        "variant": "power_boxcox_family_penalty",
        "stage": "refine",
        "success": "True",
        "message": "Optimization terminated successfully.",
        "eta": "5.970663378272638",
        "lam": "1.1805484189379591e-05",
        "reg": "0.008487569043183627",
        "beta": "0.28868489951432713",
        "alpha": "2980.9579870417283",
        "a_broad_text": "0.2532338334548216",
        "a_tech_code": "0.02",
        "a_reasoning": "0.23791296133771586",
        "tau_broad_text": "2.414249596965036",
        "tau_tech_code": "5.62069332819873",
        "tau_reasoning": "5.4884704870858885",
        "cv_rmse": "0.009167197851597276",
        "cv_foldmean_regret_at_1": "0.0028093338012695757",
        "lower_tail_optimism": "0.0012468529542485431",
        "cv_depopt_best8": "0.04180735042621921",
        "cv_rawopt_nearest_tv": "0.6118932735417746",
        "objective": "0.020230758796824694",
    },
    "power_family_penalty_global_ftotal_pairces": {
        "variant": "power_family_penalty_global_ftotal_pairces",
        "stage": "refine",
        "success": "True",
        "message": "Optimization terminated successfully.",
        "eta": "5.049151286292939",
        "lam": "0.009302144649822276",
        "reg": "0.007973626509344895",
        "beta": "0.2973791958853945",
        "tau": "6.735562085494085",
        "tau_broad_text": "2.8169808139340047",
        "tau_tech_code": "5.393093021388599",
        "tau_reasoning": "5.34045713497618",
        "a_broad_text": "0.38850726319776474",
        "a_tech_code": "0.02",
        "a_reasoning": "0.1289589000803356",
        "pair_rho": "0.6864247749354148",
        "cv_rmse": "0.009241240527415902",
        "cv_foldmean_regret_at_1": "0.0",
        "lower_tail_optimism": "0.0013465234115646462",
        "cv_depopt_best8": "0.04060547284738725",
        "cv_rawopt_nearest_tv": "0.5207328687986771",
        "objective": "0.019182378205923722",
    },
    "power_family_penalty_no_l2": {
        "variant": "power_family_penalty_no_l2",
        "stage": "refine",
        "success": "True",
        "message": "Optimization terminated successfully.",
        "eta": "5.222440513840459",
        "lam": "7.04928339546768e-06",
        "reg": "0.0",
        "beta": "0.1967681464478872",
        "a_broad_text": "0.48485414608456984",
        "a_tech_code": "0.04843166940506106",
        "a_reasoning": "1.0344800333570379",
        "tau_broad_text": "3.0915710553505598",
        "tau_tech_code": "8.0",
        "tau_reasoning": "4.860956465592155",
        "cv_rmse": "0.00872034786579222",
        "cv_foldmean_regret_at_1": "0.0020820379257202593",
        "lower_tail_optimism": "0.002776860539141558",
        "cv_depopt_best8": "0.05627328049308757",
        "cv_rawopt_nearest_tv": "0.5528221602935263",
        "objective": "0.021368429683893034",
    },
}


@dataclass(frozen=True)
class GenericFamilyPenaltyRawOptimumVariantSpec:
    """Identity for one raw-optimum deployment variant."""

    variant_name: str
    run_name: str
    run_id: int
    surrogate_variant_name: str | None = None


GENERICFAMILY_PENALTY_RAW_OPTIMUM_VARIANT_SPECS: tuple[GenericFamilyPenaltyRawOptimumVariantSpec, ...] = (
    GenericFamilyPenaltyRawOptimumVariantSpec(
        variant_name="power_family_penalty",
        run_name="baseline_genericfamily_power_family_penalty_raw_optimum",
        run_id=412,
    ),
    GenericFamilyPenaltyRawOptimumVariantSpec(
        variant_name="power_family_penalty_global_ftotal_pairces",
        run_name="baseline_genericfamily_power_family_penalty_global_ftotal_pairces_raw_optimum",
        run_id=413,
    ),
    GenericFamilyPenaltyRawOptimumVariantSpec(
        variant_name="power_boxcox_family_penalty",
        run_name="baseline_genericfamily_power_boxcox_family_penalty_raw_optimum",
        run_id=414,
    ),
    GenericFamilyPenaltyRawOptimumVariantSpec(
        variant_name="power_family_penalty_no_l2",
        run_name="baseline_genericfamily_power_family_penalty_no_l2_raw_optimum",
        run_id=415,
        surrogate_variant_name="power_family_penalty",
    ),
)
GENERICFAMILY_PENALTY_RAW_OPTIMUM_DEFAULT_VARIANTS: tuple[str, ...] = tuple(
    spec.variant_name
    for spec in GENERICFAMILY_PENALTY_RAW_OPTIMUM_VARIANT_SPECS
    if spec.variant_name != "power_family_penalty_no_l2"
)
_RAW_OPTIMUM_VARIANT_SPECS_BY_NAME = {
    spec.variant_name: spec for spec in GENERICFAMILY_PENALTY_RAW_OPTIMUM_VARIANT_SPECS
}


@dataclass(frozen=True)
class GenericFamilyPenaltyRawOptimumSummary:
    """Summary for one raw-optimum penalty-variant baseline."""

    run_name: str
    run_id: int
    variant_name: str
    objective_metric: str
    tuning_stage: str
    tuning_message: str
    tuning_objective: float
    tuning_cv_rmse: float
    tuning_cv_foldmean_regret_at_1: float
    tuning_lower_tail_optimism: float
    tuning_cv_depopt_best8: float
    tuning_cv_rawopt_nearest_tv: float
    raw_predicted_optimum_value: float
    nearest_observed_run_name: str
    nearest_observed_value: float
    nearest_observed_tv_distance: float
    optimizer_success: bool
    optimizer_message: str
    family_shares: dict[str, float]
    tuned_params: dict[str, float]
    phase_weights: dict[str, dict[str, float]]


def _raw_optimum_variant_spec(variant_name: str) -> GenericFamilyPenaltyRawOptimumVariantSpec:
    try:
        return _RAW_OPTIMUM_VARIANT_SPECS_BY_NAME[variant_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown raw-optimum variant {variant_name!r}; expected one of "
            f"{GENERICFAMILY_PENALTY_RAW_OPTIMUM_DEFAULT_VARIANTS!r}"
        ) from exc


def parse_penalty_raw_optimum_variants(raw_variants: str | Iterable[str] | None) -> tuple[str, ...]:
    """Parse a variant selector into a validated ordered tuple."""
    if raw_variants is None:
        return GENERICFAMILY_PENALTY_RAW_OPTIMUM_DEFAULT_VARIANTS
    if isinstance(raw_variants, str):
        values = tuple(part.strip() for part in raw_variants.split(",") if part.strip())
    else:
        values = tuple(str(part).strip() for part in raw_variants if str(part).strip())
    if not values:
        raise ValueError("Expected at least one raw-optimum variant")
    deduped: list[str] = []
    for value in values:
        _raw_optimum_variant_spec(value)
        if value not in deduped:
            deduped.append(value)
    return tuple(deduped)


def _load_best_rows() -> dict[str, dict[str, str]]:
    rows = dict(_EMBEDDED_BEST_ROWS)
    if BENCHMARK_BEST_CSV.exists():
        with BENCHMARK_BEST_CSV.open() as handle:
            rows.update({row["variant"]: row for row in csv.DictReader(handle)})
    if NO_L2_RETUNE_BEST_CSV.exists():
        with NO_L2_RETUNE_BEST_CSV.open() as handle:
            rows.update({row["variant"]: row for row in csv.DictReader(handle)})
    return rows


def _extract_params(row: dict[str, str]) -> dict[str, float]:
    params: dict[str, float] = {}
    for key in (
        "eta",
        "lam",
        "reg",
        "beta",
        "alpha",
        "tau",
        "tau_broad_text",
        "tau_tech_code",
        "tau_reasoning",
        "a_broad_text",
        "a_tech_code",
        "a_reasoning",
        "pair_rho",
    ):
        value = row.get(key, "")
        if value != "":
            params[key] = float(value)
    return params


@cache
def genericfamily_penalty_raw_optimum_summary(
    variant_name: str,
) -> GenericFamilyPenaltyRawOptimumSummary:
    """Return the full-data raw optimum summary for one penalty variant."""
    spec = _raw_optimum_variant_spec(variant_name)
    row = _load_best_rows()[variant_name]
    packet = load_generic_family_packet(target=OBJECTIVE_METRIC)
    params = _extract_params(row)
    surrogate_variant_name = spec.surrogate_variant_name or variant_name
    model = build_penalty_calibration_surrogate(
        packet,
        params=params,
        variant_name=surrogate_variant_name,
    ).fit(packet.base.w, packet.base.y)
    raw_result, phase0, phase1 = optimize_penalty_calibration_model(packet, model, seed=0)
    weights = np.stack([phase0, phase1], axis=0)
    distances = average_phase_tv_distance(packet.base.w, weights[None, :, :])
    nearest_idx = int(np.argmin(distances))
    return GenericFamilyPenaltyRawOptimumSummary(
        run_name=spec.run_name,
        run_id=spec.run_id,
        variant_name=spec.variant_name,
        objective_metric=OBJECTIVE_METRIC,
        tuning_stage=str(row["stage"]),
        tuning_message=str(row["message"]),
        tuning_objective=float(row["objective"]),
        tuning_cv_rmse=float(row["cv_rmse"]),
        tuning_cv_foldmean_regret_at_1=float(row["cv_foldmean_regret_at_1"]),
        tuning_lower_tail_optimism=float(row["lower_tail_optimism"]),
        tuning_cv_depopt_best8=float(row["cv_depopt_best8"]),
        tuning_cv_rawopt_nearest_tv=float(row["cv_rawopt_nearest_tv"]),
        raw_predicted_optimum_value=float(raw_result.fun),
        nearest_observed_run_name=str(packet.base.frame.iloc[nearest_idx][packet.base.name_col]),
        nearest_observed_value=float(packet.base.y[nearest_idx]),
        nearest_observed_tv_distance=float(distances[nearest_idx]),
        optimizer_success=bool(raw_result.success),
        optimizer_message=str(raw_result.message),
        family_shares=family_shares(packet, weights),
        tuned_params=params,
        phase_weights=_phase_weights_from_array(packet.base.domain_names, weights),
    )


@cache
def _cached_penalty_raw_optimum_summaries(
    variant_names: tuple[str, ...],
) -> tuple[GenericFamilyPenaltyRawOptimumSummary, ...]:
    return tuple(genericfamily_penalty_raw_optimum_summary(variant_name) for variant_name in variant_names)


def genericfamily_penalty_raw_optimum_summaries(
    variant_names: Iterable[str] | None = None,
) -> tuple[GenericFamilyPenaltyRawOptimumSummary, ...]:
    """Return ordered raw-optimum summaries for the requested variants."""
    parsed = parse_penalty_raw_optimum_variants(variant_names)
    return _cached_penalty_raw_optimum_summaries(parsed)


def genericfamily_penalty_raw_optimum_summaries_frame(
    variant_names: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Return a tabular summary for the requested raw-optimum variants."""
    return pd.DataFrame(asdict(summary) for summary in genericfamily_penalty_raw_optimum_summaries(variant_names))


def genericfamily_penalty_raw_optimum_summaries_json(
    variant_names: Iterable[str] | None = None,
) -> str:
    """Return JSON summaries for the requested raw-optimum variants."""
    return json.dumps(
        [asdict(summary) for summary in genericfamily_penalty_raw_optimum_summaries(variant_names)],
        indent=2,
        sort_keys=True,
    )


def create_genericfamily_penalty_raw_optimum_weight_config(variant_name: str) -> WeightConfig:
    """Return the weight config for one raw-optimum penalty variant."""
    summary = genericfamily_penalty_raw_optimum_summary(variant_name)
    return WeightConfig(run_id=summary.run_id, phase_weights=summary.phase_weights)
