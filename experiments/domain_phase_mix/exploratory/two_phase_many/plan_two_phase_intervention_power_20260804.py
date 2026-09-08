# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy==2.3.2",
#   "pandas==2.3.1",
#   "scipy==1.16.3",
#   "tabulate==0.9.0",
# ]
# ///
"""Quantify repeat noise and feasible phase-identification designs.

This script uses only already-exposed repeat outcomes and architecture metadata.
It does not fit a surrogate or inspect any sealed endpoint panel.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import chi2, nct
from scipy.stats import t as student_t

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
OUTPUT_DIR = SCRIPT_DIR / "reference_outputs/two_phase_intervention_power_v2_20260804"

MEASURED_FIBER_PATH = (
    SCRIPT_DIR / "reference_outputs/starcoder_wsd80_surface_refined_20260714/wsd80_measured_fiber_observations.csv"
)
GLOBAL_FIBER_PATH = (
    SCRIPT_DIR
    / "reference_outputs/starcoder_wsd80_surface_refined_20260714"
    / "wsd80_global_optimum_fiber_observations.csv"
)
SCALE_FIBER_PATH = (
    SCRIPT_DIR
    / "reference_outputs/starcoder_wsd80_scale_specific_tied_fibers_20260731/results_20260731/observations.csv"
)
CONFIRMATION_PATH = (
    SCRIPT_DIR
    / "reference_outputs/starcoder_wsd80_matched_nd_stage1_20260731/confirmation_results_20260801"
    / "confirmation_pairs.csv"
)
MATCHED_ND_DESIGN_PATH = (
    REPO_ROOT / "experiments/domain_phase_mix" / "starcoder_wsd80_matched_nd_stage1_design_20260731.json"
)

ALPHA = 0.05
POWER = 0.80
EFFECT_SIZES_BPB = (0.0039, 0.0028, 0.001545)
EFFECT_SIZE_ROLES = {
    0.0039: "selected oriented-gain point estimate",
    0.0028: "design shrinkage sensitivity target; not an inferential bound",
    0.001545: "selected oriented-gain 95% confidence-interval lower endpoint",
}
RUN_ENVELOPE = 200
REFERENCE_SEED = 20260711
PRIMARY_I2_SEEDS = 20
PRIMARY_I3_SEEDS = 13


@dataclass(frozen=True)
class NoiseEstimate:
    source: str
    estimand: str
    summary: str
    sigma_bpb: float
    coordinate_clusters: int
    independent_seed_blocks: int
    repeats_per_cluster: str
    variance_degrees_of_freedom: int
    primary_for_design: bool
    transport_assumption: str
    interpretation: str


def sha256_path(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(payload).hexdigest()


def achieved_power(effect_bpb: float, sigma_bpb: float, repeats: int) -> float:
    if effect_bpb <= 0 or sigma_bpb <= 0 or repeats < 2:
        raise ValueError("Power inputs must be positive and repeats must be at least two")
    degrees_of_freedom = repeats - 1
    critical = float(student_t.ppf(1.0 - ALPHA / 2.0, degrees_of_freedom))
    noncentrality = effect_bpb * math.sqrt(repeats) / sigma_bpb
    lower_tail = nct.cdf(-critical, degrees_of_freedom, noncentrality)
    upper_tail = nct.sf(critical, degrees_of_freedom, noncentrality)
    return float(lower_tail + upper_tail)


def minimum_repeats(effect_bpb: float, sigma_bpb: float, maximum: int = 10_000) -> int:
    for repeats in range(2, maximum + 1):
        if achieved_power(effect_bpb, sigma_bpb, repeats) >= POWER:
            return repeats
    raise ValueError(f"No powered design found by {maximum} repeats")


def minimum_detectable_effect(sigma_bpb: float, repeats: int) -> float:
    def objective(effect_bpb: float) -> float:
        return achieved_power(effect_bpb, sigma_bpb, repeats) - POWER

    upper = sigma_bpb
    while achieved_power(upper, sigma_bpb, repeats) < POWER:
        upper *= 2.0
    return float(brentq(objective, 1e-12, upper))


def pooled_rms(values: list[float]) -> float:
    if not values:
        raise ValueError("Cannot pool an empty collection")
    return float(math.sqrt(np.mean(np.square(values))))


def sigma_upper_confidence_bound(sigma_bpb: float, degrees_of_freedom: int, confidence: float) -> float:
    if sigma_bpb <= 0 or degrees_of_freedom <= 0 or not 0 < confidence < 1:
        raise ValueError("Sigma, degrees of freedom, and confidence must be valid")
    lower_chi_square = float(chi2.ppf(1.0 - confidence, degrees_of_freedom))
    return sigma_bpb * math.sqrt(degrees_of_freedom / lower_chi_square)


def repeated_coordinate_noise(measured: pd.DataFrame) -> tuple[list[NoiseEstimate], pd.DataFrame]:
    coordinate_columns = ["phase_0_starcoder", "phase_1_starcoder"]
    rows: list[dict[str, Any]] = []
    for coordinate, frame in measured.groupby(coordinate_columns, sort=True):
        if len(frame) < 2:
            continue
        rows.append(
            {
                "phase_0_starcoder": float(coordinate[0]),
                "phase_1_starcoder": float(coordinate[1]),
                "repeats": len(frame),
                "raw_bpb_sd": float(frame["wsd80_bpb"].std(ddof=1)),
            }
        )
    detail = pd.DataFrame(rows)
    sigmas = detail["raw_bpb_sd"].tolist()
    estimates = [
        NoiseEstimate(
            source="measured_fiber_repeats",
            estimand="raw_bpb",
            summary="pooled_rms",
            sigma_bpb=pooled_rms(sigmas),
            coordinate_clusters=len(detail),
            independent_seed_blocks=1,
            repeats_per_cluster=",".join(map(str, sorted(detail["repeats"].unique()))),
            variance_degrees_of_freedom=4,
            primary_for_design=False,
            transport_assumption="Raw-coordinate variance is not the planned paired estimand.",
            interpretation="Raw-coordinate variance; conservative for same-seed paired contrasts.",
        ),
        NoiseEstimate(
            source="measured_fiber_repeats",
            estimand="raw_bpb",
            summary="maximum",
            sigma_bpb=float(max(sigmas)),
            coordinate_clusters=1,
            independent_seed_blocks=1,
            repeats_per_cluster=",".join(map(str, sorted(detail["repeats"].unique()))),
            variance_degrees_of_freedom=4,
            primary_for_design=False,
            transport_assumption="Raw-coordinate variance is not the planned paired estimand.",
            interpretation="Largest observed repeated-coordinate SD.",
        ),
    ]
    return estimates, detail


def historical_paired_noise(global_fiber: pd.DataFrame) -> tuple[list[NoiseEstimate], pd.DataFrame]:
    repeated = global_fiber[global_fiber["data_seed"] != REFERENCE_SEED]
    tied_rows = repeated[repeated["fiber_index"] == 6][["data_seed", "wsd80_bpb"]]
    tied = tied_rows.rename(columns={"wsd80_bpb": "tied_bpb"})
    joined = repeated[repeated["fiber_index"] != 6].merge(tied, on="data_seed", validate="many_to_one")
    joined["paired_delta_bpb"] = joined["wsd80_bpb"] - joined["tied_bpb"]
    detail = (
        joined.groupby("fiber_index", sort=True)["paired_delta_bpb"]
        .agg(repeats="size", paired_delta_sd="std")
        .reset_index()
    )
    sigmas = detail["paired_delta_sd"].tolist()
    estimates = [
        NoiseEstimate(
            source="global_optimum_fiber_repeats",
            estimand="paired_delta",
            summary="pooled_rms",
            sigma_bpb=pooled_rms(sigmas),
            coordinate_clusters=len(detail),
            independent_seed_blocks=1,
            repeats_per_cluster=",".join(map(str, sorted(detail["repeats"].unique()))),
            variance_degrees_of_freedom=3,
            primary_for_design=False,
            transport_assumption="Historical fibers differ from the proposed intervention cells.",
            interpretation="Same-seed policy-minus-tied noise used by the frozen switch-time protocol.",
        ),
        NoiseEstimate(
            source="global_optimum_fiber_repeats",
            estimand="paired_delta",
            summary="maximum",
            sigma_bpb=float(max(sigmas)),
            coordinate_clusters=1,
            independent_seed_blocks=1,
            repeats_per_cluster=",".join(map(str, sorted(detail["repeats"].unique()))),
            variance_degrees_of_freedom=3,
            primary_for_design=False,
            transport_assumption="Historical fibers differ from the proposed intervention cells.",
            interpretation="Largest same-seed policy-minus-tied SD across repeated coordinates.",
        ),
    ]
    return estimates, detail


def antithetic_triples(
    frame: pd.DataFrame,
    *,
    source_panel: str,
    group_columns: list[str],
    contrast_column: str,
    seed_column: str,
    value_column: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    rounded = frame.assign(_contrast=frame[contrast_column].round(10))
    for group_key, group in rounded.groupby(group_columns, sort=True, dropna=False):
        group_key = group_key if isinstance(group_key, tuple) else (group_key,)
        contrasts = set(group["_contrast"])
        positive_magnitudes = sorted(value for value in contrasts if value > 0 and -value in contrasts)
        tied = group[np.isclose(group["_contrast"], 0.0)][[seed_column, value_column]].rename(
            columns={value_column: "tied_bpb"}
        )
        for magnitude in positive_magnitudes:
            plus = group[np.isclose(group["_contrast"], magnitude)][[seed_column, value_column]].rename(
                columns={value_column: "plus_bpb"}
            )
            minus = group[np.isclose(group["_contrast"], -magnitude)][[seed_column, value_column]].rename(
                columns={value_column: "minus_bpb"}
            )
            joined = plus.merge(minus, on=seed_column, validate="one_to_one").merge(
                tied, on=seed_column, validate="one_to_one"
            )
            if len(joined) < 2:
                continue
            joined["odd_bpb"] = (joined["plus_bpb"] - joined["minus_bpb"]) / 2.0
            joined["even_bpb"] = (joined["plus_bpb"] + joined["minus_bpb"]) / 2.0 - joined["tied_bpb"]
            joined["plus_oriented_bpb"] = joined["plus_bpb"] - joined["tied_bpb"]
            joined["minus_oriented_bpb"] = joined["minus_bpb"] - joined["tied_bpb"]
            joined["oracle_net_bpb"] = joined[["plus_bpb", "minus_bpb"]].min(axis=1) - joined["tied_bpb"]
            identity = {column: value for column, value in zip(group_columns, group_key, strict=True)}
            cluster_suffix = ":".join(f"{column}={identity[column]}" for column in group_columns)
            rows.append(
                {
                    "source_panel": source_panel,
                    "control_cluster_id": f"{source_panel}:{cluster_suffix}",
                    **identity,
                    "contrast_magnitude": float(magnitude),
                    "repeats": len(joined),
                    "odd_mean_bpb": float(joined["odd_bpb"].mean()),
                    "odd_sd_bpb": float(joined["odd_bpb"].std(ddof=1)),
                    "even_mean_bpb": float(joined["even_bpb"].mean()),
                    "even_sd_bpb": float(joined["even_bpb"].std(ddof=1)),
                    "plus_oriented_mean_bpb": float(joined["plus_oriented_bpb"].mean()),
                    "plus_oriented_sd_bpb": float(joined["plus_oriented_bpb"].std(ddof=1)),
                    "minus_oriented_mean_bpb": float(joined["minus_oriented_bpb"].mean()),
                    "minus_oriented_sd_bpb": float(joined["minus_oriented_bpb"].std(ddof=1)),
                    "oracle_net_mean_bpb": float(joined["oracle_net_bpb"].mean()),
                    "oracle_net_sd_bpb": float(joined["oracle_net_bpb"].std(ddof=1)),
                }
            )
    return rows


def antithetic_noise(measured: pd.DataFrame, scale: pd.DataFrame) -> tuple[list[NoiseEstimate], pd.DataFrame]:
    rounded_aggregate = measured["aggregate_starcoder_share_80_20"].round(10)
    measured = measured.assign(aggregate_starcoder_share_80_20=rounded_aggregate)
    scale = scale.assign(
        token_budget_requested=scale["token_budget_requested"].round(0),
        anchor_aggregate_starcoder=scale["anchor_aggregate_starcoder"].round(10),
    )
    rows = antithetic_triples(
        measured,
        source_panel="measured_fibers",
        group_columns=["aggregate_starcoder_share_80_20"],
        contrast_column="ordering_contrast_p1_minus_p0",
        seed_column="data_seed",
        value_column="wsd80_bpb",
    )
    rows.extend(
        antithetic_triples(
            scale,
            source_panel="scale_specific_tied_fibers",
            group_columns=["token_budget_requested", "anchor_aggregate_starcoder"],
            contrast_column="signed_contrast_phase1_minus_phase0",
            seed_column="trainer_data_seed",
            value_column="starcoder_bpb",
        )
    )
    detail = pd.DataFrame(rows)
    if len(detail) != 10:
        raise ValueError(f"Expected 10 antithetic triples, found {len(detail)}")
    independent_control_clusters = int(detail["control_cluster_id"].nunique())
    if independent_control_clusters != 8:
        raise ValueError(f"Expected 8 independent tied-control clusters, found {independent_control_clusters}")
    neighborhood = detail[
        (detail["source_panel"] == "scale_specific_tied_fibers")
        & (
            ((detail["token_budget_requested"] == 1_000_000_000) & (detail["anchor_aggregate_starcoder"] == 0.30))
            | (
                (detail["token_budget_requested"] == 2_000_000_000)
                & detail["anchor_aggregate_starcoder"].isin([0.35, 0.40])
            )
        )
    ]
    if len(neighborhood) != 3:
        raise ValueError(f"Expected 3 design-neighborhood triples, found {len(neighborhood)}")

    estimates: list[NoiseEstimate] = []
    for estimand in ("odd", "even", "plus_oriented", "oracle_net"):
        sigmas = detail[f"{estimand}_sd_bpb"].tolist()
        for summary, sigma in (("pooled_rms", pooled_rms(sigmas)), ("maximum", max(sigmas))):
            if summary == "maximum":
                degrees_of_freedom = 4
                coordinate_clusters = 1
            else:
                # Every coordinate reuses the same five trainer/data seeds.
                degrees_of_freedom = 4
                coordinate_clusters = len(detail)
            estimates.append(
                NoiseEstimate(
                    source="ten_repeated_antithetic_triples",
                    estimand=estimand,
                    summary=summary,
                    sigma_bpb=float(sigma),
                    coordinate_clusters=coordinate_clusters,
                    independent_seed_blocks=1,
                    repeats_per_cluster=",".join(map(str, sorted(detail["repeats"].unique()))),
                    variance_degrees_of_freedom=degrees_of_freedom,
                    primary_for_design=estimand in {"odd", "even", "plus_oriented"},
                    transport_assumption=(
                        "All coordinates reuse the same five seeds, and the archive spans 157.5M and matched-ND "
                        "architectures; cross-coordinate independence and variance transport are unverified."
                    ),
                    interpretation=(
                        "Odd reverses under phase swap; even is the symmetric cost; plus-oriented is a sign fixed "
                        "before the proposed intervention. Oracle net uses a per-seed minimum and is descriptive "
                        "because it is downward biased near zero."
                    ),
                )
            )
        neighborhood_sigmas = neighborhood[f"{estimand}_sd_bpb"].tolist()
        estimates.append(
            NoiseEstimate(
                source="three_design_neighborhood_triples",
                estimand=estimand,
                summary="pooled_rms",
                sigma_bpb=pooled_rms(neighborhood_sigmas),
                coordinate_clusters=len(neighborhood),
                independent_seed_blocks=1,
                repeats_per_cluster="5",
                variance_degrees_of_freedom=4,
                primary_for_design=estimand in {"odd", "even", "plus_oriented"},
                transport_assumption=(
                    "The three coordinates reuse the same five seeds; their pooled SD is transported from 1B/2B "
                    "to proposed h640/h896 or switch-time cells."
                ),
                interpretation="Closest repeated antithetic coordinates to the proposed intervention regime.",
            )
        )
    return estimates, detail


def confirmation_noise(confirmation: pd.DataFrame) -> list[NoiseEstimate]:
    return [
        NoiseEstimate(
            source="matched_nd_high_d_confirmation",
            estimand="paired_delta",
            summary="single_cell",
            sigma_bpb=float(confirmation["gain_tied_minus_untied_bpb"].std(ddof=1)),
            coordinate_clusters=1,
            independent_seed_blocks=1,
            repeats_per_cluster=str(len(confirmation)),
            variance_degrees_of_freedom=len(confirmation) - 1,
            primary_for_design=False,
            transport_assumption="This is one selected high-D coordinate and is not portable.",
            interpretation="Fresh-seed paired gain at the confirmed high-D coordinate; local rather than portable.",
        )
    ]


def power_table(estimates: list[NoiseEstimate]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for estimate in estimates:
        sigma_bases = {
            "point": estimate.sigma_bpb,
            "ucl80": sigma_upper_confidence_bound(estimate.sigma_bpb, estimate.variance_degrees_of_freedom, 0.80),
            "ucl95": sigma_upper_confidence_bound(estimate.sigma_bpb, estimate.variance_degrees_of_freedom, 0.95),
        }
        for noise_basis, sigma_bpb in sigma_bases.items():
            for effect_bpb in EFFECT_SIZES_BPB:
                level_repeats = minimum_repeats(effect_bpb, sigma_bpb)
                contrast_sigma_bpb = math.sqrt(2.0) * sigma_bpb
                contrast_repeats = minimum_repeats(effect_bpb, contrast_sigma_bpb)
                rows.append(
                    {
                        "source": estimate.source,
                        "estimand": estimate.estimand,
                        "noise_summary": estimate.summary,
                        "noise_basis": noise_basis,
                        "sigma_bpb": sigma_bpb,
                        "effect_bpb": effect_bpb,
                        "effect_role": EFFECT_SIZE_ROLES[effect_bpb],
                        "level_minimum_repeats": level_repeats,
                        "level_achieved_power": achieved_power(effect_bpb, sigma_bpb, level_repeats),
                        "contrast_sigma_bpb": contrast_sigma_bpb,
                        "contrast_minimum_repeats": contrast_repeats,
                        "contrast_achieved_power": achieved_power(effect_bpb, contrast_sigma_bpb, contrast_repeats),
                        "primary_for_design": estimate.primary_for_design,
                    }
                )
    return pd.DataFrame(rows)


def required_allocations(noise: pd.DataFrame) -> pd.DataFrame:
    neighborhood = noise[noise["source"] == "three_design_neighborhood_triples"].set_index("estimand")
    rows: list[dict[str, Any]] = []
    for estimand in ("odd", "even", "plus_oriented"):
        estimate = neighborhood.loc[estimand]
        sigma_bases = {
            "point": float(estimate["sigma_bpb"]),
            "ucl80": float(estimate["sigma_ucl80_bpb"]),
            "ucl95": float(estimate["sigma_ucl95_bpb"]),
        }
        for noise_basis, sigma_bpb in sigma_bases.items():
            contrast_sigma_bpb = math.sqrt(2.0) * sigma_bpb
            for effect_bpb in EFFECT_SIZES_BPB:
                repeats = minimum_repeats(effect_bpb, contrast_sigma_bpb)
                implied_runs = 2 * 7 * repeats
                rows.append(
                    {
                        "estimand": estimand,
                        "effect_bpb": effect_bpb,
                        "effect_role": EFFECT_SIZE_ROLES[effect_bpb],
                        "effect_mapping_is_direct": estimand == "plus_oriented",
                        "noise_basis": noise_basis,
                        "single_condition_sigma_bpb": sigma_bpb,
                        "contrast_sigma_bpb": contrast_sigma_bpb,
                        "minimum_repeats_per_condition": repeats,
                        "anchors": 2,
                        "arms_per_anchor": 7,
                        "implied_runs": implied_runs,
                        "within_200_run_envelope": implied_runs <= RUN_ENVELOPE,
                    }
                )
    return pd.DataFrame(rows)


def aligned_steps(target_tokens: float, tokens_per_step: int, alignment: int) -> int:
    raw_steps = target_tokens / tokens_per_step
    return max(alignment, round(raw_steps / alignment) * alignment)


def matched_clock_cells(design: dict[str, Any], *, target_total_tpp: float | None = None) -> pd.DataFrame:
    cells = {cell["cell_id"]: cell for cell in design["cells"]}
    base_template = cells["r0_shared_h0640_s03820"]
    h896 = cells["r1_increase_n_h0896_s03820"]
    tokens_per_step = int(design["tokens_per_step"])
    alignment = int(design["step_alignment"])

    if target_total_tpp is None:
        regime = "observed_low_tpp"
        base_steps = int(base_template["total_steps"])
    else:
        regime = "deployment_relevant_high_tpp"
        base_steps = aligned_steps(target_total_tpp * base_template["total_parameters"], tokens_per_step, alignment)
    base_tokens = base_steps * tokens_per_step
    base_total_tpp = base_tokens / base_template["total_parameters"]
    base_nonembedding_tpp = base_tokens / base_template["non_embedding_parameters"]
    candidate_specs = [
        (f"{regime}_base_h0640", base_template, base_steps, "reference"),
        (
            f"{regime}_h0896_total_tpp_match",
            h896,
            aligned_steps(base_total_tpp * h896["total_parameters"], tokens_per_step, alignment),
            "matches base total-parameter TPP",
        ),
        (
            f"{regime}_h0896_nonembedding_tpp_match",
            h896,
            aligned_steps(base_nonembedding_tpp * h896["non_embedding_parameters"], tokens_per_step, alignment),
            "matches base non-embedding TPP",
        ),
    ]
    rows: list[dict[str, Any]] = []
    for cell_id, template, steps, role in candidate_specs:
        materialized_tokens = steps * tokens_per_step
        total_tpp = materialized_tokens / template["total_parameters"]
        nonembedding_tpp = materialized_tokens / template["non_embedding_parameters"]
        rows.append(
            {
                "cell_id": cell_id,
                "regime": regime,
                "role": role,
                "hidden_size": template["hidden_size"],
                "total_steps": steps,
                "materialized_tokens": materialized_tokens,
                "total_parameters": template["total_parameters"],
                "non_embedding_parameters": template["non_embedding_parameters"],
                "embedding_parameter_fraction": (
                    1.0 - template["non_embedding_parameters"] / template["total_parameters"]
                ),
                "total_parameter_tpp": total_tpp,
                "non_embedding_parameter_tpp": nonembedding_tpp,
                "relative_total_tpp_error_vs_base": total_tpp / base_total_tpp - 1.0,
                "relative_nonembedding_tpp_error_vs_base": nonembedding_tpp / base_nonembedding_tpp - 1.0,
                "flops_per_token": template["flops_per_token"],
                "estimated_training_flops": 3.0 * template["flops_per_token"] * materialized_tokens,
            }
        )
    return pd.DataFrame(rows)


def design_envelopes(
    noise: pd.DataFrame,
    low_tpp_clock_cells: pd.DataFrame,
    high_tpp_clock_cells: pd.DataFrame,
    scale: pd.DataFrame,
) -> pd.DataFrame:
    neighborhood = noise[noise["source"] == "three_design_neighborhood_triples"].set_index("estimand")

    def contrast_mde(estimand: str, repeats: int, confidence: float | None) -> float:
        estimate = neighborhood.loc[estimand]
        sigma_bpb = float(estimate["sigma_bpb"])
        if confidence is not None:
            sigma_bpb = sigma_upper_confidence_bound(
                sigma_bpb,
                int(estimate["variance_degrees_of_freedom"]),
                confidence,
            )
        # Seeds are crossed across conditions, but zero covariance is used until measured.
        return minimum_detectable_effect(math.sqrt(2.0) * sigma_bpb, repeats)

    i2_runs = 3 * 3 * PRIMARY_I2_SEEDS

    def clock_cost(clock_cells: pd.DataFrame) -> tuple[float, float]:
        tokens = float(clock_cells["materialized_tokens"].sum() * 3 * PRIMARY_I2_SEEDS)
        flops = float(clock_cells["estimated_training_flops"].sum() * 3 * PRIMARY_I2_SEEDS)
        return tokens, flops

    low_tpp_tokens, low_tpp_flops = clock_cost(low_tpp_clock_cells)
    high_tpp_tokens, high_tpp_flops = clock_cost(high_tpp_clock_cells)

    i3_anchors = 2
    i3_switch_times = 3
    i3_arms_per_anchor = 1 + 2 * i3_switch_times
    i3_runs = i3_anchors * i3_arms_per_anchor * PRIMARY_I3_SEEDS
    two_billion = scale[np.isclose(scale["token_budget_requested"], 2_000_000_000)]
    per_run_tokens = float(two_billion["materialized_tokens"].median())
    per_run_flops = float(two_billion["estimated_training_flops"].median())

    common_power = {
        "seeds_crossed_across_conditions": True,
        "assumed_cross_condition_correlation": 0.0,
        "odd_contrast_mde_point_bpb": contrast_mde("odd", PRIMARY_I2_SEEDS, None),
        "odd_contrast_mde_ucl80_bpb": contrast_mde("odd", PRIMARY_I2_SEEDS, 0.80),
        "odd_contrast_mde_ucl95_bpb": contrast_mde("odd", PRIMARY_I2_SEEDS, 0.95),
        "even_contrast_mde_point_bpb": contrast_mde("even", PRIMARY_I2_SEEDS, None),
        "even_contrast_mde_ucl80_bpb": contrast_mde("even", PRIMARY_I2_SEEDS, 0.80),
        "even_contrast_mde_ucl95_bpb": contrast_mde("even", PRIMARY_I2_SEEDS, 0.95),
        "gain_contrast_mde_point_bpb": contrast_mde("plus_oriented", PRIMARY_I2_SEEDS, None),
        "gain_contrast_mde_ucl80_bpb": contrast_mde("plus_oriented", PRIMARY_I2_SEEDS, 0.80),
        "gain_contrast_mde_ucl95_bpb": contrast_mde("plus_oriented", PRIMARY_I2_SEEDS, 0.95),
    }
    rows = [
        {
            "design_id": "I2_low_tpp_clock_sufficiency_triangle",
            "cells_or_anchors": 3,
            "intervention_levels": 1,
            "arms_per_cell_or_anchor": 3,
            "seeds": PRIMARY_I2_SEEDS,
            "runs": i2_runs,
            "within_200_run_envelope": i2_runs <= RUN_ENVELOPE,
            "materialized_tokens": low_tpp_tokens,
            "estimated_training_flops": low_tpp_flops,
            **common_power,
            "protocol_ready": False,
            "decision": (
                "run-count feasible, but TPP 4.77-7.83 does not cover the TPP 29.83-35 failure regime; "
                "can only falsify clock sufficiency at low TPP"
            ),
        },
        {
            "design_id": "I2_high_tpp_clock_sufficiency_triangle",
            "cells_or_anchors": 3,
            "intervention_levels": 1,
            "arms_per_cell_or_anchor": 3,
            "seeds": PRIMARY_I2_SEEDS,
            "runs": i2_runs,
            "within_200_run_envelope": i2_runs <= RUN_ENVELOPE,
            "materialized_tokens": high_tpp_tokens,
            "estimated_training_flops": high_tpp_flops,
            **common_power,
            "protocol_ready": False,
            "decision": (
                "covers total TPP about 30 but costs over six times the low-TPP triangle; it provides two "
                "clock-sufficiency contrasts, not a held-cell scale-law test"
            ),
        },
    ]

    i3_power = {
        "seeds_crossed_across_conditions": True,
        "assumed_cross_condition_correlation": 0.0,
        "odd_contrast_mde_point_bpb": contrast_mde("odd", PRIMARY_I3_SEEDS, None),
        "odd_contrast_mde_ucl80_bpb": contrast_mde("odd", PRIMARY_I3_SEEDS, 0.80),
        "odd_contrast_mde_ucl95_bpb": contrast_mde("odd", PRIMARY_I3_SEEDS, 0.95),
        "even_contrast_mde_point_bpb": contrast_mde("even", PRIMARY_I3_SEEDS, None),
        "even_contrast_mde_ucl80_bpb": contrast_mde("even", PRIMARY_I3_SEEDS, 0.80),
        "even_contrast_mde_ucl95_bpb": contrast_mde("even", PRIMARY_I3_SEEDS, 0.95),
        "gain_contrast_mde_point_bpb": contrast_mde("plus_oriented", PRIMARY_I3_SEEDS, None),
        "gain_contrast_mde_ucl80_bpb": contrast_mde("plus_oriented", PRIMARY_I3_SEEDS, 0.80),
        "gain_contrast_mde_ucl95_bpb": contrast_mde("plus_oriented", PRIMARY_I3_SEEDS, 0.95),
    }
    rows.append(
        {
            "design_id": "I3_two_anchor_three_switch",
            "cells_or_anchors": i3_anchors,
            "intervention_levels": i3_switch_times,
            "arms_per_cell_or_anchor": i3_arms_per_anchor,
            "seeds": PRIMARY_I3_SEEDS,
            "runs": i3_runs,
            "within_200_run_envelope": i3_runs <= RUN_ENVELOPE,
            "materialized_tokens": i3_runs * per_run_tokens,
            "estimated_training_flops": i3_runs * per_run_flops,
            **i3_power,
            "protocol_ready": False,
            "decision": (
                "preferred before I2 because it can identify Psi, but the current seed allocation does not power "
                "a 0.0028-BPB precommitted oriented-gain contrast even at the point variance estimate"
            ),
        }
    )
    return pd.DataFrame(rows)


def render_report(
    noise: pd.DataFrame,
    power: pd.DataFrame,
    clock_cells: pd.DataFrame,
    envelopes: pd.DataFrame,
    allocations: pd.DataFrame,
    antithetic: pd.DataFrame,
    protocol_hash: str,
) -> str:
    source_mask = power["source"] == "three_design_neighborhood_triples"
    selected_power = power[source_mask & power["primary_for_design"]]
    return f"""# Two-Phase Intervention Power Audit v2

Protocol: `{protocol_hash}`

## Decision

No proposed intervention is protocol-ready.

- **I3 precedes I2.** The switch-time intervention can identify the temporal
  response `Psi`; the clock triangle can only test whether an already identified
  response is sufficient across cells. The current 182-run I3 allocation does
  not resolve the precommitted oriented-gain contrast at the 0.0028-BPB
  shrinkage target even under the point variance estimate, and it does not
  resolve both odd and even between-switch contrasts at 0.0028 BPB under the
  **95% variance upper limit**.
- **The low-TPP I2 triangle is in the wrong regime.** Its total TPP range is
  4.77-7.83, versus approximately 29.83-35.3 where the present shared-surrogate
  failure matters. It can falsify clock sufficiency only at low TPP.
- **The high-TPP I2 triangle is expensive but still limited.** It reaches total
  TPP about 30 and remains 180 runs, but costs over six times the low-TPP design.
  Three cells provide two clock-sufficiency contrasts, not a held-cell scale-law
  test and not causal identification of total versus non-embedding TPP.
- **No experiment is submitted by this artifact.** No two-anchor,
  three-switch I3 allocation within 200 runs powers a 0.0028-BPB
  precommitted-oriented-gain change even under the point variance estimate.
  Raising the budget or reducing anchors/switch times is a scientific scope
  change that requires a new protocol. Do not add `Phi(TPP)` to a surrogate
  from the exposed cross-cell outcomes or either unrun triangle.

## Noise estimates

{noise.to_markdown(index=False, floatfmt=".6f")}

The often-repeated description of the measured fiber as having 63 coordinates
with five seeds is incorrect. It had 63 reference-seed fiber coordinates, but
only 11 coordinates have five seeds. Direct odd/even power is instead based on
10 complete antithetic triples with five same-seed observations each. The even
estimand reuses tied controls, leaving eight coordinate-level control clusters,
but every coordinate reuses the same five seeds. Variance upper limits therefore
use four seed-level degrees of freedom, including for the three
design-neighborhood triples. This is deliberately conservative.

`oracle_net = min(L(+d), L(-d)) - L(0)` is descriptive only. Taking the minimum
inside each seed is downward biased near a null and can manufacture apparent
gain. Odd and even effects are the mechanistic component estimands;
`plus_oriented = L(+d)-L(0)` is the precommitted-orientation gain estimand.

## Power table

Two-sided alpha is {ALPHA:.2f}; target power is {POWER:.2f}. Effects are frozen
at 0.0039 BPB, a 0.0028-BPB shrinkage target, and the lower endpoint 0.001545
BPB from the replicated 2B confidence interval. The 0.0028 target is an
explicit design sensitivity point, not an inferential bound. Point, 80%
upper-limit, and 95% upper-limit variance bases are all reported. Both level and
between-condition repeat requirements appear; intervention decisions use the
latter.

{selected_power.to_markdown(index=False, floatfmt=".6f")}

## Candidate matched-clock cells

{clock_cells.to_markdown(index=False, floatfmt=".6f")}

The triangle uses the same architecture family but changes embedding fraction:
one h896 cell matches the h640 cell on total-parameter TPP, while another
matches it on non-embedding TPP. The h896 pair then varies token horizon and
optimizer steps at fixed architecture. This separates more aliases than the
current grid, but it cannot identify token horizon independently of steps
without a batch-size intervention. The architecture changes embedding fraction,
and all transported variance estimates remain assumptions rather than measured
properties of these cells. The high triangle is relevant to the 300M regime
under total-parameter TPP only: its non-embedding TPP range is approximately
83.7-137.3, above the 300M value 58.45.

## Designs within the one-day envelope

{envelopes.to_markdown(index=False, floatfmt=".6g")}

## Required I3 allocations

The following table holds the two-anchor, seven-arm shape fixed. The three
oriented-gain effect sizes map directly only to `plus_oriented`. Odd and even
rows are sensitivity calculations because no observed gain magnitude determines
how a transition change splits across those components.

{allocations.to_markdown(index=False, floatfmt=".6g")}

The unshrunk selected 0.0039-BPB gain is feasible at the point variance estimate
(154 runs). The negative 200-run conclusion follows from requiring the
0.0028-BPB shrinkage target rather than trusting that selected point estimate.
It is also conservative in another explicit way: a same-seed difference between
two switch conditions cancels the shared tied-control term algebraically, while
the current calculation transports policy-minus-tied SD and assumes zero
cross-condition covariance. No archive row identifies that covariance, so it
cannot be used to reduce the preregistered run count.

## Antithetic source triples

{antithetic.to_markdown(index=False, floatfmt=".6f")}

## Statistical commitments

1. Odd and even effects are estimated within seed and coordinate before
   aggregation. Per-seed best-orientation gain is not a primary estimand.
2. Intervals and tests cluster by seed and intervention cell; a shared tied
   control never crosses an outer fold.
3. Seeds are crossed across intervention conditions. Until their covariance is
   measured, between-condition power assumes zero covariance and multiplies the
   single-condition SD by `sqrt(2)`.
4. Power is checked at 0.0039, 0.0028, and 0.001545 BPB and under point, 80%,
   and 95% variance upper limits. These magnitudes map directly to the
   precommitted oriented-gain estimand, not automatically to odd or even.
   The selected 0.0039-BPB gain clears Holm correction over four primary
   anchors (`p=0.0261`) but not over all twelve repeated arms (`p=0.1194`).
5. I2 can falsify clock sufficiency but cannot identify a clock from three
   cells. Any future clock must predict a held-out cell's gain magnitude and raw
   optimum location; directional agreement or RMSE alone is insufficient.
6. On WSD80, freeze `mixture_blocked_folds`; require Programming-Languages gain
   error within the observed 0.004439-BPB worst plus-oriented repeated-triple SD
   (five seeds, four variance degrees of freedom), optimum distance at most
   0.05, and broad-text gain at most 0.005 BPB.
7. On 300M, one accumulated development look passes only if the candidate
   improves a phase-sensitive selection diagnostic beyond a
   correspondence-key clustered paired-bootstrap interval on one target,
   preserves the other target, and keeps core OOF RMSE within 5% of HPR.
8. A near-tied 300M raw optimum is admissible: none of 238 trained asymmetric
   policies beats the best trained tied policy on either target.
"""


def main() -> None:
    source_paths = [
        MEASURED_FIBER_PATH,
        GLOBAL_FIBER_PATH,
        SCALE_FIBER_PATH,
        CONFIRMATION_PATH,
        MATCHED_ND_DESIGN_PATH,
    ]
    for path in source_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    measured = pd.read_csv(MEASURED_FIBER_PATH)
    global_fiber = pd.read_csv(GLOBAL_FIBER_PATH)
    scale = pd.read_csv(SCALE_FIBER_PATH)
    confirmation = pd.read_csv(CONFIRMATION_PATH)
    matched_nd_design = json.loads(MATCHED_ND_DESIGN_PATH.read_text())

    raw_estimates, repeated_detail = repeated_coordinate_noise(measured)
    paired_estimates, paired_detail = historical_paired_noise(global_fiber)
    antithetic_estimates, antithetic_detail = antithetic_noise(measured, scale)
    estimates = raw_estimates + paired_estimates + antithetic_estimates + confirmation_noise(confirmation)
    noise = pd.DataFrame([estimate.__dict__ for estimate in estimates])
    noise["sigma_ucl80_bpb"] = noise.apply(
        lambda row: sigma_upper_confidence_bound(float(row["sigma_bpb"]), int(row["variance_degrees_of_freedom"]), 0.80),
        axis=1,
    )
    noise["sigma_ucl95_bpb"] = noise.apply(
        lambda row: sigma_upper_confidence_bound(float(row["sigma_bpb"]), int(row["variance_degrees_of_freedom"]), 0.95),
        axis=1,
    )
    power = power_table(estimates)
    allocations = required_allocations(noise)
    low_tpp_clock_cells = matched_clock_cells(matched_nd_design)
    high_tpp_clock_cells = matched_clock_cells(matched_nd_design, target_total_tpp=30.0)
    clock_cells = pd.concat([low_tpp_clock_cells, high_tpp_clock_cells], ignore_index=True)
    envelopes = design_envelopes(noise, low_tpp_clock_cells, high_tpp_clock_cells, scale)

    source_files = {str(path.relative_to(REPO_ROOT)): sha256_path(path) for path in source_paths}
    protocol = {
        "version": "2026-08-04-v2",
        "purpose": "outcome-free power and feasibility audit; no surrogate fit or sealed-panel read",
        "alpha": ALPHA,
        "power": POWER,
        "effect_sizes_bpb": EFFECT_SIZES_BPB,
        "effect_size_roles": EFFECT_SIZE_ROLES,
        "primary_estimands": ["odd", "even", "plus_oriented"],
        "oriented_gain_estimand": "plus_oriented",
        "oracle_net_role": "descriptive_only_due_per_seed_min_bias",
        "seeds_crossed_across_conditions": True,
        "assumed_cross_condition_correlation": 0.0,
        "sigma_upper_confidence_levels": [0.80, 0.95],
        "multiplicity_context": {
            "holm_p_over_four_primary_anchors": 0.0261,
            "holm_p_over_twelve_repeated_arms": 0.1194,
        },
        "high_tpp_triangle_target_total_tpp": 30.0,
        "run_envelope": RUN_ENVELOPE,
        "primary_i2_seeds": PRIMARY_I2_SEEDS,
        "primary_i3_seeds": PRIMARY_I3_SEEDS,
        "source_files": source_files,
    }
    protocol_hash = canonical_json_sha256(protocol)
    protocol["protocol_sha256"] = protocol_hash

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "protocol.json").write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    noise.to_csv(OUTPUT_DIR / "noise_estimates.csv", index=False)
    power.to_csv(OUTPUT_DIR / "power_table.csv", index=False)
    allocations.to_csv(OUTPUT_DIR / "required_allocations.csv", index=False)
    repeated_detail.to_csv(OUTPUT_DIR / "repeated_coordinate_noise.csv", index=False)
    paired_detail.to_csv(OUTPUT_DIR / "paired_delta_noise.csv", index=False)
    antithetic_detail.to_csv(OUTPUT_DIR / "antithetic_triples.csv", index=False)
    clock_cells.to_csv(OUTPUT_DIR / "matched_clock_cells.csv", index=False)
    envelopes.to_csv(OUTPUT_DIR / "design_envelopes.csv", index=False)

    decision = {
        "protocol_sha256": protocol_hash,
        "status": "no_compliant_200_run_intervention",
        "designs": {row["design_id"]: row for row in envelopes.to_dict(orient="records")},
        "preferred_order": ["I3_two_anchor_three_switch", "I2_high_tpp_clock_sufficiency_triangle"],
        "next_action": (
            "choose explicitly between a larger I3 budget and a reduced-anchor or reduced-switch scope, then "
            "freeze a new protocol; defer I2 and any Phi(TPP) model term"
        ),
    }
    (OUTPUT_DIR / "decision.json").write_text(json.dumps(decision, indent=2, sort_keys=True) + "\n")
    (OUTPUT_DIR / "report.md").write_text(
        render_report(noise, power, clock_cells, envelopes, allocations, antithetic_detail, protocol_hash)
    )
    print(json.dumps(decision, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
