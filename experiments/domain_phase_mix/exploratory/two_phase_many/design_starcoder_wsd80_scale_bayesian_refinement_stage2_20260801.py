# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
#   "tabulate",
#   "wandb",
# ]
# ///

"""Freeze the second-stage spatial refinement and paired confirmations for WSD80."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
import scipy
import sklearn
import wandb
from marin.execution.lazy import lower
from scipy import stats
from scipy.spatial import ConvexHull, Delaunay
from scipy.stats import rankdata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_fixed_model_token_scaling as scaling_launcher
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_scale_bayesian_refinement as stage1_launcher
from experiments.domain_phase_mix import launch_starcoder_wsd_80_20_surface as base_launcher
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_starcoder_wsd80_scale_bayesian_refinement_20260731 as stage1,
)
from experiments.domain_phase_mix.exploratory.two_phase_many import (
    starcoder_wsd80_training_identity as stream_identity,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[3]
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_scale_bayesian_refinement_20260731" / "stage2_design_20260801"
FROZEN_LAUNCH_DESIGN = SCRIPT_DIR.parents[1] / "starcoder_wsd80_scale_bayesian_refinement_stage2_design_20260801.json"
STAGE2_LAUNCHER_PATH = SCRIPT_DIR.parents[1] / "launch_starcoder_wsd_80_20_scale_bayesian_refinement_stage2.py"
STAGE2_ANALYZER_PATH = SCRIPT_DIR / "analyze_starcoder_wsd80_scale_bo_stage2_20260801.py"
STAGE1_DESIGN_PATH = SCRIPT_DIR.parents[1] / "starcoder_wsd80_scale_bayesian_refinement_design_20260731.json"
STAGE1_RESULTS_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_scale_bayesian_refinement_20260731" / "results_20260801"
STAGE1_OBSERVATIONS_PATH = STAGE1_RESULTS_DIR / "stage1_observations.csv"

REFERENCE_SEED = stage1.REFERENCE_SEED
EXISTING_CONFIRMATION_SEEDS = stage1.JOINT_RANDOMNESS_SEEDS
NEW_CONFIRMATION_SEEDS = (20_260_801, 20_260_802, 20_260_803, 20_260_804)
ACQUISITIONS_BY_REGION = {
    (2_000_000_000, "c10"): 6,
    (4_000_000_000, "low_aggregate"): 4,
}
EXPECTED_ACQUISITIONS = sum(ACQUISITIONS_BY_REGION.values())
EXPECTED_CONFIRMATIONS = 2 * len(EXISTING_CONFIRMATION_SEEDS + NEW_CONFIRMATION_SEEDS)
EXPECTED_RUNS = EXPECTED_ACQUISITIONS + EXPECTED_CONFIRMATIONS
LAUNCH_MANIFEST_FIELDS = (
    "run_name",
    "token_budget_requested",
    "phase_0_starcoder",
    "phase_1_starcoder",
    "boundary_step",
    "trainer_data_seed",
    "simulated_epoch_subset_seed",
    "run_kind",
    "region",
    "replicate_kind",
    "comparison_role",
    "pair_seed",
    "pair_arm",
    "comparison_source",
)

CONFIRMATION_COORDINATE = {
    "token_budget_requested": 8_000_000_000,
    "phase_0_starcoder": 0.07,
    "phase_1_starcoder": 0.87,
    "incumbent_phase_0": 0.02,
    "incumbent_phase_1": 0.82,
    "stage1_reference_gain_bpb": 0.002185,
}
CONFIRMATION_BUDGET = int(CONFIRMATION_COORDINATE["token_budget_requested"])
TRAIN_PROJECT = "marin-community/marin"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def launch_manifest(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Project a design onto fields that determine training or frozen inference strata."""
    return [{field: row.get(field) for field in LAUNCH_MANIFEST_FIELDS} for row in rows]


def _frame(
    *,
    token_budget: pd.Series | int,
    p0: pd.Series,
    p1: pd.Series,
    bpb: pd.Series,
    run_id: pd.Series,
    source: str,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "token_budget_requested": token_budget,
            "p0": pd.to_numeric(p0, errors="raise"),
            "p1": pd.to_numeric(p1, errors="raise"),
            "bpb": pd.to_numeric(bpb, errors="raise"),
            "run_id": run_id.astype(str),
            "source": source,
        }
    )


def load_reference_seed_observations() -> pd.DataFrame:
    """Load only the common reference-seed block used for spatial acquisition."""
    scaling = pd.read_csv(stage1.TOKEN_SCALING_PATH)
    tied = pd.read_csv(stage1.TIED_DIAGONAL_PATH)
    fibers = pd.read_csv(stage1.SCALE_FIBERS_PATH)
    stage1_outcomes = pd.read_csv(STAGE1_OBSERVATIONS_PATH)
    acquisition_budgets = {token_budget for token_budget, _ in ACQUISITIONS_BY_REGION}

    scaling = scaling.loc[
        scaling["trainer_data_seed"].eq(REFERENCE_SEED) & scaling["simulated_epoch_subset_seed"].eq(REFERENCE_SEED)
    ]
    tied = tied.loc[tied["token_budget_requested"].isin(acquisition_budgets)].copy()
    if not tied["run_name"].str.contains(r"_ref_s20260711$", regex=True).all():
        raise ValueError("A tied-diagonal spatial-fit row is not from the reference seed")
    fibers = fibers.loc[
        fibers["trainer_data_seed"].eq(REFERENCE_SEED) & fibers["simulated_epoch_subset_seed"].eq(REFERENCE_SEED)
    ]
    acquisitions = stage1_outcomes.loc[
        stage1_outcomes["run_kind"].eq("acquisition")
        & stage1_outcomes["trainer_data_seed"].eq(REFERENCE_SEED)
        & stage1_outcomes["simulated_epoch_subset_seed"].eq(REFERENCE_SEED)
    ]
    if len(acquisitions) != 40:
        raise ValueError(f"Expected 40 completed reference-seed Stage-1 acquisitions, found {len(acquisitions)}")

    observations = pd.concat(
        [
            _frame(
                token_budget=scaling["token_budget_requested"],
                p0=scaling["phase_0_starcoder"],
                p1=scaling["phase_1_starcoder"],
                bpb=scaling["starcoder_bpb"],
                run_id=scaling["training_wandb_id"],
                source="pre_stage1_scaling_reference",
            ),
            _frame(
                token_budget=tied["token_budget_requested"],
                p0=tied["weight"],
                p1=tied["weight"],
                bpb=tied["starcoder_bpb"],
                run_id=tied["wandb_id"],
                source="pre_stage1_tied_reference",
            ),
            _frame(
                token_budget=fibers["token_budget_requested"],
                p0=fibers["phase_0_starcoder"],
                p1=fibers["phase_1_starcoder"],
                bpb=fibers["starcoder_bpb"],
                run_id=fibers["wandb_id"],
                source="pre_stage1_fiber_reference",
            ),
            _frame(
                token_budget=acquisitions["token_budget_requested"],
                p0=acquisitions["phase_0_starcoder"],
                p1=acquisitions["phase_1_starcoder"],
                bpb=acquisitions["starcoder_bpb"],
                run_id=acquisitions["wandb_id"],
                source="stage1_acquisition_reference",
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("run_id")
    if not np.isfinite(observations[["p0", "p1", "bpb"]].to_numpy(dtype=float)).all():
        raise ValueError("Reference-seed observation table contains non-finite values")
    return observations.reset_index(drop=True)


def coordinate_summary(observations: pd.DataFrame, token_budget: int) -> pd.DataFrame:
    """Collapse duplicate reference-seed evidence at a policy coordinate."""
    selected = observations.loc[observations["token_budget_requested"].eq(token_budget)].copy()
    selected["p0_key"] = selected["p0"].round(8)
    selected["p1_key"] = selected["p1"].round(8)
    return (
        selected.groupby(["p0_key", "p1_key"], as_index=False)["bpb"]
        .agg(mean="mean", variance="var", count="count")
        .sort_values(["p0_key", "p1_key"])
        .reset_index(drop=True)
    )


def spatial_noise_estimates() -> tuple[pd.DataFrame, dict[int, float]]:
    """Estimate per-run noise from fresh-seed coordinate residuals."""
    fibers = pd.read_csv(stage1.SCALE_FIBERS_PATH)
    required = {
        "token_budget_requested",
        "phase_0_starcoder",
        "phase_1_starcoder",
        "trainer_data_seed",
        "simulated_epoch_subset_seed",
        "starcoder_bpb",
    }
    if not required.issubset(fibers.columns):
        raise ValueError(f"Fiber observations lack required columns: {sorted(required - set(fibers.columns))}")

    coordinate_columns = ["phase_0_starcoder", "phase_1_starcoder"]
    required_budgets = {token_budget for token_budget, _ in ACQUISITIONS_BY_REGION} | {CONFIRMATION_BUDGET}
    fresh = fibers.loc[
        fibers["token_budget_requested"].isin(required_budgets)
        & fibers["trainer_data_seed"].isin(EXISTING_CONFIRMATION_SEEDS)
        & fibers["trainer_data_seed"].eq(fibers["simulated_epoch_subset_seed"]),
        ["token_budget_requested", *coordinate_columns, "trainer_data_seed", "starcoder_bpb"],
    ].rename(columns={"trainer_data_seed": "seed"})

    rows: list[dict[str, object]] = []
    noise_by_budget: dict[int, float] = {}
    for token_budget, group in fresh.groupby("token_budget_requested", sort=True):
        pivot = group.pivot(index="seed", columns=coordinate_columns, values="starcoder_bpb")
        if pivot.isna().any().any():
            raise ValueError(f"Token budget {token_budget} does not form a complete fresh seed-by-coordinate panel")
        values = pivot.to_numpy(dtype=float)
        residuals = values - values.mean(axis=1, keepdims=True) - values.mean(axis=0, keepdims=True) + values.mean()
        seed_count, coordinate_count = values.shape
        degrees_of_freedom = int((seed_count - 1) * (coordinate_count - 1))
        if degrees_of_freedom <= 0:
            raise ValueError(f"Token budget {token_budget} has no two-way residual degrees of freedom")
        residual_sum_squares = float(np.square(residuals).sum())
        noise_sd = float(np.sqrt(residual_sum_squares / degrees_of_freedom))
        budget = int(token_budget)
        noise_by_budget[budget] = noise_sd
        rows.append(
            {
                "token_budget_requested": budget,
                "fresh_seeds": seed_count,
                "coordinates": coordinate_count,
                "fresh_observations": int(values.size),
                "degrees_of_freedom": degrees_of_freedom,
                "residual_sum_squares": residual_sum_squares,
                "spatial_noise_sd": noise_sd,
                "estimator": "fresh-only two-way seed-by-coordinate additive ANOVA residual SD",
                "role": "estimator_input",
            }
        )
    expected_budgets = required_budgets
    if not expected_budgets.issubset(noise_by_budget):
        raise ValueError(f"Missing spatial-noise estimates for {sorted(expected_budgets - set(noise_by_budget))}")
    reference = fibers.loc[
        fibers["token_budget_requested"].eq(CONFIRMATION_BUDGET)
        & fibers["trainer_data_seed"].eq(REFERENCE_SEED)
        & fibers["simulated_epoch_subset_seed"].eq(REFERENCE_SEED),
        [*coordinate_columns, "starcoder_bpb"],
    ].rename(columns={"starcoder_bpb": "reference_bpb"})
    fresh_8b = fresh.loc[fresh["token_budget_requested"].eq(CONFIRMATION_BUDGET)].merge(
        reference, on=coordinate_columns, how="inner", validate="many_to_one"
    )
    fresh_8b["fresh_minus_reference"] = fresh_8b["starcoder_bpb"] - fresh_8b["reference_bpb"]
    seed_effect = fresh_8b.groupby("seed")["fresh_minus_reference"].mean()
    stage1_outcomes = pd.read_csv(STAGE1_OBSERVATIONS_PATH)
    incumbent_fresh = stage1_outcomes.loc[
        stage1_outcomes["run_kind"].eq("incumbent_repeat")
        & stage1_outcomes["token_budget_requested"].eq(CONFIRMATION_BUDGET)
        & np.isclose(stage1_outcomes["phase_0_starcoder"], CONFIRMATION_COORDINATE["incumbent_phase_0"])
        & np.isclose(stage1_outcomes["phase_1_starcoder"], CONFIRMATION_COORDINATE["incumbent_phase_1"]),
        ["trainer_data_seed", "starcoder_bpb"],
    ].set_index("trainer_data_seed")["starcoder_bpb"]
    reference_observations = load_reference_seed_observations()
    incumbent_reference_rows = reference_observations.loc[
        reference_observations["token_budget_requested"].eq(CONFIRMATION_BUDGET)
        & np.isclose(reference_observations["p0"], CONFIRMATION_COORDINATE["incumbent_phase_0"])
        & np.isclose(reference_observations["p1"], CONFIRMATION_COORDINATE["incumbent_phase_1"])
    ]
    if len(incumbent_reference_rows) != 1:
        raise ValueError("Expected one reference-seed 8B incumbent observation")
    incumbent_reference = float(incumbent_reference_rows.iloc[0]["bpb"])
    corrected_incumbent = incumbent_fresh - incumbent_reference - seed_effect
    if set(corrected_incumbent.index) != set(EXISTING_CONFIRMATION_SEEDS):
        raise ValueError("Low-aggregate noise cross-check does not cover all fresh seeds")
    rows.append(
        {
            "token_budget_requested": CONFIRMATION_BUDGET,
            "fresh_seeds": len(corrected_incumbent),
            "coordinates": 1,
            "fresh_observations": len(corrected_incumbent),
            "degrees_of_freedom": len(corrected_incumbent) - 1,
            "residual_sum_squares": float(np.square(corrected_incumbent - corrected_incumbent.mean()).sum()),
            "spatial_noise_sd": float(corrected_incumbent.std(ddof=1)),
            "estimator": "low-aggregate incumbent residual after external fiber seed-effect correction",
            "role": "extrapolation_validation_only",
        }
    )
    return pd.DataFrame(rows), noise_by_budget


def stage2_trust_regions() -> tuple[stage1.TrustRegion, ...]:
    """Return only Stage-2 regions, with their actual acquisition counts."""
    regions = []
    for original in stage1.TRUST_REGIONS:
        count = ACQUISITIONS_BY_REGION.get((original.token_budget, original.name), 0)
        if count == 0:
            continue
        regions.append(
            stage1.TrustRegion(
                token_budget=original.token_budget,
                name=original.name,
                p0_min=original.p0_min,
                p0_max=original.p0_max,
                p1_min=original.p1_min,
                p1_max=original.p1_max,
                count=count,
                rationale=f"Stage-2 update of {original.rationale}",
            )
        )
    return tuple(regions)


def select_region(
    summary: pd.DataFrame,
    noise_sd: float,
    region: stage1.TrustRegion,
    extra_exclusions: np.ndarray,
) -> list[dict[str, object]]:
    """Select a diverse EI batch with observation variances in normalized-response units."""
    margin = stage1.TRAINING_MARGIN
    training = summary.loc[
        summary["p0_key"].between(max(0.0, region.p0_min - margin), min(1.0, region.p0_max + margin))
        & summary["p1_key"].between(max(0.0, region.p1_min - margin), min(1.0, region.p1_max + margin))
    ].copy()
    if len(training) < 4:
        raise ValueError(f"Trust region {region.name} at {region.token_budget} has only {len(training)} training points")
    x_train = training[["p0_key", "p1_key"]].to_numpy(dtype=float)
    y_train = training["mean"].to_numpy(dtype=float)
    response_sd = float(np.std(y_train, ddof=0))
    if response_sd <= 0.0:
        raise ValueError(f"Trust region {region.name} has zero response variance")
    if not training["count"].eq(1).all():
        raise ValueError("Reference-seed spatial fits require one observation per coordinate")
    alpha_raw = np.full(len(training), noise_sd**2)
    alpha_normalized = alpha_raw / response_sd**2
    candidates = stage1._grid(region)
    expected_improvements = []
    means = []
    standard_deviations = []
    for length_scale in stage1.KERNEL_LENGTH_SCALES:
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=length_scale,
            length_scale_bounds="fixed",
            nu=2.5,
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha_normalized,
            normalize_y=True,
            optimizer=None,
        ).fit(x_train, y_train)
        mean, sd = cast(tuple[np.ndarray, np.ndarray], model.predict(candidates, return_std=True))
        best = float(model.predict(x_train).min())
        means.append(mean)
        standard_deviations.append(sd)
        expected_improvements.append(stage1._expected_improvement(mean, sd, best))

    rank_fraction = np.mean(
        [rankdata(-values, method="average") / len(values) for values in expected_improvements], axis=0
    )
    hull = ConvexHull(x_train)
    inside_local_hull = Delaunay(x_train).find_simplex(candidates, tol=1e-12) >= 0
    hull_norms = np.linalg.norm(hull.equations[:, :-1], axis=1)
    hull_facet_distances = np.min(
        -(candidates @ hull.equations[:, :-1].T + hull.equations[:, -1]) / hull_norms,
        axis=1,
    )
    rank_fraction[~inside_local_hull] = 2.0
    existing = summary[["p0_key", "p1_key"]].to_numpy(dtype=float)
    rank_fraction[stage1._distance_to(candidates, existing) < stage1.EXISTING_EXCLUSION_RADIUS] = 2.0
    if len(extra_exclusions):
        rank_fraction[stage1._distance_to(candidates, extra_exclusions) < stage1.BATCH_EXCLUSION_RADIUS] = 2.0

    selected_indices: list[int] = []
    for _ in range(region.count):
        eligible = np.ones(len(candidates), dtype=bool)
        if selected_indices:
            eligible &= stage1._distance_to(candidates, candidates[selected_indices]) >= stage1.BATCH_EXCLUSION_RADIUS
        index = int(np.where(eligible, rank_fraction, 2.0).argmin())
        if rank_fraction[index] >= 2.0:
            raise ValueError(f"Trust region {region.name} cannot supply {region.count} diverse candidates")
        selected_indices.append(index)

    selected_coordinates = candidates[selected_indices]
    rows = []
    for local_rank, index in enumerate(selected_indices, start=1):
        other_selected = np.delete(selected_coordinates, local_rank - 1, axis=0)
        nearest_selected = (
            float(stage1._distance_to(candidates[index : index + 1], other_selected)[0]) if len(other_selected) else None
        )
        distance_to_box_edge = min(
            float(candidates[index, 0] - region.p0_min),
            float(region.p0_max - candidates[index, 0]),
            float(candidates[index, 1] - region.p1_min),
            float(region.p1_max - candidates[index, 1]),
        )
        row: dict[str, object] = {
            "token_budget_requested": region.token_budget,
            "region": region.name,
            "run_kind": "acquisition",
            "replicate_kind": "reference",
            "trainer_data_seed": REFERENCE_SEED,
            "simulated_epoch_subset_seed": REFERENCE_SEED,
            "phase_0_starcoder": round(float(candidates[index, 0]), 6),
            "phase_1_starcoder": round(float(candidates[index, 1]), 6),
            "region_acquisition_rank": local_rank,
            "committee_ei_rank_fraction": float(rank_fraction[index]),
            "committee_mean_bpb": float(np.mean([values[index] for values in means])),
            "committee_mean_sd": float(np.mean([values[index] for values in standard_deviations])),
            "noise_sd": noise_sd,
            "gp_response_sd": response_sd,
            "gp_alpha_raw_min": float(alpha_raw.min()),
            "gp_alpha_raw_max": float(alpha_raw.max()),
            "gp_alpha_normalized_min": float(alpha_normalized.min()),
            "gp_alpha_normalized_max": float(alpha_normalized.max()),
            "region_training_coordinates": len(training),
            "inside_local_training_convex_hull": bool(inside_local_hull[index]),
            "distance_to_region_box_edge": distance_to_box_edge,
            "distance_to_local_hull_facet": float(hull_facet_distances[index]),
            "nearest_existing_coordinate_distance": float(
                stage1._distance_to(candidates[index : index + 1], existing)[0]
            ),
            "nearest_selected_batch_distance": nearest_selected,
            "batch_spacing_constraint_active": bool(
                nearest_selected is not None
                and nearest_selected <= stage1.BATCH_EXCLUSION_RADIUS + stage1.GRID_STEP / 100.0
            ),
        }
        for kernel_index, values in enumerate(expected_improvements, start=1):
            row[f"expected_improvement_kernel_{kernel_index}"] = float(values[index])
        rows.append(row)
    return rows


def _weight_slug(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def _budget_slug(token_budget: int) -> str:
    return f"{token_budget // 1_000_000_000}b"


def confirmation_design_diagnostics(
    observations: pd.DataFrame,
    noise_table: pd.DataFrame,
    noise_by_budget: dict[int, float],
) -> dict[str, object]:
    """Quantify selection inflation and the frozen eight-pair confirmation sensitivity."""
    stage1_outcomes = pd.read_csv(STAGE1_OBSERVATIONS_PATH)
    low_aggregate = stage1_outcomes.loc[
        stage1_outcomes["token_budget_requested"].eq(CONFIRMATION_BUDGET)
        & stage1_outcomes["run_kind"].eq("acquisition")
        & stage1_outcomes["region"].eq("low_aggregate")
        & stage1_outcomes["trainer_data_seed"].eq(REFERENCE_SEED)
        & stage1_outcomes["simulated_epoch_subset_seed"].eq(REFERENCE_SEED)
    ].sort_values("starcoder_bpb")
    if len(low_aggregate) != 5:
        raise ValueError(f"Expected five 8B low-aggregate Stage-1 acquisitions, found {len(low_aggregate)}")
    candidate = low_aggregate.iloc[0]
    if not (
        np.isclose(candidate["phase_0_starcoder"], CONFIRMATION_COORDINATE["phase_0_starcoder"])
        and np.isclose(candidate["phase_1_starcoder"], CONFIRMATION_COORDINATE["phase_1_starcoder"])
    ):
        raise ValueError("Frozen confirmation coordinate is not the best 8B low-aggregate Stage-1 acquisition")
    incumbent_rows = observations.loc[
        observations["token_budget_requested"].eq(CONFIRMATION_BUDGET)
        & np.isclose(observations["p0"], CONFIRMATION_COORDINATE["incumbent_phase_0"])
        & np.isclose(observations["p1"], CONFIRMATION_COORDINATE["incumbent_phase_1"])
    ]
    if len(incumbent_rows) != 1:
        raise ValueError("Expected one reference-seed 8B incumbent observation")
    incumbent_bpb = float(incumbent_rows.iloc[0]["bpb"])
    selected_gain = incumbent_bpb - float(candidate["starcoder_bpb"])
    if not np.isclose(selected_gain, CONFIRMATION_COORDINATE["stage1_reference_gain_bpb"], atol=5e-7):
        raise ValueError("Hand-frozen Stage-1 reference gain does not match the source observations")
    runner_up_margin = float(low_aggregate.iloc[1]["starcoder_bpb"] - candidate["starcoder_bpb"])
    low_aggregate_batch_mean_gain = incumbent_bpb - float(low_aggregate["starcoder_bpb"].mean())

    pair_count = len(EXISTING_CONFIRMATION_SEEDS + NEW_CONFIRMATION_SEEDS)
    paired_sd = float(np.sqrt(2.0) * noise_by_budget[CONFIRMATION_BUDGET])
    degrees_of_freedom = pair_count - 1
    critical = float(stats.t.ppf(0.975, degrees_of_freedom))
    half_width = critical * paired_sd / np.sqrt(pair_count)

    def two_sided_power(effect: float) -> float:
        noncentrality = abs(effect) * np.sqrt(pair_count) / paired_sd
        return float(
            stats.nct.cdf(-critical, degrees_of_freedom, -noncentrality)
            + stats.nct.sf(critical, degrees_of_freedom, -noncentrality)
        )

    crosscheck = noise_table.loc[noise_table["role"].eq("extrapolation_validation_only"), "spatial_noise_sd"]
    if len(crosscheck) != 1:
        raise ValueError("Expected one low-aggregate noise cross-check")
    crosscheck_noise_sd = float(crosscheck.iloc[0])
    crosscheck_paired_sd = float(np.sqrt(2.0) * crosscheck_noise_sd)

    def crosscheck_power(effect: float) -> float:
        noncentrality = abs(effect) * np.sqrt(pair_count) / crosscheck_paired_sd
        return float(
            stats.nct.cdf(-critical, degrees_of_freedom, -noncentrality)
            + stats.nct.sf(critical, degrees_of_freedom, -noncentrality)
        )

    return {
        "reference_incumbent_bpb": incumbent_bpb,
        "selected_candidate_bpb": float(candidate["starcoder_bpb"]),
        "selected_reference_gain_bpb": selected_gain,
        "runner_up_margin_bpb": runner_up_margin,
        "mean_low_aggregate_acquisition_bpb": float(low_aggregate["starcoder_bpb"].mean()),
        "low_aggregate_batch_mean_gain_bpb": low_aggregate_batch_mean_gain,
        "pair_count": pair_count,
        "anticipated_paired_sd_bpb": paired_sd,
        "anticipated_ci95_half_width_bpb": half_width,
        "anticipated_power_at_selected_gain": two_sided_power(selected_gain),
        "anticipated_power_at_low_aggregate_batch_mean_gain": two_sided_power(low_aggregate_batch_mean_gain),
        "historical_incumbent_drift_pair_count": len(EXISTING_CONFIRMATION_SEEDS),
        "historical_incumbent_drift_null": (
            "zero for same-policy, same-seed, stream-identical reruns; report the observed drift without a "
            "cross-coordinate noise acceptance band"
        ),
        "primary_8b_noise_sd_bpb": noise_by_budget[CONFIRMATION_BUDGET],
        "low_aggregate_noise_crosscheck_sd_bpb": crosscheck_noise_sd,
        "low_aggregate_crosscheck_ci95_half_width_bpb": critical * crosscheck_paired_sd / np.sqrt(pair_count),
        "low_aggregate_crosscheck_power_at_selected_gain": crosscheck_power(selected_gain),
        "low_aggregate_crosscheck_power_at_batch_mean_gain": crosscheck_power(low_aggregate_batch_mean_gain),
    }


def historical_pairing_audit() -> dict[str, object]:
    """Prove that reused incumbents and new candidates share a policy-free training stream."""
    stage1_outcomes = pd.read_csv(STAGE1_OBSERVATIONS_PATH)
    historical = stage1_outcomes.loc[
        stage1_outcomes["run_kind"].eq("incumbent_repeat")
        & stage1_outcomes["token_budget_requested"].eq(CONFIRMATION_BUDGET)
        & np.isclose(stage1_outcomes["phase_0_starcoder"], CONFIRMATION_COORDINATE["incumbent_phase_0"])
        & np.isclose(stage1_outcomes["phase_1_starcoder"], CONFIRMATION_COORDINATE["incumbent_phase_1"])
        & stage1_outcomes["trainer_data_seed"].isin(EXISTING_CONFIRMATION_SEEDS)
    ].sort_values("trainer_data_seed")
    if len(historical) != len(EXISTING_CONFIRMATION_SEEDS):
        raise ValueError("Could not identify every historical incumbent used by Stage 2")

    current_specs = []
    for seed in EXISTING_CONFIRMATION_SEEDS:
        for arm, p0, p1 in (
            (
                "incumbent",
                float(CONFIRMATION_COORDINATE["incumbent_phase_0"]),
                float(CONFIRMATION_COORDINATE["incumbent_phase_1"]),
            ),
            (
                "candidate",
                float(CONFIRMATION_COORDINATE["phase_0_starcoder"]),
                float(CONFIRMATION_COORDINATE["phase_1_starcoder"]),
            ),
        ):
            run = stage1_launcher.RefinementRun(
                token_budget=CONFIRMATION_BUDGET,
                phase_0_starcoder=p0,
                phase_1_starcoder=p1,
                run_name=f"stream_audit_{arm}_s{seed}",
                trainer_data_seed=seed,
                simulated_epoch_subset_seed=seed,
            )
            current_specs.append(run.surface_spec(len(current_specs) + 1))
    steps = base_launcher.build_training_steps(
        name_prefix="scratch/wsd80-stage2-stream-audit",
        tpu_type=base_launcher.DEFAULT_TPU_TYPE,
        tpu_region=base_launcher.DEFAULT_TPU_REGION,
        tpu_zone=base_launcher.DEFAULT_TPU_ZONE,
        data_seed=scaling_launcher.REFERENCE_SEED,
        run_specs=tuple(current_specs),
        wandb_experiment_tag="wsd80_stage2_stream_audit",
        panel_tag="wsd80_stage2_stream_audit",
        experiment_budget=CONFIRMATION_BUDGET,
        target_budget=base_launcher.TARGET_BUDGET,
    )
    current_by_name = {spec.run_name: lower(step) for spec, step in zip(current_specs, steps, strict=True)}

    api = wandb.Api(timeout=240)
    rows = []
    for source in historical.itertuples(index=False):
        seed = int(source.trainer_data_seed)
        historical_run = api.run(f"{TRAIN_PROJECT}/{source.wandb_id}")
        historical_config = dict(historical_run.config)
        historical_cache_uris = {
            name: str(historical_config["data"]["components"][name]["cache_dir"])
            for name in stream_identity.wandb_training_cache_paths(historical_config)
        }
        if not all(uri.startswith("gs://marin-us-central1/tokenized/") for uri in historical_cache_uris.values()):
            raise ValueError(f"Historical run {source.wandb_id} did not use central1 training caches")
        historical_identity = stream_identity.wandb_stream_identity(historical_config)
        historical_digest = stream_identity.canonical_sha256(historical_identity)
        arm_digests = {}
        arm_policies = {}
        for arm in ("incumbent", "candidate"):
            step_spec = current_by_name[f"stream_audit_{arm}_s{seed}"]
            current_config = stream_identity.lowered_step_training_config(step_spec)
            current_identity = stream_identity.lowered_step_stream_identity(step_spec)
            differences = stream_identity.identity_differences(historical_identity, current_identity)
            if differences:
                raise ValueError(
                    f"Historical incumbent {source.wandb_id} differs from the current {arm} stream: {differences}"
                )
            arm_digests[arm] = stream_identity.canonical_sha256(current_identity)
            arm_policies[arm] = stream_identity.policy_coordinates(current_config)
        historical_policy = stream_identity.policy_coordinates(historical_config)
        expected_incumbent = [
            {"boundary_step": historical_policy[0]["boundary_step"], "starcoder_weight": 0.02},
            {"boundary_step": historical_policy[1]["boundary_step"], "starcoder_weight": 0.82},
        ]
        if stream_identity.identity_differences(historical_policy, expected_incumbent):
            raise ValueError(f"Historical run {source.wandb_id} is not the frozen incumbent policy")
        rows.append(
            {
                "seed": seed,
                "historical_wandb_id": str(source.wandb_id),
                "historical_wandb_config_sha256": stream_identity.canonical_sha256(historical_config),
                "historical_training_cache_uris": historical_cache_uris,
                "historical_stream_identity_sha256": historical_digest,
                "current_incumbent_stream_identity_sha256": arm_digests["incumbent"],
                "current_candidate_stream_identity_sha256": arm_digests["candidate"],
                "historical_policy": historical_policy,
                "current_incumbent_policy": arm_policies["incumbent"],
                "current_candidate_policy": arm_policies["candidate"],
                "identity_match": True,
            }
        )
    return {
        "definition": (
            "Policy-free identity over model, optimizer, token and step budgets, phase boundary, normalized "
            "background-mixture ratios, physical token-cache identities, and all trainer/data/subset seeds."
        ),
        "historical_config_source": "W&B configs fetched before launch and frozen by SHA-256",
        "rows": rows,
    }


def build_rows() -> tuple[list[dict[str, object]], pd.DataFrame, pd.DataFrame, dict[int, float]]:
    """Generate frozen spatial acquisitions and seed-paired candidate confirmations."""
    observations = load_reference_seed_observations()
    noise_table, noise_by_budget = spatial_noise_estimates()
    rows: list[dict[str, object]] = []
    selected_by_budget: dict[int, list[list[float]]] = {}
    for region in stage2_trust_regions():
        summary = coordinate_summary(observations, region.token_budget)
        extra_exclusions = np.asarray(selected_by_budget.get(region.token_budget, []), dtype=float).reshape(-1, 2)
        selected = select_region(summary, noise_by_budget[region.token_budget], region, extra_exclusions)
        rows.extend(selected)
        selected_by_budget.setdefault(region.token_budget, []).extend(
            [[float(row["phase_0_starcoder"]), float(row["phase_1_starcoder"])] for row in selected]
        )

    for row in rows:
        row["stage"] = 2
        row["comparison_role"] = "reference_seed_spatial_refinement"

    token_budget = int(CONFIRMATION_COORDINATE["token_budget_requested"])
    candidate = (
        float(CONFIRMATION_COORDINATE["phase_0_starcoder"]),
        float(CONFIRMATION_COORDINATE["phase_1_starcoder"]),
    )
    incumbent = (
        float(CONFIRMATION_COORDINATE["incumbent_phase_0"]),
        float(CONFIRMATION_COORDINATE["incumbent_phase_1"]),
    )

    def confirmation_row(seed: int, arm: str, comparison_source: str) -> dict[str, object]:
        coordinate = candidate if arm == "candidate" else incumbent
        return {
            "token_budget_requested": token_budget,
            "region": "stage1_candidate_confirmation",
            "run_kind": f"{arm}_confirmation",
            "replicate_kind": "joint_randomness",
            "trainer_data_seed": seed,
            "simulated_epoch_subset_seed": seed,
            "phase_0_starcoder": coordinate[0],
            "phase_1_starcoder": coordinate[1],
            "region_acquisition_rank": None,
            "committee_ei_rank_fraction": None,
            "committee_mean_bpb": None,
            "committee_mean_sd": None,
            "noise_sd": noise_by_budget[token_budget],
            "region_training_coordinates": None,
            "expected_improvement_kernel_1": None,
            "expected_improvement_kernel_2": None,
            "expected_improvement_kernel_3": None,
            "stage": 2,
            "comparison_role": "paired_candidate_vs_incumbent",
            "pair_seed": seed,
            "pair_arm": arm,
            "comparison_source": comparison_source,
            "incumbent_phase_0": incumbent[0],
            "incumbent_phase_1": incumbent[1],
            "stage1_reference_gain_bpb": float(CONFIRMATION_COORDINATE["stage1_reference_gain_bpb"]),
        }

    for seed in EXISTING_CONFIRMATION_SEEDS + NEW_CONFIRMATION_SEEDS:
        source = "historical_seed_block" if seed in EXISTING_CONFIRMATION_SEEDS else "fresh_seed_block"
        rows.append(confirmation_row(seed, "candidate", source))
        rows.append(confirmation_row(seed, "incumbent", source))

    for index, row in enumerate(rows, start=1):
        token_budget = int(row["token_budget_requested"])
        row["boundary_step"] = int(base_launcher._schedule_summary(token_budget)["boundary_step"])
        p0 = float(row["phase_0_starcoder"])
        p1 = float(row["phase_1_starcoder"])
        if row["run_kind"] == "acquisition":
            kind = "acq"
        else:
            kind = f"confirm_{row['pair_arm']}"
        seed = int(row["trainer_data_seed"])
        row["run_name"] = (
            f"d{_budget_slug(token_budget)}_bo2_{kind}{index:02d}_p0{_weight_slug(p0)}" f"_p1{_weight_slug(p1)}_s{seed}"
        )

    if len(rows) != EXPECTED_RUNS:
        raise ValueError(f"Expected {EXPECTED_RUNS} Stage-2 rows, found {len(rows)}")
    if len({str(row["run_name"]) for row in rows}) != len(rows):
        raise ValueError("Stage-2 run names are not unique")
    acquisition_coordinates = [
        (int(row["token_budget_requested"]), float(row["phase_0_starcoder"]), float(row["phase_1_starcoder"]))
        for row in rows
        if row["run_kind"] == "acquisition"
    ]
    if len(acquisition_coordinates) != len(set(acquisition_coordinates)):
        raise ValueError("Stage-2 acquisition coordinates are not unique")
    confirmation_rows = [row for row in rows if row["run_kind"] != "acquisition"]
    pair_counts = pd.Series([int(row["pair_seed"]) for row in confirmation_rows]).value_counts()
    expected_pair_counts = {seed: 2 for seed in EXISTING_CONFIRMATION_SEEDS + NEW_CONFIRMATION_SEEDS}
    if pair_counts.to_dict() != expected_pair_counts:
        raise ValueError(f"Unexpected confirmation pair counts: {pair_counts.to_dict()}")
    return rows, observations, noise_table, noise_by_budget


def write_outputs() -> None:
    """Persist the immutable launch design and its interpretation boundary."""
    rows, observations, noise_table, noise_by_budget = build_rows()
    confirmation_diagnostics = confirmation_design_diagnostics(observations, noise_table, noise_by_budget)
    pairing_audit = historical_pairing_audit()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_paths = (
        Path(__file__).resolve(),
        stage1.TOKEN_SCALING_PATH,
        stage1.TIED_DIAGONAL_PATH,
        stage1.SCALE_FIBERS_PATH,
        STAGE1_DESIGN_PATH,
        STAGE1_OBSERVATIONS_PATH,
        Path(base_launcher.__file__).resolve(),
        Path(scaling_launcher.__file__).resolve(),
        Path(stage1_launcher.__file__).resolve(),
        Path(stream_identity.__file__).resolve(),
        STAGE2_LAUNCHER_PATH,
        STAGE2_ANALYZER_PATH,
    )
    payload = {
        "experiment": "StarCoder WSD80 scale-specific Bayesian optimum refinement, stage 2",
        "design_version": "2026-08-01",
        "objective_metric": stage1.OBJECTIVE_METRIC,
        "data_use": {
            "reference_seed_observations": len(observations),
            "reference_seed": REFERENCE_SEED,
            "existing_confirmation_seeds": list(EXISTING_CONFIRMATION_SEEDS),
            "new_confirmation_seeds": list(NEW_CONFIRMATION_SEEDS),
            "source_sha256": {str(path.relative_to(REPO_ROOT)): _sha256(path) for path in source_paths},
            "spatial_fit_seed_policy": "reference trainer and subset seed only",
        },
        "training_environment": {
            "tpu_type": base_launcher.DEFAULT_TPU_TYPE,
            "tpu_region": base_launcher.DEFAULT_TPU_REGION,
            "tpu_zone": base_launcher.DEFAULT_TPU_ZONE,
            "marin_prefix": base_launcher.DEFAULT_MARIN_PREFIX,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "scipy_version": scipy.__version__,
            "sklearn_version": sklearn.__version__,
        },
        "acquisition": {
            "type": "Stage-1-refit local expected-improvement rank committee within empirical convex support",
            "kernel_length_scales": [list(values) for values in stage1.KERNEL_LENGTH_SCALES],
            "noise_sd_by_budget": {str(key): value for key, value in noise_by_budget.items()},
            "noise_estimator": "fresh-only two-way seed-by-coordinate additive ANOVA residual SD",
            "noise_scope": "Estimated on mid/high-aggregate fiber coordinates and extrapolated to low basins.",
            "acquisitions_by_region": {
                f"{token_budget}:{name}": count for (token_budget, name), count in ACQUISITIONS_BY_REGION.items()
            },
            "trust_regions": [asdict(region) for region in stage2_trust_regions()],
        },
        "confirmation": {
            "coordinate": CONFIRMATION_COORDINATE,
            "estimand": (
                "8B contemporaneous same-seed candidate minus incumbent BPB over eight non-selecting joint seeds"
            ),
            "design_diagnostics": confirmation_diagnostics,
            "analysis_plan": {
                "selection_seed_excluded": REFERENCE_SEED,
                "paired_seeds": list(EXISTING_CONFIRMATION_SEEDS + NEW_CONFIRMATION_SEEDS),
                "primary_test": (
                    "two-sided paired t-test at alpha 0.05; the directional confirmation rule has operative "
                    "one-sided alpha 0.025"
                ),
                "primary_interval": "two-sided 95% Student-t confidence interval for the mean paired difference",
                "decision_rule": "confirm only if mean candidate-minus-incumbent < 0 and CI upper bound < 0",
                "secondary_outputs": [
                    "paired SD",
                    "candidate-better sign count",
                    "all eight paired differences",
                    "same-policy historical-versus-contemporaneous incumbent drift diagnostic",
                ],
                "anticipated_paired_sd_bpb": confirmation_diagnostics["anticipated_paired_sd_bpb"],
                "anticipated_ci95_half_width_bpb": confirmation_diagnostics["anticipated_ci95_half_width_bpb"],
                "multiplicity": "one promoted coordinate and one primary hypothesis; no multiplicity adjustment",
                "null_interpretation": "failure to reject is inconclusive, not evidence of equivalence",
                "reference_seed_gain_use": "provenance only; excluded from inference",
            },
            "reason": (
                "The 8B candidate has the largest Stage-1 gain. Both arms are trained contemporaneously on all "
                "eight non-selecting seeds. Four historical incumbent repeats are retained only for a same-policy "
                "drift diagnostic and do not enter the primary comparison."
            ),
            "historical_pairing_audit": pairing_audit,
        },
        "design": {
            "run_count": len(rows),
            "acquisition_count": EXPECTED_ACQUISITIONS,
            "confirmation_training_count": EXPECTED_CONFIRMATIONS,
            "launch_manifest_sha256": stream_identity.canonical_sha256(launch_manifest(rows)),
            "counts_by_budget": {
                str(token_budget): sum(int(row["token_budget_requested"]) == token_budget for row in rows)
                for token_budget in sorted({int(row["token_budget_requested"]) for row in rows})
            },
        },
        "interpretation_boundary": (
            "Spatial probes are reference-seed local refinements, not confirmations. The 8B candidate comparison "
            "uses only eight non-selecting paired seeds. The panel does not establish global optimality outside the "
            "preregistered trust regions."
        ),
        "runs": rows,
    }
    serialized = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    (OUTPUT_DIR / "design_manifest.json").write_text(serialized, encoding="utf-8")
    FROZEN_LAUNCH_DESIGN.write_text(serialized, encoding="utf-8")
    with (OUTPUT_DIR / "run_manifest.csv").open("w", newline="") as handle:
        fieldnames = list(dict.fromkeys(key for row in rows for key in row))
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    noise_table.to_csv(OUTPUT_DIR / "spatial_noise_estimates.csv", index=False)

    acquisitions = pd.DataFrame(row for row in rows if row["run_kind"] == "acquisition")
    confirmations = pd.DataFrame(row for row in rows if row["run_kind"] != "acquisition")
    report = [
        "# StarCoder WSD80 scale Bayesian refinement: Stage-2 design",
        "",
        f"- Frozen run count: {len(rows)} ({len(acquisitions)} spatial acquisitions and "
        f"{len(confirmations)} confirmation trainings for eight paired comparisons).",
        "- Spatial GP fits use only the common reference trainer/subset seed; the systematic fresh-seed offset is "
        "not pooled into the response surface.",
        "- Spatial acquisitions remain inside the convex hull of the local reference-seed coordinates. The 4B "
        "batch lies close to its low-p0 hull facet and therefore refines p1 near p0=0.015 rather than testing a "
        "lower-p0 optimum.",
        "- The 8B Stage-1 candidate is compared with the incumbent over eight non-selecting joint seeds.",
        "- Acquisition effort is concentrated on the best observed 2B basin and the low-aggregate 4B basin; 8B is "
        "reserved for confirmation because Stage-2 candidates otherwise collapsed onto the artificial p1 boundary.",
        "- Confirmation is preregistered before outcomes: two-sided paired t-test and 95% CI, excluding the "
        "selecting seed.",
        f"- Realistic sensitivity anchor: power "
        f"{confirmation_diagnostics['anticipated_power_at_low_aggregate_batch_mean_gain']:.3f} at the mean "
        f"low-aggregate acquisition gain, with anticipated 95% CI half-width "
        f"{payload['confirmation']['analysis_plan']['anticipated_ci95_half_width_bpb']:.6f} BPB under the "
        "fresh-only 8B residual estimate.",
        f"- Selection diagnostic: runner-up margin {confirmation_diagnostics['runner_up_margin_bpb']:.6f} BPB; "
        f"selected gain {confirmation_diagnostics['selected_reference_gain_bpb']:.6f} BPB; low-aggregate batch "
        f"mean gain {confirmation_diagnostics['low_aggregate_batch_mean_gain_bpb']:.6f} BPB. The selected gain is "
        "winner's-curse biased and is retained for provenance, not prospective power.",
        f"- Low-aggregate noise cross-check: {confirmation_diagnostics['low_aggregate_noise_crosscheck_sd_bpb']:.6f} "
        f"BPB versus primary {confirmation_diagnostics['primary_8b_noise_sd_bpb']:.6f} BPB; it implies CI half-width "
        f"{confirmation_diagnostics['low_aggregate_crosscheck_ci95_half_width_bpb']:.6f} BPB and power "
        f"{confirmation_diagnostics['low_aggregate_crosscheck_power_at_batch_mean_gain']:.3f} at the "
        "low-aggregate batch mean gain. This cross-check absorbs uncertainty from an externally estimated seed "
        "effect and is not directly comparable to the primary residual estimate.",
        f"- For context only, power at the upward-biased selected Stage-1 gain is "
        f"{confirmation_diagnostics['anticipated_power_at_selected_gain']:.3f} under the primary noise estimate.",
        "- The four-pair historical-incumbent comparison is a stream-identical determinism diagnostic with a "
        "near-zero null, not a noisy cross-coordinate performance comparison. It is not a gate, and historical "
        "outcomes do not enter the primary comparison.",
        "",
        "## Spatial-noise estimate",
        "",
        noise_table.to_markdown(index=False, floatfmt=".8f"),
        "",
        "## Paired confirmations",
        "",
        confirmations[
            [
                "token_budget_requested",
                "phase_0_starcoder",
                "phase_1_starcoder",
                "trainer_data_seed",
                "pair_arm",
                "comparison_source",
                "incumbent_phase_0",
                "incumbent_phase_1",
                "stage1_reference_gain_bpb",
            ]
        ].to_markdown(index=False),
        "",
        "## Spatial acquisitions",
        "",
        acquisitions[
            [
                "token_budget_requested",
                "region",
                "phase_0_starcoder",
                "phase_1_starcoder",
                "region_acquisition_rank",
                "committee_ei_rank_fraction",
                "committee_mean_bpb",
                "committee_mean_sd",
                "inside_local_training_convex_hull",
                "distance_to_region_box_edge",
                "distance_to_local_hull_facet",
                "nearest_existing_coordinate_distance",
                "nearest_selected_batch_distance",
                "batch_spacing_constraint_active",
            ]
        ].to_markdown(index=False, floatfmt=".6f"),
        "",
        "## Interpretation boundary",
        "",
        str(payload["interpretation_boundary"]),
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    write_outputs()
