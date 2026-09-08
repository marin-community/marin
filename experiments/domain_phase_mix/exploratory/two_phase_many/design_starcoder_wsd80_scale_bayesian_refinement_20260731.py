# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "numpy",
#   "pandas",
#   "scikit-learn",
#   "scipy",
# ]
# ///

"""Freeze a replicate-aware local Bayesian-refinement panel for WSD80."""

from __future__ import annotations

import csv
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
OUTPUT_DIR = REFERENCE_OUTPUTS / "starcoder_wsd80_scale_bayesian_refinement_20260731"
FROZEN_LAUNCH_DESIGN = SCRIPT_DIR.parents[1] / "starcoder_wsd80_scale_bayesian_refinement_design_20260731.json"

DENSE_SURFACE_PATH = REFERENCE_OUTPUTS / "starcoder_wsd80_surface_refined_20260714" / "wsd80_observed_metrics.csv"
TOKEN_SCALING_PATH = (
    REFERENCE_OUTPUTS / "starcoder_wsd80_fixed_model_token_scaling_20260728" / "results_20260730" / "observations.csv"
)
TIED_DIAGONAL_PATH = (
    REFERENCE_OUTPUTS
    / "starcoder_wsd80_fixed_model_tied_diagonal_20260730"
    / "results_20260731"
    / "tied_diagonal_observations.csv"
)
SCALE_FIBERS_PATH = (
    REFERENCE_OUTPUTS / "starcoder_wsd80_scale_specific_tied_fibers_20260731" / "results_20260731" / "observations.csv"
)
SOURCE_PATHS = (DENSE_SURFACE_PATH, TOKEN_SCALING_PATH, TIED_DIAGONAL_PATH, SCALE_FIBERS_PATH)

REFERENCE_SEED = 20_260_711
JOINT_RANDOMNESS_SEEDS = (20_260_712, 20_260_713, 20_260_714, 20_260_715)
OBJECTIVE_METRIC = "eval/paloma/dolma_100_programing_languages-llama3/bpb"
GRID_STEP = 0.005
TRAINING_MARGIN = 0.15
EXISTING_EXCLUSION_RADIUS = 0.012
BATCH_EXCLUSION_RADIUS = 0.025
KERNEL_LENGTH_SCALES = ((0.06, 0.06), (0.10, 0.10), (0.20, 0.08))


@dataclass(frozen=True)
class TrustRegion:
    """One preregistered local basin and its first-stage acquisition budget."""

    token_budget: int
    name: str
    p0_min: float
    p0_max: float
    p1_min: float
    p1_max: float
    count: int
    rationale: str


TRUST_REGIONS = (
    TrustRegion(1_000_000_000, "main", 0.075, 0.135, 0.455, 0.525, 6, "Dense-panel optimum cluster."),
    TrustRegion(2_000_000_000, "c09", 0.04, 0.18, 0.43, 0.58, 6, "Five-seed c09 basin."),
    TrustRegion(2_000_000_000, "c10", 0.01, 0.14, 0.58, 0.74, 6, "Best reference-seed c10 basin."),
    TrustRegion(4_000_000_000, "low_aggregate", 0.01, 0.16, 0.65, 0.88, 6, "Low-aggregate late-code basin."),
    TrustRegion(4_000_000_000, "high_aggregate", 0.46, 0.62, 0.48, 0.74, 6, "Repeated tied/fiber basin."),
    TrustRegion(8_000_000_000, "low_aggregate", 0.01, 0.14, 0.74, 0.90, 5, "Single-seed low-aggregate incumbent basin."),
    TrustRegion(8_000_000_000, "high_aggregate", 0.66, 0.82, 0.72, 0.98, 5, "Repeated high-aggregate basin."),
)

INCUMBENT_REPEATS = {
    2_000_000_000: (0.06, 0.66),
    4_000_000_000: (0.02, 0.82),
    8_000_000_000: (0.02, 0.82),
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _frame(
    *, token_budget: pd.Series | int, p0: pd.Series, p1: pd.Series, bpb: pd.Series, run_id: pd.Series
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "token_budget_requested": token_budget,
            "p0": pd.to_numeric(p0, errors="raise"),
            "p1": pd.to_numeric(p1, errors="raise"),
            "bpb": pd.to_numeric(bpb, errors="raise"),
            "run_id": run_id.astype(str),
        }
    )


def load_observations() -> pd.DataFrame:
    """Load and de-duplicate every run-level observation used by the design."""
    dense = pd.read_csv(DENSE_SURFACE_PATH)
    scaling = pd.read_csv(TOKEN_SCALING_PATH)
    tied = pd.read_csv(TIED_DIAGONAL_PATH)
    fibers = pd.read_csv(SCALE_FIBERS_PATH)
    observations = pd.concat(
        [
            _frame(
                token_budget=1_000_000_000,
                p0=dense["phase_0_starcoder"],
                p1=dense["phase_1_starcoder"],
                bpb=dense["wsd80_bpb"],
                run_id=dense["wandb_run_id"],
            ),
            _frame(
                token_budget=scaling["token_budget_requested"],
                p0=scaling["phase_0_starcoder"],
                p1=scaling["phase_1_starcoder"],
                bpb=scaling["starcoder_bpb"],
                run_id=scaling["training_wandb_id"],
            ),
            _frame(
                token_budget=tied["token_budget_requested"],
                p0=tied["weight"],
                p1=tied["weight"],
                bpb=tied["starcoder_bpb"],
                run_id=tied["wandb_id"],
            ),
            _frame(
                token_budget=fibers["token_budget_requested"],
                p0=fibers["phase_0_starcoder"],
                p1=fibers["phase_1_starcoder"],
                bpb=fibers["starcoder_bpb"],
                run_id=fibers["wandb_id"],
            ),
        ],
        ignore_index=True,
    ).drop_duplicates("run_id")
    if observations[["p0", "p1"]].lt(0.0).any().any() or observations[["p0", "p1"]].gt(1.0).any().any():
        raise ValueError("An observed StarCoder coordinate is outside [0, 1]^2")
    if not np.isfinite(observations[["p0", "p1", "bpb"]].to_numpy(dtype=float)).all():
        raise ValueError("Observation table contains non-finite values")
    return observations.reset_index(drop=True)


def coordinate_summary(observations: pd.DataFrame, token_budget: int) -> tuple[pd.DataFrame, float]:
    """Collapse run-level repeats and estimate a pooled within-coordinate noise SD."""
    selected = observations.loc[observations["token_budget_requested"].eq(token_budget)].copy()
    selected["p0_key"] = selected["p0"].round(8)
    selected["p1_key"] = selected["p1"].round(8)
    summary = (
        selected.groupby(["p0_key", "p1_key"], as_index=False)["bpb"]
        .agg(mean="mean", variance="var", count="count")
        .sort_values(["p0_key", "p1_key"])
        .reset_index(drop=True)
    )
    repeated = summary.loc[summary["count"].ge(3) & summary["variance"].notna()]
    if repeated.empty:
        raise ValueError(f"Token budget {token_budget} has no repeated coordinates for noise estimation")
    degrees = repeated["count"] - 1
    noise_variance = float(np.sum(degrees * repeated["variance"]) / np.sum(degrees))
    return summary, float(np.sqrt(noise_variance))


def _expected_improvement(mean: np.ndarray, sd: np.ndarray, best: float) -> np.ndarray:
    safe_sd = np.maximum(sd, 1e-12)
    z = (best - mean) / safe_sd
    return (best - mean) * norm.cdf(z) + safe_sd * norm.pdf(z)


def _grid(region: TrustRegion) -> np.ndarray:
    p0 = np.arange(region.p0_min, region.p0_max + GRID_STEP / 2, GRID_STEP)
    p1 = np.arange(region.p1_min, region.p1_max + GRID_STEP / 2, GRID_STEP)
    grid_p0, grid_p1 = np.meshgrid(p0, p1, indexing="ij")
    return np.column_stack([grid_p0.ravel(), grid_p1.ravel()])


def _distance_to(points: np.ndarray, references: np.ndarray) -> np.ndarray:
    return np.sqrt(((points[:, None, :] - references[None, :, :]) ** 2).sum(axis=2)).min(axis=1)


def select_region(summary: pd.DataFrame, noise_sd: float, region: TrustRegion) -> list[dict[str, object]]:
    """Select a diverse EI batch under a frozen local GP kernel committee."""
    margin = TRAINING_MARGIN
    training = summary.loc[
        summary["p0_key"].between(max(0.0, region.p0_min - margin), min(1.0, region.p0_max + margin))
        & summary["p1_key"].between(max(0.0, region.p1_min - margin), min(1.0, region.p1_max + margin))
    ].copy()
    if len(training) < 4:
        raise ValueError(f"Trust region {region.name} at {region.token_budget} has only {len(training)} training points")
    x_train = training[["p0_key", "p1_key"]].to_numpy(dtype=float)
    y_train = training["mean"].to_numpy(dtype=float)
    alpha = np.where(
        training["count"].gt(1),
        training["variance"].fillna(noise_sd**2).to_numpy(dtype=float) / training["count"].to_numpy(dtype=float),
        noise_sd**2,
    )
    candidates = _grid(region)
    expected_improvements = []
    means = []
    standard_deviations = []
    for length_scale in KERNEL_LENGTH_SCALES:
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * Matern(
            length_scale=length_scale,
            length_scale_bounds="fixed",
            nu=2.5,
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            optimizer=None,
        ).fit(x_train, y_train)
        mean, sd = model.predict(candidates, return_std=True)
        best = float(model.predict(x_train).min())
        means.append(mean)
        standard_deviations.append(sd)
        expected_improvements.append(_expected_improvement(mean, sd, best))

    rank_fraction = np.mean(
        [rankdata(-values, method="average") / len(values) for values in expected_improvements], axis=0
    )
    existing = summary[["p0_key", "p1_key"]].to_numpy(dtype=float)
    rank_fraction[_distance_to(candidates, existing) < EXISTING_EXCLUSION_RADIUS] = 2.0

    selected_indices: list[int] = []
    for _ in range(region.count):
        eligible = np.ones(len(candidates), dtype=bool)
        if selected_indices:
            eligible &= _distance_to(candidates, candidates[selected_indices]) >= BATCH_EXCLUSION_RADIUS
        index = int(np.where(eligible, rank_fraction, 2.0).argmin())
        if rank_fraction[index] >= 2.0:
            raise ValueError(f"Trust region {region.name} cannot supply {region.count} diverse candidates")
        selected_indices.append(index)

    rows = []
    for local_rank, index in enumerate(selected_indices, start=1):
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
            "region_training_coordinates": len(training),
        }
        for kernel_index, values in enumerate(expected_improvements, start=1):
            row[f"expected_improvement_kernel_{kernel_index}"] = float(values[index])
        rows.append(row)
    return rows


def _weight_slug(value: float) -> str:
    return f"{value:.4f}".replace(".", "p")


def _budget_slug(token_budget: int) -> str:
    return f"{token_budget // 1_000_000_000}b"


def build_rows() -> tuple[list[dict[str, object]], dict[int, float], pd.DataFrame]:
    """Build all acquisition and incumbent-repeat rows."""
    observations = load_observations()
    summaries: dict[int, pd.DataFrame] = {}
    noise_by_budget: dict[int, float] = {}
    rows: list[dict[str, object]] = []
    for region in TRUST_REGIONS:
        if region.token_budget not in summaries:
            summaries[region.token_budget], noise_by_budget[region.token_budget] = coordinate_summary(
                observations, region.token_budget
            )
        rows.extend(select_region(summaries[region.token_budget], noise_by_budget[region.token_budget], region))

    for token_budget, (p0, p1) in INCUMBENT_REPEATS.items():
        summary = summaries[token_budget]
        match = summary.loc[np.isclose(summary["p0_key"], p0) & np.isclose(summary["p1_key"], p1)]
        if len(match) != 1 or int(match.iloc[0]["count"]) != 1:
            raise ValueError(f"Incumbent repeat target {(token_budget, p0, p1)} is not a one-seed coordinate")
        for repeat_index, seed in enumerate(JOINT_RANDOMNESS_SEEDS, start=1):
            rows.append(
                {
                    "token_budget_requested": token_budget,
                    "region": "single_seed_incumbent",
                    "run_kind": "incumbent_repeat",
                    "replicate_kind": "joint_randomness",
                    "trainer_data_seed": seed,
                    "simulated_epoch_subset_seed": seed,
                    "phase_0_starcoder": p0,
                    "phase_1_starcoder": p1,
                    "region_acquisition_rank": repeat_index,
                    "committee_ei_rank_fraction": None,
                    "committee_mean_bpb": float(match.iloc[0]["mean"]),
                    "committee_mean_sd": None,
                    "noise_sd": noise_by_budget[token_budget],
                    "region_training_coordinates": None,
                    "expected_improvement_kernel_1": None,
                    "expected_improvement_kernel_2": None,
                    "expected_improvement_kernel_3": None,
                }
            )

    for index, row in enumerate(rows, start=1):
        token_budget = int(row["token_budget_requested"])
        p0 = float(row["phase_0_starcoder"])
        p1 = float(row["phase_1_starcoder"])
        kind = "acq" if row["run_kind"] == "acquisition" else "rep"
        seed = int(row["trainer_data_seed"])
        row["run_name"] = (
            f"d{_budget_slug(token_budget)}_bo1_{kind}{index:02d}_p0{_weight_slug(p0)}" f"_p1{_weight_slug(p1)}_s{seed}"
        )
    if len(rows) != 52:
        raise ValueError(f"Expected 52 first-stage runs, got {len(rows)}")
    if len({str(row["run_name"]) for row in rows}) != len(rows):
        raise ValueError("Bayesian-refinement run names are not unique")
    acquisition_coordinates = [
        (int(row["token_budget_requested"]), float(row["phase_0_starcoder"]), float(row["phase_1_starcoder"]))
        for row in rows
        if row["run_kind"] == "acquisition"
    ]
    if len(acquisition_coordinates) != len(set(acquisition_coordinates)):
        raise ValueError("Acquisition coordinates are not unique within a token budget")
    return rows, noise_by_budget, observations


def write_outputs() -> None:
    """Write the frozen design, flat rows, diagnostics, and human-readable report."""
    rows, noise_by_budget, observations = build_rows()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    source_hashes = {str(path.relative_to(SCRIPT_DIR)): _sha256(path) for path in SOURCE_PATHS}
    payload = {
        "experiment": "StarCoder WSD80 scale-specific Bayesian optimum refinement, stage 1",
        "design_version": "2026-07-31",
        "objective_metric": OBJECTIVE_METRIC,
        "data_use": {
            "run_level_observations": len(observations),
            "source_sha256": source_hashes,
            "repeats_are_collapsed_for_fit": True,
            "noise_estimator": "pooled within-coordinate variance over coordinates with at least three seeds",
        },
        "acquisition": {
            "type": "replicate-aware local expected-improvement rank committee",
            "kernel": "constant times Matern-5/2 with fixed anisotropic length scales",
            "kernel_length_scales": [list(values) for values in KERNEL_LENGTH_SCALES],
            "grid_step": GRID_STEP,
            "training_margin": TRAINING_MARGIN,
            "existing_exclusion_radius": EXISTING_EXCLUSION_RADIUS,
            "batch_exclusion_radius": BATCH_EXCLUSION_RADIUS,
            "noise_sd_by_budget": {str(key): value for key, value in noise_by_budget.items()},
            "trust_regions": [asdict(region) for region in TRUST_REGIONS],
            "stage_policy": (
                "This is a first sequential batch. Refit after outcomes before selecting another batch, "
                "especially at 8B."
            ),
        },
        "design": {
            "run_count": len(rows),
            "acquisition_count": sum(row["run_kind"] == "acquisition" for row in rows),
            "incumbent_repeat_count": sum(row["run_kind"] == "incumbent_repeat" for row in rows),
            "counts_by_budget": {
                str(token_budget): sum(int(row["token_budget_requested"]) == token_budget for row in rows)
                for token_budget in sorted(noise_by_budget)
            },
            "incumbent_repeat_coordinates": {str(key): list(value) for key, value in INCUMBENT_REPEATS.items()},
        },
        "interpretation_boundary": (
            "The panel refines all currently competitive local basins but does not prove global optimality over "
            "the full square. "
            "Single-seed incumbent repeats are confirmatory controls against winner's-curse selection."
        ),
        "runs": rows,
    }
    (OUTPUT_DIR / "design_manifest.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    FROZEN_LAUNCH_DESIGN.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    with (OUTPUT_DIR / "run_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    report = [
        "# StarCoder WSD80 Bayesian optimum refinement: stage 1",
        "",
        f"- Frozen run count: {len(rows)} ({payload['design']['acquisition_count']} new coordinates and "
        f"{payload['design']['incumbent_repeat_count']} incumbent repeats).",
        f"- Counts by budget: {payload['design']['counts_by_budget']}.",
        "- The acquisition uses coordinate-level means and pooled repeat noise rather than the lowest single "
        "observation.",
        "- The 4B and 8B designs preserve both low-aggregate and high-aggregate basins because their differences "
        "are below repeat noise.",
        "- This is deliberately sequential: no second 8B batch should be chosen until this stage is observed.",
        "",
        "## 8B caution",
        "",
        "The raw low-aggregate incumbent `(0.02, 0.82)` has one seed. The repeated high-aggregate point near "
        "`(0.71, 0.91)` has a lower five-seed mean, so a single-basin optimization would be selection-biased. "
        "The panel spends four runs replicating the low-aggregate incumbent and divides ten new coordinates across "
        "both basins.",
        "",
        "## Interpretation boundary",
        "",
        str(payload["interpretation_boundary"]),
        "",
    ]
    (OUTPUT_DIR / "report.md").write_text("\n".join(report))


if __name__ == "__main__":
    write_outputs()
