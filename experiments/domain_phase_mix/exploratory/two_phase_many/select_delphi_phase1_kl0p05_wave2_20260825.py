# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

# /// script
# requires-python = ">=3.12"
# dependencies = [
#   "fsspec==2026.1.0",
#   "gcsfs==2026.1.0",
#   "numpy==2.3.5",
#   "pandas==2.2.2",
#   "scikit-learn==1.8.0",
#   "scipy==1.17.0",
# ]
# ///
"""Select the frozen 80-row KL0.05 phase-1 Wave-2 panel.

The scientific contract can be written without reading endpoint outcomes::

    PYTHONPATH=. uv run \
      experiments/domain_phase_mix/exploratory/two_phase_many/\
select_delphi_phase1_kl0p05_wave2_20260825.py --contract-only

The normal invocation requires complete Wave-1 and n=5 noise artifacts. It
selects 40 model-guided rows and appends the 40 outcome-blind coverage rows
that were frozen before Wave-1 outcomes were available.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import optimize, stats
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from experiments.domain_phase_mix.exploratory.two_phase_many import (
    design_delphi_phase1_common_branches_20260824 as branch_design,
)

SCRIPT_DIR = Path(__file__).resolve().parent
REFERENCE_OUTPUTS = SCRIPT_DIR / "reference_outputs"
POOL_DIR = REFERENCE_OUTPUTS / "delphi_phase1_kl0p05_wave2_pool_20260825"
WAVE1_DIR = REFERENCE_OUTPUTS / "delphi_phase1_kl0p05_wave1_results_20260825"
NOISE_DIR = REFERENCE_OUTPUTS / "delphi_phase1_kl0p05_noise_results_20260825"
DEFAULT_OUTPUT_DIR = REFERENCE_OUTPUTS / "delphi_phase1_kl0p05_wave2_selection_20260825"
FROZEN_CONTRACT_PATH = DEFAULT_OUTPUT_DIR / "frozen_contract.json"
DEFAULT_PREFIX_WEIGHTS = REFERENCE_OUTPUTS / "delphi_phase0_prefix_candidates_20260824" / "candidate_weights.csv"
TARGET_PREFIX = "shared_bounded_ensemble_kl0p05"
TARGET_METRIC = "uncheatable_bpb"
TARGET_METRIC_LONG = "eval/uncheatable_eval/bpb"
EXPECTED_WAVE1_FIT_ROWS = 100
EXPECTED_FIXED_COVERAGE_ROWS = 40
EXPECTED_REFEREE_ROWS = 8
GUIDED_LCB_ROWS = 32
GUIDED_DISAGREEMENT_ROWS = 8
EXPECTED_WAVE2_ROWS = EXPECTED_FIXED_COVERAGE_ROWS + GUIDED_LCB_ROWS + GUIDED_DISAGREEMENT_ROWS
OUTER_FOLDS = 5
INNER_FOLDS = 4
PARTITION_SEEDS = (0, 1, 2)
RIDGE_ALPHAS = (1e-6, 1e-4, 1e-2, 0.1, 1.0, 10.0)
MAX_MODEL_COLUMNS = 14
MINIMUM_MEAN_SPEARMAN = 0.4
MAXIMUM_BASELINE_RMSE_RATIO = 0.9
NOISE_SD_MULTIPLIER = 3.0
MINIMUM_RESPONSE_SNR = 4.0
MAXIMUM_CROSS_WAVE_ANCHOR_DELTA = 0.0002
LCB_STANDARD_DEVIATIONS = 1.5
MINIMUM_SPATIAL_TEST_ROWS = 8
MINIMUM_SPATIAL_TRAIN_ROWS = 2 * MAX_MODEL_COLUMNS
MINIMUM_PREDICTION_SPREAD = 1e-12
WAVE1_MEDIAN_NEAREST_NEIGHBOR_HELLINGER = 0.4026943185930977
GUIDED_SUPPORT_RADIUS = WAVE1_MEDIAN_NEAREST_NEIGHBOR_HELLINGER
GUIDED_DIVERSITY_FLOOR = WAVE1_MEDIAN_NEAREST_NEIGHBOR_HELLINGER / 2.0
POOL_MANIFEST_SHA256 = "b8cba48f3ffa60ce95bd6f28c5f3476b4bf219fd68c9805c8e55bdb1ce01c534"
POOL_COUNTS_SHA256 = "f19bdcf1dd8e4f666f137013fe07a91c5cc02947e66e422c03475994c3e14a14"
POOL_METADATA_SHA256 = "6a5672255989f10de6959e7b25822e12f297d5fb9439b04a840c373b1be0ba49"
COVERAGE_SUMMARY_SHA256 = "fa041c9dd8eddb09ca501d73c025f430907c21684f8d4e7edfe5e3711455e2ed"
COVERAGE_WEIGHTS_SHA256 = "c0d2bf2e17feb6cc936e1f7bcd19bb4e36ef275893f4f4676f8f289cdf24394c"
PREFIX_WEIGHTS_SHA256 = "fef07d4188ef05f4df4a43d1eda6a12f7d2daf69a1ae1eb777863fd20db732b6"
WAVE1_CONTINUATION_SHA256 = (
    "9305b5c1598c9eb11e7f898f709bfb193f37802efaba40a43fbecd0d52c12355",
    "2860d0e1f177f1728580ec1cdda05e049734e7977b868a8c0abd05d9d8bd0ec3",
)
WAVE1_DESIGN_PATHS = (
    REFERENCE_OUTPUTS / "delphi_phase1_common_branches_20260824" / "continuation_weights.csv",
    REFERENCE_OUTPUTS / "delphi_phase1_kl0p05_wave1_extension_20260825" / "continuation_weights.csv",
)
DEPENDENCY_VERSIONS = {
    "numpy": "2.3.5",
    "pandas": "2.2.2",
    "scikit-learn": "1.8.0",
    "scipy": "1.17.0",
}
FROZEN_CONTRACT_SHA256 = "0ba8d66e1b58e351f747cdfa8fd037ecd60d20ea315965ed732c4466d6d61b91"
MODEL_NAMES = (
    "hellinger_linear_14",
    "valley_quadratic_14",
    "incremental_dsp_14",
    "hybrid_14",
)


@dataclass(frozen=True)
class ModelScore:
    name: str
    mean_spearman: float
    rmse: float
    baseline_rmse: float
    baseline_rmse_ratio: float
    mean_fold_regret_at_1: float
    mean_fold_regret_at_3: float
    regret_tolerance: float
    eligible: bool


@dataclass(frozen=True)
class RidgeFit:
    scaler: StandardScaler
    model: Ridge
    alpha: float


@dataclass(frozen=True)
class FeatureBank:
    hellinger_pca: PCA
    benefit_pca: PCA
    damage_pca: PCA
    phase_0_exposure: np.ndarray
    phase_1_scales: np.ndarray

    def transform(self, weights: np.ndarray, model_name: str) -> np.ndarray:
        hellinger = self.hellinger_pca.transform(hellinger_coordinates(weights))
        benefit, damage = incremental_mechanism_blocks(
            weights,
            self.phase_0_exposure,
            self.phase_1_scales,
        )
        benefit_scores = self.benefit_pca.transform(benefit)
        damage_scores = self.damage_pca.transform(damage)
        if model_name == "hellinger_linear_14":
            result = hellinger[:, :14]
        elif model_name == "valley_quadratic_14":
            first = hellinger[:, :3]
            result = np.column_stack(
                [
                    hellinger[:, :8],
                    first[:, 0] ** 2,
                    first[:, 1] ** 2,
                    first[:, 2] ** 2,
                    first[:, 0] * first[:, 1],
                    first[:, 0] * first[:, 2],
                    first[:, 1] * first[:, 2],
                ]
            )
        elif model_name == "incremental_dsp_14":
            result = np.column_stack([benefit_scores[:, :7], damage_scores[:, :7]])
        elif model_name == "hybrid_14":
            result = np.column_stack([hellinger[:, :8], benefit_scores[:, :3], damage_scores[:, :3]])
        else:
            raise ValueError(f"Unknown frozen model {model_name!r}")
        if result.shape[1] > MAX_MODEL_COLUMNS:
            raise ValueError(f"Model {model_name} exceeds the {MAX_MODEL_COLUMNS}-column cap")
        return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract-only", action="store_true")
    parser.add_argument("--pool-dir", type=Path, default=POOL_DIR)
    parser.add_argument("--wave1-dir", type=Path, default=WAVE1_DIR)
    parser.add_argument("--noise-dir", type=Path, default=NOISE_DIR)
    parser.add_argument("--prefix-weights", type=Path, default=DEFAULT_PREFIX_WEIGHTS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--expected-wave1-materialization-sha256")
    parser.add_argument("--expected-noise-materialization-sha256")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_file_hash(path: Path, expected: str, label: str) -> None:
    actual = file_sha256(path)
    if actual != expected:
        raise ValueError(f"{label} changed: {actual} != {expected}")


def frozen_contract() -> dict[str, object]:
    return {
        "contract_version": "delphi_phase1_kl0p05_wave2_20260825_v1",
        "endpoint_outcomes_used_to_define_contract": False,
        "target_prefix": TARGET_PREFIX,
        "target_metric": TARGET_METRIC,
        "fit_budget": {
            "wave1_rows": EXPECTED_WAVE1_FIT_ROWS,
            "guided_wave2_rows": GUIDED_LCB_ROWS + GUIDED_DISAGREEMENT_ROWS,
            "fixed_coverage_wave2_rows": EXPECTED_FIXED_COVERAGE_ROWS,
            "total_wave2_rows": EXPECTED_WAVE2_ROWS,
            "confirmations_and_noise_controls_included": False,
        },
        "models": {
            "names": list(MODEL_NAMES),
            "maximum_columns_excluding_intercept": MAX_MODEL_COLUMNS,
            "semantic_bucket_partitions": False,
            "feature_basis_fit": "outcome-blind 50000-row candidate pool",
            "incremental_benefit": "(1+E0)^-1/2 - (1+E0+E1)^-1/2",
            "incremental_damage": "bounded_excess(E0+E1)-bounded_excess(E0), knee=1 epoch",
        },
        "validation": {
            "outer_folds": OUTER_FOLDS,
            "inner_folds": INNER_FOLDS,
            "partition_seeds": list(PARTITION_SEEDS),
            "spatial_geometry": "capacity-balanced KMeans blocks in Hellinger coordinates",
            "minimum_spatial_test_rows": MINIMUM_SPATIAL_TEST_ROWS,
            "minimum_spatial_train_rows": MINIMUM_SPATIAL_TRAIN_ROWS,
            "ridge_alphas": list(RIDGE_ALPHAS),
            "minimum_mean_spearman": MINIMUM_MEAN_SPEARMAN,
            "maximum_baseline_rmse_ratio": MAXIMUM_BASELINE_RMSE_RATIO,
            "maximum_mean_fold_regret_at_3": "3 * conservative n=5 full-branch data-seed sample SD",
            "noise_control_actions": ["control_proportional", "fit_maximin_26"],
            "minimum_response_q90_q10_to_noise_sd": MINIMUM_RESPONSE_SNR,
            "maximum_cross_wave_anchor_absolute_bpb": MAXIMUM_CROSS_WAVE_ANCHOR_DELTA,
            "coordinate_distance_gate": False,
            "identifiability_caveat": (
                "outer folds fit at most 14 columns on 80 rows; inner folds select alpha on 60 rows"
            ),
        },
        "acquisition": {
            "guided_lcb_rows": GUIDED_LCB_ROWS,
            "lcb_standard_deviations": LCB_STANDARD_DEVIATIONS,
            "uncertainty_ensemble": "15 spatial subfits: 5 outer folds x 3 partition seeds",
            "minimum_prediction_spread": MINIMUM_PREDICTION_SPREAD,
            "guided_disagreement_rows": GUIDED_DISAGREEMENT_ROWS,
            "fixed_coverage_rows": EXPECTED_FIXED_COVERAGE_ROWS,
            "referee_holdouts_inside_fixed_coverage": EXPECTED_REFEREE_ROWS,
            "guided_support_radius_hellinger": GUIDED_SUPPORT_RADIUS,
            "guided_minimum_distance_hellinger": GUIDED_DIVERSITY_FLOOR,
            "guided_geometry_scale": "median nearest-neighbor Hellinger distance among the frozen 100 Wave-1 rows",
            "guided_geometry_inputs": [
                {"path": str(path.relative_to(SCRIPT_DIR)), "sha256": sha256}
                for path, sha256 in zip(WAVE1_DESIGN_PATHS, WAVE1_CONTINUATION_SHA256, strict=True)
            ],
            "fallback": "40 additional outcome-blind Hellinger maximin rows",
            "fallback_conditions": [
                "no model passes all frozen gates",
                "winner subfit spread or all-model disagreement is at most minimum_prediction_spread",
            ],
            "outcome_blind_geometry_preflight": {
                "eligible_pool_rows_after_fixed_exclusions": 45_441,
                "eligible_annulus_rows_after_fixed_exclusions": 21_850,
                "fallback_rows_materialized": 40,
                "fallback_minimum_pairwise_hellinger": 0.4080151587060811,
                "fallback_minimum_hellinger_to_existing_design": 0.4020897826051688,
            },
        },
        "frozen_input_hashes": {
            "pool_manifest": POOL_MANIFEST_SHA256,
            "pool_counts": POOL_COUNTS_SHA256,
            "pool_metadata": POOL_METADATA_SHA256,
            "coverage_summary": COVERAGE_SUMMARY_SHA256,
            "coverage_weights": COVERAGE_WEIGHTS_SHA256,
            "prefix_weights": PREFIX_WEIGHTS_SHA256,
            "wave1_continuation_designs": list(WAVE1_CONTINUATION_SHA256),
        },
        "implementation": {
            "dependency_versions": DEPENDENCY_VERSIONS,
            "outcome_materializations": "required by caller-supplied manifest SHA-256 and recursively verified",
        },
    }


def canonical_json_bytes(payload: dict[str, object]) -> bytes:
    return (json.dumps(payload, indent=2, sort_keys=True) + "\n").encode()


def validate_dependency_versions() -> None:
    observed = {
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "scikit-learn": sklearn.__version__,
        "scipy": scipy.__version__,
    }
    if observed != DEPENDENCY_VERSIONS:
        raise ValueError(f"Selector dependency versions changed: {observed} != {DEPENDENCY_VERSIONS}")


def validate_frozen_contract() -> dict[str, object]:
    generated = canonical_json_bytes(frozen_contract())
    generated_sha256 = hashlib.sha256(generated).hexdigest()
    if generated_sha256 != FROZEN_CONTRACT_SHA256:
        raise ValueError(f"Generated contract changed: {generated_sha256} != {FROZEN_CONTRACT_SHA256}")
    if not FROZEN_CONTRACT_PATH.exists():
        raise ValueError(f"Frozen contract is missing: {FROZEN_CONTRACT_PATH}")
    if FROZEN_CONTRACT_PATH.read_bytes() != generated:
        raise ValueError("Committed frozen contract differs from the selector's contract")
    return json.loads(generated)


def hellinger_coordinates(weights: np.ndarray) -> np.ndarray:
    if weights.ndim != 2 or np.any(weights < 0.0):
        raise ValueError("Mixture weights must be a nonnegative matrix")
    if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
        raise ValueError("Mixture weights do not sum to one")
    return np.sqrt(weights) / np.sqrt(2.0)


def frozen_wave1_design_weights(buckets: tuple[str, ...]) -> np.ndarray:
    frames = []
    for path, expected_sha256 in zip(WAVE1_DESIGN_PATHS, WAVE1_CONTINUATION_SHA256, strict=True):
        validate_file_hash(path, expected_sha256, f"Wave-1 continuation design {path}")
        frame = pd.read_csv(path)
        fit = frame[frame.fit_budget.astype(str).str.lower().eq("true")]
        wide = fit.pivot(index="continuation_id", columns="bucket", values="phase_1_weight")
        if set(wide.columns) != set(buckets):
            raise ValueError(f"Wave-1 continuation buckets changed in {path}")
        frames.append(wide.loc[:, list(buckets)])
    combined = pd.concat(frames)
    if len(combined) != EXPECTED_WAVE1_FIT_ROWS or combined.index.duplicated().any():
        raise ValueError("Frozen Wave-1 continuation identities changed")
    return combined.to_numpy(dtype=float)


def median_nearest_neighbor_hellinger(weights: np.ndarray) -> float:
    coordinates = hellinger_coordinates(weights)
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    np.fill_diagonal(distances, np.inf)
    return float(np.median(distances.min(axis=1)))


def bounded_excess(exposure: np.ndarray) -> np.ndarray:
    excess = np.maximum(exposure - 1.0, 0.0)
    return excess / (1.0 + excess)


def incremental_mechanism_blocks(
    weights: np.ndarray,
    phase_0_exposure: np.ndarray,
    phase_1_scales: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    phase_1_exposure = weights * phase_1_scales[None, :]
    total = phase_0_exposure[None, :] + phase_1_exposure
    benefit = (1.0 + phase_0_exposure[None, :]) ** -0.5 - (1.0 + total) ** -0.5
    damage = bounded_excess(total) - bounded_excess(phase_0_exposure[None, :])
    return benefit, damage


def build_feature_bank(
    pool_weights: np.ndarray,
    phase_0_weights: np.ndarray,
    phase_0_scales: np.ndarray,
    phase_1_scales: np.ndarray,
) -> FeatureBank:
    phase_0_exposure = phase_0_weights * phase_0_scales
    benefit, damage = incremental_mechanism_blocks(pool_weights, phase_0_exposure, phase_1_scales)
    return FeatureBank(
        hellinger_pca=PCA(n_components=14, svd_solver="full").fit(hellinger_coordinates(pool_weights)),
        benefit_pca=PCA(n_components=7, svd_solver="full").fit(benefit),
        damage_pca=PCA(n_components=7, svd_solver="full").fit(damage),
        phase_0_exposure=phase_0_exposure,
        phase_1_scales=phase_1_scales,
    )


def spatial_folds(weights: np.ndarray, folds: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    if folds < 2:
        raise ValueError("Spatial cross-validation needs at least two folds")
    coordinates = hellinger_coordinates(weights)
    centers = KMeans(n_clusters=folds, random_state=20_260_825 + seed, n_init=50).fit(coordinates).cluster_centers_
    capacities = np.full(folds, len(weights) // folds, dtype=int)
    capacities[: len(weights) % folds] += 1
    slot_labels = np.repeat(np.arange(folds), capacities)
    slot_centers = centers[slot_labels]
    squared_distances = ((coordinates[:, None, :] - slot_centers[None, :, :]) ** 2).sum(axis=2)
    row_indices, slot_indices = optimize.linear_sum_assignment(squared_distances)
    labels = np.empty(len(weights), dtype=int)
    labels[row_indices] = slot_labels[slot_indices]
    rows = np.arange(len(weights))
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for label in range(folds):
        test = rows[labels == label]
        train = rows[labels != label]
        if len(test) < MINIMUM_SPATIAL_TEST_ROWS or len(train) < MINIMUM_SPATIAL_TRAIN_ROWS:
            raise ValueError(f"Spatial fold is too small for model selection: train={len(train)}, test={len(test)}")
        result.append((train, test))
    return result


def fit_ridge(features: np.ndarray, response: np.ndarray, alpha: float) -> RidgeFit:
    scaler = StandardScaler().fit(features)
    model = Ridge(alpha=alpha).fit(scaler.transform(features), response)
    return RidgeFit(scaler=scaler, model=model, alpha=alpha)


def predict_ridge(fitted: RidgeFit, features: np.ndarray) -> np.ndarray:
    return fitted.model.predict(fitted.scaler.transform(features))


def select_alpha(features: np.ndarray, weights: np.ndarray, response: np.ndarray, seed: int) -> float:
    folds = spatial_folds(weights, INNER_FOLDS, seed)
    scores = []
    for alpha in RIDGE_ALPHAS:
        squared_errors = []
        for train, test in folds:
            fitted = fit_ridge(features[train], response[train], alpha)
            squared_errors.extend((predict_ridge(fitted, features[test]) - response[test]) ** 2)
        scores.append((float(np.mean(squared_errors)), alpha))
    return min(scores, key=lambda item: (item[0], item[1]))[1]


def rank_correlation(observed: np.ndarray, predicted: np.ndarray) -> float:
    result = stats.spearmanr(observed, predicted).statistic
    return float(result) if np.isfinite(result) else 0.0


def model_is_eligible(
    mean_spearman: float,
    rmse: float,
    baseline_rmse: float,
    mean_regret_at_3: float,
    regret_tolerance: float,
) -> bool:
    return (
        mean_spearman >= MINIMUM_MEAN_SPEARMAN
        and rmse <= MAXIMUM_BASELINE_RMSE_RATIO * baseline_rmse
        and mean_regret_at_3 <= regret_tolerance
    )


def benchmark_model(
    name: str,
    features: np.ndarray,
    weights: np.ndarray,
    response: np.ndarray,
    noise_sd: float,
) -> tuple[ModelScore, pd.DataFrame, pd.DataFrame]:
    prediction_rows = []
    fold_rows = []
    seed_metrics = []
    for partition_seed in PARTITION_SEEDS:
        predictions = np.full(len(response), np.nan)
        baselines = np.full(len(response), np.nan)
        for fold, (train, test) in enumerate(spatial_folds(weights, OUTER_FOLDS, partition_seed)):
            alpha = select_alpha(features[train], weights[train], response[train], 100 * partition_seed + fold)
            fitted = fit_ridge(features[train], response[train], alpha)
            predictions[test] = predict_ridge(fitted, features[test])
            baselines[test] = response[train].mean()
            order = test[np.argsort(predictions[test], kind="stable")]
            optimum = float(response[test].min())
            regret_at_1 = float(response[order[0]] - optimum)
            regret_at_3 = float(response[order[: min(3, len(order))]].min() - optimum)
            fold_rows.append(
                {
                    "model": name,
                    "partition_seed": partition_seed,
                    "outer_fold": fold,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "selected_alpha": alpha,
                    "regret_at_1": regret_at_1,
                    "regret_at_3": regret_at_3,
                }
            )
            for row in test:
                prediction_rows.append(
                    {
                        "model": name,
                        "partition_seed": partition_seed,
                        "outer_fold": fold,
                        "row": int(row),
                        "observed": float(response[row]),
                        "predicted": float(predictions[row]),
                        "baseline": float(baselines[row]),
                    }
                )
        if not np.isfinite(predictions).all() or not np.isfinite(baselines).all():
            raise ValueError(f"Model {name} did not produce complete OOF predictions")
        seed_metrics.append(
            {
                "spearman": rank_correlation(response, predictions),
                "rmse": float(np.sqrt(np.mean((predictions - response) ** 2))),
                "baseline_rmse": float(np.sqrt(np.mean((baselines - response) ** 2))),
            }
        )
    folds = pd.DataFrame(fold_rows)
    rmse = float(np.mean([row["rmse"] for row in seed_metrics]))
    baseline_rmse = float(np.mean([row["baseline_rmse"] for row in seed_metrics]))
    regret_tolerance = NOISE_SD_MULTIPLIER * noise_sd
    mean_spearman = float(np.mean([row["spearman"] for row in seed_metrics]))
    mean_regret_at_1 = float(folds.regret_at_1.mean())
    mean_regret_at_3 = float(folds.regret_at_3.mean())
    score = ModelScore(
        name=name,
        mean_spearman=mean_spearman,
        rmse=rmse,
        baseline_rmse=baseline_rmse,
        baseline_rmse_ratio=rmse / baseline_rmse,
        mean_fold_regret_at_1=mean_regret_at_1,
        mean_fold_regret_at_3=mean_regret_at_3,
        regret_tolerance=regret_tolerance,
        eligible=model_is_eligible(
            mean_spearman,
            rmse,
            baseline_rmse,
            mean_regret_at_3,
            regret_tolerance,
        ),
    )
    return score, pd.DataFrame(prediction_rows), folds


def spatial_subfit_prediction_ensemble(
    model_names: list[str],
    feature_bank: FeatureBank,
    train_weights: np.ndarray,
    response: np.ndarray,
    candidate_weights: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, list[float]]]:
    predictions: dict[str, np.ndarray] = {}
    alphas: dict[str, list[float]] = {}
    for model_name in model_names:
        train_features = feature_bank.transform(train_weights, model_name)
        candidate_features = feature_bank.transform(candidate_weights, model_name)
        for partition_seed in PARTITION_SEEDS:
            for fold, (train, _) in enumerate(spatial_folds(train_weights, OUTER_FOLDS, partition_seed)):
                alpha = select_alpha(
                    train_features[train],
                    train_weights[train],
                    response[train],
                    10_000 + 100 * partition_seed + fold,
                )
                fitted = fit_ridge(train_features[train], response[train], alpha)
                key = f"{model_name}/seed{partition_seed}/fold{fold}"
                predictions[key] = predict_ridge(fitted, candidate_features)
                alphas.setdefault(model_name, []).append(alpha)
    return predictions, alphas


def minimum_distances(coordinates: np.ndarray, references: np.ndarray, chunk_size: int = 2_000) -> np.ndarray:
    result = np.empty(len(coordinates), dtype=float)
    for start in range(0, len(coordinates), chunk_size):
        chunk = coordinates[start : start + chunk_size]
        distances = np.linalg.norm(chunk[:, None, :] - references[None, :, :], axis=2)
        result[start : start + len(chunk)] = distances.min(axis=1)
    return result


def greedy_ranked_selection(
    coordinates: np.ndarray,
    references: np.ndarray,
    eligible: np.ndarray,
    score: np.ndarray,
    count: int,
    *,
    descending: bool,
) -> np.ndarray:
    available = eligible.copy()
    minimum = minimum_distances(coordinates, references)
    ranking = np.argsort(score, kind="stable")
    if descending:
        ranking = ranking[::-1]
    selected = []
    for candidate in ranking:
        if not available[candidate] or minimum[candidate] < GUIDED_DIVERSITY_FLOOR - 1e-12:
            continue
        selected.append(int(candidate))
        available[candidate] = False
        minimum = np.minimum(minimum, np.linalg.norm(coordinates - coordinates[candidate], axis=1))
        if len(selected) == count:
            break
    if len(selected) != count:
        raise ValueError(f"Only {len(selected)} candidates satisfy the guided diversity contract; need {count}")
    return np.asarray(selected, dtype=int)


def outcome_blind_maximin(
    coordinates: np.ndarray,
    references: np.ndarray,
    eligible: np.ndarray,
    count: int,
) -> np.ndarray:
    available = eligible.copy()
    minimum = minimum_distances(coordinates, references)
    selected = []
    for _ in range(count):
        candidates = np.flatnonzero(available)
        if not len(candidates):
            raise ValueError("Outcome-blind fallback candidate set exhausted")
        pick = int(candidates[np.argmax(minimum[candidates])])
        if minimum[pick] < GUIDED_DIVERSITY_FLOOR - 1e-12:
            raise ValueError("Outcome-blind fallback cannot satisfy the diversity floor")
        selected.append(pick)
        available[pick] = False
        minimum = np.minimum(minimum, np.linalg.norm(coordinates - coordinates[pick], axis=1))
    return np.asarray(selected, dtype=int)


def load_materialization_manifest(
    directory: Path,
    expected_sha256: str,
    required_artifacts: set[str],
) -> dict[str, object]:
    path = directory / "materialization_manifest.json"
    validate_file_hash(path, expected_sha256, f"materialization manifest {path}")
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or payload.get("complete") is not True:
        raise ValueError(f"Materialization is not complete: {path}")
    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, dict) or not required_artifacts.issubset(artifacts):
        raise ValueError(f"Materialization manifest is missing required artifacts: {path}")
    for name, record in artifacts.items():
        if Path(str(name)).name != name or not isinstance(record, dict):
            raise ValueError(f"Malformed materialization artifact entry: {name!r}")
        sha256 = record.get("sha256")
        if not isinstance(sha256, str):
            raise ValueError(f"Materialization artifact lacks a SHA-256: {name}")
        validate_file_hash(directory / name, sha256, f"materialized artifact {name}")
    return payload


def validate_noise_wave1_crosslink(
    wave1_materialization: dict[str, object],
    noise_materialization: dict[str, object],
) -> None:
    wave1_artifacts = wave1_materialization.get("artifacts")
    noise_provenance = noise_materialization.get("provenance")
    if not isinstance(wave1_artifacts, dict) or not isinstance(noise_provenance, dict):
        raise ValueError("Materialization manifests lack Wave-1/noise cross-link provenance")
    expected_links = {
        "wave1_results_sha256": "branch_results.csv",
        "wave1_metrics_sha256": "uncheatable_metrics_long.csv",
    }
    for provenance_key, artifact_name in expected_links.items():
        artifact = wave1_artifacts.get(artifact_name)
        if not isinstance(artifact, dict) or not isinstance(artifact.get("sha256"), str):
            raise ValueError(f"Wave-1 materialization lacks {artifact_name} provenance")
        if noise_provenance.get(provenance_key) != artifact["sha256"]:
            raise ValueError(f"Noise materialization does not reference the frozen Wave-1 {artifact_name}")


def load_frozen_inputs(args: argparse.Namespace, buckets: tuple[str, ...]) -> tuple[
    dict[str, object],
    np.ndarray,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    paths = {
        "pool_manifest": args.pool_dir / "manifest.json",
        "pool_counts": args.pool_dir / "candidate_pool_counts.npy",
        "pool_metadata": args.pool_dir / "candidate_pool_metadata.csv",
        "coverage_summary": args.pool_dir / "coverage_summary.csv",
        "coverage_weights": args.pool_dir / "coverage_weights.csv",
    }
    expected = {
        "pool_manifest": POOL_MANIFEST_SHA256,
        "pool_counts": POOL_COUNTS_SHA256,
        "pool_metadata": POOL_METADATA_SHA256,
        "coverage_summary": COVERAGE_SUMMARY_SHA256,
        "coverage_weights": COVERAGE_WEIGHTS_SHA256,
    }
    for label, path in paths.items():
        validate_file_hash(path, expected[label], label)
    validate_file_hash(args.prefix_weights, PREFIX_WEIGHTS_SHA256, "prefix weights")
    if args.expected_wave1_materialization_sha256 is None or args.expected_noise_materialization_sha256 is None:
        raise ValueError("Normal selection requires both expected materialization manifest SHA-256 values")
    wave1_materialization = load_materialization_manifest(
        args.wave1_dir,
        args.expected_wave1_materialization_sha256,
        {
            "branch_results.csv",
            "branch_fit_matrix.csv",
            "uncheatable_metrics_long.csv",
            "cross_wave_anchor.csv",
            "materialization_coverage.json",
        },
    )
    noise_materialization = load_materialization_manifest(
        args.noise_dir,
        args.expected_noise_materialization_sha256,
        {"noise_summary_n5.csv", "materialization_coverage.json"},
    )
    validate_noise_wave1_crosslink(wave1_materialization, noise_materialization)
    pool_manifest = json.loads(paths["pool_manifest"].read_text())
    pool = np.load(paths["pool_counts"], allow_pickle=False).astype(float) / branch_design.MIXTURE_BLOCK_SIZE
    pool_metadata = pd.read_csv(paths["pool_metadata"])
    coverage_summary = pd.read_csv(paths["coverage_summary"])
    coverage_weights = pd.read_csv(paths["coverage_weights"])
    wave1_coverage = json.loads((args.wave1_dir / "materialization_coverage.json").read_text())
    noise_coverage = json.loads((args.noise_dir / "materialization_coverage.json").read_text())
    if not wave1_coverage.get("complete") or int(wave1_coverage.get("completed_fit_rows", 0)) != EXPECTED_WAVE1_FIT_ROWS:
        raise ValueError("Wave 1 is not complete")
    observed_wave_hashes = tuple(row["continuation_weights_sha256"] for row in wave1_coverage["manifests"])
    if observed_wave_hashes != WAVE1_CONTINUATION_SHA256:
        raise ValueError(f"Wave-1 continuation contracts changed: {observed_wave_hashes}")
    if not noise_coverage.get("complete") or not noise_coverage.get("n5_summary_available"):
        raise ValueError("The n=5 noise-control summary is not complete")
    wave1 = pd.read_csv(args.wave1_dir / "branch_fit_matrix.csv")
    anchor = pd.read_csv(args.wave1_dir / "cross_wave_anchor.csv")
    noise_summary = pd.read_csv(args.noise_dir / "noise_summary_n5.csv")
    if len(pool) != int(pool_manifest["candidate_pool_rows"]):
        raise ValueError("Candidate pool row count changed")
    if len(coverage_summary) != EXPECTED_FIXED_COVERAGE_ROWS:
        raise ValueError("Fixed coverage row count changed")
    if int(coverage_summary.referee_holdout.sum()) != EXPECTED_REFEREE_ROWS:
        raise ValueError("Referee holdout count changed")
    if len(wave1) != EXPECTED_WAVE1_FIT_ROWS or not wave1.fit_budget.astype(str).str.lower().eq("true").all():
        raise ValueError("Wave-1 fit matrix changed")
    frozen_design = frozen_wave1_design_weights(buckets)
    observed_weights = wave1_weights(wave1, buckets)
    if not np.array_equal(observed_weights, frozen_design):
        raise ValueError("Materialized Wave-1 mixtures differ from the frozen continuation designs")
    derived_support_radius = median_nearest_neighbor_hellinger(frozen_design)
    if not np.isclose(derived_support_radius, GUIDED_SUPPORT_RADIUS, rtol=0.0, atol=1e-15):
        raise ValueError(f"Guided support radius changed: {derived_support_radius} != {GUIDED_SUPPORT_RADIUS}")
    return (
        pool_manifest,
        pool,
        pool_metadata,
        coverage_summary,
        coverage_weights,
        wave1,
        anchor,
        noise_summary,
        pd.DataFrame([wave1_materialization, noise_materialization]),
    )


def wave1_weights(frame: pd.DataFrame, buckets: tuple[str, ...]) -> np.ndarray:
    columns = [f"phase_1_{bucket}" for bucket in buckets]
    if not set(columns).issubset(frame.columns):
        raise ValueError("Wave-1 phase-1 weight columns are incomplete")
    weights = frame.loc[:, columns].to_numpy(dtype=float)
    hellinger_coordinates(weights)
    return weights


def target_prefix_weights(path: Path, buckets: tuple[str, ...]) -> np.ndarray:
    frame = pd.read_csv(path)
    group = frame[frame.candidate_id.eq(TARGET_PREFIX)]
    if tuple(group.bucket) != buckets or len(group) != len(buckets):
        raise ValueError("Frozen KL0.05 prefix weights changed")
    weights = group.phase_0_weight.to_numpy(dtype=float)
    if not np.array_equal(branch_design.runtime_counts(weights), group.phase_0_count.to_numpy(dtype=int)):
        raise ValueError("Frozen KL0.05 prefix is not runtime exact")
    return weights


def noise_and_signal_gates(
    response: np.ndarray,
    anchor: pd.DataFrame,
    noise_summary: pd.DataFrame,
) -> tuple[float, float, float, bool, dict[str, object]]:
    rows = noise_summary[noise_summary.metric.eq(TARGET_METRIC_LONG)]
    if len(rows) != 2 or not rows.n.eq(5).all():
        raise ValueError("Expected two n=5 Uncheatable noise groups")
    expected_groups = {f"{TARGET_PREFIX}/{group}" for group in ("control_proportional", "fit_maximin_26")}
    if set(rows.noise_group_id.astype(str)) != expected_groups:
        raise ValueError(f"Noise-control identities changed: {sorted(rows.noise_group_id.astype(str))}")
    noise_sd = float(rows.sample_sd.max())
    if not np.isfinite(noise_sd) or noise_sd <= 0.0:
        raise ValueError(f"Noise-control sample SD must be positive and finite; found {noise_sd}")
    signal = float(np.quantile(response, 0.9) - np.quantile(response, 0.1))
    signal_to_noise = signal / max(noise_sd, np.finfo(float).eps)
    if len(anchor) != 1:
        raise ValueError("Cross-wave anchor is missing or ambiguous")
    anchor_delta = abs(float(anchor.iloc[0].uncheatable_bpb_wave1b_minus_wave1a))
    passed = signal_to_noise >= MINIMUM_RESPONSE_SNR and anchor_delta <= MAXIMUM_CROSS_WAVE_ANCHOR_DELTA
    return (
        noise_sd,
        signal,
        anchor_delta,
        passed,
        {
            "conservative_noise_sd": noise_sd,
            "response_q90_minus_q10": signal,
            "response_signal_to_noise": signal_to_noise,
            "minimum_response_signal_to_noise": MINIMUM_RESPONSE_SNR,
            "cross_wave_anchor_absolute_bpb": anchor_delta,
            "maximum_cross_wave_anchor_absolute_bpb": MAXIMUM_CROSS_WAVE_ANCHOR_DELTA,
            "passed": passed,
        },
    )


def coverage_coordinates(coverage: pd.DataFrame, pool: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    indices = coverage.pool_index.to_numpy(dtype=int)
    if len(np.unique(indices)) != len(indices) or np.any(indices < 0) or np.any(indices >= len(pool)):
        raise ValueError("Fixed coverage pool indices are invalid")
    return indices, hellinger_coordinates(pool[indices])


def select_guided_indices(
    pool: np.ndarray,
    wave1: np.ndarray,
    coverage: pd.DataFrame,
    winner_predictions: np.ndarray | None,
    all_predictions: np.ndarray | None,
) -> tuple[np.ndarray, list[str], dict[str, np.ndarray], str]:
    coordinates = hellinger_coordinates(pool)
    wave1_coordinates = hellinger_coordinates(wave1)
    coverage_indices, fixed_coordinates = coverage_coordinates(coverage, pool)
    nearest_wave1 = minimum_distances(coordinates, wave1_coordinates)
    eligible = nearest_wave1 <= GUIDED_SUPPORT_RADIUS + 1e-12
    eligible[coverage_indices] = False
    references = np.vstack([wave1_coordinates, fixed_coordinates])
    if winner_predictions is None or all_predictions is None:
        selected = outcome_blind_maximin(
            coordinates,
            references,
            eligible,
            GUIDED_LCB_ROWS + GUIDED_DISAGREEMENT_ROWS,
        )
        return (
            selected,
            ["fallback_maximin"] * len(selected),
            {"nearest_wave1": nearest_wave1},
            "outcome_blind_fallback_no_eligible_model",
        )
    if winner_predictions.ndim != 2 or all_predictions.ndim != 2:
        raise ValueError("Candidate prediction ensembles must be matrices")
    if winner_predictions.shape[1] != len(pool) or all_predictions.shape[1] != len(pool):
        raise ValueError("Candidate prediction ensembles have the wrong pool width")
    if not np.isfinite(winner_predictions).all() or not np.isfinite(all_predictions).all():
        raise ValueError("Candidate prediction ensembles contain non-finite values")
    predicted_mean = winner_predictions.mean(axis=0)
    predicted_std = winner_predictions.std(axis=0, ddof=1)
    disagreement = all_predictions.std(axis=0, ddof=1) if len(all_predictions) > 1 else predicted_std
    lcb = predicted_mean - LCB_STANDARD_DEVIATIONS * predicted_std
    diagnostics = {
        "predicted_mean": predicted_mean,
        "predicted_std": predicted_std,
        "model_disagreement": disagreement,
        "lcb": lcb,
        "nearest_wave1": nearest_wave1,
    }
    if (
        float(predicted_std[eligible].max(initial=0.0)) <= MINIMUM_PREDICTION_SPREAD
        or float(disagreement[eligible].max(initial=0.0)) <= MINIMUM_PREDICTION_SPREAD
    ):
        selected = outcome_blind_maximin(
            coordinates,
            references,
            eligible,
            GUIDED_LCB_ROWS + GUIDED_DISAGREEMENT_ROWS,
        )
        return (
            selected,
            ["fallback_maximin"] * len(selected),
            diagnostics,
            "outcome_blind_fallback_degenerate_instability",
        )
    lcb_indices = greedy_ranked_selection(
        coordinates,
        references,
        eligible,
        lcb,
        GUIDED_LCB_ROWS,
        descending=False,
    )
    after_lcb_references = np.vstack([references, coordinates[lcb_indices]])
    disagreement_eligible = eligible.copy()
    disagreement_eligible[lcb_indices] = False
    disagreement_indices = greedy_ranked_selection(
        coordinates,
        after_lcb_references,
        disagreement_eligible,
        disagreement,
        GUIDED_DISAGREEMENT_ROWS,
        descending=True,
    )
    selected = np.concatenate([lcb_indices, disagreement_indices])
    return (
        selected,
        ["guided_lcb"] * len(lcb_indices) + ["guided_disagreement"] * len(disagreement_indices),
        diagnostics,
        "model_guided",
    )


def build_wave2_weights(
    buckets: tuple[str, ...],
    pool: np.ndarray,
    pool_metadata: pd.DataFrame,
    selected: np.ndarray,
    tranches: list[str],
    diagnostics: dict[str, np.ndarray],
    coverage_summary: pd.DataFrame,
    coverage_weights: pd.DataFrame,
    panel: branch_design.CanonicalPanelGeometry,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    phase_1_exposure = panel.phase1 * panel.c1[None, :]
    total_exposure = panel.phase0 * panel.c0[None, :] + phase_1_exposure
    phase_1_caps = phase_1_exposure.max(axis=0)
    total_caps = total_exposure.max(axis=0)
    summary_rows = []
    weight_rows = []
    tranche_positions: dict[str, int] = {}
    for pool_index, tranche in zip(selected, tranches, strict=True):
        position = tranche_positions.get(tranche, 0)
        tranche_positions[tranche] = position + 1
        continuation_id = f"fit_wave2_{tranche}_{position:02d}"
        weights = pool[pool_index]
        counts = branch_design.runtime_counts(weights)
        metadata = pool_metadata.iloc[pool_index]
        summary = {
            "continuation_id": continuation_id,
            "role": f"wave2_{tranche}",
            "selection_tranche": tranche,
            "fit_budget": True,
            "referee_holdout": False,
            "pool_index": int(pool_index),
            "tv_to_proportional": float(metadata.tv_to_proportional),
            "hellinger_to_proportional": float(metadata.hellinger_to_proportional),
            "max_phase_1_materialized_epoch": float(np.max(weights * panel.c1)),
        }
        for key, values in diagnostics.items():
            summary[key] = float(values[pool_index])
        summary_rows.append(summary)
        for bucket_position, bucket in enumerate(buckets):
            weight_rows.append(
                {
                    **summary,
                    "bucket": bucket,
                    "phase_1_count": int(counts[bucket_position]),
                    "phase_1_weight": float(weights[bucket_position]),
                    "phase_1_materialized_epochs": float(weights[bucket_position] * panel.c1[bucket_position]),
                    "historical_phase_1_bucket_epoch_cap": float(phase_1_caps[bucket_position]),
                    "historical_total_bucket_epoch_cap": float(total_caps[bucket_position]),
                }
            )
    guided_summary = pd.DataFrame(summary_rows)
    guided_weights = pd.DataFrame(weight_rows)
    fixed_summary = coverage_summary.copy()
    fixed_summary["selection_tranche"] = np.where(
        fixed_summary.role.eq("wave2_near_fill"),
        "fixed_near_coverage",
        "fixed_global_coverage",
    )
    fixed_weights = coverage_weights.copy()
    fixed_weights["selection_tranche"] = np.where(
        fixed_weights.role.eq("wave2_near_fill"),
        "fixed_near_coverage",
        "fixed_global_coverage",
    )
    for column in guided_weights.columns:
        if column not in fixed_weights:
            fixed_weights[column] = np.nan
    for column in fixed_weights.columns:
        if column not in guided_weights:
            guided_weights[column] = np.nan
    combined_weights = pd.concat([guided_weights[fixed_weights.columns], fixed_weights], ignore_index=True)
    for column in guided_summary.columns:
        if column not in fixed_summary:
            fixed_summary[column] = np.nan
    for column in fixed_summary.columns:
        if column not in guided_summary:
            guided_summary[column] = np.nan
    combined_summary = pd.concat([guided_summary[fixed_summary.columns], fixed_summary], ignore_index=True)
    if combined_summary.continuation_id.nunique() != EXPECTED_WAVE2_ROWS:
        raise ValueError("Wave-2 continuation identities changed")
    if len(combined_weights) != EXPECTED_WAVE2_ROWS * len(buckets):
        raise ValueError("Wave-2 long-form design row count changed")
    if int(combined_summary.referee_holdout.sum()) != EXPECTED_REFEREE_ROWS:
        raise ValueError("Wave-2 referee count changed")
    return combined_summary, combined_weights


def main() -> None:
    args = parse_args()
    validate_dependency_versions()
    contract = validate_frozen_contract()
    if args.contract_only:
        print(json.dumps({"contract_sha256": FROZEN_CONTRACT_SHA256, **contract}, indent=2, sort_keys=True))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)

    panel = branch_design.load_canonical_panel_geometry()
    buckets = panel.buckets
    (
        pool_manifest,
        pool,
        pool_metadata,
        fixed_coverage,
        fixed_coverage_weights,
        wave1,
        anchor,
        noise_summary,
        coverage_records,
    ) = load_frozen_inputs(args, buckets)
    train_weights = wave1_weights(wave1, buckets)
    response = wave1[TARGET_METRIC].to_numpy(dtype=float)
    prefix_weights = target_prefix_weights(args.prefix_weights, buckets)
    feature_bank = build_feature_bank(pool, prefix_weights, panel.c0, panel.c1)
    noise_sd, response_signal, anchor_delta, global_gates_passed, global_gates = noise_and_signal_gates(
        response,
        anchor,
        noise_summary,
    )

    scores = []
    prediction_frames = []
    fold_frames = []
    for model_name in MODEL_NAMES:
        features = feature_bank.transform(train_weights, model_name)
        score, predictions, folds = benchmark_model(model_name, features, train_weights, response, noise_sd)
        scores.append(score)
        prediction_frames.append(predictions)
        fold_frames.append(folds)
    model_scores = pd.DataFrame([asdict(score) for score in scores])
    eligible = [score for score in scores if score.eligible and global_gates_passed]
    winner = min(
        eligible,
        key=lambda score: (score.mean_fold_regret_at_3, score.rmse, -score.mean_spearman, score.name),
        default=None,
    )
    if winner is None:
        predictions_by_fit: dict[str, np.ndarray] = {}
        selected_indices, tranches, diagnostics, selection_mode = select_guided_indices(
            pool,
            train_weights,
            fixed_coverage,
            None,
            None,
        )
        fitted_alphas: dict[str, list[float]] = {}
    else:
        predictions_by_fit, fitted_alphas = spatial_subfit_prediction_ensemble(
            [score.name for score in eligible],
            feature_bank,
            train_weights,
            response,
            pool,
        )
        winner_predictions = np.stack(
            [values for key, values in predictions_by_fit.items() if key.startswith(f"{winner.name}/")]
        )
        all_predictions = np.stack(list(predictions_by_fit.values()))
        selected_indices, tranches, diagnostics, selection_mode = select_guided_indices(
            pool,
            train_weights,
            fixed_coverage,
            winner_predictions,
            all_predictions,
        )

    combined_summary, combined_weights = build_wave2_weights(
        buckets,
        pool,
        pool_metadata,
        selected_indices,
        tranches,
        diagnostics,
        fixed_coverage,
        fixed_coverage_weights,
        panel,
    )
    combined_summary.to_csv(args.output_dir / "continuation_summary.csv", index=False)
    combined_weights.to_csv(args.output_dir / "continuation_weights.csv", index=False)
    model_scores.to_csv(args.output_dir / "model_scores.csv", index=False)
    pd.concat(prediction_frames, ignore_index=True).to_csv(args.output_dir / "oof_predictions.csv", index=False)
    pd.concat(fold_frames, ignore_index=True).to_csv(args.output_dir / "fold_selection_metrics.csv", index=False)
    candidate_predictions = pool_metadata.copy()
    fixed_indices = fixed_coverage.pool_index.to_numpy(dtype=int)
    candidate_predictions["selected_guided_wave2"] = False
    candidate_predictions["selected_fixed_wave2"] = False
    candidate_predictions.loc[selected_indices, "selected_guided_wave2"] = True
    candidate_predictions.loc[fixed_indices, "selected_fixed_wave2"] = True
    candidate_predictions["selected_wave2"] = (
        candidate_predictions.selected_guided_wave2 | candidate_predictions.selected_fixed_wave2
    )
    for key, values in diagnostics.items():
        candidate_predictions[key] = values
    candidate_predictions.to_csv(args.output_dir / "candidate_predictions.csv", index=False)

    input_hashes = {
        "wave1_materialization_manifest": args.expected_wave1_materialization_sha256,
        "noise_materialization_manifest": args.expected_noise_materialization_sha256,
    }
    manifest = {
        "selection_mode": selection_mode,
        "target_prefix": TARGET_PREFIX,
        "target_metric": TARGET_METRIC,
        "winner": winner.name if winner is not None else None,
        "eligible_models": [score.name for score in eligible],
        "fitted_alphas": fitted_alphas,
        "global_gates": global_gates,
        "noise_sd": noise_sd,
        "response_q90_minus_q10": response_signal,
        "cross_wave_anchor_absolute_bpb": anchor_delta,
        "selected_guided_pool_indices": selected_indices.tolist(),
        "selected_guided_tranches": tranches,
        "guided_rows": len(selected_indices),
        "fixed_coverage_rows": len(fixed_coverage),
        "total_wave2_rows": len(combined_summary),
        "referee_holdout_rows": int(combined_summary.referee_holdout.sum()),
        "contract_sha256": FROZEN_CONTRACT_SHA256,
        "continuation_summary_sha256": file_sha256(args.output_dir / "continuation_summary.csv"),
        "continuation_weights_sha256": file_sha256(args.output_dir / "continuation_weights.csv"),
        "model_scores_sha256": file_sha256(args.output_dir / "model_scores.csv"),
        "input_hashes": input_hashes,
        "pool_manifest": pool_manifest,
        "materialization_coverage": coverage_records.to_dict(orient="records"),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps({key: value for key, value in manifest.items() if key != "pool_manifest"}, indent=2, sort_keys=True)
    )


if __name__ == "__main__":
    main()
