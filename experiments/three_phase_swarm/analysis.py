# Copyright 2025 The Marin Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Analysis utilities for three-phase data mixture swarm experiments.

Provides tools for:
- Collecting results from W&B
- Running regression analysis on weight configurations
- Comparing phase importance
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class RegressionResults:
    """Results from regression analysis."""

    feature_names: list[str]
    coefficients: dict[str, float]
    r2_score: float
    n_samples: int


def load_weight_configs(config_path: str | Path) -> dict:
    """Load weight configurations from a JSON file.

    Args:
        config_path: Path to the weight_configs.json file.

    Returns:
        Dictionary containing seed, n_runs, summary, and configs.
    """
    with open(config_path) as f:
        return json.load(f)


def collect_results_from_wandb(
    run_ids: list[str],
    entity: str = "marin",
    project: str = "marin",
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Collect training results from W&B for all runs.

    Args:
        run_ids: List of W&B run IDs.
        entity: W&B entity name.
        project: W&B project name.
        metrics: List of metric keys to collect. If None, uses defaults.

    Returns:
        DataFrame with run configurations and metrics.
    """
    try:
        import pandas as pd
        import wandb
    except ImportError as err:
        raise ImportError("pandas and wandb are required for W&B analysis") from err

    if metrics is None:
        metrics = [
            "eval/loss",
            "eval/paloma/c4_en/bpb",
            "eval/paloma/wikipedia_en/bpb",
            "eval_harness/gsm8k/acc",
            "eval_harness/mmlu/acc",
            "eval_harness/hellaswag/acc",
        ]

    api = wandb.Api()
    results = []

    for run_id in run_ids:
        try:
            run = api.run(f"{entity}/{project}/{run_id}")

            # Extract metrics from run summary
            row = {"run_id": run_id}
            for metric in metrics:
                row[metric] = run.summary.get(metric)

            # Extract weight configuration from run config if available
            config = run.config
            for phase in ["phase1", "phase2", "phase3"]:
                weights_key = f"{phase}_weights"
                if weights_key in config:
                    for partition, weight in config[weights_key].items():
                        row[f"{phase}_{partition}_weight"] = weight

            results.append(row)

        except Exception as e:
            logger.warning(f"Failed to fetch run {run_id}: {e}")
            continue

    return pd.DataFrame(results)


def build_features_from_configs(
    configs: list[dict],
    phase: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Build feature matrix from weight configurations.

    Args:
        configs: List of configuration dictionaries.
        phase: If specified, only use weights from this phase ("phase1", "phase2", "phase3").
               If None, use all phases.

    Returns:
        Tuple of (feature_matrix, feature_names).
    """
    from experiments.three_phase_swarm.weight_sampler import ThreePartitionWeightSampler

    partitions = ThreePartitionWeightSampler.PARTITIONS
    phases = ["phase1", "phase2", "phase3"] if phase is None else [phase]

    feature_names = []
    for p in phases:
        for partition in partitions:
            feature_names.append(f"{p}_{partition}")

    features = []
    for config in configs:
        row = []
        for p in phases:
            weights = config[f"{p}_weights"]
            for partition in partitions:
                row.append(weights.get(partition, 0.0))
        features.append(row)

    return np.array(features), feature_names


def run_regression_analysis(
    features: np.ndarray,
    targets: np.ndarray,
    feature_names: list[str],
    alpha: float = 1.0,
) -> RegressionResults:
    """Run Ridge regression to identify important features.

    Args:
        features: Feature matrix of shape (n_samples, n_features).
        targets: Target vector of shape (n_samples,).
        feature_names: Names of the features.
        alpha: Regularization strength for Ridge regression.

    Returns:
        RegressionResults with coefficients and R2 score.
    """
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
    except ImportError as err:
        raise ImportError("scikit-learn is required for regression analysis") from err

    # Remove samples with missing targets
    mask = ~np.isnan(targets)
    X = features[mask]
    y = targets[mask]

    if len(y) < 10:
        raise ValueError(f"Not enough samples for regression: {len(y)}")

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Fit Ridge regression
    model = Ridge(alpha=alpha)
    model.fit(X_scaled, y)

    return RegressionResults(
        feature_names=feature_names,
        coefficients=dict(zip(feature_names, model.coef_, strict=True)),
        r2_score=model.score(X_scaled, y),
        n_samples=len(y),
    )


def compare_phase_importance(
    configs: list[dict],
    targets: np.ndarray,
    target_name: str = "eval/loss",
) -> dict[str, RegressionResults]:
    """Compare the relative importance of each training phase.

    Runs separate regressions for each phase to see which phase's
    weight distribution has the strongest predictive power.

    Args:
        configs: List of configuration dictionaries.
        targets: Target metric values.
        target_name: Name of the target metric (for logging).

    Returns:
        Dictionary mapping phase names to RegressionResults.
    """
    results = {}

    for phase in ["phase1", "phase2", "phase3"]:
        features, feature_names = build_features_from_configs(configs, phase=phase)
        try:
            phase_results = run_regression_analysis(features, targets, feature_names)
            results[phase] = phase_results
            logger.info(f"{phase} R2 for {target_name}: {phase_results.r2_score:.4f} " f"(n={phase_results.n_samples})")
        except ValueError as e:
            logger.warning(f"Skipping {phase}: {e}")

    return results


def analyze_sft_importance(
    configs: list[dict],
    targets: np.ndarray,
    target_name: str = "eval/loss",
) -> dict:
    """Analyze the importance of SFT data weight in each phase.

    This directly tests the hypothesis that SFT data weight in Phase 3
    correlates most strongly with final performance.

    Args:
        configs: List of configuration dictionaries.
        targets: Target metric values.
        target_name: Name of the target metric.

    Returns:
        Dictionary with SFT correlations per phase.
    """
    results = {}

    for phase in ["phase1", "phase2", "phase3"]:
        # Extract SFT weights for this phase
        sft_weights = np.array([config[f"{phase}_weights"].get("sft", 0.0) for config in configs])

        # Remove samples with missing targets
        mask = ~np.isnan(targets)
        sft_w = sft_weights[mask]
        y = targets[mask]

        if len(y) < 10:
            continue

        # Compute correlation
        correlation = np.corrcoef(sft_w, y)[0, 1]

        results[phase] = {
            "correlation": float(correlation),
            "mean_sft_weight": float(np.mean(sft_w)),
            "std_sft_weight": float(np.std(sft_w)),
            "n_samples": len(y),
        }

        logger.info(f"{phase} SFT weight correlation with {target_name}: {correlation:.4f}")

    return results


def print_analysis_summary(
    phase_results: dict[str, RegressionResults],
    sft_analysis: dict,
    target_name: str = "eval/loss",
):
    """Print a summary of the analysis results.

    Args:
        phase_results: Results from compare_phase_importance.
        sft_analysis: Results from analyze_sft_importance.
        target_name: Name of the target metric.
    """
    print(f"\n{'='*60}")
    print(f"Analysis Summary for {target_name}")
    print("=" * 60)

    print("\n1. Phase Importance (R2 scores):")
    print("-" * 40)
    for phase, results in sorted(phase_results.items()):
        print(f"  {phase}: R2 = {results.r2_score:.4f} (n={results.n_samples})")

    print("\n2. Top Coefficients per Phase:")
    print("-" * 40)
    for phase, results in sorted(phase_results.items()):
        sorted_coefs = sorted(results.coefficients.items(), key=lambda x: abs(x[1]), reverse=True)
        print(f"  {phase}:")
        for name, coef in sorted_coefs[:3]:
            print(f"    {name}: {coef:.4f}")

    print("\n3. SFT Weight Correlation by Phase:")
    print("-" * 40)
    for phase, stats in sorted(sft_analysis.items()):
        print(f"  {phase}: corr={stats['correlation']:.4f}, " f"mean_weight={stats['mean_sft_weight']:.3f}")

    # Highlight key finding
    if sft_analysis:
        phase3_corr = sft_analysis.get("phase3", {}).get("correlation", 0)
        phase1_corr = sft_analysis.get("phase1", {}).get("correlation", 0)

        print("\n" + "=" * 60)
        if abs(phase3_corr) > abs(phase1_corr):
            print(
                "KEY FINDING: Phase 3 SFT weight has stronger correlation "
                f"({phase3_corr:.4f}) than Phase 1 ({phase1_corr:.4f})"
            )
        else:
            print(
                "NOTE: Phase 1 SFT weight has stronger correlation "
                f"({phase1_corr:.4f}) than Phase 3 ({phase3_corr:.4f})"
            )
    print("=" * 60 + "\n")
