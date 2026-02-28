# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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

"""50 additional training runs for the three-phase StarCoder experiment.

This is a follow-up to three_phase_starcoder_experiment.py, adding 50 new
random-sampled runs to improve statistical power for fitting functional forms
over a (N=3, M=2) weight space.

Key differences from v1:
- Loads existing v1 configs from GCS and uses rejection sampling (via
  WeightSampler.sample_n_configs(existing_configs=...)) to ensure new points
  are well-separated from existing ones.
- No baselines (they already exist in v1).
- Supports --dry-run mode for visualizing sampling coverage before submitting.

Sampling uses the same Dirichlet parameters as v1 (min_weight=0.05,
min_strength=0.1, max_strength=0.5, temp=0.5).

Usage:
    # Dry-run: visualize sampling coverage
    python -m experiments.domain_phase_mix.three_phase_starcoder_experiment_v2 --dry-run

    # Dry-run with custom min distance
    python -m experiments.domain_phase_mix.three_phase_starcoder_experiment_v2 --dry-run --min-dist 0.04

    # Submit training runs
    python -m experiments.domain_phase_mix.three_phase_starcoder_experiment_v2

    # Run analysis (after training completes)
    python -m experiments.domain_phase_mix.three_phase_starcoder_experiment_v2 --analyze
"""

import argparse
import logging
import os
import sys

import numpy as np

from experiments.domain_phase_mix.analysis import create_analysis_step
from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.three_phase_starcoder_experiment import (
    ANALYSIS_METRICS,
    BASELINES,
    EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS,
    TOKENIZER_CACHE_BASE,
    TOKENIZER_NAME,
    _load_original_weight_configs,
    create_baseline_weight_configs,
    create_three_phase_experiment,
)
from experiments.domain_phase_mix.weight_sampler import DirichletSamplingParams, SamplingStrategy
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import executor_main
from marin.utils import create_cache_tokenizer_step

logger = logging.getLogger("ray")


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

NAME = "pinlin_calvin_xu/data_mixture/three_phase_starcoder_2"

# GCS path prefix for loading v1 weight configs (used by _load_original_weight_configs)
V1_NAME = "pinlin_calvin_xu/data_mixture/three_phase_starcoder_1"

# Same Dirichlet parameters as v1, but with the max_ratio cap removed so that
# StarCoder weight can reach 1.0 (v1 capped it at ~54% via natural_proportion * 15).
SAMPLING_PARAMS = DirichletSamplingParams(
    strategy=SamplingStrategy.DIRICHLET,
    min_weight=0.05,
    min_config_distance=0.10,
    min_strength=0.1,
    max_strength=0.5,
    temp=0.5,
    max_ratio=float("inf"),
)


# ============================================================================
# LOADING EXISTING CONFIGS
# ============================================================================


def load_all_existing_configs() -> list[WeightConfig]:
    """Load all existing configs (random + baselines) from the v1 experiment."""
    random_configs = _load_original_weight_configs(V1_NAME)
    baseline_configs = create_baseline_weight_configs(BASELINES)
    return random_configs + baseline_configs


# ============================================================================
# DRY-RUN VISUALIZATION
# ============================================================================


def dry_run(
    n_new: int,
    min_config_distance: float,
    seed: int,
):
    """Sample configs and print coverage stats without creating training steps.

    For a 3-phase, 2-domain experiment there are 3 weight dimensions (one per
    phase), so we produce pairwise scatter plots of StarCoder fractions.

    Args:
        n_new: Number of new configs to sample.
        min_config_distance: Minimum config distance for rejection sampling.
        seed: Random seed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiment = create_three_phase_experiment(name=NAME)

    # Load existing random configs + baselines
    random_configs = _load_original_weight_configs(V1_NAME)
    baseline_configs = create_baseline_weight_configs(BASELINES)
    all_existing = random_configs + baseline_configs
    logger.info(f"Loaded {len(random_configs)} random configs + {len(baseline_configs)} baselines from v1")

    # Use v2 sampling params (no max_ratio cap) with the requested min distance
    experiment.sampling_params = SAMPLING_PARAMS
    experiment.sampling_params.min_config_distance = min_config_distance

    # Sample new configs using the framework's rejection sampling
    sampler = experiment.create_weight_sampler(seed=seed)
    new_configs = sampler.sample_n_configs(n_new, deduplicate=True, existing_configs=all_existing)

    # Extract starcoder proportions for each phase
    def get_sc_props(config: WeightConfig) -> tuple[float, float, float]:
        p0 = config.phase_weights["phase_0"].get("starcoder", 0.0)
        p1 = config.phase_weights["phase_1"].get("starcoder", 0.0)
        p2 = config.phase_weights["phase_2"].get("starcoder", 0.0)
        return p0, p1, p2

    existing_props = np.array([get_sc_props(c) for c in random_configs])
    baseline_props = np.array([get_sc_props(c) for c in baseline_configs])
    new_props = np.array([get_sc_props(c) for c in new_configs])

    # Compute nearest-neighbor distances for new configs
    all_configs = all_existing + new_configs
    nn_dists_new = []
    for nc in new_configs:
        dists = [sampler._config_distance(nc, other) for other in all_configs if other is not nc]
        nn_dists_new.append(min(dists))

    # Print stats
    print(f"\n{'=' * 60}")
    print("Dry-run sampling summary")
    print(f"{'=' * 60}")
    print(f"Existing random (v1):     {len(random_configs)}")
    print(f"Existing baselines (v1):  {len(baseline_configs)}")
    print(f"New configs (v2):         {len(new_configs)}")
    print(f"Total:                    {len(all_existing) + len(new_configs)}")
    print(f"Min config distance:      {min_config_distance}")
    print(f"Seed:                     {seed}")
    print("\nNearest-neighbor distances (new configs):")
    print(f"  min:  {min(nn_dists_new):.4f}")
    print(f"  mean: {np.mean(nn_dists_new):.4f}")
    print(f"  max:  {max(nn_dists_new):.4f}")
    print(f"{'=' * 60}\n")

    # Pairwise scatter plots for the 3 phase dimensions
    phase_pairs = [("phase_0", "phase_1", 0, 1), ("phase_0", "phase_2", 0, 2), ("phase_1", "phase_2", 1, 2)]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, (xlabel, ylabel, xi, yi) in zip(axes, phase_pairs, strict=True):
        ax.scatter(
            existing_props[:, xi],
            existing_props[:, yi],
            c="royalblue",
            marker="o",
            s=40,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
            label=f"v1 random (n={len(random_configs)})",
            zorder=3,
        )
        ax.scatter(
            baseline_props[:, xi],
            baseline_props[:, yi],
            c="seagreen",
            marker="D",
            s=60,
            alpha=0.9,
            edgecolors="k",
            linewidths=0.5,
            label=f"v1 baselines (n={len(baseline_configs)})",
            zorder=4,
        )
        ax.scatter(
            new_props[:, xi],
            new_props[:, yi],
            c="darkorange",
            marker="o",
            s=40,
            alpha=0.7,
            edgecolors="k",
            linewidths=0.5,
            label=f"v2 new (n={len(new_configs)})",
            zorder=3,
        )
        ax.set_xlabel(f"StarCoder fraction ({xlabel})", fontsize=10)
        ax.set_ylabel(f"StarCoder fraction ({ylabel})", fontsize=10)
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        ax.set_aspect("equal")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Sampling coverage: {len(random_configs)} existing + {len(baseline_configs)} baselines + "
        f"{len(new_configs)} new (min_dist={min_config_distance})",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = os.path.join(os.path.dirname(__file__), "exploratory", "v2_three_phase_sampling_preview.png")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter plot to {out_path}")


# ============================================================================
# TRAINING & ANALYSIS
# ============================================================================


def run_training(
    n_new: int,
    min_config_distance: float,
    seed: int,
):
    """Sample new configs and submit training runs.

    Args:
        n_new: Number of new configs to sample.
        min_config_distance: Minimum config distance for rejection sampling.
        seed: Random seed.
    """
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    experiment = create_three_phase_experiment(name=NAME)

    # Set environment variables
    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = TOKENIZER_CACHE_BASE
    os.environ["HF_ALLOW_CODE_EVAL"] = "1"

    # Cache steps
    cache_tokenizer_step = create_cache_tokenizer_step(
        tokenizer_name=TOKENIZER_NAME,
        gcs_path=os.path.join(TOKENIZER_CACHE_BASE, TOKENIZER_NAME.replace("/", "--")),
        name_prefix=NAME,
    )
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=EVAL_TASKS,
        gcs_path=EVAL_DATASETS_CACHE_PATH,
        name_prefix=NAME,
    )

    # Load existing configs for rejection sampling
    all_existing = load_all_existing_configs()
    logger.info(f"Loaded {len(all_existing)} existing configs (random + baselines) from v1")

    # Use v2 sampling params (no max_ratio cap) with the requested min distance
    experiment.sampling_params = SAMPLING_PARAMS
    experiment.sampling_params.min_config_distance = min_config_distance

    # Use create_swarm_steps with existing_configs for rejection sampling
    weight_configs_step, training_steps = experiment.create_swarm_steps(
        n_runs=n_new,
        seed=seed,
        name_prefix=NAME,
        existing_configs=all_existing,
    )

    # Analysis step depends on all training steps
    analysis_step = create_analysis_step(
        weight_configs_step=weight_configs_step,
        name_prefix=NAME,
        metrics=ANALYSIS_METRICS,
        depends_on=list(training_steps),
    )

    logger.info(f"Created {len(training_steps)} training steps for v2")

    all_steps = [
        cache_tokenizer_step,
        cache_eval_datasets_step,
        weight_configs_step,
        *training_steps,
        analysis_step,
    ]
    executor_main(
        steps=all_steps,
        description=f"Three-phase starcoder v2: {n_new} additional runs",
    )


def run_analysis():
    """Collect results from W&B for the v2 experiment."""
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    experiment = create_three_phase_experiment(name=NAME)

    # Load v2 weight configs from GCS
    configs = _load_original_weight_configs(NAME)
    logger.info(f"Loaded {len(configs)} v2 configs")

    weight_configs_step = experiment.create_weight_configs_step(
        configs=configs,
        summary={},
        seed=0,
        name_prefix=f"{NAME}_analysis",
    )
    analysis_step = create_analysis_step(
        weight_configs_step=weight_configs_step,
        name_prefix=NAME,
        metrics=ANALYSIS_METRICS,
    )

    executor_main(
        steps=[weight_configs_step, analysis_step],
        description=f"Analysis for {NAME}",
    )


# ============================================================================
# CLI
# ============================================================================


def _parse_args():
    parser = argparse.ArgumentParser(description="50 additional training runs for three-phase StarCoder experiment.")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=75,
        help="Number of new training runs (default: 75).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for weight sampling (default: 100, distinct from v1's 42).",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.10,
        help="Minimum config distance for rejection sampling (default: 0.10).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample configs and visualize without creating training steps.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis only (collect results from W&B).",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    if args.dry_run:
        dry_run(
            n_new=args.n_runs,
            min_config_distance=args.min_dist,
            seed=args.seed,
        )
    elif args.analyze:
        run_analysis()
    else:
        run_training(
            n_new=args.n_runs,
            min_config_distance=args.min_dist,
            seed=args.seed,
        )
