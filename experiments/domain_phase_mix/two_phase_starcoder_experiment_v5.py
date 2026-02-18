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

"""50 additional training runs for the two-phase StarCoder experiment.

This is a follow-up to two_phase_starcoder_experiment.py (v4), adding 50 new
random-sampled runs to improve statistical power for 15-parameter functional
forms (pushes n/k from ~4 to ~7).

Key differences from v4:
- Loads existing v4 configs from GCS and uses rejection sampling to ensure
  new points are well-separated from existing ones.
- No min_phase_change constraint (was artificially excluding configs where
  both phases have similar weights).
- No baselines (they already exist in v4).
- Supports --dry-run mode for visualizing sampling coverage before submitting.

Sampling uses the same Dirichlet parameters as v4 (min_weight=0.05,
min_strength=0.1, max_strength=0.5, temp=0.5) with two overrides:
- min_phase_change=0.0 (allows configs where both phases are similar)
- max_ratio cap removed (v4 had 15x, limiting StarCoder to ~56%)
min_config_distance is handled by a custom rejection loop against all
existing v4 configs (random + baselines) plus newly accepted points.

Usage:
    # Dry-run: visualize sampling coverage
    python -m experiments.domain_phase_mix.two_phase_starcoder_experiment_v5 --dry-run

    # Dry-run with custom min distance
    python -m experiments.domain_phase_mix.two_phase_starcoder_experiment_v5 --dry-run --min-dist 0.04

    # Submit training runs
    python -m experiments.domain_phase_mix.two_phase_starcoder_experiment_v5

    # Run analysis (after training completes)
    python -m experiments.domain_phase_mix.two_phase_starcoder_experiment_v5 --analyze
"""

import argparse
import json
import logging
import os
import sys

import fsspec
import numpy as np

from experiments.domain_phase_mix.analysis import create_analysis_step
from experiments.domain_phase_mix.config import WeightConfig
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.two_phase_starcoder_experiment import (
    ANALYSIS_METRICS,
    BASELINES,
    EVAL_DATASETS_CACHE_PATH,
    EVAL_TASKS,
    TOKENIZER_CACHE_BASE,
    TOKENIZER_NAME,
    create_baseline_weight_configs,
    create_two_phase_experiment,
)
from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import executor_main
from marin.utils import create_cache_tokenizer_step

logger = logging.getLogger("ray")


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_5"

# GCS path to existing v4 weight configs
V4_WEIGHT_CONFIGS_PATTERN = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_4/weight_configs-*/weight_configs.json"


# ============================================================================
# LOADING EXISTING CONFIGS
# ============================================================================


def load_existing_configs() -> list[WeightConfig]:
    """Load weight configs from the v4 experiment on GCS.

    Returns:
        List of WeightConfig objects from v4.
    """
    prefix = os.environ.get("MARIN_PREFIX", "gs://marin-us-central1")
    fs, base = fsspec.core.url_to_fs(prefix)
    matches = fs.glob(f"{base}/{V4_WEIGHT_CONFIGS_PATTERN}")

    if not matches:
        raise FileNotFoundError(
            f"No weight_configs found at {prefix}/{V4_WEIGHT_CONFIGS_PATTERN}. " "Run the v4 training swarm first."
        )

    if len(matches) > 1:
        logger.warning(f"Found multiple weight_configs: {matches}. Using the first one.")

    path = f"{fs.protocol}://{matches[0]}" if isinstance(fs.protocol, str) else f"{fs.protocol[0]}://{matches[0]}"
    logger.info(f"Loading existing weight configs from {path}")

    with fsspec.open(path) as f:
        data = json.load(f)

    return [WeightConfig.from_dict(c) for c in data["configs"]]


# ============================================================================
# REJECTION SAMPLING AGAINST EXISTING POINTS
# ============================================================================


def sample_new_configs(
    existing_configs: list[WeightConfig],
    n_new: int,
    experiment: MixtureExperiment,
    seed: int = 100,
    min_config_distance: float = 0.001,
) -> list[WeightConfig]:
    """Sample n_new configs that are well-separated from existing ones.

    Uses rejection sampling: each candidate is checked against all existing
    configs AND all previously accepted new configs. Candidates closer than
    min_config_distance (average L1/2 across phases) are rejected.

    Args:
        existing_configs: Configs from the v4 experiment to avoid.
        n_new: Number of new configs to sample.
        experiment: MixtureExperiment to create the sampler from.
        seed: Random seed for sampling.
        min_config_distance: Minimum average L1/2 distance to any existing
            or newly-accepted config.

    Returns:
        List of n_new WeightConfig objects.

    Raises:
        ValueError: If not enough configs could be sampled.
    """
    sampler = experiment.create_weight_sampler(seed=seed)
    # Override sampling params for v5:
    # - No min_phase_change (v4 had this implicitly nonzero)
    # - Remove max_ratio cap (v4 had 15x which limited StarCoder to ~56%)
    sampler.params.min_phase_change = 0.0
    sampler.upper_bounds = {name: 1.0 for name in sampler.domain_names}

    all_accepted = list(existing_configs)  # Distance check pool
    new_configs: list[WeightConfig] = []
    attempts = 0
    max_attempts = n_new * 10000

    while len(new_configs) < n_new and attempts < max_attempts:
        candidate = sampler.sample_config(run_id=len(new_configs))

        # Check distance to ALL points (existing + newly accepted)
        min_dist = min(sampler._config_distance(candidate, prev) for prev in all_accepted)

        if min_dist >= min_config_distance:
            candidate.run_id = len(new_configs)
            new_configs.append(candidate)
            all_accepted.append(candidate)

        attempts += 1

    if len(new_configs) < n_new:
        raise ValueError(
            f"Only sampled {len(new_configs)}/{n_new} after {max_attempts} attempts. "
            f"Try reducing --min-dist (currently {min_config_distance})."
        )

    logger.info(f"Sampled {n_new} new configs in {attempts} attempts " f"(rejection rate: {1 - n_new / attempts:.1%})")
    return new_configs


# ============================================================================
# DRY-RUN VISUALIZATION
# ============================================================================


def dry_run(
    n_new: int,
    min_config_distance: float,
    seed: int,
):
    """Sample configs and visualize coverage without creating training steps.

    Produces a scatter plot of (p0_starcoder, p1_starcoder) for existing (blue)
    and new (orange) configs.

    Args:
        n_new: Number of new configs to sample.
        min_config_distance: Minimum config distance for rejection sampling.
        seed: Random seed.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    experiment = create_two_phase_experiment(name=NAME)

    # Load existing random configs + baselines
    existing = load_existing_configs()
    baselines = create_baseline_weight_configs(BASELINES)
    all_existing = existing + baselines
    logger.info(f"Loaded {len(existing)} random configs + {len(baselines)} baselines from v4")

    # Sample new (rejection against both random and baseline configs)
    new_configs = sample_new_configs(
        existing_configs=all_existing,
        n_new=n_new,
        experiment=experiment,
        seed=seed,
        min_config_distance=min_config_distance,
    )

    # Extract starcoder proportions for plotting
    def get_sc_props(config: WeightConfig) -> tuple[float, float]:
        p0 = config.phase_weights["phase_0"].get("starcoder", 0.0)
        p1 = config.phase_weights["phase_1"].get("starcoder", 0.0)
        return p0, p1

    existing_props = np.array([get_sc_props(c) for c in existing])
    baseline_props = np.array([get_sc_props(c) for c in baselines])
    new_props = np.array([get_sc_props(c) for c in new_configs])

    # Compute nearest-neighbor distances for new configs
    sampler = experiment.create_weight_sampler(seed=seed)
    all_configs = all_existing + new_configs
    nn_dists_new = []
    for nc in new_configs:
        dists = [sampler._config_distance(nc, other) for other in all_configs if other is not nc]
        nn_dists_new.append(min(dists))

    # Print stats
    print(f"\n{'=' * 60}")
    print("Dry-run sampling summary")
    print(f"{'=' * 60}")
    print(f"Existing random (v4):     {len(existing)}")
    print(f"Existing baselines (v4):  {len(baselines)}")
    print(f"New configs (v5):         {len(new_configs)}")
    print(f"Total:                    {len(all_existing) + len(new_configs)}")
    print(f"Min config distance:      {min_config_distance}")
    print(f"Seed:                     {seed}")
    print("\nNearest-neighbor distances (new configs):")
    print(f"  min:  {min(nn_dists_new):.4f}")
    print(f"  mean: {np.mean(nn_dists_new):.4f}")
    print(f"  max:  {max(nn_dists_new):.4f}")
    print(f"{'=' * 60}\n")

    # Plot
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(
        existing_props[:, 0],
        existing_props[:, 1],
        c="royalblue",
        marker="o",
        s=40,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
        label=f"Existing v4 random (n={len(existing)})",
        zorder=3,
    )
    ax.scatter(
        baseline_props[:, 0],
        baseline_props[:, 1],
        c="seagreen",
        marker="D",
        s=60,
        alpha=0.9,
        edgecolors="k",
        linewidths=0.5,
        label=f"Existing v4 baselines (n={len(baselines)})",
        zorder=4,
    )
    ax.scatter(
        new_props[:, 0],
        new_props[:, 1],
        c="darkorange",
        marker="o",
        s=40,
        alpha=0.7,
        edgecolors="k",
        linewidths=0.5,
        label=f"New v5 (n={len(new_configs)})",
        zorder=3,
    )
    ax.set_xlabel("$p_0$ (StarCoder fraction, phase 0)", fontsize=12)
    ax.set_ylabel("$p_1$ (StarCoder fraction, phase 1)", fontsize=12)
    ax.set_title(
        f"Sampling coverage: {len(existing)} existing + {len(baselines)} baselines + "
        f"{len(new_configs)} new (min_dist={min_config_distance})",
        fontsize=11,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_aspect("equal")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    out_path = os.path.join(os.path.dirname(__file__), "exploratory", "v5_sampling_preview.png")
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

    experiment = create_two_phase_experiment(name=NAME)

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

    # Load existing random configs + baselines for rejection sampling
    existing = load_existing_configs()
    baselines = create_baseline_weight_configs(BASELINES)
    all_existing = existing + baselines
    logger.info(f"Loaded {len(existing)} random configs + {len(baselines)} baselines from v4")

    new_configs = sample_new_configs(
        existing_configs=all_existing,
        n_new=n_new,
        experiment=experiment,
        seed=seed,
        min_config_distance=min_config_distance,
    )

    # Save weight configs step
    sampler = experiment.create_weight_sampler(seed=seed)
    summary = sampler.summarize_configs(new_configs)

    weight_configs_step = experiment.create_weight_configs_step(
        configs=new_configs,
        summary=summary,
        seed=seed,
        name_prefix=NAME,
    )

    # Create training steps
    training_steps = []
    for config in new_configs:
        step = experiment.create_training_step(config, name_prefix=NAME)
        training_steps.append(step)

    # Analysis step depends on all training steps so it waits for them to finish
    analysis_step = create_analysis_step(
        weight_configs_step=weight_configs_step,
        name_prefix=NAME,
        metrics=ANALYSIS_METRICS,
        depends_on=training_steps,
    )

    logger.info(f"Created {len(training_steps)} training steps for v5")

    all_steps = [
        cache_tokenizer_step,
        cache_eval_datasets_step,
        weight_configs_step,
        *training_steps,
        analysis_step,
    ]
    executor_main(
        steps=all_steps,
        description=f"Two-phase starcoder v5: {n_new} additional runs",
    )


def run_analysis():
    """Collect results from W&B for the v5 experiment."""
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    experiment = create_two_phase_experiment(name=NAME)

    # Load the v5 weight configs from GCS
    prefix = os.environ.get("MARIN_PREFIX", "gs://marin-us-central1")
    fs, base = fsspec.core.url_to_fs(prefix)
    pattern = f"{NAME}/weight_configs-*/weight_configs.json"
    matches = fs.glob(f"{base}/{pattern}")

    if not matches:
        raise FileNotFoundError(f"No v5 weight_configs found at {prefix}/{pattern}. " "Run the training first.")

    path = f"{fs.protocol}://{matches[0]}" if isinstance(fs.protocol, str) else f"{fs.protocol[0]}://{matches[0]}"
    logger.info(f"Loading v5 weight configs from {path}")

    with fsspec.open(path) as f:
        data = json.load(f)

    configs = [WeightConfig.from_dict(c) for c in data["configs"]]
    logger.info(f"Loaded {len(configs)} v5 configs")

    weight_configs_step = experiment.create_weight_configs_step(
        configs=configs,
        summary={},
        seed=0,
        name_prefix=f"{NAME}_analysis",
    )
    analysis_step = create_analysis_step(
        weight_configs_step=weight_configs_step,
        name_prefix=NAME,  # Use v5 name for W&B tag matching
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
    parser = argparse.ArgumentParser(description="50 additional training runs for two-phase StarCoder experiment.")
    parser.add_argument(
        "--n-runs",
        type=int,
        default=50,
        help="Number of new training runs (default: 50).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="Random seed for weight sampling (default: 100, distinct from v4's 42).",
    )
    parser.add_argument(
        "--min-dist",
        type=float,
        default=0.05,
        help="Minimum config distance for rejection sampling (default: 0.05).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Sample configs and visualize without creating training steps.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis only (collect results from W&B and export CSV).",
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
