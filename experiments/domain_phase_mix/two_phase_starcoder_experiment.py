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

"""Two-phase starcoder experiment for sanity-checking RegMix.

This experiment replicates the two_stage experiment setup using RegMix infrastructure:
- Two domains: Nemotron (common web, ~5.7T tokens) and StarCoder (code, ~217B tokens)
- Two phases: [0, 0.5) pretraining and [0.5, 1.0) code-focused
- Evaluates on code-related benchmarks (CODE_TASKS + starcoder validation)
- Uses Dirichlet sampling for weight exploration

Key differences from three_phase_experiment:
- Only 2 domains and 2 phases (simpler, cleaner signal)
- Uses Nemotron (large common data) and StarCoder (rare code data) from Dolma
- Code evals should directly benefit from more StarCoder data

Usage:
    # Run training with random weight sampling (Dirichlet)
    python -m experiments.domain_phase_mix.two_phase_starcoder_experiment [--n_runs N] [--seed SEED]

    # Run predefined baseline runs
    python -m experiments.domain_phase_mix.two_phase_starcoder_experiment --baseline_runs

    # Run analysis (after training completes)
    python -m experiments.domain_phase_mix.two_phase_starcoder_experiment --analyze
"""

import logging
import os

from marin.evaluation.eval_dataset_cache import create_cache_eval_datasets_step
from marin.execution.executor import executor_main
from marin.utils import create_cache_tokenizer_step

from experiments.domain_phase_mix.analysis import create_analysis_step
from experiments.llama import llama3_tokenizer
from experiments.domain_phase_mix.config import PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.domains import (
    NEMOTRON_FULL_DOMAIN,
    STARCODER_DOMAIN,
)
from experiments.domain_phase_mix.experiment import MixtureExperiment
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy
from experiments.domain_phase_mix.weight_sampler import DirichletSamplingParams, SamplingStrategy
from experiments.evals.task_configs import CORE_TASKS, CODE_TASKS

logger = logging.getLogger("ray")


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

NAME = "pinlin_calvin_xu/data_mixture/two_phase_starcoder_3"

# Token budget: 1B tokens (as specified in RegMix paper)
EXPERIMENT_BUDGET = 1_000_000_000  # Actual tokens we train with

# Target budget: set to size of Nemotron (~5.7T tokens)
# StarCoder can be epoched at most ~26x (5.7T / 217B)
TARGET_BUDGET = 5_729_908_864_777  # Nemotron full token count

# Batch and sequence configuration
BATCH_SIZE = 128
SEQ_LEN = 2048

# Phase boundaries (fractions of total training)
PHASE_BOUNDARIES = [0.5]  # Creates 2 phases: [0, 0.5), [0.5, 1.0]

# Domain names (must match names from get_nemotron_starcoder_domains())
DOMAIN_NAMES = ["nemotron_full", "starcoder"]

# Combine CORE_TASKS + CODE_TASKS for evaluation
EVAL_TASKS = CORE_TASKS + CODE_TASKS

# GCS paths for pre-cached data to avoid HuggingFace rate limiting
# Use us-central1 to match the cluster region
EVAL_DATASETS_CACHE_PATH = "gs://marin-us-central1/raw/eval-datasets/code-tasks"
TOKENIZER_CACHE_BASE = "gs://marin-us-central1/raw/tokenizers"

# Tokenizer used by regmix_60m_proxy and all domains
TOKENIZER_NAME = llama3_tokenizer  # "meta-llama/Meta-Llama-3.1-8B"

# Use uniform sampling strategy for weight exploration
SAMPLING_PARAMS = DirichletSamplingParams(
    strategy=SamplingStrategy.UNIFORM,
    min_weight=0.01,
    # min_phase_change=0.15,
)


# ============================================================================
# BASELINE CONFIGURATIONS
# ============================================================================

# Predefined baseline configurations for baseline runs
# Each entry is a tuple of phase weights: [[phase_0_weights], [phase_1_weights]]
# Weights correspond to domains in order: [nemotron_full, starcoder]
BASELINES: list[tuple[list[float], list[float]]] = [
    # Pure transitions
    ([1, 0], [0, 1]),  # Nemotron-only -> StarCoder-only
    ([0.5, 0.5], [0.5, 0.5]),  # Balanced throughout
    # Two-stage inspired (matching replay_ratio=0.8, rare_stage2_allocation=0.9)
    ([0.99, 0.01], [0.2, 0.8]),  # Original two_stage defaults
    ([0.95, 0.05], [0.2, 0.8]),  # Slightly more code in phase 1
    ([0.99, 0.01], [0.5, 0.5]),  # More balanced phase 2
    # Single-domain baselines
    ([1, 0], [1, 0]),  # Nemotron only (no code)
    ([0, 1], [0, 1]),  # StarCoder only (no web)
]


# ============================================================================
# EXPERIMENT DEFINITION
# ============================================================================


def get_nemotron_starcoder_domains():
    """Get domains for two-phase starcoder experiment.

    Uses Nemotron (large common web data, ~5.7T tokens) and StarCoder (rare code data, ~217B tokens).
    This ensures StarCoder is truly "rare" relative to the common data.

    Returns:
        List of [NEMOTRON_FULL_DOMAIN, STARCODER_DOMAIN]
    """
    return [NEMOTRON_FULL_DOMAIN, STARCODER_DOMAIN]


def create_two_phase_experiment(
    name: str = NAME,
    experiment_budget: int = EXPERIMENT_BUDGET,
    target_budget: int = TARGET_BUDGET,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
) -> MixtureExperiment:
    """Create the two-phase starcoder experiment.

    This sets up:
    - 2 domains: Nemotron (common web, ~5.7T tokens) and StarCoder (code, ~217B tokens)
    - 2 phases: [0, 0.5), [0.5, 1.0)
    - RegMix 60M proxy model
    - Simulated epoching with max 32x epoching on smallest dataset (StarCoder)

    Args:
        name: Experiment name.
        experiment_budget: Actual token budget for training.
        target_budget: Target budget for simulated epoching.
        batch_size: Training batch size.
        seq_len: Sequence length.

    Returns:
        MixtureExperiment configured for two-phase starcoder experiment.
    """
    # Create two-phase schedule
    phase_schedule = PhaseSchedule.from_boundaries(
        boundaries=PHASE_BOUNDARIES,
        names=["phase_0", "phase_1"],
    )

    # Get domains: Nemotron (common) and StarCoder (rare code)
    domains = get_nemotron_starcoder_domains()

    return MixtureExperiment(
        name=name,
        domains=domains,
        phase_schedule=phase_schedule,
        model_config=regmix_60m_proxy,
        batch_size=batch_size,
        seq_len=seq_len,
        experiment_budget=experiment_budget,
        target_budget=target_budget,
        eval_harness_tasks=EVAL_TASKS,
        sampling_params=SAMPLING_PARAMS,
        eval_datasets_cache_path=EVAL_DATASETS_CACHE_PATH,
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def create_baseline_weight_configs(
    baselines: list[tuple[list[float], list[float]]] = BASELINES,
    phase_names: list[str] | None = None,
    domain_names: list[str] | None = None,
) -> list[WeightConfig]:
    """Create WeightConfig objects from predefined baseline weights.

    Args:
        baselines: List of baseline configurations. Each is a tuple of 2 lists,
            one per phase, with weights for each domain.
        phase_names: Names of phases. Defaults to ["phase_0", "phase_1"].
        domain_names: Names of domains. Defaults to DOMAIN_NAMES.

    Returns:
        List of WeightConfig objects with unique run_ids starting from 90000.
    """
    phase_names = phase_names or ["phase_0", "phase_1"]
    domain_names = domain_names or DOMAIN_NAMES

    # Start baseline run_ids at 90000 to avoid conflicts with swarm run_ids (0-99)
    BASELINE_RUN_ID_START = 90000
    configs = []
    for i, (phase0, phase1) in enumerate(baselines):
        phase_weights = {
            phase_names[0]: dict(zip(domain_names, phase0, strict=True)),
            phase_names[1]: dict(zip(domain_names, phase1, strict=True)),
        }
        configs.append(WeightConfig(run_id=BASELINE_RUN_ID_START + i, phase_weights=phase_weights))

    return configs


def run_baselines(
    name_prefix: str = NAME,
    baselines: list[tuple[list[float], list[float]]] | None = None,
):
    """Run predefined baseline trial runs.

    Args:
        name_prefix: Prefix for run names.
        baselines: List of baseline configurations. If None, uses BASELINES.
    """
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    baselines = baselines or BASELINES
    experiment = create_two_phase_experiment(name=name_prefix)

    # Set environment variable for tokenizer GCS caching (used by defaults.py)
    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = TOKENIZER_CACHE_BASE

    # Create step to pre-cache tokenizer to GCS (runs once before training)
    cache_tokenizer_step = create_cache_tokenizer_step(
        tokenizer_name=TOKENIZER_NAME,
        gcs_path=os.path.join(TOKENIZER_CACHE_BASE, TOKENIZER_NAME.replace("/", "--")),
        name_prefix=name_prefix,
    )

    # Create step to pre-cache eval datasets to GCS (runs once before training)
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=EVAL_TASKS,
        gcs_path=EVAL_DATASETS_CACHE_PATH,
        name_prefix=name_prefix,
    )

    # Create weight configs from baselines
    weight_configs = create_baseline_weight_configs(baselines)

    logger.info(f"Running {len(weight_configs)} baseline configurations:")
    for config in weight_configs:
        logger.info(f"  baseline_run_{config.run_id}: {config.phase_weights}")

    # Create training steps with baseline_run naming
    training_steps = []
    for config in weight_configs:
        step = experiment.create_training_step(
            config,
            name_prefix=name_prefix,
            run_name=f"base_{config.run_id:05d}",
        )
        training_steps.append(step)

    executor_main(
        steps=[cache_tokenizer_step, cache_eval_datasets_step, *training_steps],
        description=f"Baseline runs for {name_prefix}",
    )


def main(
    n_runs: int = 100,
    seed: int = 42,
    name_prefix: str = NAME,
    analyze: bool = False,
    baseline_runs: bool = False,
):
    """Main entry point for running the swarm experiment.

    Args:
        n_runs: Number of training runs.
        seed: Random seed for weight sampling.
        name_prefix: Prefix for run names.
        analyze: If True, only run analysis step (collect results from W&B).
        baseline_runs: If True, run predefined baseline trial runs instead of random sampling.

    Note:
        Additional executor options like --max_concurrent and --force_run_failed
        are handled by executor_main via draccus CLI parsing.
    """
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    # Handle trial runs mode
    if baseline_runs:
        run_baselines(name_prefix=name_prefix)
        return

    experiment = create_two_phase_experiment(name=name_prefix)

    # Set environment variable for tokenizer GCS caching (used by defaults.py)
    os.environ["MARIN_TOKENIZER_CACHE_PATH"] = TOKENIZER_CACHE_BASE

    # Create step to pre-cache tokenizer to GCS (runs once before training)
    cache_tokenizer_step = create_cache_tokenizer_step(
        tokenizer_name=TOKENIZER_NAME,
        gcs_path=os.path.join(TOKENIZER_CACHE_BASE, TOKENIZER_NAME.replace("/", "--")),
        name_prefix=name_prefix,
    )

    # Create step to pre-cache eval datasets to GCS (runs once before training)
    cache_eval_datasets_step = create_cache_eval_datasets_step(
        eval_tasks=EVAL_TASKS,
        gcs_path=EVAL_DATASETS_CACHE_PATH,
        name_prefix=name_prefix,
    )

    weight_configs_step, training_steps = experiment.create_swarm_steps(
        n_runs=n_runs, seed=seed, name_prefix=name_prefix
    )

    analysis_step = create_analysis_step(
        weight_configs_step=weight_configs_step,
        name_prefix=name_prefix,
    )

    if analyze:
        # Only run analysis
        logger.info("Running analysis only (collecting results from W&B)")
        # Include baseline configs in the weight_configs step by re-creating it
        # with baseline configs appended.
        baseline_weight_configs = create_baseline_weight_configs(BASELINES)
        weight_configs_step_with_baselines, _ = experiment.create_swarm_steps(
            n_runs=n_runs,
            seed=seed,
            name_prefix=f"{name_prefix}_with_baselines",
            additional_configs=baseline_weight_configs,
        )
        analysis_step_with_baselines = create_analysis_step(
            weight_configs_step=weight_configs_step_with_baselines,
            name_prefix=name_prefix,  # Keep original name for W&B tag matching
        )
        all_steps = [weight_configs_step_with_baselines, analysis_step_with_baselines]
        executor_main(
            steps=all_steps,
            description=f"Analysis for {name_prefix}",
        )
        return

    # Log experiment details
    tokens_per_step = BATCH_SIZE * SEQ_LEN
    total_steps = EXPERIMENT_BUDGET // tokens_per_step
    phase1_end = int(total_steps * PHASE_BOUNDARIES[0])

    logger.info(
        f"Created {len(training_steps)} training steps + 1 tokenizer cache step + "
        f"1 eval datasets cache step + 1 weight configs step + 1 analysis step"
    )
    logger.info(f"Total tokens per run: {EXPERIMENT_BUDGET:,}")
    logger.info(f"Total steps per run: {total_steps:,}")
    logger.info(f"Phase boundary: step {phase1_end} (50%)")
    logger.info(f"Target budget (simulated epoching): {TARGET_BUDGET:,}")
    logger.info("Max epoching on smallest dataset: 32x")

    # Run all steps through executor_main, which handles --max_concurrent and
    # --force_run_failed via draccus CLI parsing
    # Cache steps run first to pre-cache tokenizer and datasets before training
    all_steps = [
        cache_tokenizer_step,
        cache_eval_datasets_step,
        weight_configs_step,
        *training_steps,
        analysis_step,
    ]
    executor_main(
        steps=all_steps,
        description=f"Two-phase starcoder experiment: {n_runs} runs",
    )


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Two-phase starcoder experiment for RegMix sanity check.")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=100,
        help="Number of training runs (default: 100).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for weight sampling (default: 42).",
    )
    parser.add_argument(
        "--name_prefix",
        type=str,
        default=NAME,
        help="Prefix for run names.",
    )
    parser.add_argument(
        "--analyze",
        action="store_true",
        help="Run analysis only (collect results from W&B and export CSV).",
    )
    parser.add_argument(
        "--baseline_runs",
        action="store_true",
        help="Run predefined baseline trial runs instead of random sampling.",
    )
    # Note: --max_concurrent and --force_run_failed are handled by executor_main
    # via draccus CLI parsing, so they don't need to be defined here.
    return parser.parse_known_args()


if __name__ == "__main__":
    import sys

    args, remaining = _parse_args()
    sys.argv = [sys.argv[0], *remaining]

    main(
        n_runs=args.n_runs,
        seed=args.seed,
        name_prefix=args.name_prefix,
        analyze=args.analyze,
        baseline_runs=args.baseline_runs,
    )
