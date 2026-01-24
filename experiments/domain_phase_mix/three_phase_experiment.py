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

"""Three-phase data mixture swarm experiment.

This experiment creates ~100 proxy model training runs with:
- RegMix 60M model configuration (1B tokens)
- Three training phases with independently sampled mixture weights
- Phase boundaries at 33% and 67% of training
- Three data domains: pretrain (Nemotron), midtrain (full Dolmino), SFT

Usage:
    # Run training with random weight sampling
    python -m experiments.domain_phase_mix.three_phase_experiment [--n_runs N] [--seed SEED]

    # Run predefined baseline runs
    python -m experiments.domain_phase_mix.three_phase_experiment --baseline_runs

    # Run analysis (after training completes)
    python -m experiments.domain_phase_mix.three_phase_experiment --analyze
"""

import logging
import os

from experiments.evals.task_configs import CORE_TASKS
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy
from experiments.domain_phase_mix.analysis import create_analysis_step
from marin.execution.executor import executor_main

from experiments.domain_phase_mix.config import PhaseSchedule, WeightConfig
from experiments.domain_phase_mix.domains import get_three_partition_domains
from experiments.domain_phase_mix.experiment import MixtureExperiment

logger = logging.getLogger("ray")


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

NAME = "pinlin_calvin_xu/data_mixture/3_partitions_3_phases_6"

# Token budget: 1B tokens (as specified in RegMix paper)
EXPERIMENT_BUDGET = 1_000_000_000  # Actual tokens we train with

# Target budget: simulates a large-scale run (e.g., 2.5T tokens)
# This controls the simulated epoching ratio
TARGET_BUDGET = 5_700_000_000_000  # 5.7T tokens

# Batch and sequence configuration
BATCH_SIZE = 128
SEQ_LEN = 2048

# Phase boundaries (fractions of total training)
PHASE_BOUNDARIES = [0.33, 0.67]  # Creates 3 phases: [0, 0.33), [0.33, 0.67), [0.67, 1.0]

# Domain names (must match names from get_three_partition_domains())
DOMAIN_NAMES = ["nemotron_full", "dolmino", "openthoughts_sft"]

# ============================================================================
# BASELINE CONFIGURATIONS
# ============================================================================

# Predefined baseline configurations for baseline runs
# Each entry is a tuple of phase weights: [[phase_0_weights], [phase_1_weights], [phase_2_weights]]
# Weights correspond to domains in order: [nemotron_full, dolmino, openthoughts_sft]
BASELINES: list[tuple[list[float], list[float], list[float]]] = [
    # Pure domain transitions: nemotron -> dolmino -> openthoughts_sft
    ([1, 0, 0], [0, 1, 0], [0, 0, 1]),
    # Nemotron -> dolmino -> half nemotron + half sft
    ([1, 0, 0], [0, 1, 0], [0.5, 0, 0.5]),
    # Nemotron -> dolmino -> balanced final phase
    ([1, 0, 0], [0, 1, 0], [0.25, 0.25, 0.5]),
    ([1, 0, 0], [0, 1, 0], [0.5, 0.5, 0]),
    ([1, 0, 0], [1, 0, 0], [1, 0, 0]),
    ([1, 0, 0], [1, 0, 0], [0, 1, 0]),
    # RegMix-optimized for eval/paloma/c4_en/bpb (from regmix_regression.py)
    ([0.5608, 0.3962, 0.0430], [0.7221, 0.2168, 0.0611], [0.6633, 0.2397, 0.0969]),
    # RegMix k-fold CV optimized for eval/paloma/c4_en/bpb (10M samples, from regmix_regression_kfold.py)
    ([0.4977, 0.4568, 0.0455], [0.6575, 0.3140, 0.0285], [0.6756, 0.2445, 0.0799]),
    # RegMix k-fold CV optimized for lm_eval/arc_challenge/choice_logprob (10M samples)
    ([0.5203, 0.4339, 0.0458], [0.1468, 0.8256, 0.0277], [0.5405, 0.3860, 0.0735]),
]


# ============================================================================
# EXPERIMENT DEFINITION
# ============================================================================


def create_three_phase_experiment(
    name: str = NAME,
    experiment_budget: int = EXPERIMENT_BUDGET,
    target_budget: int = TARGET_BUDGET,
    batch_size: int = BATCH_SIZE,
    seq_len: int = SEQ_LEN,
) -> MixtureExperiment:
    """Create the three-phase swarm experiment.

    This sets up:
    - 3 domains: Nemotron HQ (pretrain), full Dolmino (midtrain), Math SFT
    - 3 phases: [0, 0.33), [0.33, 0.67), [0.67, 1.0)
    - RegMix 60M proxy model
    - Simulated epoching with 2.5T target budget

    Args:
        name: Experiment name.
        experiment_budget: Actual token budget for training.
        target_budget: Target budget for simulated epoching.
        batch_size: Training batch size.
        seq_len: Sequence length.

    Returns:
        MixtureExperiment configured for three-phase swarm.
    """
    # Create three-phase schedule
    phase_schedule = PhaseSchedule.from_boundaries(
        boundaries=PHASE_BOUNDARIES,
        names=["phase_0", "phase_1", "phase_2"],
    )

    # Get the three partition domains
    domains = get_three_partition_domains()

    return MixtureExperiment(
        name=name,
        domains=domains,
        phase_schedule=phase_schedule,
        model_config=regmix_60m_proxy,
        batch_size=batch_size,
        seq_len=seq_len,
        experiment_budget=experiment_budget,
        target_budget=target_budget,
        eval_harness_tasks=CORE_TASKS,
    )


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================


def create_baseline_weight_configs(
    baselines: list[tuple[list[float], list[float], list[float]]] = BASELINES,
    phase_names: list[str] | None = None,
    domain_names: list[str] | None = None,
) -> list[WeightConfig]:
    """Create WeightConfig objects from predefined baseline weights.

    Args:
        baselines: List of baseline configurations. Each is a tuple of 3 lists,
            one per phase, with weights for each domain.
        phase_names: Names of phases. Defaults to ["phase_0", "phase_1", "phase_2"].
        domain_names: Names of domains. Defaults to DOMAIN_NAMES.

    Returns:
        List of WeightConfig objects with unique run_ids starting from 90000.
    """
    phase_names = phase_names or ["phase_0", "phase_1", "phase_2"]
    domain_names = domain_names or DOMAIN_NAMES

    # Start baseline run_ids at 90000 to avoid conflicts with swarm run_ids (0-99)
    BASELINE_RUN_ID_START = 90000
    configs = []
    for i, (phase0, phase1, phase2) in enumerate(baselines):
        phase_weights = {
            phase_names[0]: dict(zip(domain_names, phase0, strict=True)),
            phase_names[1]: dict(zip(domain_names, phase1, strict=True)),
            phase_names[2]: dict(zip(domain_names, phase2, strict=True)),
        }
        configs.append(WeightConfig(run_id=BASELINE_RUN_ID_START + i, phase_weights=phase_weights))

    return configs


def run_baselines(
    name_prefix: str = NAME,
    baselines: list[tuple[list[float], list[float], list[float]]] | None = None,
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
    experiment = create_three_phase_experiment(name=name_prefix)

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
        steps=training_steps,
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

    experiment = create_three_phase_experiment(name=name_prefix)

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
        # with baseline configs appended. Use a different name suffix to ensure
        # the step is re-run even if the original weight_configs step exists.
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
    phase2_end = int(total_steps * PHASE_BOUNDARIES[1])

    logger.info(f"Created {len(training_steps)} training steps + 1 weight configs step + 1 analysis step")
    logger.info(f"Total tokens per run: {EXPERIMENT_BUDGET:,}")
    logger.info(f"Total steps per run: {total_steps:,}")
    logger.info(f"Phase boundaries: step {phase1_end} (33%), step {phase2_end} (67%)")

    # Run all steps through executor_main, which handles --max_concurrent and
    # --force_run_failed via draccus CLI parsing
    all_steps = [weight_configs_step, *training_steps, analysis_step]
    executor_main(
        steps=all_steps,
        description=f"Three-phase data mixture swarm: {n_runs} runs",
    )


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Three-phase data mixture swarm experiment.")
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
