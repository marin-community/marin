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
    # Run training
    python -m experiments.domain_phase_mix.three_phase_experiment [--n_runs N] [--seed SEED]

    # Run analysis (after training completes)
    python -m experiments.domain_phase_mix.three_phase_experiment --analyze
"""

import logging
import os

from experiments.evals.task_configs import CORE_TASKS
from experiments.domain_phase_mix.proxy_sweep import regmix_60m_proxy
from experiments.domain_phase_mix.analysis import create_analysis_step
from marin.execution.executor import executor_main

from experiments.domain_phase_mix.config import PhaseSchedule
from experiments.domain_phase_mix.domains import get_three_partition_domains
from experiments.domain_phase_mix.experiment import MixtureExperiment

logger = logging.getLogger("ray")


# ============================================================================
# EXPERIMENT CONFIGURATION
# ============================================================================

NAME = "pinlin_calvin_xu/data_mixture/3_partitions_3_phases_3"

# Token budget: 1B tokens (as specified in RegMix paper)
EXPERIMENT_BUDGET = 1_000_000_000  # Actual tokens we train with

# Target budget: simulates a large-scale run (e.g., 2.5T tokens)
# This controls the simulated epoching ratio
TARGET_BUDGET = 2_500_000_000_000  # 2.5T tokens

# Batch and sequence configuration
BATCH_SIZE = 128
SEQ_LEN = 2048

# Phase boundaries (fractions of total training)
PHASE_BOUNDARIES = [0.33, 0.67]  # Creates 3 phases: [0, 0.33), [0.33, 0.67), [0.67, 1.0]


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


def main(
    n_runs: int = 100,
    seed: int = 42,
    name_prefix: str = NAME,
    analyze: bool = False,
    batch_size: int | None = None,
):
    """Main entry point for running the swarm experiment.

    Args:
        n_runs: Number of training runs.
        seed: Random seed for weight sampling.
        name_prefix: Prefix for run names.
        analyze: If True, only run analysis step (collect results from W&B).
        batch_size: If set, run training steps in batches of this size to limit parallelism.
    """
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
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
        all_steps = [weight_configs_step, analysis_step]
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

    # Run weight configs step first
    executor_main(
        steps=[weight_configs_step],
        description=f"Save weight configurations for {name_prefix}",
    )

    # Run training steps (optionally in batches)
    if batch_size is not None:
        logger.info(f"Running training steps in batches of {batch_size}")
        for i in range(0, len(training_steps), batch_size):
            batch = training_steps[i : i + batch_size]
            logger.info(f"Running batch {i // batch_size + 1}: steps {i} to {i + len(batch) - 1}")
            executor_main(
                steps=batch,
                description=f"Batch {i // batch_size + 1}: {len(batch)} training runs",
            )
    else:
        executor_main(
            steps=list(training_steps),
            description=f"Three-phase data mixture swarm: {n_runs} runs",
        )

    # Run analysis step last
    executor_main(
        steps=[analysis_step],
        description=f"Analysis for {name_prefix}",
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
        "--batch_size",
        type=int,
        default=None,
        help="Run training steps in batches of this size to limit parallelism.",
    )
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
        batch_size=args.batch_size,
    )
