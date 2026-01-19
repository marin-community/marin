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

"""Main swarm runner for three-phase data mixture experiments.

Orchestrates the creation of ~100 proxy model training runs with:
- RegMix 60M model configuration (1B tokens)
- Three training phases with independently sampled mixture weights
- Phase boundaries at 33% and 67% of training

Usage:
    python -m experiments.three_phase_swarm.swarm_runner [--n_runs N] [--seed SEED]
"""

import json
import logging
import os
from collections.abc import Sequence
from pathlib import Path

from levanter.optim import MuonHConfig

from experiments.defaults import simulated_epoching_train
from experiments.evals.task_configs import CORE_TASKS
from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.swarm_proxy_runs.proxy_sweep import regmix_60m_proxy
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

from experiments.three_phase_swarm.datasets import (
    expand_partition_weights,
    get_all_components,
)
from experiments.three_phase_swarm.weight_sampler import (
    ThreePartitionWeightConfig,
    ThreePartitionWeightSampler,
)


logger = logging.getLogger("ray")


# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================

# Token budget: 1B tokens (as specified in RegMix paper)
EXPERIMENT_BUDGET = 1_000_000_000  # Actual tokens we train with

# Target budget: simulates a large-scale run (e.g., 10T tokens)
# This controls the simulated epoching ratio
TARGET_BUDGET = 2.5_000_000_000_000  # 2.5T tokens

# Batch and sequence configuration
BATCH_SIZE = 128
SEQ_LEN = 2048  # regmix_60m_proxy.max_seq_len

# Compute total training steps
TOKENS_PER_STEP = BATCH_SIZE * SEQ_LEN  # 262,144
TOTAL_STEPS = EXPERIMENT_BUDGET // TOKENS_PER_STEP  # 3,814

# Phase boundaries (as fraction of total steps)
PHASE1_END_FRACTION = 0.33
PHASE2_END_FRACTION = 0.67

# Compute phase boundary steps
PHASE1_END_STEP = int(TOTAL_STEPS * PHASE1_END_FRACTION)  # 1,258
PHASE2_END_STEP = int(TOTAL_STEPS * PHASE2_END_FRACTION)  # 2,555

# Convert to sequence indices for lm_varying_mixture_data_config
# The weights_list uses sequence indices: sequence_index = step * batch_size
PHASE2_START_SEQ = PHASE1_END_STEP * BATCH_SIZE  # 161,024
PHASE3_START_SEQ = PHASE2_END_STEP * BATCH_SIZE  # 327,040


# ============================================================================
# OPTIMIZER CONFIGURATION
# ============================================================================

# MuonH optimizer config from proxy_sweep.py
MUON_CONFIG = MuonHConfig(
    learning_rate=0.02,
    adam_lr=0.008,
    min_lr_ratio=0,
    momentum=0.95,
    beta1=0.9,
    beta2=0.98,
    epsilon=1e-15,
    muon_epsilon=1e-5,
    max_grad_norm=1,
    warmup=1000,
)


# ============================================================================
# CORE FUNCTIONS
# ============================================================================


def create_three_phase_mixture_config(config: ThreePartitionWeightConfig):
    """Create a varying mixture config for three-phase training.

    Args:
        config: ThreePartitionWeightConfig with weights for each phase.

    Returns:
        LMMixtureDatasetConfig with time-varying weights.
    """
    # Get all dataset components
    all_components = get_all_components()

    # Expand partition weights to component weights for each phase
    phase1_weights = expand_partition_weights(config.phase1_weights)
    phase2_weights = expand_partition_weights(config.phase2_weights)
    phase3_weights = expand_partition_weights(config.phase3_weights)

    # Build weights list with phase transitions
    weights_list = [
        (0, phase1_weights),
        (PHASE2_START_SEQ, phase2_weights),
        (PHASE3_START_SEQ, phase3_weights),
    ]

    return lm_varying_mixture_data_config(
        components=all_components,
        weights_list=weights_list,
        permutation_type="feistel",
        shuffle=True,
    )


def create_train_config(run_id: int) -> SimpleTrainConfig:
    """Create training configuration for a single run.

    Args:
        run_id: Run identifier (used for data seed).

    Returns:
        SimpleTrainConfig for the training run.
    """
    return SimpleTrainConfig(
        resources=ResourceConfig.with_tpu("v5p-8"),
        train_batch_size=BATCH_SIZE,
        num_train_steps=TOTAL_STEPS,
        learning_rate=MUON_CONFIG.learning_rate,
        optimizer_config=MUON_CONFIG,
        steps_per_eval=500,
        steps_per_export=TOTAL_STEPS,  # Only save at the end
        data_seed=run_id,  # Different seed per run for reproducibility
    )


def create_training_step(
    config: ThreePartitionWeightConfig,
    name_prefix: str = "three_phase_swarm",
) -> ExecutorStep:
    """Create a training step for a single weight configuration.

    Args:
        config: ThreePartitionWeightConfig with weights for each phase.
        name_prefix: Prefix for the run name.

    Returns:
        ExecutorStep for the training run.
    """
    mixture_config = create_three_phase_mixture_config(config)
    train_config = create_train_config(config.run_id)

    return simulated_epoching_train(
        name=f"{name_prefix}/run_{config.run_id:03d}",
        tokenized=mixture_config,
        model_config=regmix_60m_proxy,
        train_config=train_config,
        target_budget=TARGET_BUDGET,
        tags=["three_phase_swarm", f"run_{config.run_id:03d}"],
        use_default_validation=True,
        eval_harness_tasks=CORE_TASKS,
    )


def create_swarm_steps(
    n_runs: int = 100,
    seed: int = 42,
    name_prefix: str = "three_phase_swarm",
    save_configs: bool = True,
    output_dir: str | None = None,
) -> Sequence[ExecutorStep]:
    """Create all training steps for the swarm.

    Args:
        n_runs: Number of training runs to create.
        seed: Random seed for weight sampling.
        name_prefix: Prefix for run names.
        save_configs: Whether to save weight configurations to disk.
        output_dir: Directory to save configurations (if save_configs is True).

    Returns:
        List of ExecutorSteps for all training runs.
    """
    # Sample weight configurations
    sampler = ThreePartitionWeightSampler(seed=seed)
    configs = sampler.sample_n_configs(n_runs, deduplicate=True)

    # Log summary statistics
    summary = sampler.summarize_configs(configs)
    logger.info(f"Sampled {summary['n_configs']} unique configurations")
    for phase in ["phase1", "phase2", "phase3"]:
        logger.info(f"  {phase}:")
        for partition, stats in summary[phase].items():
            logger.info(
                f"    {partition}: mean={stats['mean']:.3f}, "
                f"std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]"
            )

    # Save configurations if requested
    if save_configs:
        if output_dir is None:
            output_dir = f"experiments/three_phase_swarm/configs/{name_prefix}"
        os.makedirs(output_dir, exist_ok=True)

        configs_path = Path(output_dir) / "weight_configs.json"
        configs_data = {
            "seed": seed,
            "n_runs": n_runs,
            "summary": summary,
            "configs": [c.to_dict() for c in configs],
        }
        with open(configs_path, "w") as f:
            json.dump(configs_data, f, indent=2)
        logger.info(f"Saved weight configurations to {configs_path}")

    # Create training steps
    steps = []
    for config in configs:
        step = create_training_step(config, name_prefix)
        steps.append(step)

    return steps


def main(
    n_runs: int = 100,
    seed: int = 42,
    name_prefix: str = "three_phase_swarm",
):
    """Main entry point for running the swarm experiment.

    Args:
        n_runs: Number of training runs.
        seed: Random seed for weight sampling.
        name_prefix: Prefix for run names.
    """
    if os.getenv("CI", None) is not None:
        logger.info("Skipping experiment execution on CI environment.")
        return

    steps = create_swarm_steps(n_runs=n_runs, seed=seed, name_prefix=name_prefix)

    logger.info(f"Created {len(steps)} training steps")
    logger.info(f"Total tokens per run: {EXPERIMENT_BUDGET:,}")
    logger.info(f"Total steps per run: {TOTAL_STEPS:,}")
    logger.info(f"Phase boundaries: step {PHASE1_END_STEP} (33%), step {PHASE2_END_STEP} (67%)")

    executor_main(
        steps=steps,
        description=f"Three-phase data mixture swarm: {n_runs} runs with independently sampled weights per phase",
    )


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(
        description="Three-phase data mixture swarm experiment."
    )
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
        default="three_phase_swarm",
        help="Prefix for run names (default: three_phase_swarm).",
    )
    return parser.parse_known_args()


if __name__ == "__main__":
    import sys

    args, remaining = _parse_args()
    # Pass remaining args to executor_main
    sys.argv = [sys.argv[0], *remaining]

    main(n_runs=args.n_runs, seed=args.seed, name_prefix=args.name_prefix)
