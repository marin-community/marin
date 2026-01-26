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

"""Base experiment runner for n-domain, n-phase mixture experiments.

This module provides the core infrastructure for running mixture experiments
with arbitrary numbers of domains and phases.
"""

import json
import logging
import os
from collections.abc import Sequence
from dataclasses import dataclass

import fsspec
from levanter.main.train_lm import LmConfig
from levanter.optim import MuonHConfig

from experiments.defaults import simulated_epoching_train
from experiments.evals.task_configs import EvalTaskConfig, CORE_TASKS
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path
from marin.processing.tokenize.data_configs import lm_varying_mixture_data_config

from experiments.domain_phase_mix.config import (
    ExperimentConfig,
    Domain,
    PhaseSchedule,
    WeightConfig,
)
from experiments.domain_phase_mix.weight_sampler import WeightSampler, DirichletSamplingParams


# Default MuonH optimizer configuration from proxy_sweep.py
# This is adapted from 130M Qwen3 config for small models
DEFAULT_MUON_CONFIG = MuonHConfig(
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

logger = logging.getLogger("ray")


# ============================================================================
# WEIGHT CONFIG STEP
# ============================================================================


@dataclass(frozen=True)
class SaveWeightConfigsConfig:
    """Configuration for saving weight configs to GCS."""

    output_path: str
    experiment_name: str
    seed: int
    n_runs: int
    domains: tuple[str, ...]
    phases: tuple[str, ...]
    summary: str  # JSON-encoded summary
    configs: str  # JSON-encoded list of configs


def save_weight_configs(config: SaveWeightConfigsConfig):
    """Save weight configurations to GCS.

    This is an ExecutorStep function that writes weight_configs.json.
    """
    configs_data = {
        "experiment_name": config.experiment_name,
        "seed": config.seed,
        "n_runs": config.n_runs,
        "domains": list(config.domains),
        "phases": list(config.phases),
        "summary": json.loads(config.summary),
        "configs": json.loads(config.configs),
    }

    output_file = os.path.join(config.output_path, "weight_configs.json")
    with fsspec.open(output_file, "w") as f:
        json.dump(configs_data, f, indent=2)

    logger.info(f"Saved weight configurations to {output_file}")


class MixtureExperiment:
    """Base class for running mixture experiments.

    This class provides the core infrastructure for:
    - Generating weight configurations
    - Creating mixture data configs
    - Building training steps
    - Running swarm experiments

    Subclasses can customize model configs, training parameters, and domains.
    """

    def __init__(
        self,
        name: str,
        domains: list[Domain],
        phase_schedule: PhaseSchedule,
        model_config: LmConfig,
        batch_size: int = 128,
        seq_len: int = 2048,
        learning_rate: float = 0.02,
        num_train_steps: int | None = None,
        steps_per_eval: int = 1000,
        experiment_budget: int | None = None,
        target_budget: int | None = None,
        resources: ResourceConfig | None = None,
        eval_harness_tasks: Sequence[EvalTaskConfig] = CORE_TASKS,
        sampling_params: DirichletSamplingParams | None = None,
        mixture_block_size: int = 2048,
        optimizer_config: MuonHConfig | None = None,
    ):
        """Initialize the experiment.

        Args:
            name: Name of the experiment (used for output paths).
            domains: List of data domains to use.
            phase_schedule: Schedule defining training phases.
            model_config: Model configuration (LlamaConfig, etc.).
            batch_size: Training batch size.
            seq_len: Sequence length.
            learning_rate: Learning rate for training.
            num_train_steps: Number of training steps. If None, computed from experiment_budget.
            experiment_budget: Total tokens to train on. If None, computed from num_train_steps.
            target_budget: Target budget for simulated epoching. If None, no simulated epoching.
            resources: Resource configuration for training.
            eval_harness_tasks: Evaluation tasks to run.
            sampling_params: Parameters for weight sampling.
            mixture_block_size: Block size for mixture dataset. Phase transitions must
                occur at block boundaries. Default 2048.
            optimizer_config: Optimizer configuration. Defaults to MuonH optimizer
                with hyperparameters from proxy_sweep.py.
        """
        self.name = name
        self.domains = domains
        self.phase_schedule = phase_schedule
        self.model_config = model_config
        self.batch_size = batch_size
        self.seq_len = seq_len
        self.learning_rate = learning_rate
        self.steps_per_eval = steps_per_eval
        self.resources = resources or ResourceConfig.with_tpu("v5p-8")
        self.eval_harness_tasks = eval_harness_tasks
        self.sampling_params = sampling_params or DirichletSamplingParams()
        self.mixture_block_size = mixture_block_size
        self.optimizer_config = optimizer_config or DEFAULT_MUON_CONFIG

        # Compute training steps and budget
        self.tokens_per_step = batch_size * seq_len
        if num_train_steps is not None:
            self.num_train_steps = num_train_steps
            self.experiment_budget = num_train_steps * self.tokens_per_step
        elif experiment_budget is not None:
            self.experiment_budget = experiment_budget
            self.num_train_steps = experiment_budget // self.tokens_per_step
        else:
            raise ValueError("Must specify either num_train_steps or experiment_budget")

        self.target_budget = target_budget

        # Build experiment config
        self.experiment_config = ExperimentConfig(
            name=name,
            domains=domains,
            phase_schedule=phase_schedule,
            total_steps=self.num_train_steps,
            batch_size=batch_size,
            seq_len=seq_len,
            target_budget=target_budget,
        )

    def create_weight_sampler(self, seed: int = 42) -> WeightSampler:
        """Create a weight sampler for this experiment."""
        return WeightSampler.from_experiment_config(
            self.experiment_config,
            seed=seed,
            params=self.sampling_params,
        )

    def create_mixture_config(self, weight_config: WeightConfig):
        """Create a varying mixture config for training.

        Args:
            weight_config: Weight configuration for all phases.

        Returns:
            LMMixtureDatasetConfig with time-varying weights.
        """
        # Get all dataset components
        all_components = self.experiment_config.get_all_components()

        # Build weights list for each phase transition
        # NOTE: weights_list uses step indices, not sequence indices!
        # LMMixtureDatasetConfig.train_set() calls rescale_mixture_schedule_for_batch_schedule()
        # which converts step indices to sequence indices using the batch schedule.
        # The step indices must be chosen so that step * batch_size is a multiple of mixture_block_size.
        weights_list = []
        for phase in self.phase_schedule.phases:
            # Get domain weights for this phase
            domain_weights = weight_config.get_weights_for_phase(phase.name)

            # Expand to component weights
            component_weights = self.experiment_config.expand_domain_weights(domain_weights)

            # Get step index for phase start (aligned for block_size constraints)
            start_step = phase.get_start_step_aligned(
                self.num_train_steps, self.batch_size, self.mixture_block_size
            )

            weights_list.append((start_step, component_weights))

        return lm_varying_mixture_data_config(
            components=all_components,
            weights_list=weights_list,
            permutation_type="feistel",
            shuffle=True,
            mixture_block_size=self.mixture_block_size,
        )

    def create_train_config(self, run_id: int, **kwargs) -> SimpleTrainConfig:
        """Create training configuration for a single run.

        Args:
            run_id: Run identifier (used for data seed).
            **kwargs: Additional arguments to override defaults.

        Returns:
            SimpleTrainConfig for the training run.
        """
        defaults = {
            "resources": self.resources,
            "train_batch_size": self.batch_size,
            "num_train_steps": self.num_train_steps,
            "learning_rate": self.learning_rate,
            "optimizer_config": self.optimizer_config,
            "steps_per_eval": self.steps_per_eval,
            "steps_per_export": self.num_train_steps,  # Save only at the end
            "data_seed": run_id,
        }
        defaults.update(kwargs)
        return SimpleTrainConfig(**defaults)

    def create_training_step(
        self,
        weight_config: WeightConfig,
        name_prefix: str | None = None,
        run_name: str | None = None,
        **train_kwargs,
    ) -> ExecutorStep:
        """Create a training step for a single weight configuration.

        Args:
            weight_config: Weight configuration for all phases.
            name_prefix: Prefix for the run name. Defaults to experiment name.
            run_name: Custom run name (without prefix). Defaults to "run_{run_id:05d}".
            **train_kwargs: Additional training configuration overrides.

        Returns:
            ExecutorStep for the training run.
        """
        prefix = name_prefix or self.name
        run_name = run_name or f"run_{weight_config.run_id:05d}"
        full_name = f"{prefix}/{run_name}"

        mixture_config = self.create_mixture_config(weight_config)
        train_config = self.create_train_config(weight_config.run_id, **train_kwargs)

        if self.target_budget is not None:
            return simulated_epoching_train(
                name=full_name,
                tokenized=mixture_config,
                model_config=self.model_config,
                train_config=train_config,
                target_budget=self.target_budget,
                tags=[self.name, run_name],
                use_default_validation=True,
                eval_harness_tasks=self.eval_harness_tasks,
                wandb_name=full_name,
            )
        else:
            from experiments.defaults import default_train

            return default_train(
                name=full_name,
                tokenized=mixture_config,
                model_config=self.model_config,
                train_config=train_config,
                tags=[self.name, run_name],
                use_default_validation=True,
                eval_harness_tasks=self.eval_harness_tasks,
                wandb_name=full_name,
            )

    def create_weight_configs_step(
        self,
        configs: list[WeightConfig],
        summary: dict,
        seed: int,
        name_prefix: str | None = None,
    ) -> ExecutorStep:
        """Create an ExecutorStep that saves weight configurations to GCS.

        Args:
            configs: List of weight configurations.
            summary: Summary statistics from the sampler.
            seed: Random seed used for sampling.
            name_prefix: Prefix for the step name.

        Returns:
            ExecutorStep that saves weight_configs.json.
        """
        prefix = name_prefix or self.name

        return ExecutorStep(
            name=f"{prefix}/weight_configs",
            description=f"Save weight configurations for {len(configs)} runs",
            fn=save_weight_configs,
            config=SaveWeightConfigsConfig(
                output_path=this_output_path(),
                experiment_name=self.name,
                seed=seed,
                n_runs=len(configs),
                domains=tuple(self.experiment_config.domain_names),
                phases=tuple(self.phase_schedule.phase_names),
                summary=json.dumps(summary),
                configs=json.dumps([c.to_dict() for c in configs]),
            ),
        )

    def create_swarm_steps(
        self,
        n_runs: int = 100,
        seed: int = 42,
        name_prefix: str | None = None,
        additional_configs: Sequence[WeightConfig] | None = None,
    ) -> tuple[ExecutorStep, Sequence[ExecutorStep]]:
        """Create all steps for the swarm experiment.

        Args:
            n_runs: Number of training runs to create.
            seed: Random seed for weight sampling.
            name_prefix: Prefix for run names.
            additional_configs: Additional weight configs to include (e.g., baseline configs).
                These are added to the weight_configs.json but no training steps are created for them.

        Returns:
            Tuple of (weight_configs_step, training_steps).
            The weight_configs_step saves configurations to GCS.
        """
        prefix = name_prefix or self.name

        # Sample weight configurations
        sampler = self.create_weight_sampler(seed=seed)
        configs = sampler.sample_n_configs(n_runs, deduplicate=True)

        # Include additional configs (e.g., baselines) in the saved configs
        all_configs = list(configs)
        if additional_configs:
            all_configs.extend(additional_configs)

        # Log summary
        summary = sampler.summarize_configs(configs)
        logger.info(f"Sampled {summary['n_configs']} unique configurations")
        for phase_name in self.phase_schedule.phase_names:
            logger.info(f"  {phase_name}:")
            for domain_name in self.experiment_config.domain_names:
                stats = summary[phase_name][domain_name]
                logger.info(
                    f"    {domain_name}: mean={stats['mean']:.3f}, "
                    f"std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]"
                )

        # Create weight configs step (saves to GCS)
        # Use all_configs (includes additional configs like baselines) for saving
        weight_configs_step = self.create_weight_configs_step(
            configs=all_configs,
            summary=summary,
            seed=seed,
            name_prefix=prefix,
        )

        # Create training steps
        training_steps = []
        for config in configs:
            step = self.create_training_step(config, name_prefix=prefix)
            training_steps.append(step)

        return weight_configs_step, training_steps

    def run(
        self,
        n_runs: int = 100,
        seed: int = 42,
        name_prefix: str | None = None,
    ):
        """Run the experiment.

        Args:
            n_runs: Number of training runs.
            seed: Random seed for weight sampling.
            name_prefix: Prefix for run names.
        """
        if os.getenv("CI", None) is not None:
            logger.info("Skipping experiment execution on CI environment.")
            return

        weight_configs_step, training_steps = self.create_swarm_steps(n_runs=n_runs, seed=seed, name_prefix=name_prefix)

        logger.info(f"Created {len(training_steps)} training steps")
        logger.info(f"Experiment: {self.name}")
        logger.info(f"Domains: {self.experiment_config.domain_names}")
        logger.info(f"Phases: {self.phase_schedule.phase_names}")
        logger.info(f"Total tokens per run: {self.experiment_budget:,}")
        logger.info(f"Total steps per run: {self.num_train_steps:,}")
        if self.target_budget:
            logger.info(f"Target budget (simulated): {self.target_budget:,}")

        # Include weight_configs_step first, then all training steps
        all_steps = [weight_configs_step, *training_steps]

        executor_main(
            steps=all_steps,
            description=f"{self.name}: {n_runs} runs with {self.experiment_config.n_phases} phases "
            f"and {self.experiment_config.n_domains} domains",
        )
