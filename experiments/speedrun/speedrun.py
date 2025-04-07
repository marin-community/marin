"""
Default configurations and utilities for the Marin speedrun experiment.
"""

import dataclasses
import logging
from dataclasses import dataclass
from enum import Enum

from levanter.data.text import LMMixtureDatasetConfig
from levanter.models.lm_model import LmConfig

from experiments.evals.task_configs import CORE_TASKS_PLUS_MMLU
from experiments.defaults import default_train
from levanter.compat.hf_checkpoints import load_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    unwrap_versioned_value,
)
from marin.training.training import TrainLmOnPodConfig

logger = logging.getLogger("ray")


class ComputeBudget(Enum):
    """Predefined compute budgets for speed run tracks, in FLOPs."""
    TINY = 3e18    # Smallest track, ~3e18 FLOPs
    SMALL = 6e18   # Small track, ~6e18 FLOPs
    MEDIUM = 3e19  # Medium track, ~3e19 FLOPs
    

@dataclass
class SpeedrunConfig:
    """Configuration for a speed run submission, defining model, training, and compute constraints.
    Attributes:
        compute_budget: The chosen compute budget track (e.g., SMALL = 6e18 FLOPs).
        model_config: The language model configuration (e.g., LLaMA architecture).
        train_config: Training configuration with batch size, steps, learning rate, etc.
        tokenized_dataset: Dataset to use (e.g., pre-tokenized input or mixture config).
        vocab_size: Size of the tokenizer vocabulary (defaults to LLaMA 3's vocab size).
        hyperparameter_scaling: Optional function to dynamically set hyperparameters based on model/budget.
    """

    compute_budget: ComputeBudget
    model_config: LmConfig
    train_config: SimpleTrainConfig | TrainLmOnPodConfig
    tokenized_dataset: InputName | ExecutorStep | LMMixtureDatasetConfig
    hyperparameter_scaling: Optional[Callable[[LmConfig, ComputeBudget], dict]] = None
    vocab_size: int = load_tokenizer(unwrap_versioned_value(tokenized_dataset.tokenizer)).vocab_size

    def estimate_training_flops_using_levanter(self) -> float:
        """Estimate training FLOPs using Levanter's built-in calculation."""
        flops_per_token = self.model_config.flops_per_token(self.vocab_size)
        batch_size = self.train_config.train_batch_size
        total_tokens = batch_size * self.model_config.seq_len * self.train_config.num_train_steps
        return float(flops_per_token * total_tokens * 3)

    def estimate_flops_via_6nd(self) -> float:
        """Estimate training FLOPs using 6 times N times D, where N is the number of parameters, and D is the number of tokens."""
        
        num_params = compute_num_parameters(self.model_config, self.vocab_size)
        batch_size = self.train_config.train_batch_size
        total_tokens = batch_size * self.model_config.seq_len * self.train_config.num_train_steps
        return 6 * num_params * total_tokens

    def adjust_to_exact_budget(self):
        """
        Adjust num_train_steps to ensure FLOPs exactly match the compute budget.

        Modifies train_config.num_train_steps based on the budget, model FLOPs per token,
        batch size, and sequence length.
        """
        flops_per_token = self.model_config.flops_per_token(self.vocab_size) * 3
        target_flops = self.compute_budget.value
        batch_size = self.train_config.train_batch_size
        seq_len = self.train_config.seq_len
        total_tokens = target_flops / flops_per_token
        self.train_config.num_train_steps = int(total_tokens / (batch_size * seq_len))
        actual_flops = self.estimate_training_flops_using_levanter()
        logger.info(f"Adjusted to {self.train_config.num_train_steps} steps for {actual_flops:e} FLOPs "
                    f"(target: {target_flops:e})")

    def apply_scaling(self):
        """
        Apply user-defined hyperparameter scaling if provided.

        Updates train_config with values from hyperparameter_scaling function, if present.
        """
        if self.hyperparameter_scaling:
            params = self.hyperparameter_scaling(self.model_config, self.compute_budget)
            self.train_config.learning_rate = params.get("learning_rate", self.train_config.learning_rate)
            self.train_config.train_batch_size = params.get("train_batch_size", self.train_config.train_batch_size)
            self.train_config.weight_decay = params.get("weight_decay", self.train_config.weight_decay)
            # Add more as needed (e.g., seq_len, warmup steps)

    def validate(self) -> tuple[bool, str | None]:
        """Validates the configuration against compute budget constraints, and makes adjustments as needed."""
        self.apply_scaling()
        self.adjust_to_exact_budget()
        levanter_flops_estimate = self.estimate_training_flops_using_levanter()
        estimate_flops_6nd = self.estimate_flops_via_6nd()
        logger.info(f"FLOPs estimate: {levanter_flops_estimate:e} vs. budget {self.compute_budget.value:e}")
        logger.info(f"FLOPs estimate: {estimate_flops_6nd:e} vs. budget {self.compute_budget.value:e}")
        return levanter_flops_estimate <= self.compute_budget.value, f"FLOPs {levanter_flops_estimate:e} vs. budget {self.compute_budget.value:e}"


def get_compute_budgets():
    return {budget.name: budget.value for budget in ComputeBudget}


def default_speedrun(
    name: str,
    config: SpeedrunConfig,
    tags: list[str] | None = None,
) -> ExecutorStep:
    """Returns an ExecutorStep for a speedrun with the given configuration.

    Args:
        name: name of the training run. Will form the basis of the output path for the executor step.
        config: SpeedrunConfig containing model, training, and dataset configuration
        tags: Optional additional tags for tracking

    Returns:
        training step configured for the speedrun

    Raises:
        ValueError: If the configuration is invalid
    """
    # Validate configuration
    is_valid, error = config.validate()
    if not is_valid:
        raise ValueError(f"Invalid speedrun configuration: {error}")

    # Set up tags
    run_tags = ["speedrun", f"budget_{config.compute_budget.name}"]
    if tags:
        run_tags.extend(tags)

    # Create training step
    model_config = dataclasses.replace(config.model_config)

    return default_train(
        name=name,
        tokenized=config.tokenized_dataset,
        model_config=model_config,
        train_config=config.train_config,
        tags=run_tags,
        eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
    )
