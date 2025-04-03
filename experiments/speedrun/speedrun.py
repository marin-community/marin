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
from experiments.llama import llama3_tokenizer_vocab_size
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import (
    ExecutorStep,
    InputName,
)
from marin.training.training import TrainLmOnPodConfig

logger = logging.getLogger(__name__)


class ComputeBudget(Enum):
    # TODO: these are rough guesses from looking at a few of our wandb runs and the compute budgets Meta used
    # we could decide these based on analyzing FLOPs from models we've trained so far and setting thresholds
    # based on that
    TINY = 3e18
    SMALL = 6e18  # ~150M params
    MEDIUM = 3e19
    # LARGE = 3e20


@dataclass
class SpeedrunConfig:
    """Configuration for a speedrun submission."""

    compute_budget: ComputeBudget
    model_config: LmConfig
    train_config: SimpleTrainConfig | TrainLmOnPodConfig
    tokenized_dataset: InputName | ExecutorStep | LMMixtureDatasetConfig
    vocab_size: int = llama3_tokenizer_vocab_size

    def estimate_training_flops_using_levanter(self) -> float:
        """Estimate training FLOPs using Levanter's built-in calculation."""
        flops_per_token = self.model_config.flops_per_token(self.vocab_size)
        batch_size = self.train_config.train_batch_size
        if callable(batch_size):
            batch_size = batch_size(0)
        total_tokens = batch_size * self.model_config.seq_len * self.train_config.num_train_steps
        return float(flops_per_token * total_tokens * 2)

    def _estimate_training_flops(self) -> float:
        """Estimates total training FLOPs based on model config and training setup.

        Current implementation uses JAX's cost analysis to estimate FLOPs for a forward pass.
        There's an alternative implementation using Levanter's built-in flops_per_token
        in estimate_training_flops_using_levanter() which may be more accurate.

        The current implementation:
        1. Defines a simplified forward pass with key operations:
           - Token embedding
           - Self-attention (4 * b * s * d * d FLOPs)
           - FFN (8 * b * s * d * d FLOPs)
           - Final projection
        2. Uses JAX's cost analysis to count FLOPs
        3. Multiplies by 2 for backward pass, steps, and batch size

        Note: This is a rough estimate and may need refinement.
        """
        from functools import partial

        import jax
        import jax.numpy as jnp

        # Get model parameters
        d_model = self.model_config.d_model
        n_heads = self.model_config.n_heads
        n_layers = self.model_config.n_layers
        vocab_size = self.model_config.vocab_size
        seq_len = self.model_config.max_sequence_length

        # Define a simple forward pass function to analyze
        @partial(jax.jit, static_argnums=(1,))
        def forward_step(params, batch_size):
            # Simplified forward pass operations
            # 1. Token embedding: batch_size * seq_len * d_model
            embed = jnp.ones((batch_size, seq_len, d_model))

            # 2. For each layer:
            for _ in range(n_layers):
                # Self-attention: 4 * batch_size * seq_len * d_model * d_model
                attn = jnp.einsum("bld,bmd->blm", embed, embed)
                # FFN: 8 * batch_size * seq_len * d_model * d_model
                ffn = jnp.einsum("bld,dd->bld", embed, jnp.ones((d_model, d_model)))

            # 3. Final projection to vocab
            logits = jnp.einsum("bld,vd->blv", embed, jnp.ones((vocab_size, d_model)))
            return logits

        # Get batch size and steps
        batch_size = self.train_config.train_batch_size
        if callable(batch_size):  # Handle IntSchedule
            batch_size = batch_size(0)  # Use initial batch size
        num_steps = self.train_config.num_train_steps

        # Analyze cost of a single forward pass
        dummy_params = {}
        cost = jax.jit(forward_step).lower(dummy_params, batch_size).cost_analysis()
        flops_per_fwd = cost.get("flops", 0)

        # Total FLOPs = FLOPs/fwd * 2 (for backward pass) * steps * batch_size
        total_flops = flops_per_fwd * 2 * num_steps * batch_size

        return float(total_flops)

    def validate(self) -> tuple[bool, str | None]:
        """Validates the configuration against compute budget constraints."""
        estimated_flops = self.estimate_training_flops_using_levanter()
        logger.info(f"Estimated FLOPs: {estimated_flops}")
        if estimated_flops > self.compute_budget.value:
            return False, f"Estimated FLOPs {estimated_flops:e} exceeds budget {self.compute_budget.value:e}"
        return True, None


def get_compute_budgets():
    """Returns available compute budget categories."""
    return {budget.name: budget.value for budget in ComputeBudget}


def default_speedrun(
    name: str,
    config: SpeedrunConfig,
    tags: list[str] | None = None,
) -> ExecutorStep:
    """Returns an ExecutorStep for a speedrun with the given configuration.

    Args:
        name: Name for the training run
        config: SpeedrunConfig containing model, training, and dataset configuration
        tags: Optional additional tags for tracking

    Returns:
        ExecutorStep configured for the speedrun

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

    from experiments.defaults import default_train

    # Create training step
    # Make a deep copy of the model config to ensure proper serialization
    model_config = dataclasses.replace(config.model_config)

    return default_train(
        name=name,
        tokenized=config.tokenized_dataset,
        model_config=model_config,
        train_config=config.train_config,
        tags=run_tags,
        eval_harness_tasks=CORE_TASKS_PLUS_MMLU,
    )
