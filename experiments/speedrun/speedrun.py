"""
Default configurations and utilities for the Marin speedrun experiment.
"""
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from levanter.models.lm_model import LmConfig
from levanter.data.text import LMMixtureDatasetConfig
from marin.execution.executor import (
    ExecutorStep,
    InputName,
    executor_main,
)
from marin.evaluation.evaluation_config import EvalTaskConfig
from marin.training.training import TrainLmOnPodConfig
from experiments.simple_train_config import SimpleTrainConfig

class ComputeBudget(Enum):
    # TODO: these are rough guesses from looking at a few of our wandb runs and the compute budgets Meta used
    # we could decide these based on analyzing FLOPs from models we've trained so far and setting thresholds
    # based on that
    TINY = 3e18
    SMALL = 6e18 # ~150M params
    MEDIUM = 3e19
    # LARGE = 3e20

@dataclass
class SpeedrunConfig:
    """Configuration for a speedrun submission."""
    compute_budget: ComputeBudget
    model_config: LmConfig
    train_config: SimpleTrainConfig | TrainLmOnPodConfig
    tokenized_dataset: InputName | ExecutorStep | LMMixtureDatasetConfig
    
    def _estimate_training_flops(self) -> float:
        """
        
        TODO (Nikil): I had windsurf write this; need to go through and check/correct.

        Estimates total training FLOPs based on model config and training setup.
        
        Uses JAX's cost analysis to estimate FLOPs for a single forward pass,
        then multiplies by training steps and batch size.
        
        Note: This is a rough estimate and may need refinement.
        """
        import jax
        import jax.numpy as jnp
        from functools import partial
        
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
                attn = jnp.einsum('bld,bmd->blm', embed, embed)
                # FFN: 8 * batch_size * seq_len * d_model * d_model
                ffn = jnp.einsum('bld,dd->bld', embed, jnp.ones((d_model, d_model)))
            
            # 3. Final projection to vocab
            logits = jnp.einsum('bld,vd->blv', embed, jnp.ones((vocab_size, d_model)))
            return logits
        
        # Get batch size and steps
        batch_size = self.train_config.train_batch_size
        if hasattr(batch_size, '__call__'):  # Handle IntSchedule
            batch_size = batch_size(0)  # Use initial batch size
        num_steps = self.train_config.num_train_steps
        
        # Analyze cost of a single forward pass
        dummy_params = {}
        cost = jax.jit(forward_step).lower(dummy_params, batch_size).cost_analysis()
        flops_per_fwd = cost.get('flops', 0)
        
        # Total FLOPs = FLOPs/fwd * 2 (for backward pass) * steps * batch_size
        total_flops = flops_per_fwd * 2 * num_steps * batch_size
        
        return float(total_flops)

    def validate(self) -> tuple[bool, Optional[str]]:
        """validates the configuration."""
        # validate compute budget based on model size and training steps
        estimated_flops = self._estimate_training_flops()
        if estimated_flops > self.compute_budget.value:
            return False, f"Estimated FLOPs {estimated_flops} exceeds budget {self.compute_budget.value}"
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
    return default_train(
        name=name,
        tokenized=config.tokenized_dataset,
        model_config=config.model_config,
        train_config=config.train_config,
        tags=run_tags,
        eval_harness_tasks=SPEEDRUN_CORE_TASKS,
    )
