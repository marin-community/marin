# nodryrun
import logging
import jax
from levanter.models.mixtral import MixtralConfig
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun
import os

# Enable JAX debugging
jax.config.update("jax_debug_nans", True)
jax.config.update("jax_debug_infs", True)

# This config uses MixtralConfig directly for MoE functionality
moe_300m_config = MixtralConfig(
    seq_len=1024,
    hidden_dim=768,
    intermediate_dim=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    gradient_checkpointing=True,
    scan_layers=True,
    n_routed_experts=32,
    num_experts_per_tok=4,
    # Disable MoE auxiliary loss logging to prevent JAX tracer leaks
    lbl_coef=None,  # Disables load balancing loss logging
    rzl_coef=None,  # Disables router z-loss logging
)

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Pranshu Chaturvedi",
        affiliation="Stanford University",
        url="https://stanford.edu/~pranshu",
    ),
    description="Training a 300M parameter Mixtral-style MoE model on a TPU with debugging.",
    model_config=moe_300m_config,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v4-8"),
        train_batch_size=256,
        num_train_steps=100,  # Reduced for debugging
        learning_rate=5e-4,
        weight_decay=0.1,
        steps_per_eval=50,
    ),
)

if __name__ == "__main__":
    executor_main(steps=default_speedrun("pranshu_mixtral_300m_debug", speedrun_config))
