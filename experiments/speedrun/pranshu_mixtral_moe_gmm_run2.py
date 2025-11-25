# nodryrun
import logging
import dataclasses
from typing import Optional

import jax
import jax.random as jrandom
import haliax as hax
import haliax.nn as hnn
from haliax.jax_utils import maybe_rng_split, shaped_rng_split
from haliax.nn.scan import Stacked, BlockSeq

# Import from the custom mixtral implementation
from experiments.speedrun.custom_mixtral import (
    CustomMixtralConfig,
    MixtralLMHeadModel,
)

from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

# Configuration that enables GMM
moe_300m_config_gmm = CustomMixtralConfig(
    seq_len=1024,
    hidden_dim=768,
    intermediate_dim=768,
    num_layers=12,
    num_heads=12,
    num_kv_heads=12,
    gradient_checkpointing=True,
    scan_layers=True,
    n_routed_experts=8,
    num_experts_per_tok=2,
    use_gmm=True, # https://github.com/AI-Hypercomputer/maxtext/issues/1183  We would expect to see differences in training loss, parameter norms etc. with a GMM implementation over a ragged dot implementation
    # Disable MoE auxiliary loss logging to prevent JAX tracer leaks
    lbl_coef=None,  # disables load balancing loss logging
    rzl_coef=None,  # disables router z-loss logging
)

train_batch_size = 1536

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Pranshu Chaturvedi",
        affiliation="Stanford University", 
        url="https://stanford.edu/~pranshu",
    ),
    description="Training a 300M parameter Mixtral-style MoE model on a TPU with GMM (Grouped Matrix Multiply)",
    model_config=moe_300m_config_gmm,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v5p-32", slice_count=1),
        train_batch_size=train_batch_size,
        num_train_steps=6000,
        learning_rate=2e-4,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)

if __name__ == "__main__":
    # Add logging to confirm GMM is being used
    logger.info("Running Mixtral with GMM-enabled MoE layers from custom_mixtral.py")
    logger.info(f"Model type: {moe_300m_config_gmm.model_type}")
    logger.info(f"Using GMM: {moe_300m_config_gmm.use_gmm}")
    
    executor_main(steps=default_speedrun(f"pranshu_mixtral_300m_gmm_run_central1_fresh_bs{train_batch_size}", speedrun_config))

"""
2025-09-30 01:42:14,010 INFO resources_utils.py:122 -- TPU type: v5p, suffix: 32
2025-09-30 01:42:14,009 INFO pranshu_mixtral_moe_gmm_sweep.py:149 -- Prepared pranshu_mixtral_gmm_300m_default with 247.4M parameters
2025-09-30 01:42:14,009 INFO pranshu_mixtral_moe_gmm_sweep.py:132 -- Variant mixtral_1_5b: params=1342.0M ratio=5.42 batch=640 total_tokens=37.75B
2025-09-30 01:42:14,009 INFO pranshu_mixtral_moe_gmm_sweep.py:132 -- Variant mixtral_1b: params=993.0M ratio=4.01 batch=768 total_tokens=25.17B
2025-09-30 01:42:14,009 INFO pranshu_mixtral_moe_gmm_sweep.py:132 -- Variant mixtral_300m: params=247.4M ratio=1.00 batch=1536 total_tokens=6.29B
"""