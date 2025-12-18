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

# nodryrun

"""Standard MoE run for fair comparison - no GMM, just standard ragged dot."""

import logging

from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from levanter.models.mixtral import MixtralConfig
from marin.execution.executor import executor_main
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

# Enable detailed logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# This config uses standard MixtralConfig for MoE functionality
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

# Log configuration details
logger.info("=" * 60)
logger.info("Standard MoE Configuration (Ragged Dot Implementation)")
logger.info("=" * 60)
logger.info(f"Model: {moe_300m_config.__class__.__name__}")
logger.info(f"Experts: {moe_300m_config.n_routed_experts}")
logger.info(f"Experts per token: {moe_300m_config.num_experts_per_tok}")
logger.info(f"Hidden dim: {moe_300m_config.hidden_dim}")
logger.info(f"Intermediate dim: {moe_300m_config.intermediate_dim}")
logger.info(f"Num layers: {moe_300m_config.num_layers}")
logger.info(f"Auxiliary losses disabled: lbl_coef={moe_300m_config.lbl_coef}, rzl_coef={moe_300m_config.rzl_coef}")
logger.info("=" * 60)

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Pranshu Chaturvedi",
        affiliation="Stanford University",
        url="https://stanford.edu/~pranshu",
    ),
    description="Training a 300M parameter Mixtral-style MoE model on a TPU with standard ragged dot implementation",
    model_config=moe_300m_config,
    train_config=SimpleTrainConfig(
        ResourceConfig.with_tpu("v4-8", slice_count=1),
        train_batch_size=256,
        num_train_steps=4000,
        learning_rate=5e-4,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)

if __name__ == "__main__":
    # Add logging to confirm standard implementation is being used
    logger.info("Running Mixtral with standard ragged dot MoE implementation")
    logger.info("This should produce identical results to the original pranshu_mixtral_moe_run.py")

    executor_main(steps=default_speedrun("pranshu_mixtral_300m_standard_comparison", speedrun_config))
