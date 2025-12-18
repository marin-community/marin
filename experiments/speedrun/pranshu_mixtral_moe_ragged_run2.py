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
import logging

from experiments.simple_train_config import SimpleTrainConfig
from experiments.speedrun.custom_mixtral import MixtralConfig
from marin.execution.executor import executor_main
from marin.resources import TpuPodConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

# Configuration that uses RAGGED DOT (default behavior)
moe_300m_config_ragged = MixtralConfig(
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
    use_gmm=False,  # This ensures ragged dot is used (default behavior)
    # Disable MoE auxiliary loss logging to prevent JAX tracer leaks
    lbl_coef=None,  # disables load balancing loss logging
    rzl_coef=None,  # disables router z-loss logging
)

speedrun_config = SpeedrunConfig(
    author=Author(
        name="Pranshu Chaturvedi",
        affiliation="Stanford University",
        url="https://stanford.edu/~pranshu",
    ),
    description="Training a 300M parameter Mixtral-style MoE model on a TPU with Ragged Dot (baseline)",
    model_config=moe_300m_config_ragged,
    train_config=SimpleTrainConfig(
        TpuPodConfig(tpu_type="v4-8"),
        train_batch_size=256,
        num_train_steps=4000,
        learning_rate=5e-4,
        weight_decay=0.1,
        steps_per_eval=1000,
    ),
)

if __name__ == "__main__":
    # Add logging to confirm ragged dot is being used
    logger.info("Running Mixtral with Ragged Dot MoE layers from custom_mixtral.py")
    logger.info(f"Model type: {moe_300m_config_ragged.model_type}")
    logger.info(f"Using GMM: {moe_300m_config_ragged.use_gmm}")

    executor_main(steps=default_speedrun("pranshu_mixtral_300m_ragged_run2", speedrun_config))
