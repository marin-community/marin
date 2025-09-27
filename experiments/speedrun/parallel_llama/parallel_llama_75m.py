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

"""
Speedrun experiment with a 75M parameter Parallel Llama model.
This configuration uses proper 75M model dimensions suitable for H100 GPU training.
"""

import logging

from experiments.speedrun.parallel_llama.exp1571_parallel_llama import ParallelLlamaConfig
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main
from marin.resources import GpuConfig
from marin.speedrun.speedrun import Author, SpeedrunConfig, default_speedrun

logger = logging.getLogger("ray")

# Define the parallel Llama 75M model configuration
# Configuration for a ~75M parameter model with parallel attention/MLP computation
# Matches the standard llama_75m config from experiments/llama.py
parallel_llama_75m = ParallelLlamaConfig(
    seq_len=1024,
    hidden_dim=256,
    intermediate_dim=896,
    num_heads=4,
    num_kv_heads=4,
    num_layers=8,
    use_bias=False,
    use_layer_norm_weight=True,
    initializer_range=0.02,
    layer_norm_epsilon=1e-5,
    tie_word_embeddings=False,
    use_parallel_blocks=True,
    cross_entropy_block_size=32000
)

train_config = SimpleTrainConfig(
    resources=GpuConfig(gpu_count=1, accelerator_type="H100"),
    train_batch_size=64,
    num_train_steps=10000,
    learning_rate=3e-3,
    weight_decay=0.1,
    steps_per_eval=2000,
)

# Speedrun configuration
speedrun_config = SpeedrunConfig(
    author=Author(
        name="Harry Shin",
        affiliation="Independent",
        url="https://www.linkedin.com/in/harry-shin-34743216a/",
    ),
    description=(
        "75M parameter Parallel Llama model with custom transformer blocks. "
        "Features truly parallel MLP/Attention computation with shared layer normalization. "
    ),
    model_config=parallel_llama_75m,
    train_config=train_config,
)

speedrun_config.print_run_info()


def main():
    """Main function to run the parallel Llama 75M speedrun."""
    steps = default_speedrun("parallel_llama_75m", speedrun_config)
    executor_main(steps=steps)


if __name__ == "__main__":
    main()
