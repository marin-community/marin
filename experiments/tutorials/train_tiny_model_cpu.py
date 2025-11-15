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
This is a tutorial on how to train a tiny model on a small dataset using CPU.

This script demonstrates how to:
1. Train a tiny model on TinyStories using CPU
2. Use CPU-specific training configuration
3. Run a quick training experiment

For GPU training, see train_tiny_model_gpu.py
"""

from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import executor_main, versioned
from marin.resources import CpuOnlyConfig

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_nano
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

# 1. Choose a dataset
tinystories_hf_id = "roneneldan/TinyStories"

# 2. Tokenize the dataset with sampling
# For this tutorial, we limit to 1000 documents per shard
tinystories_tokenized = default_tokenize(
    name=tinystories_hf_id,
    dataset=tinystories_hf_id,
    tokenizer=marin_tokenizer,
    format=TextLmDatasetFormat(),
    sample_count=versioned(1000),
)


# 3. Define training configuration
nano_train_config = SimpleTrainConfig(
    # Here we define the hardware resources we need.
    resources=CpuOnlyConfig(num_cpus=1),
    train_batch_size=4,
    num_train_steps=100,
    # set hyperparameters
    learning_rate=6e-4,
    weight_decay=0.1,
    # keep eval quick for tutorial
    max_eval_batches=4,
)

# 4. Train the model
nano_tinystories_model = default_train(
    name="marin-nano-tinystories",
    # Steps can depend on other steps: nano_tinystories_model depends on tinystories_tokenized
    tokenized=tinystories_tokenized,
    model_config=llama_nano,
    train_config=nano_train_config,
    # wandb tags
    tags=["llama", "nano", "tinystories", "tutorial"],
    # We can run many [eval_harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks in the loop
    # during training, but there's no point in running evals on such a tiny model
    eval_harness_tasks=[],
    # to keep tutorial fast, skip default validation sets
    use_default_validation=False,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            nano_tinystories_model,
        ]
    )
