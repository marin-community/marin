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

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.execution import step, versioned
from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_nano
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

# 1. Choose a dataset
tinystories_hf_id = "roneneldan/TinyStories"

# 3. Define training configuration
nano_train_config = SimpleTrainConfig(
    # Here we define the hardware resources we need.
    resources=ResourceConfig.with_cpu(),
    train_batch_size=4,
    num_train_steps=100,
    # set hyperparameters
    learning_rate=6e-4,
    weight_decay=0.1,
    # keep eval quick for tutorial
    max_eval_batches=4,
)


def tokenize_tinystories():
    """Tokenize the TinyStories dataset with sampling."""
    return default_tokenize(
        name=tinystories_hf_id,
        dataset=tinystories_hf_id,
        tokenizer=marin_tokenizer,
        format=TextLmDatasetFormat(),
        sample_count=1000,
    )


@step(name="tutorials/train_tiny_model_cpu/all")
def run_cpu_training():
    """Entry point for CPU training tutorial."""
    # 2. Tokenize the dataset with sampling
    # For this tutorial, we limit to 1000 documents per shard
    tinystories_tokenized = tokenize_tinystories()

    # Train the model
    nano_tinystories_model = default_train(
        name="marin-nano-tinystories",
        # Steps can depend on other steps: nano_tinystories_model depends on tinystories_tokenized
        tokenized=tinystories_tokenized,
        model_config=versioned(llama_nano),
        train_config=nano_train_config,
        # wandb tags
        tags=["llama", "nano", "tinystories", "tutorial"],
        # We can run many [eval_harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks in the loop
        # during training, but there's no point in running evals on such a tiny model
        eval_harness_tasks=[],
        # to keep tutorial fast, skip default validation sets
        use_default_validation=False,
    )

    return nano_tinystories_model


if __name__ == "__main__":
    executor_main(
        steps=[run_cpu_training()],
        description="Train a tiny model on CPU using TinyStories dataset",
    )
