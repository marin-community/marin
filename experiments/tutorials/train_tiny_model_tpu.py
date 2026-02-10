# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
This is a tutorial on how to train a tiny model on a small dataset using TPU.

For CPU training, see train_tiny_model_cpu.py
For GPU training, see train_tiny_model_gpu.py
"""

from fray.v2 import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import executor_main, versioned

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_30m
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

RESOURCES = ResourceConfig.with_tpu(
    "v5litepod-16",
    slice_count=4,
    cpu=32,
    ram="128g",
    disk="50g",
)

# 1. Choose a dataset
tinystories_hf_id = "roneneldan/TinyStories"

# 2. Tokenize the dataset with sampling
# For this tutorial, we limit to 1000 documents per shard
tinystories_tokenized = default_tokenize(
    name=tinystories_hf_id,
    dataset=tinystories_hf_id,
    tokenizer=marin_tokenizer,
    format=TextLmDatasetFormat(),
    sample_count=1000,
)


# 3. Define training configuration
small_train_config = SimpleTrainConfig(
    # Here we define the hardware resources we need.
    resources=RESOURCES,
    train_batch_size=128,
    num_train_steps=200,
    # set hyperparameters
    learning_rate=6e-4,
    weight_decay=0.1,
)

# 4. Train the model
tinystories_model_30m = default_train(
    name="marin-tinystories-30m",
    # Steps can depend on other steps: tinystories_model_30m depends on tinystories_tokenized
    tokenized=tinystories_tokenized,
    model_config=versioned(llama_30m),
    train_config=small_train_config,
    # wandb tags
    tags=["llama", "30m", "tinystories", "tutorial"],
    # We can run many [eval_harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks in the loop
    # during training, but there's no point in running evals on such a tiny model
    eval_harness_tasks=[],
    # to keep tutorial fast, skip default validation sets
    use_default_validation=False,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            tinystories_model_30m,
        ]
    )
