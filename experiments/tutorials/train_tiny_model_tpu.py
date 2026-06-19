# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
This is a tutorial on how to train a tiny model on a small dataset using TPU.

For CPU training, see train_tiny_model_cpu.py
For GPU training, see train_tiny_model_gpu.py
"""

import dataclasses

import draccus
from fray import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.execution.types import versioned

from experiments.defaults import train
from experiments.launch import LaunchConfig, launch, override_resources
from experiments.llama import llama_30m
from experiments.marin_models import marin_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tokenization import default_tokenize

RESOURCES = ResourceConfig.with_tpu(
    "v5litepod-16",
    slice_count=1,
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
    num_train_steps=10000,
    # set hyperparameters
    learning_rate=6e-4,
    weight_decay=0.1,
)


@draccus.wrap()
def main(config: LaunchConfig):
    # 4. `launch` runs `train` on the `--cluster` coordinator (or in-process for
    # `--local`); `train` resolves the output path and blocks until the job ends.
    train_config = dataclasses.replace(small_train_config, resources=override_resources(RESOURCES, config))
    launch(
        config,
        train,
        name="marin-tinystories-30m",
        # Steps can depend on other steps: the training job depends on tinystories_tokenized
        tokenized=tinystories_tokenized,
        model_config=versioned(llama_30m),
        train_config=train_config,
        # wandb tags
        tags=["llama", "30m", "tinystories", "tutorial"],
        # We can run many [eval_harness](https://github.com/EleutherAI/lm-evaluation-harness) tasks in the loop
        # during training, but there's no point in running evals on such a tiny model
        eval_harness_tasks=[],
        # to keep tutorial fast, skip default validation sets
        use_default_validation=False,
    )


if __name__ == "__main__":
    main()
