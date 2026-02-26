# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave canary ferry: llama-150M on SlimPajama-6B, 1B tokens, 8x H100.

Uses block-shuffle for better I/O locality on remote object stores (R2/S3),
reducing random-read pressure while maintaining good data mixing via
hierarchical Feistel permutation.

Usage (via Iris):
    iris --config=lib/iris/examples/coreweave.yaml job run \
        -e MARIN_PREFIX s3://marin-test -e WANDB_API_KEY $WANDB_API_KEY \
        -- python -m experiments.ferries.canary_ferry_cw
"""

import dataclasses
import datetime
import os

from fray.cluster import ResourceConfig
from levanter.data.text import BlockShuffleConfig, TextLmDatasetFormat
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_data_config

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama3_tokenizer, llama_150m
from experiments.simple_train_config import SimpleTrainConfig

CANARY_DATE = os.environ.get("CANARY_DATE", datetime.date.today().isoformat())

BATCH_SIZE = 4096
SEQ_LEN = 1024
TARGET_TOKENS = 1_000_000_000
NUM_STEPS = TARGET_TOKENS // (BATCH_SIZE * SEQ_LEN)

_slimpajama_step = default_tokenize(
    name="slimpajama-6b-cw",
    dataset="DKYoon/SlimPajama-6B",
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(),
)
# Default worker_resources is 5GB which OOMs on SlimPajama-6B tokenization.
slimpajama_tokenized = dataclasses.replace(
    _slimpajama_step,
    config=dataclasses.replace(
        _slimpajama_step.config,
        worker_resources=ResourceConfig(ram="64g", disk="64g"),
    ),
)

data_config = lm_data_config(
    training_set=slimpajama_tokenized,
    shuffle=BlockShuffleConfig(
        io_block_size=256,
        window_blocks=256,
        perm_type="feistel",
    ),
)

train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_gpu(count=8, cpu=32, ram="256g", disk="256g"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_STEPS,
    learning_rate=3e-3,
    weight_decay=0.1,
    train_seq_len=SEQ_LEN,
    steps_per_eval=50,
)

training_step = default_train(
    name=f"canary-ferry-cw-{CANARY_DATE}",
    tokenized=data_config,
    model_config=llama_150m,
    train_config=train_config,
    tags=["canary", "ferry", "llama", "150m", "slimpajama", "coreweave"],
    eval_harness_tasks=[],
)


def main():
    executor_main(steps=[training_step])


if __name__ == "__main__":
    main()
