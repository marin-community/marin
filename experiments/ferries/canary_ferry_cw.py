# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""CoreWeave canary ferry: llama-150M on SlimPajama-6B, 1B tokens, 8x H100.

Usage (via Iris):
    iris --config=lib/iris/examples/coreweave.yaml job run \
        --gpu H100x8 --memory 256GB --disk 1TB \
        -e MARIN_PREFIX s3://marin-test -e WANDB_API_KEY $WANDB_API_KEY \
        -- python -m experiments.ferries.canary_ferry_cw
"""

import datetime
import os

from fray.cluster import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from marin.execution.executor import executor_main

from experiments.defaults import default_tokenize, default_train
from experiments.llama import llama_150m, llama3_tokenizer
from experiments.simple_train_config import SimpleTrainConfig

CANARY_DATE = os.environ.get("CANARY_DATE", datetime.date.today().isoformat())

BATCH_SIZE = 64
SEQ_LEN = 1024
TARGET_TOKENS = 1_000_000_000
NUM_STEPS = TARGET_TOKENS // (BATCH_SIZE * SEQ_LEN)

slimpajama_tokenized = default_tokenize(
    name="slimpajama-6b-cw",
    dataset="DKYoon/SlimPajama-6B",
    tokenizer=llama3_tokenizer,
    format=TextLmDatasetFormat(),
)

train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_gpu(count=8, cpu=128, ram="256g", disk="256g"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_STEPS,
    learning_rate=3e-3,
    weight_decay=0.1,
    train_seq_len=SEQ_LEN,
    steps_per_eval=500,
)

training_step = default_train(
    name=f"canary-ferry-cw-{CANARY_DATE}",
    tokenized=slimpajama_tokenized,
    model_config=llama_150m,
    train_config=train_config,
    tags=["canary", "ferry", "llama", "150m", "slimpajama", "coreweave"],
    eval_harness_tasks=[],
    use_default_validation=False,
)


def main():
    executor_main(steps=[training_step])


if __name__ == "__main__":
    main()
