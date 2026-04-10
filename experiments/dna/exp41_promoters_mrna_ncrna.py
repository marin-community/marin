# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Experiment 41: Promoters from mRNA vs ncRNA.

Question: Is it better to train on promoters from mRNA transcripts only
or to also add promoters from ncRNA transcripts?

https://github.com/Open-Athena/bolinas-dna/issues/41
"""

import dataclasses

from experiments.dna.defaults import (
    DNA_TOKENIZER_V1,
    FAST_RUN_CONFIG_V1,
    PROMOTERS_MRNA_DATASET_V1,
    PROMOTERS_MRNA_NCRNA_DATASET_V1,
    dna_effective_seq_len,
    dna_tokenize_rw_v1,
    dna_train,
)
from experiments.qwen3 import qwen3_0_6b_hd128
from marin.execution.executor import executor_main

SEQ_LEN = 512
model_config = dataclasses.replace(qwen3_0_6b_hd128, max_seq_len=dna_effective_seq_len(SEQ_LEN, DNA_TOKENIZER_V1))

DATASETS = [
    PROMOTERS_MRNA_DATASET_V1,
    PROMOTERS_MRNA_NCRNA_DATASET_V1,
]


def dataset_name(dataset: str) -> str:
    """Extract dataset name from HuggingFace path (org/name -> name)."""
    return dataset.split("/")[-1]


training_steps = []
for dataset in DATASETS:
    name = dataset_name(dataset)

    tokenized = dna_tokenize_rw_v1(
        name=f"{name}-rw01",
        dataset=dataset,
    )

    train_step = dna_train(
        name=f"exp41-{name}-r02",
        tokenized=tokenized,
        model_config=model_config,
        train_config=FAST_RUN_CONFIG_V1,
        tags=["dna", "exp41", "promoters", "fast"],
    )
    training_steps.append(train_step)

if __name__ == "__main__":
    executor_main(steps=training_steps)
