# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
YOLO experiment for animal mRNA + promoters - comparing standard vs repeat downweighting.

https://github.com/Open-Athena/bolinas-dna/issues/22
"""

import dataclasses

from experiments.dna.defaults import (
    DNA_TOKENIZER_V1,
    MRNA_PLUS_PROMOTERS_DATASET_V1,
    YOLO_RUN_CONFIG_V1,
    dna_effective_seq_len,
    dna_tokenize_rw_v1,
    dna_tokenize_std_v1,
    dna_train,
)
from experiments.qwen3 import qwen3_1_7b
from marin.execution.executor import executor_main

SEQ_LEN = 512
model_config = dataclasses.replace(qwen3_1_7b, max_seq_len=dna_effective_seq_len(SEQ_LEN, DNA_TOKENIZER_V1))

# =============================================================================
# Standard (no repeat weighting)
# =============================================================================

data_standard = dna_tokenize_std_v1("animal-mRNA-plus-promoters", MRNA_PLUS_PROMOTERS_DATASET_V1)

train_standard = dna_train(
    name="animal-mRNA-plus-promoters-yolo-r01",
    tokenized=data_standard,
    model_config=model_config,
    train_config=YOLO_RUN_CONFIG_V1,
    tags=["dna", "animal-mRNA-plus-promoters", "yolo"],
)

# =============================================================================
# Repeat downweight (0.01)
# =============================================================================

data_downweight = dna_tokenize_rw_v1("animal-mRNA-plus-promoters-repeat-weight-0.01", MRNA_PLUS_PROMOTERS_DATASET_V1)

train_downweight = dna_train(
    name="animal-mRNA-plus-promoters-yolo-repeat-weight-0.01-r01",
    tokenized=data_downweight,
    model_config=model_config,
    train_config=YOLO_RUN_CONFIG_V1,
    tags=["dna", "animal-mRNA-plus-promoters", "yolo"],
)

if __name__ == "__main__":
    executor_main(steps=[train_standard, train_downweight])
