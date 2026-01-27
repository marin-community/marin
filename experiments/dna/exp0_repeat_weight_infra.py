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
Infrastructure experiment testing repeat weight downweighting strategies.

This experiment compares three loss weighting approaches on animal promoter sequences:
1. Standard: TextLmDatasetFormat with uniform loss weights (no soft masking)
2. Repeat weight 1.0: DNALmDatasetFormat with uniform weights (control)
3. Repeat weight 0.01: DNALmDatasetFormat with 0.01 loss weight on soft-masked positions
"""

from levanter.data.text import DNALmDatasetFormat, TextLmDatasetFormat

from experiments.dna.defaults import (
    DNA_TOKENIZER_V1,
    DNA_WINDOW_SIZE_BYTES_V1,
    PROMOTERS_DATASET_V1,
    SHORT_RUN_CONFIG_V1,
    dna_qwen3_0_6b_v1,
    dna_tokenize,
    dna_train,
)
from marin.execution.executor import executor_main

# =============================================================================
# Standard (no repeat weighting) - uses TextLmDatasetFormat
# =============================================================================

data_standard = dna_tokenize(
    name="animal-promoters-standard",
    dataset=PROMOTERS_DATASET_V1,
    tokenizer=DNA_TOKENIZER_V1,
    data_format=TextLmDatasetFormat(text_key="seq"),
    window_size_bytes=DNA_WINDOW_SIZE_BYTES_V1,
)

train_standard = dna_train(
    name="animal-promoters-standard-r08",
    tokenized=data_standard,
    model_config=dna_qwen3_0_6b_v1,
    train_config=SHORT_RUN_CONFIG_V1,
    tags=["dna", "animal-promoters"],
)

# =============================================================================
# Repeat weight 1.0 (control - uniform weighting via DNALmDatasetFormat)
# =============================================================================

data_rw_1_0 = dna_tokenize(
    name="animal-promoters-repeat-weight-1.0",
    dataset=PROMOTERS_DATASET_V1,
    tokenizer=DNA_TOKENIZER_V1,
    data_format=DNALmDatasetFormat(soft_mask_weight=1.0),
    window_size_bytes=DNA_WINDOW_SIZE_BYTES_V1,
)

train_rw_1_0 = dna_train(
    name="animal-promoters-repeat-weight-1.0-r01",
    tokenized=data_rw_1_0,
    model_config=dna_qwen3_0_6b_v1,
    train_config=SHORT_RUN_CONFIG_V1,
    tags=["dna", "animal-promoters"],
)

# =============================================================================
# Repeat weight 0.01 (strong downweighting)
# =============================================================================

data_rw_0_01 = dna_tokenize(
    name="animal-promoters-repeat-weight-0.01",
    dataset=PROMOTERS_DATASET_V1,
    tokenizer=DNA_TOKENIZER_V1,
    data_format=DNALmDatasetFormat(soft_mask_weight=0.01),
    window_size_bytes=DNA_WINDOW_SIZE_BYTES_V1,
)

train_rw_0_01 = dna_train(
    name="animal-promoters-repeat-weight-0.01-r01",
    tokenized=data_rw_0_01,
    model_config=dna_qwen3_0_6b_v1,
    train_config=SHORT_RUN_CONFIG_V1,
    tags=["dna", "animal-promoters"],
)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    executor_main(steps=[train_standard, train_rw_1_0, train_rw_0_01])
