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
Mixture experiment for animal promoters and CDS regions.

This experiment trains on a mixture of promoter and CDS sequences with two
different weighting schemes to study the effect of data composition:

1. "equal": 50/50 weight split between promoters and CDS
   - Since CDS dataset is ~8.4x larger, CDS sequences will be seen less often
   - Each promoter sequence is seen ~8.4x more than each CDS sequence

2. "proportional": Weights proportional to dataset sizes
   - Each sequence from both datasets is seen approximately equally often
   - Natural sampling based on dataset sizes

https://github.com/Open-Athena/bolinas-dna/issues/13
"""

from levanter.data.text import DNALmDatasetFormat

from experiments.defaults import default_train
from experiments.dna.defaults import (
    CDS_DATASET_V1,
    DNA_TOKENIZER_V1,
    DNA_WINDOW_SIZE_BYTES_V1,
    PROMOTERS_DATASET_V1,
    YOLO_RUN_CONFIG_V1,
    dna_qwen3_1_7b_v1,
    dna_tokenize,
)
from marin.execution.executor import executor_main
from marin.processing.tokenize import lm_mixture_data_config

# =============================================================================
# Dataset sizes (number of sequences)
# =============================================================================

PROMOTERS_SIZE = 22_588_618
CDS_SIZE = 189_627_296
# CDS is ~8.4x larger than promoters

# =============================================================================
# Weight configurations for controlled experiment
# =============================================================================

WEIGHT_CONFIGS = {
    "equal": {
        "promoters": 0.5,
        "cds": 0.5,
    },
    "proportional": {
        "promoters": PROMOTERS_SIZE,  # raw counts get normalized
        "cds": CDS_SIZE,
    },
}

# =============================================================================
# Tokenize each dataset (shared across runs)
# =============================================================================

tokenized_datasets = {
    "promoters": dna_tokenize(
        name="animal-promoters-repeat-weight-0.01",
        dataset=PROMOTERS_DATASET_V1,
        tokenizer=DNA_TOKENIZER_V1,
        data_format=DNALmDatasetFormat(soft_mask_weight=0.01),
        window_size_bytes=DNA_WINDOW_SIZE_BYTES_V1,
    ),
    "cds": dna_tokenize(
        name="animal-cds-repeat-weight-0.01",
        dataset=CDS_DATASET_V1,
        tokenizer=DNA_TOKENIZER_V1,
        data_format=DNALmDatasetFormat(soft_mask_weight=0.01),
        window_size_bytes=DNA_WINDOW_SIZE_BYTES_V1,
    ),
}

# =============================================================================
# Create training steps for each weight configuration
# =============================================================================

training_steps = []
for config_name, weights in WEIGHT_CONFIGS.items():
    mixture_config = lm_mixture_data_config(
        components=tokenized_datasets,
        weights=weights,
    )

    training_step = default_train(
        name=f"promoter-cds-mixture-{config_name}-r01",
        tokenized=mixture_config,
        model_config=dna_qwen3_1_7b_v1,
        train_config=YOLO_RUN_CONFIG_V1,
        tags=["dna", "mixture", config_name],
        eval_harness_tasks=[],
        use_default_validation=False,
    )
    training_steps.append(training_step)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    executor_main(steps=[*training_steps])
