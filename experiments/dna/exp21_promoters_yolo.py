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
YOLO experiment for animal promoters - comparing standard vs repeat downweighting.

https://github.com/Open-Athena/bolinas-dna/issues/21
"""

from experiments.dna.defaults import (
    PROMOTERS_DATASET_V1,
    YOLO_RUN_CONFIG_V1,
    dna_qwen3_1_7b_v1,
    dna_tokenize_rw_v1,
    dna_tokenize_std_v1,
    dna_train,
)
from marin.execution.executor import executor_main

# =============================================================================
# Standard (no repeat weighting)
# =============================================================================

data_standard = dna_tokenize_std_v1("animal-promoters-standard", PROMOTERS_DATASET_V1)

train_standard = dna_train(
    name="animal-promoters-yolo-standard-r01",
    tokenized=data_standard,
    model_config=dna_qwen3_1_7b_v1,
    train_config=YOLO_RUN_CONFIG_V1,
    tags=["dna", "animal-promoters", "yolo", "standard"],
)

# =============================================================================
# Repeat downweight (0.01)
# =============================================================================

data_downweight = dna_tokenize_rw_v1("animal-promoters-repeat-weight-0.01", PROMOTERS_DATASET_V1)

train_downweight = dna_train(
    name="animal-promoters-yolo-r01",
    tokenized=data_downweight,
    model_config=dna_qwen3_1_7b_v1,
    train_config=YOLO_RUN_CONFIG_V1,
    tags=["dna", "animal-promoters", "yolo"],
)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    executor_main(steps=[train_standard, train_downweight])
