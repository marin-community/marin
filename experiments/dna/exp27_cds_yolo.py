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
YOLO experiment for animal CDS regions with loss downweighting on repetitive DNA elements.

https://github.com/Open-Athena/bolinas-dna/issues/27
"""

from experiments.dna.defaults import (
    CDS_DATASET_V1,
    YOLO_RUN_CONFIG_V1,
    dna_qwen3_1_7b_v1,
    dna_tokenize_rw_v1,
    dna_train,
)
from marin.execution.executor import executor_main

# =============================================================================
# CDS with repeat downweight (0.01)
# =============================================================================

data_tokenized = dna_tokenize_rw_v1("animal-cds-repeat-weight-0.01", CDS_DATASET_V1)

training_step = dna_train(
    name="animal-cds-yolo-repeat-weight-0.01-r01",
    tokenized=data_tokenized,
    model_config=dna_qwen3_1_7b_v1,
    train_config=YOLO_RUN_CONFIG_V1,
    tags=["dna", "animal-cds", "yolo"],
)

# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    executor_main(steps=[training_step])
