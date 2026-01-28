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
Experiment 37: Context size comparison (256 context).

Question: How does smaller context size (256 vs 512) affect model performance?
Batch size is doubled compared to 512 context to match tokens per batch.
Compare results with exp41 which uses 512 context.

https://github.com/Open-Athena/bolinas-dna/issues/37
"""

import dataclasses

from experiments.dna.defaults import (
    FAST_RUN_CONFIG_V1,
    PROMOTERS_MRNA_256_DATASET_V1,
    dna_qwen3_0_6b_256_v1,
    dna_tokenize_rw_v1,
    dna_train,
)
from marin.execution.executor import executor_main


def dataset_name(dataset: str) -> str:
    """Extract dataset name from HuggingFace path (org/name -> name)."""
    return dataset.split("/")[-1]


# Double batch size for 256 context to match tokens/batch with 512 context
train_config_256 = dataclasses.replace(FAST_RUN_CONFIG_V1, train_batch_size=FAST_RUN_CONFIG_V1.train_batch_size * 2)

name = dataset_name(PROMOTERS_MRNA_256_DATASET_V1)

tokenized = dna_tokenize_rw_v1(
    name=f"{name}-rw01",
    dataset=PROMOTERS_MRNA_256_DATASET_V1,
)

train_step = dna_train(
    name=f"exp37-{name}-r02",
    tokenized=tokenized,
    model_config=dna_qwen3_0_6b_256_v1,
    train_config=train_config_256,
    tags=["dna", "exp37", "context-size", "fast"],
)

if __name__ == "__main__":
    executor_main(steps=[train_step])
