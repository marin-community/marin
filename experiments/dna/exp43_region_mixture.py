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
Experiment 43: Mixing 5 different genomic regions.

Question: How do models perform when trained on different genomic regions
and their mixtures?

This file implements Part 1: Individual training runs on 4 regions
(CDS, 5' UTR, 3' UTR, ncRNA) with 256 context size.

TODO: Part 2 - Add mixture training runs combining regions with different weights.

https://github.com/Open-Athena/bolinas-dna/issues/43
"""

import dataclasses

from experiments.dna.defaults import (
    FAST_RUN_CONFIG_V1,
    dna_qwen3_0_6b_256_v1,
    dna_tokenize_rw_v1,
    dna_train,
)
from marin.execution.executor import executor_main

DATASETS = {
    "cds": "bolinas-dna/genomes-v4-genome_set-animals-intervals-v5_256_128",
    "five_prime_utr": "bolinas-dna/genomes-v4-genome_set-animals-intervals-v6_256_128",
    "three_prime_utr": "bolinas-dna/genomes-v4-genome_set-animals-intervals-v7_256_128",
    "ncrna": "bolinas-dna/genomes-v4-genome_set-animals-intervals-v8_256_128",
}


def dataset_name(dataset: str) -> str:
    """Extract dataset name from HuggingFace path (org/name -> name)."""
    return dataset.split("/")[-1]


# Double batch size for 256 context to match tokens/batch with 512 context
train_config_256 = dataclasses.replace(FAST_RUN_CONFIG_V1, train_batch_size=FAST_RUN_CONFIG_V1.train_batch_size * 2)

training_steps = []
for region, dataset in DATASETS.items():
    name = dataset_name(dataset)

    tokenized = dna_tokenize_rw_v1(
        name=f"{name}-rw01",
        dataset=dataset,
    )

    train_step = dna_train(
        name=f"exp43-{region}-r01",
        tokenized=tokenized,
        model_config=dna_qwen3_0_6b_256_v1,
        train_config=train_config_256,
        tags=["dna", "exp43", "regions", "fast"],
    )
    training_steps.append(train_step)

if __name__ == "__main__":
    executor_main(steps=training_steps)
