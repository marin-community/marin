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
Experiment 53: Alternative datasets based on distance from CDS.

Question: Can we improve performance by using distance-based heuristics to identify
5' and 3' regions relative to CDS (coding sequences), similar to SpeciesLM?

https://github.com/Open-Athena/bolinas-dna/issues/53
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
    "three_prime_utr_baseline": "bolinas-dna/genomes-v4-genome_set-animals-intervals-v12_256_128",
    "upstream_of_cds_512": "bolinas-dna/genomes-v4-genome_set-animals-intervals-v13_256_128",
    "downstream_of_cds_512": "bolinas-dna/genomes-v4-genome_set-animals-intervals-v14_256_128",
    "downstream_of_cds_256": "bolinas-dna/genomes-v4-genome_set-animals-intervals-v15_256_128",
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
        name=f"exp53-{region}-r01",
        tokenized=tokenized,
        model_config=dna_qwen3_0_6b_256_v1,
        train_config=train_config_256,
        tags=["dna", "exp53", "distance_from_cds", "fast"],
    )
    training_steps.append(train_step)

if __name__ == "__main__":
    executor_main(steps=training_steps)
