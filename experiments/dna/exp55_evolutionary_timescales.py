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
Experiment 55: Promoters from different evolutionary timescales.

Question: What are the tradeoffs between data quantity and evolutionary relevance
when training on promoter sequences from different timescales?

https://github.com/Open-Athena/bolinas-dna/issues/55
"""

import dataclasses

from experiments.dna.defaults import (
    FAST_RUN_CONFIG_V1,
    dna_qwen3_0_6b_256_v1,
    dna_tokenize_rw_v1,
    dna_train,
)
from marin.execution.executor import executor_main

TIMESCALES = ["animals", "vertebrates", "mammals", "primates", "humans"]

DATASETS = {timescale: f"bolinas-dna/genomes-v4-genome_set-{timescale}-intervals-v1_256_128" for timescale in TIMESCALES}


def dataset_name(dataset: str) -> str:
    """Extract dataset name from HuggingFace path (org/name -> name)."""
    return dataset.split("/")[-1]


# Double batch size for 256 context to match tokens/batch with 512 context
# Increase steps to 17K to cover at least 1 epoch of largest dataset (68M samples)
train_config = dataclasses.replace(
    FAST_RUN_CONFIG_V1,
    train_batch_size=FAST_RUN_CONFIG_V1.train_batch_size * 2,
    num_train_steps=17_000,
)

training_steps = []
for timescale, dataset in DATASETS.items():
    name = dataset_name(dataset)

    tokenized = dna_tokenize_rw_v1(
        name=f"{name}-rw01",
        dataset=dataset,
    )

    train_step = dna_train(
        name=f"exp55-{timescale}-r01",
        tokenized=tokenized,
        model_config=dna_qwen3_0_6b_256_v1,
        train_config=train_config,
        tags=["dna", "exp55", "evolutionary_timescales", "fast"],
    )
    training_steps.append(train_step)

if __name__ == "__main__":
    executor_main(steps=training_steps)
