# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

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
Experiment 64: k-mer tokenization for promoter sequences (mammals dataset).

Question: How does k-mer tokenization (4-mer, 8-mer) compare to character-level
tokenization for DNA language models trained on promoter sequences?

Baseline: exp55's mammals 256-context run (character-level tokenizer, Qwen3 0.6B,
batch size 4096, 10K steps).

With non-overlapping k-mers, 256 bp sequences become:
  - 4-mer: 256 / 4 = 64 tokens
  - 8-mer: 256 / 8 = 32 tokens

We use TextLmDatasetFormat (not DNALmDatasetFormat) because k-mer tokenizers
handle the sequence directly, without character-level repeat masking.

https://github.com/Open-Athena/bolinas-dna/issues/64
"""

import dataclasses

from levanter.data.text import TextLmDatasetFormat

from experiments.defaults import default_tokenize
from experiments.dna.defaults import (
    DNA_WINDOW_SIZE_BYTES_V1,
    FAST_RUN_CONFIG_V1,
    dna_train,
)
from experiments.qwen3 import qwen3_0_6b_hd128
from marin.execution.executor import executor_main

KMER_TOKENIZERS = {
    "4mer": {"hf_id": "bolinas-dna/tokenizer-4-mer", "max_seq_len": 64},
    "8mer": {"hf_id": "bolinas-dna/tokenizer-8-mer", "max_seq_len": 32},
}

PROMOTERS_MRNA_256_MAMMALS = "bolinas-dna/genomes-v4-genome_set-mammals-intervals-v1_256_128"

# Double batch size for 256 context to match base pairs per batch with 512 context (same as exp55)
train_config = dataclasses.replace(FAST_RUN_CONFIG_V1, train_batch_size=FAST_RUN_CONFIG_V1.train_batch_size * 2)

dataset_name = PROMOTERS_MRNA_256_MAMMALS.split("/")[-1]

train_steps = []
for kmer_label, kmer_cfg in KMER_TOKENIZERS.items():
    tokenized = default_tokenize(
        name=f"{dataset_name}-{kmer_label}",
        dataset=PROMOTERS_MRNA_256_MAMMALS,
        tokenizer=kmer_cfg["hf_id"],
        format=TextLmDatasetFormat(text_key="seq"),
        window_size_bytes=DNA_WINDOW_SIZE_BYTES_V1,
    )

    model_config = dataclasses.replace(qwen3_0_6b_hd128, max_seq_len=kmer_cfg["max_seq_len"])

    train_step = dna_train(
        name=f"exp64-{dataset_name}-{kmer_label}-r01",
        tokenized=tokenized,
        model_config=model_config,
        train_config=train_config,
        tags=["dna", "exp64", "kmer", kmer_label, "fast"],
    )
    train_steps.append(train_step)

if __name__ == "__main__":
    executor_main(steps=train_steps)
