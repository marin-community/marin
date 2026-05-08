# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Train the 1.4B Llama protein model with a loss mask that focuses training on
distance-bin predictions only.

Same data, tokenizer, model, and hyperparameters as
``experiments/protein/train_protein_1b.py`` — the only change is the loss
weight. A distance statement has 6 tokens:

    <distance> <p_i> <p_j> <atom_i> <atom_j> <d_value>

For next-token prediction, ``loss_weight[i]`` weights the loss for predicting
``tokens[i+1]``. We zero the weights at positions ``s, s+1, s+2, s+3`` (where
``s`` is the index of a ``<distance>`` token), so that only the prediction of
``<d_value>`` (the distance bin) contributes to the loss inside a distance
statement. All tokens outside distance statements (amino acids, contacts,
headers, pLDDT, sequence separators) continue to train normally.

Usage:
    python experiments/protein/train_protein_1b_distance_masked.py
"""

import dataclasses

from levanter.data.text import LmDataConfig
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_train
from experiments.protein.protein_distogram_eval import distogram_eval_benchmark
from experiments.protein.protein_train_common import (
    PROTEIN_TOKENIZER,
    distance_masked_components,
)
from experiments.simple_train_config import SimpleTrainConfig
from fray.v2 import ResourceConfig
from marin.execution.executor import executor_main, versioned

# -- Resources --

RESOURCES = ResourceConfig.with_tpu(
    "v5p-8",
    slice_count=1,
    cpu=32,
    ram="128g",
    disk="50g",
)

# -- Model (1.4B, matching reference experiment architecture) --

protein_llama_1b = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=16,
    # Vocab is small (2840 tokens) so the embedding table is tiny.
    # Levanter rounds vocab_size for TPU partitioning at training time.
)

# -- Protein distogram benchmark (loss-based eval on specific PDB targets) --
# See experiments/protein/protein_distogram_eval.py for how to add targets.
# The eval components use `PrebuiltLmDatasetFormat` with a pre-computed
# loss_weights column in the parquet, so the distance-mask loss_weight_fn
# (applied to the train + val components) does NOT apply to them — they see
# only the final GT distance-bin position as their training target, which is
# what we want.
PROTEIN_MPNN_REDESIGNS_SOURCE = "gs://marin-us-east5/protein-structure/protein-mpnn-redesigns/v1/redesigns.jsonl"
# 100 pairs per (target, N) keeps a full eval pass fast. The default of 1000 made
# eval take ~78 min on 132 datasets, longer than most training intervals. Drop
# to 100 and the full eval runs in ~8 min — acceptable overhead at
# steps_per_eval=1000 below.
DISTOGRAM_EVAL_PAIRS_PER_TARGET = 100
distogram_eval = distogram_eval_benchmark(
    PROTEIN_TOKENIZER,
    redesigns_source=PROTEIN_MPNN_REDESIGNS_SOURCE,
    n_pairs=DISTOGRAM_EVAL_PAIRS_PER_TARGET,
)

protein_docs_data = LmDataConfig(
    components={
        **distance_masked_components(),
        **distogram_eval.components,
    },
    # Only the main training corpus has non-zero train weight. The distogram
    # components are validation-only.
    train_weights={"protein-docs-cd": 1.0, "protein-docs-cd-val": 0.0},
    tokenizer=PROTEIN_TOKENIZER,
    cache_dir=None,
    block_cross_document_attention=True,
)

# -- Training config --
#
# LR bumped 3x from the prior 3.5e-4 unmasked run to 1.05e-3. The distance-masked
# loss reduces the effective per-batch gradient signal (only ~1/6 of distance-statement
# tokens contribute), so a bigger learning rate is warranted. Batch, warmup, and
# step count kept the same as train_protein_1b.py for comparability.

train_config = SimpleTrainConfig(
    resources=RESOURCES,
    train_batch_size=128,
    num_train_steps=50_000,
    learning_rate=versioned(1.05e-3),
    weight_decay=0.01,
    warmup=0.1,
    train_seq_len=8192,
    steps_per_eval=1000,
    env_vars={
        "WANDB_ENTITY": "timodonnell",
    },
)

# -- Train --

protein_model_1b_distance_masked = default_train(
    name="protein-contacts-1b-1.05e-3-distance-masked",
    tokenized=protein_docs_data,
    model_config=protein_llama_1b,
    train_config=train_config,
    tags=["protein", "contacts-and-distances", "1b", "llama", "distance-masked"],
    eval_harness_tasks=[],
    use_default_validation=False,
    wandb_group="protein-training",
    wandb_name=None,
)

# Allow levanter to auto-build tree-caches for the protein-distogram eval
# components. Each (target, N) parquet → small levanter cache (seconds). We
# parameterize each component's `cache_dir` off its build-step output path so
# the cache lives inside the step's (versioned) output and is reused across
# training runs. Marin's default is False to fail fast on missing caches for
# big text tokenizations; for our pre-tokenized eval parquets the build is
# trivial and auto-build is the right default.
protein_model_1b_distance_masked = dataclasses.replace(
    protein_model_1b_distance_masked,
    config=dataclasses.replace(protein_model_1b_distance_masked.config, auto_build_caches=True),
)

if __name__ == "__main__":
    executor_main(
        steps=[
            *distogram_eval.build_steps,
            protein_model_1b_distance_masked,
        ]
    )
