# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""
Train a 1.4B Llama model on the protein-docs contacts-and-distances-v1-5x dataset.

This trains a model from scratch on a custom protein structure vocabulary (~2840 tokens)
encoding amino acid sequences, inter-residue contacts, and atomic distances.

Prerequisites:
    Push the tokenizer to HuggingFace Hub first:
        python experiments/protein/create_protein_tokenizer.py --push timodonnell/protein-docs-tokenizer

Usage:
    python experiments/protein/train_protein_1b.py
"""

import dataclasses

from levanter.data.text import LmDataConfig, TextLmDatasetFormat
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig
from fray.v2 import ResourceConfig
from marin.execution.executor import executor_main, versioned
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

# -- Tokenizer (must be pushed to HF Hub before running) --

PROTEIN_TOKENIZER = "timodonnell/protein-docs-tokenizer"

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

# -- Dataset (fsspec paths to HF Hub parquets) --

HF_DATASET_BASE = "hf://datasets/timodonnell/protein-docs@main/contacts-and-distances-v1-5x"

# -- Tokenize --

protein_docs_tokenized = default_tokenize(
    name="protein-docs-cd",
    dataset=f"{HF_DATASET_BASE}/train/",
    tokenizer=PROTEIN_TOKENIZER,
    format=TextLmDatasetFormat(text_key="document"),
)

protein_docs_val_tokenized = default_tokenize(
    name="protein-docs-cd-val",
    dataset=f"{HF_DATASET_BASE}/val/",
    tokenizer=PROTEIN_TOKENIZER,
    format=TextLmDatasetFormat(text_key="document"),
    is_validation=True,
)

# Use pack=True to avoid concat-and-split, which would create partial documents.
# Protein docs are nonsensical if you only see the latter half without the header.
train_component = dataclasses.replace(
    step_to_lm_mixture_component(protein_docs_tokenized, include_raw_paths=True),
    pack=True,
)
val_component = dataclasses.replace(
    step_to_lm_mixture_component(protein_docs_val_tokenized, include_raw_paths=True),
    pack=True,
)

protein_docs_data = LmDataConfig(
    components={"protein-docs-cd": train_component, "protein-docs-cd-val": val_component},
    train_weights={"protein-docs-cd": 1.0, "protein-docs-cd-val": 0.0},
    tokenizer=PROTEIN_TOKENIZER,
    cache_dir=None,
    block_cross_document_attention=True,
)

# -- Training config --

train_config = SimpleTrainConfig(
    resources=RESOURCES,
    train_batch_size=128,
    num_train_steps=50_000,
    learning_rate=versioned(3.5e-4),
    weight_decay=0.01,
    warmup=0.1,
    train_seq_len=8192,
    steps_per_eval=500,
    env_vars={
        "WANDB_ENTITY": "timodonnell",
    },
)

# -- Train --

protein_model_1b = default_train(
    name="protein-contacts-1b-3.5e-4",
    tokenized=protein_docs_data,
    model_config=protein_llama_1b,
    train_config=train_config,
    tags=["protein", "contacts-and-distances", "1b", "llama"],
    # Standard NLP evals don't apply to protein structure prediction
    eval_harness_tasks=[],
    # Skip default Paloma validation sets
    use_default_validation=False,
    wandb_group="protein-training",
    wandb_name=None,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            protein_model_1b,
        ]
    )
