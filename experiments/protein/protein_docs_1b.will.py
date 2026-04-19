# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protein docs: Llama ~1.4B pretraining on tokenized protein contact docs (timodonnell/protein-docs).

Train a 1.4B Llama model on the protein-docs contacts-and-distances-v1-5x dataset.

This trains a model from scratch on a custom protein structure vocabulary (~2840 tokens)
encoding amino acid sequences, inter-residue contacts, and atomic distances.

Prerequisites:
    Push the tokenizer to HuggingFace Hub first:
        python experiments/protein/create_protein_tokenizer.py --push timodonnell/protein-docs-tokenizer

Run with Iris::

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --extra marin:tpu --tpu v5p-8 -- \
        python experiments/tatt/protein_docs_1b.py
"""

import dataclasses

from levanter.data.text import LmDataConfig, TextLmDatasetFormat
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_download, default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

TOKENIZER = "timodonnell/protein-docs-tokenizer"

model = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=16,
    # Vocab is small (2840 tokens) so the embedding table is tiny.
    # Levanter rounds vocab_size for TPU partitioning at training time.
)

BATCH_SIZE = 128
SEQ_LEN = 8192
NUM_STEPS = 50_000

train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_STEPS,
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup=0.1,
    train_seq_len=SEQ_LEN,
    steps_per_eval=500,
    env_vars={
        "WANDB_ENTITY": "timodonnell",
    },
)

protein_docs_download = default_download(
    name="raw/protein-docs",
    hf_dataset_id="timodonnell/protein-docs",
    revision="05b4797b9e4d7c6108e36f379673d1f5af1abd3c",
)

protein_docs_tokenized = default_tokenize(
    name="protein-docs-cd",
    dataset=protein_docs_download / "contacts-and-distances-v1-5x/train",
    tokenizer=TOKENIZER,
    format=TextLmDatasetFormat(text_key="document"),
)

protein_docs_val_tokenized = default_tokenize(
    name="protein-docs-cd-val",
    dataset=protein_docs_download / "contacts-and-distances-v1-5x/val",
    tokenizer=TOKENIZER,
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
    tokenizer=TOKENIZER,
    cache_dir=None,
    shuffle=True,
    permutation_type="feistel",
    block_cross_document_attention=True,
)

training_step = default_train(
    name="protein-docs-contacts-1b",
    tokenized=protein_docs_data,
    model_config=model,
    train_config=train_config,
    tags=["protein-docs", "contacts-and-distances", "1b", "llama"],
    use_default_validation=False,
    eval_harness_tasks=[],
    wandb_group="protein-training",
    wandb_name="protein-contacts-1b",
)

if __name__ == "__main__":
    executor_main(steps=[training_step])
