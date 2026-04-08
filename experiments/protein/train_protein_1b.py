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

import os

from fray.v2 import ResourceConfig
from levanter.data.text import TextLmDatasetFormat
from levanter.models.llama import LlamaConfig
from marin.execution.executor import ExecutorStep, executor_main, this_output_path, versioned
from marin.execution.remote import remote
from marin.processing.tokenize import TokenizeConfig, tokenize

from experiments.defaults import default_train
from experiments.simple_train_config import SimpleTrainConfig

# -- Tokenizer (must be pushed to HF Hub before running) --

PROTEIN_TOKENIZER = "timodonnell/protein-docs-tokenizer"

# -- Dataset (fsspec paths to HF Hub parquets) --

HF_DATASET_BASE = "hf://datasets/timodonnell/protein-docs@main/contacts-and-distances-v1-5x"
TRAIN_PATH = f"{HF_DATASET_BASE}/train/"
VAL_PATH = f"{HF_DATASET_BASE}/val/"

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

# -- Tokenize --

protein_tokenized = ExecutorStep(
    name=os.path.join("tokenized", "protein-docs", "contacts-and-distances-v1-5x"),
    description=f"Tokenize protein-docs contacts-and-distances-v1-5x using {PROTEIN_TOKENIZER}.",
    fn=remote(
        tokenize,
        resources=ResourceConfig.with_cpu(cpu=4, ram="16g", disk="10g"),
        pip_dependency_groups=["cpu"],
        env_vars={
            "TRANSFORMERS_NO_TORCH": "1",
            "TRANSFORMERS_NO_TORCHVISION": "1",
            "USE_TORCH": "0",
            "TORCH_DISABLE_GLOBAL_DEPS": "1",
        },
    ),
    config=TokenizeConfig(
        train_paths=[TRAIN_PATH],
        validation_paths=[VAL_PATH],
        cache_path=this_output_path(),
        tokenizer=versioned(PROTEIN_TOKENIZER),
        format=TextLmDatasetFormat(text_key="document"),
    ),
)

# -- Training config --

train_config = SimpleTrainConfig(
    resources=RESOURCES,
    train_batch_size=versioned(128),
    num_train_steps=versioned(50_000),
    learning_rate=2e-4,
    weight_decay=0.01,
    warmup=0.1,
    train_seq_len=8192,
    env_vars={
        "WANDB_ENTITY": "timodonnell",
    },
)

# -- Train --

protein_model_1b = default_train(
    name="protein-contacts-1b",
    tokenized=protein_tokenized,
    model_config=versioned(protein_llama_1b),
    train_config=train_config,
    tags=["protein", "contacts-and-distances", "1b", "llama"],
    # Standard NLP evals don't apply to protein structure prediction
    eval_harness_tasks=[],
    # Skip default Paloma validation sets
    use_default_validation=False,
    wandb_group="protein-training",
    wandb_name="protein-contacts-1b",
)

if __name__ == "__main__":
    executor_main(
        steps=[
            protein_model_1b,
        ]
    )
