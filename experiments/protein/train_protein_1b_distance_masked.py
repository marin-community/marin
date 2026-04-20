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

import jax
import jax.numpy as jnp
from levanter.data.text import LmDataConfig, TextLmDatasetFormat
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_tokenize, default_train
from experiments.protein.create_protein_tokenizer import create_protein_tokenizer
from experiments.simple_train_config import SimpleTrainConfig
from fray.v2 import ResourceConfig
from marin.execution.executor import executor_main, versioned
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

# -- Tokenizer (must be pushed to HF Hub before running) --

PROTEIN_TOKENIZER = "timodonnell/protein-docs-tokenizer"

# Token id for the <distance> marker. Looked up locally via the canonical vocab
# order so this matches what the pushed HF tokenizer produces.
DISTANCE_TOKEN_ID: int = create_protein_tokenizer().convert_tokens_to_ids("<distance>")

# A distance statement is 6 tokens: <distance> <p_i> <p_j> <atom_i> <atom_j> <d_value>.
# For next-token prediction, loss_weight[i] weights the loss on predicting tokens[i+1].
# We zero loss_weight at positions s..s+3 (where s is the <distance> index) so that only
# the prediction of <d_value> (at loss_weight[s+4]) contributes inside a distance statement.
_NUM_NON_BIN_STATEMENT_POSITIONS = 4


def _distance_bin_only_loss_weight(tokens: jax.Array) -> jax.Array:
    """Loss weight that keeps distance-bin predictions only inside distance statements."""
    is_distance = tokens == DISTANCE_TOKEN_ID
    mask_zero = is_distance
    for shift in range(1, _NUM_NON_BIN_STATEMENT_POSITIONS):
        mask_zero = mask_zero | jnp.roll(is_distance, shift)
    return jnp.where(mask_zero, 0.0, 1.0).astype(jnp.float32)


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
)

# -- Dataset (fsspec paths to HF Hub parquets) --

HF_DATASET_BASE = "hf://datasets/timodonnell/protein-docs@main/contacts-and-distances-v1-5x"

# -- Tokenize (shared cache with train_protein_1b.py since inputs are identical) --

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
# Apply the distance-bin-only loss mask to both splits so eval metrics match
# the training objective.
train_component = dataclasses.replace(
    step_to_lm_mixture_component(protein_docs_tokenized, include_raw_paths=True),
    pack=True,
    loss_weight_fn=_distance_bin_only_loss_weight,
)
val_component = dataclasses.replace(
    step_to_lm_mixture_component(protein_docs_val_tokenized, include_raw_paths=True),
    pack=True,
    loss_weight_fn=_distance_bin_only_loss_weight,
)

protein_docs_data = LmDataConfig(
    components={"protein-docs-cd": train_component, "protein-docs-cd-val": val_component},
    train_weights={"protein-docs-cd": 1.0, "protein-docs-cd-val": 0.0},
    tokenizer=PROTEIN_TOKENIZER,
    cache_dir=None,
    block_cross_document_attention=True,
)

# -- Training config (identical to the unmasked 1b run) --

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

protein_model_1b_distance_masked = default_train(
    name="protein-contacts-1b-3.5e-4-distance-masked",
    tokenized=protein_docs_data,
    model_config=protein_llama_1b,
    train_config=train_config,
    tags=["protein", "contacts-and-distances", "1b", "llama", "distance-masked"],
    eval_harness_tasks=[],
    use_default_validation=False,
    wandb_group="protein-training",
    wandb_name=None,
)

if __name__ == "__main__":
    executor_main(
        steps=[
            protein_model_1b_distance_masked,
        ]
    )
