# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a 1B Llama protein model on every document type in protein-docs.

Closely mirrors the original ``train_protein_1b.py`` recipe (1B Llama, LR
3.5e-4, batch 128, 50K steps, ``pack=True`` + block-cross-document attention,
us-east5-a TPU pin) but differs in three ways:

* **Full mixture, not just contacts-and-distances.** The dataset
  ``timodonnell/protein-docs`` ships three distinct document types
  (``contacts-and-distances-v1-5x``, ``deterministic-positives-only``,
  ``random-3-bins``); this run trains on all three at equal weight.
* **No loss masking.** Every position contributes to the loss uniformly,
  so all document types are treated the same. (The ``-distance-masked``
  recipes zero loss on every position except ``<d_value>`` inside a
  ``<distance>`` statement; not the case here.)
* **Block-shuffle with Feistel permutation, stated explicitly.** Matches
  ``DEFAULT_LM_DATA_SHUFFLE`` but written out so the choice is visible.

Per-subset val components mean ``eval/protein-docs-cd-val/loss``,
``eval/protein-docs-det-val/loss``, and ``eval/protein-docs-r3b-val/loss``
are all logged separately. Lets us compare this run head-to-head against
single-subset baselines like ``protein-contacts-1b-3.5e-4-distance-masked-7d355e``
(cd-only, masked) at the same val set per type.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a -- \\
        python -m experiments.protein.train_protein_1b_all_docs_unmasked
"""

import dataclasses

from fray import ResourceConfig
from levanter.data.text import BlockShuffleConfig, LmDataConfig, TextLmDatasetFormat
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main, versioned
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

# Architecture matches train_protein_1b.py exactly.
protein_llama_1b = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=16,
)

HF_DATASET_BASE = "hf://datasets/timodonnell/protein-docs@main"

# All-doc-types tokenizer (2849 tokens) — superset of the legacy
# ``timodonnell/protein-docs-tokenizer`` (2840). Old runs at vocab=2840 keep
# using the legacy URL so their checkpoints stay loadable; new training
# (this script) needs the extra tokens for ``deterministic-positives-only``
# and ``random-3-bins`` documents.
PROTEIN_TOKENIZER = "timodonnell/protein-docs-all-doc-types-tokenizer"

# Don't reuse protein_train_common's tokenize steps — those use the legacy
# 2840-vocab tokenizer. Tokenize all three subsets fresh under this script's
# extended-vocab tokenizer so loss computation is consistent across them.
protein_docs_cd_tokenized = default_tokenize(
    name="protein-docs-cd-all-doc-types",
    dataset=f"{HF_DATASET_BASE}/contacts-and-distances-v1-5x/train/",
    tokenizer=PROTEIN_TOKENIZER,
    format=TextLmDatasetFormat(text_key="document"),
)
protein_docs_cd_val_tokenized = default_tokenize(
    name="protein-docs-cd-val-all-doc-types",
    dataset=f"{HF_DATASET_BASE}/contacts-and-distances-v1-5x/val/",
    tokenizer=PROTEIN_TOKENIZER,
    format=TextLmDatasetFormat(text_key="document"),
    is_validation=True,
)
protein_docs_det_tokenized = default_tokenize(
    name="protein-docs-det",
    dataset=f"{HF_DATASET_BASE}/deterministic-positives-only/train/",
    tokenizer=PROTEIN_TOKENIZER,
    format=TextLmDatasetFormat(text_key="document"),
)
protein_docs_det_val_tokenized = default_tokenize(
    name="protein-docs-det-val",
    dataset=f"{HF_DATASET_BASE}/deterministic-positives-only/val/",
    tokenizer=PROTEIN_TOKENIZER,
    format=TextLmDatasetFormat(text_key="document"),
    is_validation=True,
)
protein_docs_r3b_tokenized = default_tokenize(
    name="protein-docs-r3b",
    dataset=f"{HF_DATASET_BASE}/random-3-bins/train/",
    tokenizer=PROTEIN_TOKENIZER,
    format=TextLmDatasetFormat(text_key="document"),
)
protein_docs_r3b_val_tokenized = default_tokenize(
    name="protein-docs-r3b-val",
    dataset=f"{HF_DATASET_BASE}/random-3-bins/val/",
    tokenizer=PROTEIN_TOKENIZER,
    format=TextLmDatasetFormat(text_key="document"),
    is_validation=True,
)


def _component(tok_step):
    """Component with pack=True and no loss_weight_fn (uniform per-token loss).

    pack=True is essential — protein docs are nonsensical without their header,
    so we must avoid concat-and-split fragmenting documents.
    """
    return dataclasses.replace(
        step_to_lm_mixture_component(tok_step, include_raw_paths=True),
        pack=True,
    )


# Three train + three val components. Equal weights mean each document type
# contributes roughly the same number of training tokens; val components are
# weight 0 so they never feed gradients but do produce eval/<name>/loss.
components = {
    "protein-docs-cd": _component(protein_docs_cd_tokenized),
    "protein-docs-cd-val": _component(protein_docs_cd_val_tokenized),
    "protein-docs-det": _component(protein_docs_det_tokenized),
    "protein-docs-det-val": _component(protein_docs_det_val_tokenized),
    "protein-docs-r3b": _component(protein_docs_r3b_tokenized),
    "protein-docs-r3b-val": _component(protein_docs_r3b_val_tokenized),
}
train_weights = {
    "protein-docs-cd": 1.0,
    "protein-docs-cd-val": 0.0,
    "protein-docs-det": 1.0,
    "protein-docs-det-val": 0.0,
    "protein-docs-r3b": 1.0,
    "protein-docs-r3b-val": 0.0,
}

# Hierarchical block-shuffle with a Feistel permutation. Matches
# levanter.data.text.DEFAULT_LM_DATA_SHUFFLE; restated here so the choice is
# explicit (rather than inherited from a default that could drift).
SHUFFLE = BlockShuffleConfig(
    io_block_size=256,
    window_blocks=512,
    perm_type="feistel",
)

protein_docs_data = LmDataConfig(
    components=components,
    train_weights=train_weights,
    tokenizer=PROTEIN_TOKENIZER,
    cache_dir=None,
    block_cross_document_attention=True,
    shuffle=SHUFFLE,
)

# v5p-32 (32 chips, 4× v5p-8) using the same minimal config that
# experiments.ferries.* run with successfully. Earlier we tried adding cpu/ram/
# zone overrides; that broke JAX multi-host bootstrap with
# "multihost_broadcast_sync requires jax distributed client to be initialized"
# during wandb tracker init. Defaults are fine here.
PROTEIN_RESOURCES_V5P_32 = ResourceConfig.with_tpu("v5p-32")

# Scaling vs the reference 1B recipe (v5p-8, batch 128, lr 3.5e-4):
#   * train_batch_size: 128 → 512  (linear scale with chip count)
#   * learning_rate:    3.5e-4 → 7e-4  (square-root scale; conservative —
#     420m_deep just had a divergence event at 3.5e-4 batch 128, so we
#     don't want to push linear-scaled 1.4e-3)
#   * num_train_steps:  unchanged at 200K (4× the original total tokens of
#     ~52B → ~210B; long but matches the user's bumped horizon)
train_config = SimpleTrainConfig(
    resources=PROTEIN_RESOURCES_V5P_32,
    train_batch_size=512,
    num_train_steps=200_000,
    learning_rate=versioned(7e-4),
    weight_decay=0.01,
    warmup=0.1,
    train_seq_len=8192,
    steps_per_eval=500,
    steps_per_export=5000,
    env_vars={"WANDB_ENTITY": "timodonnell"},
)

protein_model_1b_all_docs_unmasked = default_train(
    name="protein-contacts-1b-3.5e-4-all-docs-unmasked",
    tokenized=protein_docs_data,
    model_config=protein_llama_1b,
    train_config=train_config,
    tags=["protein", "all-docs", "1b", "llama", "unmasked"],
    eval_harness_tasks=[],
    use_default_validation=False,
    wandb_group="protein-training",
    wandb_name=None,
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_1b_all_docs_unmasked])
