# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the 100M protein model with the same recipe as
``train_protein_100m_distance_masked.py``, but without the distance-bin-only
loss mask.

Identical architecture, learning rate (3.5e-4), schedule (50K steps, batch
128), ``pack=True``, ``block_cross_document_attention``, us-east5-a TPU pin,
``steps_per_eval=500``, ``steps_per_export=5000``. The only change is no
``loss_weight_fn`` on the train+val components, so every position contributes
to the loss instead of just the ``<d_value>`` bin slot inside ``<distance>``
statements.

Pairs with ``train_protein_1b_unmasked.py`` to test the loss-mask ablation
at two different scales — for the perplexity↔accuracy study, this gives a
second "unmasked" data point to compare against the masked size sweep.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a -- \\
        python -m experiments.protein.train_protein_100m_unmasked
"""

import dataclasses

from levanter.data.text import LmDataConfig
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_train
from experiments.protein.protein_train_common import (
    PROTEIN_RESOURCES_USE5,
    PROTEIN_TOKENIZER,
    protein_docs_tokenized,
    protein_docs_val_tokenized,
)
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main, versioned
from marin.processing.tokenize.data_configs import step_to_lm_mixture_component

protein_llama_100m = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=768,
    intermediate_dim=3072,
    num_heads=12,
    num_kv_heads=4,
    num_layers=12,
)

# Same components as `protein_train_common.distance_masked_components()` but
# WITHOUT the loss_weight_fn: every position contributes to the loss, not just
# the <d_value> bin slot inside <distance> statements.
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

train_config = SimpleTrainConfig(
    resources=PROTEIN_RESOURCES_USE5,
    train_batch_size=128,
    num_train_steps=50_000,
    learning_rate=versioned(3.5e-4),
    weight_decay=0.01,
    warmup=0.1,
    train_seq_len=8192,
    steps_per_eval=500,
    steps_per_export=5000,
    env_vars={"WANDB_ENTITY": "timodonnell"},
)

protein_model_100m_unmasked = default_train(
    name="protein-contacts-100m-3.5e-4-unmasked",
    tokenized=protein_docs_data,
    model_config=protein_llama_100m,
    train_config=train_config,
    tags=["protein", "contacts-and-distances", "100m", "llama", "unmasked"],
    eval_harness_tasks=[],
    use_default_validation=False,
    wandb_group="protein-training",
    wandb_name=None,
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_100m_unmasked])
