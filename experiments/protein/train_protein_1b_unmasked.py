# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train the 1B protein model with the same recipe as the finished 7d355e run,
but without the distance-bin-only loss mask.

Identical to ``continue_train_protein_1b_distance_masked.py`` (architecture,
learning rate, schedule, batch, ``pack=True``, ``block_cross_document_attention``,
us-east5-a TPU pin, ``steps_per_eval=500``, ``steps_per_export=5000``) — the
only change is that no ``loss_weight_fn`` is applied to the train+val
components, so every token position contributes to the loss instead of just
the ``<d_value>`` bin slot inside ``<distance>`` statements.

Compares head-to-head against ``protein-contacts-1b-3.5e-4-distance-masked-7d355e``
on downstream evals (distogram MAE, contact F1, etc.) to test whether the
loss mask was actually load-bearing.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a -- \\
        python -m experiments.protein.train_protein_1b_unmasked
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

protein_llama_1b = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=16,
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

protein_model_1b_unmasked = default_train(
    name="protein-contacts-1b-3.5e-4-unmasked",
    tokenized=protein_docs_data,
    model_config=protein_llama_1b,
    train_config=train_config,
    tags=["protein", "contacts-and-distances", "1b", "llama", "unmasked"],
    eval_harness_tasks=[],
    use_default_validation=False,
    wandb_group="protein-training",
    wandb_name=None,
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_1b_unmasked])
