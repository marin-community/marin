# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protein docs: Qwen3 ~600M pretraining on tokenized protein contact docs (timodonnell/protein-docs).

Trains on all 22.3B tokens on v5p-8 with AdamH. LR scaled from the 30M config
using sqrt(batch_size) / sqrt(token_count).

Run with Iris::

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --extra marin:tpu --tpu v5p-8 -- \
        python experiments/tatt/protein_docs_600m.py
"""

import dataclasses

from levanter.data.text import LmDataConfig, TextLmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamHConfig

from experiments.defaults import default_download, default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_data_config, step_to_lm_mixture_component

TOKENIZER = "timodonnell/protein-docs-tokenizer"

model = Qwen3Config(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=16,
    num_kv_heads=16,
    num_layers=21,
    rope=Llama3RotaryEmbeddingsConfig(),
)

BATCH_SIZE = 32
SEQ_LEN = 8192
TARGET_TOKENS = 22_300_000_000
NUM_STEPS = TARGET_TOKENS // (BATCH_SIZE * SEQ_LEN)

# LR scaled via CompletedAdamHHeuristic(effective_batch=64, tokens=22.3e9)
# effective_batch=64 because batch_size=32 * seq_len=8192 / ref_seq_len=4096
LR = 0.00327

train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_STEPS,
    learning_rate=LR,
    train_seq_len=SEQ_LEN,
    z_loss_weight=1.0e-07,
    optimizer_config=AdamHConfig(
        learning_rate=LR,
        adam_lr=0.000220,
        min_lr_ratio=0.0,
        warmup=0.1,
        decay=0.9,
        lr_schedule="linear",
        beta1=0.9,
        beta2=0.9999,
        epsilon=5.53e-08,
        max_grad_norm=0.1,
        nesterov=False,
    ),
    steps_per_eval=500,
)

protein_docs_download = default_download(
    name="raw/protein-docs",
    hf_dataset_id="timodonnell/protein-docs",
    revision="cdd2e2b4af3c52835b7b5d9fa2819c51abe55b91",
)

protein_docs_tokenized = default_tokenize(
    name="protein-docs",
    dataset=protein_docs_download / "contacts-and-distances-v1-5x/train",
    tokenizer=TOKENIZER,
    format=TextLmDatasetFormat(text_key="document"),
)

protein_docs_val_tokenized = default_tokenize(
    name="protein-docs-val",
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
    components={"protein-docs": train_component, "protein-docs-val": val_component},
    train_weights={"protein-docs": 1.0, "protein-docs-val": 0.0},
    tokenizer=TOKENIZER,
    cache_dir=None,
    shuffle=True,
    permutation_type="feistel",
    block_cross_document_attention=True,
)

training_step = default_train(
    name="protein-docs-1b-v4",
    tokenized=protein_docs_data,
    model_config=model,
    train_config=train_config,
    tags=["protein-docs", "1b", "qwen3", "adamh"],
    use_default_validation=False,
    eval_harness_tasks=[],
)

if __name__ == "__main__":
    executor_main(steps=[training_step])
