# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Protein docs: Qwen3 ~30M pretraining on tokenized protein contact docs (timodonnell/protein-docs).

Trains to 1B tokens on v5p-8 with AdamH.

Run with Iris::

    uv run iris --config lib/iris/examples/marin.yaml job run \
        --extra marin:tpu --tpu v5p-8 -- \
        python experiments/tatt/protein_docs_30m.py
"""

from levanter.data.text import TextLmDatasetFormat
from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamHConfig

from experiments.defaults import default_download, default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_data_config

TOKENIZER = "WillHeld/contactdoc-tokenizer"

model = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=512,
    intermediate_dim=2048,
    num_heads=4,
    num_kv_heads=4,
    num_layers=6,
    rope=Llama3RotaryEmbeddingsConfig(),
)

BATCH_SIZE = 64
SEQ_LEN = 4096
TARGET_TOKENS = 1_000_000_000
NUM_STEPS = TARGET_TOKENS // (BATCH_SIZE * SEQ_LEN)

train_config = SimpleTrainConfig(
    resources=ResourceConfig.with_tpu("v5p-8"),
    train_batch_size=BATCH_SIZE,
    num_train_steps=NUM_STEPS,
    learning_rate=0.00864,
    train_seq_len=SEQ_LEN,
    z_loss_weight=1.10e-05,
    optimizer_config=AdamHConfig(
        learning_rate=0.00864,
        adam_lr=0.000502,
        min_lr_ratio=0.0,
        warmup=0.1,
        decay=0.2,
        lr_schedule="linear",
        beta1=0.894,
        beta2=0.999,
        epsilon=2.32e-07,
        max_grad_norm=0.1,
        nesterov=False,
    ),
    steps_per_eval=500,
)

protein_docs_download = default_download(
    name="raw/protein-docs",
    hf_dataset_id="timodonnell/protein-docs",
    revision="b3719628abb8bf3d7f02d8283cf37420c5146a4d",
)

protein_docs_tokenized = default_tokenize(
    name="protein-docs",
    dataset=protein_docs_download / "deterministic-positives-only/train",
    tokenizer=TOKENIZER,
    format=TextLmDatasetFormat(text_key="document"),
)

protein_docs_val_tokenized = default_tokenize(
    name="protein-docs-val",
    dataset=protein_docs_download / "deterministic-positives-only/val",
    tokenizer=TOKENIZER,
    format=TextLmDatasetFormat(text_key="document"),
    is_validation=True,
)

protein_docs_data = lm_data_config(
    protein_docs_tokenized,
    validation_sets={"protein-docs-val": protein_docs_val_tokenized},
)

training_step = default_train(
    name="protein-docs-30m",
    tokenized=protein_docs_data,
    model_config=model,
    train_config=train_config,
    tags=["protein-docs", "30m", "qwen3", "adamh"],
    use_default_validation=False,
    eval_harness_tasks=[],
)

if __name__ == "__main__":
    executor_main(steps=[training_step])
