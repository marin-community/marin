# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Tomol25: Qwen3 ~30M pretraining on tokenized molecular data (WillHeld/Tomol25).

Trains to 1B tokens on v5p-8 with AdamH.
"""

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamHConfig

from experiments.defaults import default_tokenize, default_train
from experiments.simple_train_config import SimpleTrainConfig
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main
from marin.processing.tokenize.data_configs import lm_data_config

TOKENIZER = "WillHeld/marin-tomol"

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

tomol_tokenized = default_tokenize(
    name="tomol25",
    dataset="WillHeld/Tomol25",
    tokenizer=TOKENIZER,
)

tomol_data = lm_data_config(tomol_tokenized)

training_step = default_train(
    name="tomol25-30m",
    tokenized=tomol_data,
    model_config=model,
    train_config=train_config,
    tags=["tomol", "30m", "qwen3", "adamh"],
    use_default_validation=False,
    eval_harness_tasks=[],
)

if __name__ == "__main__":
    executor_main(steps=[training_step])
