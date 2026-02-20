# Copyright 2025 The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Canary ferry: Qwen3 ~30M (hid512) daily pretraining canary.

Trains to 1B tokens on v5p-8 with AdamH. Designed to run daily to catch infra
and pretraining regressions early. See #2873 for the proposal.

Optimal hyperparameters from mega-sweep-bs64-1b-hid512-v3 (trial 26, macro_loss=3.754):
  lr=0.00864, beta1=0.894, adam_lr=0.000502, beta2=0.999, eps=2.32e-07,
  max_grad_norm=0.1, z_loss_weight=1.10e-05
"""

from levanter.layers.rotary import Llama3RotaryEmbeddingsConfig
from levanter.models.qwen import Qwen3Config
from levanter.optim import AdamHConfig

from experiments.defaults import default_train
from experiments.simple_train_config import SimpleTrainConfig
from experiments.tootsie.exp1295_32b import nemotron_mix
from fray.cluster import ResourceConfig
from marin.execution.executor import executor_main

# --- Model: Qwen3 ~30M (hidden_dim=512) ---
model = Qwen3Config(
    max_seq_len=4096,
    hidden_dim=512,
    intermediate_dim=2048,
    num_heads=4,
    num_kv_heads=4,
    num_layers=6,
    rope=Llama3RotaryEmbeddingsConfig(),
)

# --- Training: 1B tokens, bs64, seq_len=4096 ---
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

training_step = default_train(
    name="canary-ferry",
    tokenized=nemotron_mix,
    model_config=model,
    train_config=train_config,
    tags=["canary", "ferry", "qwen3", "adamh", "hid512", "1b"],
    eval_harness_tasks=[],
)

if __name__ == "__main__":
    executor_main(steps=[training_step])
