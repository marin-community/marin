# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""Full DPO fine-tune on v6e-64 (fallback if v6e-32 OOMs).

Batch 64, pd=1 (64 chips x 1 ex/chip), lr=5e-7, 1 epoch, beta=0.1.
"""
from experiments.posttrain.full_dpo.common import (
    make_full_dpo_step,
    tokenized_eval,
    tokenized_train,
)
from marin.execution.executor import executor_main

training_step = make_full_dpo_step(
    tpu_type="v6e-64",
    per_device=1,
    learning_rate=5e-7,
    num_epochs=1.0,
    train_batch_size=64,
    steps_per_hf_export=500,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_eval, training_step])
