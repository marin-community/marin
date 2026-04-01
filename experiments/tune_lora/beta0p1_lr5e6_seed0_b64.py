# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executor-native LoRA DPO tuning run: beta=0.1, lr=5e-6, seed=0, batch=64."""

from experiments.tune_lora.common import LoraTuneSpec, run_executor

SPEC = LoraTuneSpec(
    slug="bloom_speceval_v2_marin_instruct_lora_beta0p1_lr5e6_seed0_b64_v5p8",
    learning_rate=5e-6,
    seed=0,
)


if __name__ == "__main__":
    run_executor(SPEC)
