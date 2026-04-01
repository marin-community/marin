# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executor-native LoRA DPO tuning run: beta=0.1, lr=1e-5, seed=0, batch=64."""

from experiments.tune_lora.common import LoraTuneSpec, run_executor

SPEC = LoraTuneSpec(
    slug="bloom_speceval_v2_marin_instruct_lora_beta0p1_lr1e5_seed0_b64_v5p8",
    learning_rate=1e-5,
    seed=0,
)


if __name__ == "__main__":
    run_executor(SPEC)
