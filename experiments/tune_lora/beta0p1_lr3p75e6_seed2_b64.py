# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Executor-native LoRA DPO tuning run: beta=0.1, lr=3.75e-6, seed=2, batch=64."""

from experiments.tune_lora.common import LoraTuneSpec, run_executor

SPEC = LoraTuneSpec(
    slug="bloom_speceval_v2_marin_instruct_lora_beta0p1_lr3p75e6_seed2_b64_v5p8",
    learning_rate=3.75e-6,
    seed=2,
)


if __name__ == "__main__":
    run_executor(SPEC)
