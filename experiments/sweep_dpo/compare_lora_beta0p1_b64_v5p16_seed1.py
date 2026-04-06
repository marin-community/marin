# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Full-DPO comparison sweep: beta=0.1, batch=64, one epoch, v5p-16, seed=1."""

from experiments.sweep_dpo.compare_lora_beta0p1_b64_v5p16_common import run_executor

if __name__ == "__main__":
    run_executor(seed=1)
