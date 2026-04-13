# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""v6e-8 LoRA DPO probe — us-east5, per_device=4 (grad accum)."""
from experiments.tune_lora.v6e8_probe_multiregion import make_v6e8_probe, tokenized_eval, tokenized_train
from marin.execution.executor import executor_main

step = make_v6e8_probe(regions=["us-east5"], name_suffix="_ga_ue5")

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_eval, step])
