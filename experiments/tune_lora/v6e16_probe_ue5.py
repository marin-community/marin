# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""v6e-16 LoRA DPO probe — us-east5."""
from experiments.tune_lora.v6e8_probe_multiregion import make_v6e_probe, tokenized_eval, tokenized_train
from marin.execution.executor import executor_main

step = make_v6e_probe(
    tpu_type="v6e-16",
    regions=["us-east5"],
    name_suffix="_v6e16_ue5",
    per_device=4,
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_eval, step])
