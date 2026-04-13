# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""v6e-8 LoRA DPO with xprof profiling — 20 steps, us-east5."""
from levanter.callbacks.profiler import ProfilerConfig

from experiments.tune_lora.v6e8_probe_multiregion import make_v6e_probe, tokenized_eval, tokenized_train
from marin.execution.executor import executor_main

step = make_v6e_probe(
    tpu_type="v6e-8",
    regions=["us-east5"],
    name_suffix="_profile_ue5",
    per_device=4,
    num_train_steps=20,
    profiler=ProfilerConfig(enabled=True, start_step=5, num_steps=10),
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_eval, step])
