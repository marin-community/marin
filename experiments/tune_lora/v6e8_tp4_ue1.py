# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0
"""v6e-8 LoRA DPO with TP=4+FSDP=2 — us-east1."""
from levanter.callbacks.profiler import ProfilerConfig
from levanter.trainer import MeshConfig

from experiments.tune_lora.v6e8_probe_multiregion import make_v6e_probe, tokenized_eval, tokenized_train
from marin.execution.executor import executor_main

step = make_v6e_probe(
    regions=["us-east1"],
    name_suffix="_tp4_ue1",
    per_device=-1,
    num_train_steps=20,
    profiler=ProfilerConfig(enabled=True, start_step=5, num_steps=10),
    mesh=MeshConfig(axes={"data": -1, "model": 4}),
)

if __name__ == "__main__":
    executor_main(steps=[tokenized_train, tokenized_eval, step])
