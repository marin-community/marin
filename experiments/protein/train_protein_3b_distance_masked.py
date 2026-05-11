# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a ~3.03B-param Llama protein model with the distance-masked loss.

Largest member of the protein-docs distance-masked size sweep. Shape matches
Pythia-2.8B (h=2560, l=32, dff=10240, heads=40).

Recipe matches ``continue_train_protein_1b_distance_masked.py``: 50K steps,
batch 128, LR 3.5e-4, distance-masked loss only.

Caveat: at 3B params the 50K-step horizon is undertrained relative to
Chinchilla; bump ``num_train_steps`` if you want a compute-balanced point.
A v5p-8 should still fit this model but at lower MFU than the smaller runs;
v5p-16 would help throughput.

The 3B model spikes host-RAM during checkpoint serialization beyond the 128 GB
budget shared by the smaller runs, so train_lm gets OOM-killed at every temp
checkpoint save (~250 steps). Iris auto-recovers, but progress slows ~10x.
This script overrides ``ram="256g"`` for 3b only and pins
``override_output_path`` to the existing checkpoint dir so the bumped run
resumes from the same checkpoint instead of starting over (which a resources
change would otherwise force via the executor hash).

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a -- \\
        python -m experiments.protein.train_protein_3b_distance_masked
"""

from fray import ResourceConfig
from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main

from experiments.protein.protein_train_common import build_distance_masked_train_step

protein_llama_3b = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=2560,
    intermediate_dim=10240,
    num_heads=40,
    num_kv_heads=8,
    num_layers=32,
)

PROTEIN_RESOURCES_USE5_3B = ResourceConfig.with_tpu(
    "v5p-8",
    slice_count=1,
    cpu=32,
    ram="256g",
    disk="50g",
    zone="us-east5-a",
)

protein_model_3b_distance_masked = build_distance_masked_train_step(
    name="protein-contacts-3b-distance-masked",
    model_config=protein_llama_3b,
    learning_rate=3.5e-4,
    extra_tags=("3b",),
    resources=PROTEIN_RESOURCES_USE5_3B,
    override_output_path="gs://marin-us-east5/checkpoints/protein-contacts-3b-distance-masked-ef3aa5",
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_3b_distance_masked])
