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

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a -- \\
        python -m experiments.protein.train_protein_3b_distance_masked
"""

from levanter.models.llama import LlamaConfig

from experiments.protein.protein_train_common import build_distance_masked_train_step
from marin.execution.executor import executor_main

protein_llama_3b = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=2560,
    intermediate_dim=10240,
    num_heads=40,
    num_kv_heads=8,
    num_layers=32,
)

protein_model_3b_distance_masked = build_distance_masked_train_step(
    name="protein-contacts-3b-distance-masked",
    model_config=protein_llama_3b,
    learning_rate=3.5e-4,
    extra_tags=("3b",),
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_3b_distance_masked])
