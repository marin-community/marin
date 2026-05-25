# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a ~1.47B-param Llama protein model with the distance-masked loss.

Shape matches Pythia-1.4B (h=2048, l=24, dff=8192, heads=32). Same hidden
width as the existing ``protein_llama_1b`` (l=16) — this run is the
fixed-width depth ablation against it.

Recipe matches ``continue_train_protein_1b_distance_masked.py``: 50K steps,
batch 128, LR 3.5e-4, distance-masked loss only.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a -- \\
        python -m experiments.protein.train_protein_1_5b_distance_masked
"""

from levanter.models.llama import LlamaConfig
from marin.execution.executor import executor_main

from experiments.protein.protein_train_common import build_distance_masked_train_step

protein_llama_1_5b = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=24,
)

# Pinned to the 70f8f5 output dir from the original launch. Without this, a
# checkpoint at step ~28933 was orphaned when something upstream in the marin
# executor's hash computation drifted and produced a new hash (ce18f8). The
# checkpoint is still at the 70f8f5 path; pin so we resume from it.
protein_model_1_5b_distance_masked = build_distance_masked_train_step(
    name="protein-contacts-1_5b-distance-masked",
    model_config=protein_llama_1_5b,
    learning_rate=3.5e-4,
    extra_tags=("1_5b",),
    override_output_path="gs://marin-us-east5/checkpoints/protein-contacts-1_5b-distance-masked-70f8f5",
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_1_5b_distance_masked])
