# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a ~383M-param Llama protein model with the distance-masked loss.

Shape matches Pythia-410M (h=1024, l=24, dff=4096, heads=16). Recipe matches
``continue_train_protein_1b_distance_masked.py``: 50K steps, batch 128, LR
3.5e-4, distance-masked loss only.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a -- \\
        python -m experiments.protein.train_protein_400m_distance_masked
"""

from levanter.models.llama import LlamaConfig

from experiments.protein.protein_train_common import build_distance_masked_train_step
from marin.execution.executor import executor_main

protein_llama_400m = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=1024,
    intermediate_dim=4096,
    num_heads=16,
    num_kv_heads=8,
    num_layers=24,
)

protein_model_400m_distance_masked = build_distance_masked_train_step(
    name="protein-contacts-400m-distance-masked",
    model_config=protein_llama_400m,
    learning_rate=3.5e-4,
    extra_tags=("400m",),
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_400m_distance_masked])
