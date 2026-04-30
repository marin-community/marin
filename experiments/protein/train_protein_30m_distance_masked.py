# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a ~28M-param Llama protein model with the distance-masked loss.

Smallest member of the protein-docs distance-masked size sweep. Shape matches
Pythia-70M (h=512, l=6, dff=2048, heads=8). Recipe matches
``continue_train_protein_1b_distance_masked.py``: 50K steps, batch 128, LR
3.5e-4, distance-masked loss only (no in-training distogram benchmark), TPU
pinned to ``us-east5-a``.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a -- \\
        python -m experiments.protein.train_protein_30m_distance_masked
"""

from levanter.models.llama import LlamaConfig

from experiments.protein.protein_train_common import build_distance_masked_train_step
from marin.execution.executor import executor_main

protein_llama_30m = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=512,
    intermediate_dim=2048,
    num_heads=8,
    num_kv_heads=8,
    num_layers=6,
)

protein_model_30m_distance_masked = build_distance_masked_train_step(
    name="protein-contacts-30m-distance-masked",
    model_config=protein_llama_30m,
    learning_rate=3.5e-4,
    extra_tags=("30m",),
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_30m_distance_masked])
