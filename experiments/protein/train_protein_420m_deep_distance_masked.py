# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Train a ~420M-param Llama protein model with a very deep aspect ratio.

Same hidden width (768) as the 100M model, but 4x the depth (48 layers vs 12).
Aspect ratio l/h = 0.0625 — meaningfully deeper than every other model in the
sweep (next deepest is the 400M at 0.0234, then the 30M at 0.0117).

Recipe matches ``continue_train_protein_1b_distance_masked.py`` except
``learning_rate`` is dropped from 3.5e-4 → 2.5e-4 (~0.7x) since deeper nets
are typically more sensitive to LR. Easy to bump if loss is uneventful, or
drop further if it diverges.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a -- \\
        python -m experiments.protein.train_protein_420m_deep_distance_masked
"""

from levanter.models.llama import LlamaConfig

from experiments.protein.protein_train_common import build_distance_masked_train_step
from marin.execution.executor import executor_main

protein_llama_420m_deep = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=768,
    intermediate_dim=3072,
    num_heads=12,
    num_kv_heads=4,
    num_layers=48,
)

protein_model_420m_deep_distance_masked = build_distance_masked_train_step(
    name="protein-contacts-420m-deep-distance-masked",
    model_config=protein_llama_420m_deep,
    learning_rate=2.5e-4,
    extra_tags=("420m", "deep"),
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_420m_deep_distance_masked])
