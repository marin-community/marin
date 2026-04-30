# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Resume training of the `3.5e-4-distance-masked-7d355e` run.

This file reproduces the exact training config from the original
`train_protein_1b_distance_masked.py` at commit f162b00e3 (LR=3.5e-4, no
distogram benchmark wiring) and pins the output path to the existing run's
directory so levanter auto-resumes from the latest checkpoint.

The sibling file `train_protein_1b_distance_masked.py` has since been bumped to
LR=1.05e-3 and wired with the distogram benchmark; that's a different
experiment living at its own output path. This file is deliberately kept
separate so we don't disturb the surviving 3.5e-4 run while continuing its
training.

The inner TPU task is pinned to `us-east5-a` (same region as the checkpoint
bucket) via the ``RESOURCES`` config below.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=16GB --disk=16GB --cpu=1 --extra=tpu --zone=us-east5-a -- \\
        python -m experiments.protein.continue_train_protein_1b_distance_masked
"""

from levanter.data.text import LmDataConfig
from levanter.models.llama import LlamaConfig

from experiments.defaults import default_train
from experiments.protein.protein_train_common import (
    PROTEIN_RESOURCES_USE5,
    PROTEIN_TOKENIZER,
    distance_masked_components,
)
from experiments.simple_train_config import SimpleTrainConfig
from marin.execution.executor import executor_main, versioned

protein_llama_1b = LlamaConfig(
    max_seq_len=8192,
    hidden_dim=2048,
    intermediate_dim=8192,
    num_heads=32,
    num_kv_heads=8,
    num_layers=16,
)

protein_docs_data = LmDataConfig(
    components=distance_masked_components(),
    train_weights={"protein-docs-cd": 1.0, "protein-docs-cd-val": 0.0},
    tokenizer=PROTEIN_TOKENIZER,
    cache_dir=None,
    block_cross_document_attention=True,
)

train_config = SimpleTrainConfig(
    resources=PROTEIN_RESOURCES_USE5,
    train_batch_size=128,
    num_train_steps=50_000,
    learning_rate=versioned(3.5e-4),
    weight_decay=0.01,
    warmup=0.1,
    train_seq_len=8192,
    steps_per_eval=500,
    steps_per_export=5000,
    env_vars={"WANDB_ENTITY": "timodonnell"},
)

# Pin the output directory to the existing 7d355e dir so levanter's checkpointer
# finds the step-8740 checkpoint and resumes from there. Without this pin the
# config would be assigned a fresh hash suffix and we'd start from scratch.
EXISTING_OUTPUT = "gs://marin-us-east5/checkpoints/protein-contacts-1b-3.5e-4-distance-masked-7d355e"

protein_model_1b_distance_masked = default_train(
    name="protein-contacts-1b-3.5e-4-distance-masked",
    tokenized=protein_docs_data,
    model_config=protein_llama_1b,
    train_config=train_config,
    tags=["protein", "contacts-and-distances", "1b", "llama", "distance-masked"],
    eval_harness_tasks=[],
    use_default_validation=False,
    wandb_group="protein-training",
    wandb_name=None,
    override_output_path=EXISTING_OUTPUT,
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_1b_distance_masked])
