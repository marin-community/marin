# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the 3B distance-masked training run to HuggingFace format.

Targets the run produced by
``experiments/protein/train_protein_3b_distance_masked.py``. ``CHECKPOINT_STEP``
is used only for the output directory name (``hf/step-{CHECKPOINT_STEP}``);
``discover_latest=True`` resolves whichever loadable checkpoint is current.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=32GB --disk=16GB --cpu=4 \\
        -- python -m experiments.protein.export_protein_3b_distance_masked
"""

from experiments.protein.protein_train_common import build_hf_export_step
from experiments.protein.train_protein_3b_distance_masked import (
    protein_llama_3b,
    protein_model_3b_distance_masked,
)
from marin.execution.executor import executor_main

CHECKPOINT_STEP = 1000
CHECKPOINT_PATH = "gs://marin-us-east5/checkpoints/protein-contacts-3b-distance-masked-ef3aa5/checkpoints"

hf_export = build_hf_export_step(
    train_step=protein_model_3b_distance_masked,
    model_config=protein_llama_3b,
    checkpoint_step=CHECKPOINT_STEP,
    name_prefix="protein-contacts-3b-distance-masked",
    checkpoint_path_override=CHECKPOINT_PATH,
)


if __name__ == "__main__":
    executor_main(steps=[hf_export])
