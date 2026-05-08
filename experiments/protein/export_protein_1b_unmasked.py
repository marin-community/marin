# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the 1B unmasked training run to HuggingFace format.

Targets the run produced by ``experiments/protein/train_protein_1b_unmasked.py``
(loss applied to every position, no distance-bin-only mask).
``CHECKPOINT_STEP`` is used only for the output directory name
(``hf/step-{CHECKPOINT_STEP}``); ``discover_latest=True`` resolves whichever
loadable checkpoint is current.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=32GB --disk=16GB --cpu=4 \\
        -- python -m experiments.protein.export_protein_1b_unmasked
"""

from experiments.protein.protein_train_common import build_hf_export_step
from experiments.protein.train_protein_1b_unmasked import (
    protein_llama_1b,
    protein_model_1b_unmasked,
)
from marin.execution.executor import executor_main

CHECKPOINT_STEP = 3500
CHECKPOINT_PATH = "gs://marin-us-east5/checkpoints/protein-contacts-1b-3.5e-4-unmasked-8efbcb/checkpoints"

hf_export = build_hf_export_step(
    train_step=protein_model_1b_unmasked,
    model_config=protein_llama_1b,
    checkpoint_step=CHECKPOINT_STEP,
    name_prefix="protein-contacts-1b-3.5e-4-unmasked",
    checkpoint_path_override=CHECKPOINT_PATH,
)


if __name__ == "__main__":
    executor_main(steps=[hf_export])
