# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the 100M unmasked training run to HuggingFace format.

Targets the run produced by ``experiments/protein/train_protein_100m_unmasked.py``
(loss applied to every position, no distance-bin-only mask). ``CHECKPOINT_STEP``
is used only for the output directory name (``hf/step-{CHECKPOINT_STEP}``);
``discover_latest=True`` resolves whichever loadable checkpoint is current.

``CHECKPOINT_PATH`` is a literal gs:// path so the export step doesn't depend
on the train step's marin-executor status — useful for snapshot exports while
training is still running.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=32GB --disk=16GB --cpu=4 \\
        -- python -m experiments.protein.export_protein_100m_unmasked
"""

from experiments.protein.protein_train_common import build_hf_export_step
from experiments.protein.train_protein_100m_unmasked import (
    protein_llama_100m,
    protein_model_100m_unmasked,
)
from marin.execution.executor import executor_main

CHECKPOINT_STEP = 50000
CHECKPOINT_PATH = "gs://marin-us-east5/checkpoints/protein-contacts-100m-3.5e-4-unmasked-7c3ef7/checkpoints"

hf_export = build_hf_export_step(
    train_step=protein_model_100m_unmasked,
    model_config=protein_llama_100m,
    checkpoint_step=CHECKPOINT_STEP,
    name_prefix="protein-contacts-100m-3.5e-4-unmasked",
    checkpoint_path_override=CHECKPOINT_PATH,
)


if __name__ == "__main__":
    executor_main(steps=[hf_export])
