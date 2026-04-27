# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the distance-masked 1.4B protein training run to HuggingFace format.

Targets the run at
``gs://marin-us-east5/checkpoints/protein-contacts-1b-3.5e-4-distance-masked-7d355e``
(wandb run `protein-contacts-1b-3.5e-4-distance-masked-7d355e`). Writes the HF
checkpoint to ``{TRAINING_OUTPUT}/hf/step-{CHECKPOINT_STEP}`` so multiple
checkpoints can coexist.

Usage::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=32GB --disk=16GB --cpu=4 \\
        -- python -m experiments.protein.export_protein_1b_distance_masked
"""

from copy import deepcopy

from levanter.trainer import TrainerConfig

from experiments.protein.train_protein_1b_distance_masked import (
    PROTEIN_TOKENIZER,
    protein_llama_1b,
    protein_model_1b_distance_masked,
)
from marin.execution.executor import executor_main
from marin.export import convert_checkpoint_to_hf_step

TRAINING_OUTPUT = "gs://marin-us-east5/checkpoints/protein-contacts-1b-3.5e-4-distance-masked-7d355e"
CHECKPOINT_STEP = 31337  # latest temp checkpoint with metadata.json as of 2026-04-27 13:16Z
# Permanent checkpoints in this run lack metadata.json (only the latest temp does), so
# we point at the parent dir and let levanter pick whatever loadable ckpt is current at job
# start. CHECKPOINT_STEP is used only for the output directory name.


def _trainer_from_training_step() -> TrainerConfig:
    trainer = protein_model_1b_distance_masked.config.train_config.trainer
    if not isinstance(trainer, TrainerConfig):
        raise TypeError(f"Expected TrainerConfig, got {type(trainer)!r}")
    return deepcopy(trainer)


hf_export = convert_checkpoint_to_hf_step(
    name=f"hf/protein-contacts-1b-3.5e-4-distance-masked-step-{CHECKPOINT_STEP}",
    checkpoint_path=f"{TRAINING_OUTPUT}/checkpoints",
    trainer=_trainer_from_training_step(),
    model=protein_llama_1b,
    tokenizer=PROTEIN_TOKENIZER,
    use_cpu=True,
    discover_latest=True,
    override_output_path=f"{TRAINING_OUTPUT}/hf/step-{CHECKPOINT_STEP}",
)


if __name__ == "__main__":
    executor_main(steps=[hf_export])
