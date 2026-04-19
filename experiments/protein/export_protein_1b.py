# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Convert the trained protein-contacts-1b Levanter checkpoint to HuggingFace format.

The training run from `experiments/protein/train_protein_1b.py` writes a
Levanter-native (tensorstore/ocdbt) checkpoint at
``gs://marin-us-east5/checkpoints/protein-contacts-1b-2.5e-4-780930/checkpoints/step-22477``
but no HF export. vLLM needs an HF checkpoint to load, so this step materializes
one at ``.../hf``.

Runs on CPU only (the 1.4B model is small enough to load into host memory)::

    uv run iris --config=lib/iris/examples/marin.yaml job run \\
        --memory=8GB --disk=8GB --cpu=1 \\
        -e HF_TOKEN <your-hf-token> \\
        -- \\
        python -m experiments.protein.export_protein_1b
"""

from copy import deepcopy

from levanter.trainer import TrainerConfig

from experiments.protein.train_protein_1b import (
    PROTEIN_TOKENIZER,
    protein_llama_1b,
    protein_model_1b,
)
from marin.execution.executor import executor_main
from marin.export import convert_checkpoint_to_hf_step

TRAINING_OUTPUT = "gs://marin-us-east5/checkpoints/protein-contacts-1b-2.5e-4-780930"
CHECKPOINT_STEP = 22477


def _trainer_from_training_step() -> TrainerConfig:
    trainer = protein_model_1b.config.train_config.trainer
    if not isinstance(trainer, TrainerConfig):
        raise TypeError(f"Expected TrainerConfig, got {type(trainer)!r}")
    return deepcopy(trainer)


protein_model_1b_hf = convert_checkpoint_to_hf_step(
    name="hf/protein-contacts-1b",
    checkpoint_path=f"{TRAINING_OUTPUT}/checkpoints/step-{CHECKPOINT_STEP}",
    trainer=_trainer_from_training_step(),
    model=protein_llama_1b,
    tokenizer=PROTEIN_TOKENIZER,
    use_cpu=True,
    # Pin output to the path `eval_protein_contacts.py` expects, matching the
    # original training run's bucket layout.
    override_output_path=f"{TRAINING_OUTPUT}/hf",
)


if __name__ == "__main__":
    executor_main(steps=[protein_model_1b_hf])
