# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Training-configuration checks for Grug pipeline checkpoints."""

import json

from levanter import mpmd_checkpoint

from experiments.grug.moe_pipeline.pipeline import GrugMoeAutomaticPipelineState


def save_checkpoint(root: str, state: GrugMoeAutomaticPipelineState, *, step: int, contract: dict) -> str:
    """Save pipeline state and the training configuration required to resume it."""
    return mpmd_checkpoint.save_checkpoint(root, state, step=step, metadata=contract)


def restore_checkpoint(
    root: str, state: GrugMoeAutomaticPipelineState, shardings, *, contract: dict
) -> tuple[GrugMoeAutomaticPipelineState, int]:
    """Resume only when the saved training configuration matches this run."""
    expected = json.loads(json.dumps(contract))

    def validate_metadata(metadata: dict) -> None:
        if metadata != expected:
            raise ValueError("Checkpoint training configuration does not match")

    return mpmd_checkpoint.restore_checkpoint(root, state, shardings, validate_metadata=validate_metadata)
