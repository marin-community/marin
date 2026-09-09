# Copyright The Marin Authors
# SPDX-License-Identifier: Apache-2.0

"""Validate compact optimizer-progress evidence without loading checkpoint tensors."""

from experiments.post_training import async_rl_audit as audit


def validate_progress(progress, *, training_sha256, trainer_state_sha256):
    """Verify a separately qualified saved-state progress receipt before export."""
    body = {key: value for key, value in progress.items() if key != "progress_sha256"}
    updates, minibatches = progress["optimizer_updates"], progress["minibatches"]
    if (
        progress["schema"] != "math_eval_checkpoint_progress_v1"
        or progress["progress_sha256"] != audit.canonical_sha(body)
        or progress["training_manifest_sha256"] != training_sha256
        or progress["trainer_state_sha256"] != trainer_state_sha256
        or type(updates) is not int
        or updates <= 0
        or type(minibatches) is not int
        or not 1 <= minibatches <= 16
        or updates % minibatches
        or progress["global_step"] != updates // minibatches
        or progress["train_batch_size"] != progress["policy_mini_batch_size"] * minibatches
    ):
        raise ValueError("Checkpoint progress receipt changed or belongs to another training run")
